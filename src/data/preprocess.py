"""
preprocess.py — Windowing, scaling, smoothing, and train/test preparation.

Provides:
  - MinMax / StandardScaler wrappers (fit on train, transform test)
  - Savitzky-Golay smoothing (Paper A idea)
  - Sliding-window generator for sequence models
  - Train/validation split by engine unit
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from pathlib import Path


# ── Sensor selection ──────────────────────────────────────────────────────
# Sensors commonly dropped in literature (near-constant for FD001):
#   sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
DROP_SENSORS_FD001 = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

# Informative sensors (14 remaining)
INFORMATIVE_SENSORS_FD001 = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def select_sensors(
    df: pd.DataFrame,
    drop_list: Optional[List[str]] = None,
    keep_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select sensor columns. Provide either drop_list OR keep_list, not both.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if keep_list is not None:
        drop_cols = [c for c in sensor_cols if c not in keep_list]
    elif drop_list is not None:
        drop_cols = [c for c in sensor_cols if c in drop_list]
    else:
        return df
    return df.drop(columns=drop_cols, errors="ignore")


# ── Scaling ───────────────────────────────────────────────────────────────
def fit_scaler(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    method: str = "minmax",
    save_path: Optional[str] = None,
) -> object:
    """
    Fit a scaler on training data.

    Parameters
    ----------
    method : str
        'minmax' (scales to [0,1]) or 'standard' (zero-mean, unit-var).
    save_path : str or None
        If given, save the fitted scaler with joblib.

    Returns
    -------
    Fitted scaler object.
    """
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler method: {method}")

    scaler.fit(df_train[feature_cols])

    if save_path:
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to {save_path}")

    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: object,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a fitted scaler to a dataframe (in-place on feature_cols).
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


# ── Savitzky-Golay Smoothing (Paper A) ───────────────────────────────────
def apply_savgol_smoothing(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_length: int = 11,
    polyorder: int = 3,
) -> pd.DataFrame:
    """
    Apply Savitzky-Golay filter per engine unit per sensor.

    This is inspired by Paper A:
      "A method for predicting remaining useful life using enhanced
       Savitzky-Golay filter and improved deep learning framework"

    Parameters
    ----------
    window_length : int
        Must be odd. Default 11 (adjustable).
    polyorder : int
        Polynomial order. Default 3.

    Returns
    -------
    DataFrame with smoothed sensor values.
    """
    df = df.copy()
    for unit_id in df["unit_id"].unique():
        mask = df["unit_id"] == unit_id
        unit_data = df.loc[mask, sensor_cols]

        for col in sensor_cols:
            series = unit_data[col].values
            # Only apply if series is long enough
            if len(series) >= window_length:
                df.loc[mask, col] = savgol_filter(
                    series, window_length=window_length, polyorder=polyorder
                )
            # else: keep raw values

    return df


# ── Sliding Window Generator ─────────────────────────────────────────────
def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
    target_col: str = "RUL",
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for each engine unit.

    For each unit, a window of `window_size` consecutive cycles is slid
    across the time series. The target is the RUL at the last cycle
    in the window.

    Parameters
    ----------
    window_size : int
        Number of time steps per input sequence.
    stride : int
        Step size between consecutive windows. Default 1.

    Returns
    -------
    X : np.ndarray, shape (N, window_size, n_features)
    y : np.ndarray, shape (N,)
    """
    X_list, y_list = [], []

    for unit_id in df["unit_id"].unique():
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")
        features = unit_df[feature_cols].values
        targets = unit_df[target_col].values

        n_steps = len(features)
        if n_steps < window_size:
            continue

        for i in range(0, n_steps - window_size + 1, stride):
            X_list.append(features[i : i + window_size])
            y_list.append(targets[i + window_size - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def create_test_sequences(
    df_test: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
) -> np.ndarray:
    """
    Create one sequence per test engine: the LAST `window_size` cycles.

    Returns
    -------
    X_test : np.ndarray, shape (n_engines, window_size, n_features)
    """
    X_list = []
    for unit_id in sorted(df_test["unit_id"].unique()):
        unit_df = df_test[df_test["unit_id"] == unit_id].sort_values("cycle")
        features = unit_df[feature_cols].values

        if len(features) >= window_size:
            X_list.append(features[-window_size:])
        else:
            # Pad with first row if too short
            pad_len = window_size - len(features)
            padding = np.tile(features[0], (pad_len, 1))
            X_list.append(np.vstack([padding, features]))

    return np.array(X_list, dtype=np.float32)


# ── Train/Val Split ──────────────────────────────────────────────────────
def train_val_split(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data by engine units (not by rows).

    This prevents data leakage — all cycles of one engine stay together.
    """
    unit_ids = df["unit_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unit_ids)

    n_val = max(1, int(len(unit_ids) * val_fraction))
    val_units = set(unit_ids[:n_val])

    df_val = df[df["unit_id"].isin(val_units)].copy()
    df_train = df[~df["unit_id"].isin(val_units)].copy()

    return df_train, df_val


# ── Full preprocessing pipeline ──────────────────────────────────────────
def preprocess_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    window_size: int = 30,
    rul_cap: Optional[int] = 125,
    use_savgol: bool = False,
    scaler_method: str = "minmax",
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    End-to-end preprocessing: sensor selection → scaling → smoothing → windowing.

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test,
        scaler, feature_cols, config
    """
    if feature_cols is None:
        feature_cols = INFORMATIVE_SENSORS_FD001

    # Cap RUL if present
    if "RUL" in df_train.columns and rul_cap is not None:
        df_train = df_train.copy()
        df_train["RUL"] = df_train["RUL"].clip(upper=rul_cap)

    # Train/val split (by engine)
    df_tr, df_vl = train_val_split(df_train, val_fraction, random_state)

    # Fit scaler on training split
    scaler = fit_scaler(df_tr, feature_cols, method=scaler_method)
    df_tr = apply_scaler(df_tr, scaler, feature_cols)
    df_vl = apply_scaler(df_vl, scaler, feature_cols)
    df_test_scaled = apply_scaler(df_test, scaler, feature_cols)

    # Optional Savitzky-Golay smoothing
    if use_savgol:
        df_tr = apply_savgol_smoothing(df_tr, feature_cols)
        df_vl = apply_savgol_smoothing(df_vl, feature_cols)
        df_test_scaled = apply_savgol_smoothing(df_test_scaled, feature_cols)

    # Create sequences
    X_train, y_train = create_sequences(df_tr, feature_cols, window_size)
    X_val, y_val = create_sequences(df_vl, feature_cols, window_size)
    X_test = create_test_sequences(df_test_scaled, feature_cols, window_size)

    config = {
        "window_size": window_size,
        "rul_cap": rul_cap,
        "use_savgol": use_savgol,
        "scaler_method": scaler_method,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "config": config,
    }


if __name__ == "__main__":
    from data_loader import load_train, load_test

    df_train = load_train(fd_number=1, rul_cap=125)
    df_test, rul_true = load_test(fd_number=1)

    result = preprocess_pipeline(df_train, df_test, window_size=30)

    print(f"X_train shape: {result['X_train'].shape}")
    print(f"y_train shape: {result['y_train'].shape}")
    print(f"X_val shape:   {result['X_val'].shape}")
    print(f"y_val shape:   {result['y_val'].shape}")
    print(f"X_test shape:  {result['X_test'].shape}")
    print(f"Config: {result['config']}")
