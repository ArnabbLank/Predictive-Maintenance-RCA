"""
features.py — Hand-crafted feature engineering for baseline ML models.

Converts raw time-series per engine into a single feature vector
(tabular format) suitable for scikit-learn models.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def _rolling_stats(
    unit_df: pd.DataFrame,
    sensor_cols: List[str],
    window: int,
) -> dict:
    """Compute rolling statistics over the last `window` cycles."""
    tail = unit_df.tail(window)
    features = {}
    for col in sensor_cols:
        vals = tail[col].values
        features[f"{col}_last"] = vals[-1]
        features[f"{col}_mean"] = np.mean(vals)
        features[f"{col}_std"] = np.std(vals) if len(vals) > 1 else 0.0
        features[f"{col}_min"] = np.min(vals)
        features[f"{col}_max"] = np.max(vals)

        # Linear trend (slope) via polyfit
        if len(vals) >= 2:
            x = np.arange(len(vals))
            slope = np.polyfit(x, vals, 1)[0]
            features[f"{col}_slope"] = slope
        else:
            features[f"{col}_slope"] = 0.0

    return features


def _exponential_weighted_features(
    unit_df: pd.DataFrame,
    sensor_cols: List[str],
    span: int = 10,
) -> dict:
    """Compute exponential weighted mean and std."""
    features = {}
    for col in sensor_cols:
        ewm = unit_df[col].ewm(span=span, adjust=False)
        features[f"{col}_ewm_mean"] = ewm.mean().iloc[-1]
        features[f"{col}_ewm_std"] = ewm.std().iloc[-1] if len(unit_df) > 1 else 0.0
    return features


def _rate_of_change(
    unit_df: pd.DataFrame,
    sensor_cols: List[str],
    periods: int = 5,
) -> dict:
    """Compute rate of change (diff) over last `periods` cycles."""
    features = {}
    for col in sensor_cols:
        if len(unit_df) >= periods + 1:
            diff = unit_df[col].iloc[-1] - unit_df[col].iloc[-1 - periods]
            features[f"{col}_roc_{periods}"] = diff
        else:
            features[f"{col}_roc_{periods}"] = 0.0
    return features


def extract_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window: int = 30,
    include_ewm: bool = True,
    include_roc: bool = True,
) -> pd.DataFrame:
    """
    Extract tabular features for each engine unit.

    For each unit, takes the LAST `window` cycles and computes:
      - Last value, mean, std, min, max, slope per sensor
      - (optional) Exponential weighted mean/std
      - (optional) Rate of change over last 5 cycles
      - Operational metadata: total cycles, max cycle

    Returns
    -------
    pd.DataFrame with one row per engine unit.
    """
    records = []
    for unit_id in sorted(df["unit_id"].unique()):
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")

        feat = {"unit_id": unit_id}
        feat["total_cycles"] = len(unit_df)
        feat["max_cycle"] = unit_df["cycle"].max()

        # Rolling stats over last window cycles
        feat.update(_rolling_stats(unit_df, sensor_cols, window))

        # Exponential weighted features
        if include_ewm:
            feat.update(_exponential_weighted_features(unit_df, sensor_cols))

        # Rate of change
        if include_roc:
            feat.update(_rate_of_change(unit_df, sensor_cols, periods=5))

        # RUL label (last cycle's RUL)
        if "RUL" in unit_df.columns:
            feat["RUL"] = unit_df["RUL"].iloc[-1]

        records.append(feat)

    return pd.DataFrame(records)


def extract_windowed_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window: int = 30,
    stride: int = 10,
) -> pd.DataFrame:
    """
    Sliding-window feature extraction (multiple samples per engine).

    Unlike `extract_features` which produces ONE row per engine,
    this creates multiple rows using a sliding window approach.
    """
    records = []
    for unit_id in sorted(df["unit_id"].unique()):
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle").reset_index(drop=True)

        if len(unit_df) < window:
            continue

        for start in range(0, len(unit_df) - window + 1, stride):
            window_df = unit_df.iloc[start : start + window]
            feat = {"unit_id": unit_id, "window_end_cycle": window_df["cycle"].iloc[-1]}
            feat.update(_rolling_stats(window_df, sensor_cols, window))

            if "RUL" in window_df.columns:
                feat["RUL"] = window_df["RUL"].iloc[-1]

            records.append(feat)

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_train
    from preprocess import INFORMATIVE_SENSORS_FD001

    df = load_train(fd_number=1, rul_cap=125)
    feats = extract_features(df, INFORMATIVE_SENSORS_FD001, window=30)
    print(f"Feature matrix shape: {feats.shape}")
    print(f"Columns ({len(feats.columns)}): {list(feats.columns[:10])}...")
    print(feats.head())
