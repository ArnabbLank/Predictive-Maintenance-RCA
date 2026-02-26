"""
data_loader.py — Load and parse C-MAPSS dataset files.

The C-MAPSS dataset consists of:
  - train_FD00X.txt: Training data (run-to-failure)
  - test_FD00X.txt:  Test data (cut-off before failure)
  - RUL_FD00X.txt:   True RUL for the last cycle of each test engine

Each row has 26 columns (space-delimited, no header):
  [unit_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1..sensor_21]
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict

# ── Column names ──────────────────────────────────────────────────────────
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
INDEX_COLS = ["unit_id", "cycle"]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# ── Default data directory ────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "cmapss"


def load_train(
    fd_number: int = 1,
    data_dir: Optional[str] = None,
    rul_cap: Optional[int] = 125,
) -> pd.DataFrame:
    """
    Load a C-MAPSS training file and compute RUL labels.

    Parameters
    ----------
    fd_number : int
        Sub-dataset number (1–4). Default is 1 (FD001).
    data_dir : str or None
        Path to data directory. Defaults to project ``data/`` folder.
    rul_cap : int or None
        If set, cap the RUL label at this value (piece-wise linear).
        Common values: 125 (most papers) or None (raw linear).

    Returns
    -------
    pd.DataFrame
        Training data with columns: unit_id, cycle, op_settings, sensors, RUL.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    filepath = data_dir / f"train_FD{fd_number:03d}.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Training file not found: {filepath}\n"
            f"Run `bash data/download_data.sh` to download the dataset."
        )

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=ALL_COLS,
        index_col=False,
    )

    # Compute RUL = max_cycle_for_unit - current_cycle
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    # Optional RUL capping (piece-wise linear)
    if rul_cap is not None:
        df["RUL"] = df["RUL"].clip(upper=rul_cap)

    return df


def load_test(
    fd_number: int = 1,
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load a C-MAPSS test file and the corresponding true RUL values.

    Parameters
    ----------
    fd_number : int
        Sub-dataset number (1–4).
    data_dir : str or None
        Path to data directory.

    Returns
    -------
    df_test : pd.DataFrame
        Test data (same columns as training, but no RUL column).
    rul_true : np.ndarray
        1-D array of true RUL values for the last cycle of each unit.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    test_path = data_dir / f"test_FD{fd_number:03d}.txt"
    rul_path = data_dir / f"RUL_FD{fd_number:03d}.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")

    df_test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=ALL_COLS,
        index_col=False,
    )

    rul_true = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )["RUL"].values

    return df_test, rul_true


def load_all(
    fd_number: int = 1,
    data_dir: Optional[str] = None,
    rul_cap: Optional[int] = 125,
) -> Dict[str, object]:
    """
    Convenience function to load train, test, and RUL labels together.

    Returns
    -------
    dict with keys: 'train', 'test', 'rul_true'
    """
    df_train = load_train(fd_number, data_dir, rul_cap)
    df_test, rul_true = load_test(fd_number, data_dir)
    return {
        "train": df_train,
        "test": df_test,
        "rul_true": rul_true,
    }


def get_dataset_info(fd_number: int = 1, data_dir: Optional[str] = None) -> dict:
    """
    Quick summary stats for a given sub-dataset.
    """
    data = load_all(fd_number, data_dir)
    df_train = data["train"]
    df_test = data["test"]
    rul_true = data["rul_true"]

    info = {
        "fd_number": fd_number,
        "train_engines": df_train["unit_id"].nunique(),
        "train_rows": len(df_train),
        "train_max_cycle": df_train.groupby("unit_id")["cycle"].max().describe().to_dict(),
        "test_engines": df_test["unit_id"].nunique(),
        "test_rows": len(df_test),
        "rul_true_stats": {
            "min": int(rul_true.min()),
            "max": int(rul_true.max()),
            "mean": float(rul_true.mean()),
            "median": float(np.median(rul_true)),
        },
        "sensor_columns": SENSOR_COLS,
        "setting_columns": SETTING_COLS,
    }
    return info


# ── Main (quick sanity check) ────────────────────────────────────────────
if __name__ == "__main__":
    import json

    info = get_dataset_info(fd_number=1)
    print(json.dumps(info, indent=2, default=str))
    print(f"\n✅ FD001 loaded successfully: "
          f"{info['train_engines']} train engines, "
          f"{info['test_engines']} test engines")
