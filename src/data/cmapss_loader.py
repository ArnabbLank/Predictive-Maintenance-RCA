"""C-MAPSS dataset loader and preprocessor."""
import numpy as np
import pandas as pd
from pathlib import Path

class CMAPSSLoader:
    def __init__(self, data_dir="data/raw/cmapss"):
        self.data_dir = Path(data_dir)
        self.sensor_cols = [f"s{i}" for i in range(1, 22)]
        self.setting_cols = ["setting1", "setting2", "setting3"]
        
    def load_dataset(self, subset="FD001"):
        """Load train/test data for specified subset (FD001-FD004)."""
        train_path = self.data_dir / f"train_{subset}.txt"
        test_path = self.data_dir / f"test_{subset}.txt"
        rul_path = self.data_dir / f"RUL_{subset}.txt"
        
        cols = ["unit", "cycle"] + self.setting_cols + self.sensor_cols
        
        train = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
        test = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)
        rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])
        
        return self._add_rul(train), test, rul_true
    
    def _add_rul(self, df):
        """Add RUL column to training data."""
        max_cycles = df.groupby("unit")["cycle"].max().reset_index()
        max_cycles.columns = ["unit", "max_cycle"]
        df = df.merge(max_cycles, on="unit")
        df["rul"] = df["max_cycle"] - df["cycle"]
        df.drop("max_cycle", axis=1, inplace=True)
        return df
    
    def normalize(self, train, test):
        """Normalize sensor readings using train statistics."""
        feature_cols = self.setting_cols + self.sensor_cols
        mean = train[feature_cols].mean()
        std = train[feature_cols].std()
        
        train[feature_cols] = (train[feature_cols] - mean) / std
        test[feature_cols] = (test[feature_cols] - mean) / std
        
        return train, test, (mean, std)
