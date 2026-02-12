#!/usr/bin/env python3
"""
Download NASA C-MAPSS dataset

Run this script to download the turbofan engine degradation dataset.
"""

import urllib.request
import zipfile
from pathlib import Path
import sys


def download_cmapss():
    """Download and extract C-MAPSS dataset"""
    
    data_dir = Path("data/raw/cmapss")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_dir / "CMAPSSData.zip"
    
    # Try multiple sources
    urls = [
        "https://data.nasa.gov/download/brfb-gzcv/application%2Fzip",
        "https://phmsociety.org/wp-content/uploads/2024/10/CMAPSSData.zip"
    ]
    
    print("Downloading NASA C-MAPSS dataset...")
    
    for url in urls:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, zip_path)
            print("✓ Download complete")
            break
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    else:
        print("\n❌ All download sources failed.")
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("2. Download CMAPSSData.zip")
        print(f"3. Extract to: {data_dir.absolute()}")
        sys.exit(1)
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    zip_path.unlink()
    print("✓ Extraction complete")
    
    # Verify files
    expected_files = [
        "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
        "test_FD001.txt", "test_FD002.txt", "test_FD003.txt", "test_FD004.txt",
        "RUL_FD001.txt", "RUL_FD002.txt", "RUL_FD003.txt", "RUL_FD004.txt"
    ]
    
    missing = [f for f in expected_files if not (data_dir / f).exists()]
    
    if missing:
        print(f"\n⚠️  Warning: Missing files: {missing}")
    else:
        print("\n✓ All dataset files present")
        print(f"Dataset location: {data_dir.absolute()}")


if __name__ == "__main__":
    download_cmapss()
