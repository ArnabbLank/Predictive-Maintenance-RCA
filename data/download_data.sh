#!/usr/bin/env bash
# ============================================================
# download_data.sh — Download C-MAPSS dataset into data/
# ============================================================
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_FILE="$DATA_DIR/CMAPSSData.zip"
ZENODO_URL="https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1"
NASA_URL="https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

echo "=== C-MAPSS Data Downloader ==="

# Check if data already exists
if [ -f "$DATA_DIR/train_FD001.txt" ]; then
    echo "Data already exists in $DATA_DIR — skipping download."
    exit 0
fi

# Try Zenodo first (more stable), then NASA
echo "Downloading from Zenodo…"
if curl -L -o "$ZIP_FILE" "$ZENODO_URL" --connect-timeout 30 --max-time 300; then
    echo "Download complete from Zenodo."
elif curl -L -o "$ZIP_FILE" "$NASA_URL" --connect-timeout 30 --max-time 300; then
    echo "Download complete from NASA."
else
    echo "ERROR: Could not download from either source."
    echo "Please manually download CMAPSSData.zip and place it in: $DATA_DIR"
    exit 1
fi

# Unzip
echo "Extracting…"
unzip -o "$ZIP_FILE" -d "$DATA_DIR"
echo "Done. Files in $DATA_DIR:"
ls -la "$DATA_DIR"/*.txt
