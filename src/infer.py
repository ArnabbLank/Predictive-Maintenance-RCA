"""
infer.py — Run inference on a single engine or batch of engines.

Provides a high-level API for:
  - Point prediction (any model)
  - Uncertainty quantification (CNN-Transformer with MC Dropout)
  - JSON output suitable for the Copilot agent
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_test
from preprocess import (
    INFORMATIVE_SENSORS_FD001,
    create_test_sequences,
    fit_scaler,
    apply_scaler,
    apply_savgol_smoothing,
)
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTMModel
from models.cnn_transformer import CNNTransformerModel


def load_model(
    model_type: str,
    checkpoint_path: str,
    n_features: int = 14,
    seq_len: int = 30,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    if model_type == "lstm":
        model = LSTMModel(n_features=n_features)
    elif model_type == "cnn_lstm":
        model = CNNLSTMModel(n_features=n_features, seq_len=seq_len)
    elif model_type == "cnn_transformer":
        model = CNNTransformerModel(n_features=n_features, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model


def predict_single(
    model: torch.nn.Module,
    engine_window: np.ndarray,
    device: str = "cpu",
    mc_samples: int = 0,
) -> dict:
    """
    Predict RUL for a single engine window.

    Parameters
    ----------
    engine_window : np.ndarray, shape (seq_len, n_features) or (1, seq_len, n_features)
    mc_samples : int
        If > 0 and model supports it, use MC Dropout.

    Returns
    -------
    dict with keys: rul_mean, rul_std (if MC), confidence_interval
    """
    if engine_window.ndim == 2:
        engine_window = engine_window[np.newaxis, :]

    x = torch.tensor(engine_window, dtype=torch.float32).to(device)

    if mc_samples > 0 and isinstance(model, CNNTransformerModel):
        mean, std, _ = model.predict_with_uncertainty(x, n_samples=mc_samples)
        return {
            "rul_mean": float(mean[0]),
            "rul_std": float(std[0]),
            "confidence_interval_95": [
                float(max(0, mean[0] - 1.96 * std[0])),
                float(mean[0] + 1.96 * std[0]),
            ],
            "uncertainty_level": _classify_uncertainty(std[0]),
        }
    else:
        model.eval()
        with torch.no_grad():
            pred = model(x).squeeze().item()
        return {
            "rul_mean": float(pred),
            "rul_std": None,
            "confidence_interval_95": None,
            "uncertainty_level": "unknown",
        }


def _classify_uncertainty(std: float) -> str:
    """Classify uncertainty level for the Copilot."""
    if std < 5:
        return "low"
    elif std < 15:
        return "medium"
    elif std < 30:
        return "high"
    else:
        return "very_high"


def main():
    parser = argparse.ArgumentParser(description="Run RUL inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "cnn_lstm", "cnn_transformer"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--engine_id", type=int, default=None,
                        help="Specific engine ID (None = all)")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--mc_samples", type=int, default=100)
    args = parser.parse_args()

    df_test, rul_true = load_test(fd_number=args.fd)

    # Quick scaling (fit on test data range for demo — in practice use train scaler)
    feature_cols = INFORMATIVE_SENSORS_FD001
    X_test = create_test_sequences(df_test, feature_cols, args.window)

    model = load_model(args.model, args.checkpoint, len(feature_cols), args.window)

    if args.engine_id is not None:
        idx = args.engine_id - 1
        result = predict_single(model, X_test[idx], mc_samples=args.mc_samples)
        result["engine_id"] = args.engine_id
        result["true_rul"] = int(rul_true[idx])
        print(json.dumps(result, indent=2))
    else:
        results = []
        for i in range(len(X_test)):
            r = predict_single(model, X_test[i], mc_samples=args.mc_samples)
            r["engine_id"] = i + 1
            r["true_rul"] = int(rul_true[i])
            results.append(r)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
