"""
eval.py — Evaluate trained models on test data. Compute metrics and generate plots.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_train, load_test
from preprocess import preprocess_pipeline
from train import compute_metrics
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTMModel
from models.cnn_transformer import CNNTransformerModel


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    rul_true: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Evaluate a model on the test set."""
    model.eval()
    model = model.to(device)

    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_tensor).squeeze(-1).cpu().numpy()

    metrics = compute_metrics(rul_true, preds)
    metrics["predictions"] = preds.tolist()
    return metrics


def evaluate_with_uncertainty(
    model: CNNTransformerModel,
    X_test: np.ndarray,
    rul_true: np.ndarray,
    n_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate CNN-Transformer with MC Dropout uncertainty."""
    model = model.to(device)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    mean_pred, std_pred, all_preds = model.predict_with_uncertainty(
        X_tensor, n_samples=n_samples
    )

    metrics = compute_metrics(rul_true, mean_pred)
    metrics["predictions"] = mean_pred.tolist()
    metrics["uncertainties"] = std_pred.tolist()

    # Uncertainty-error correlation
    abs_errors = np.abs(mean_pred - rul_true)
    correlation = np.corrcoef(std_pred, abs_errors)[0, 1]
    metrics["uncertainty_error_correlation"] = float(correlation)

    return metrics


def plot_predictions(
    rul_true: np.ndarray,
    rul_pred: np.ndarray,
    uncertainties: np.ndarray = None,
    title: str = "RUL Prediction vs True",
    save_path: str = None,
):
    """Plot predicted vs true RUL with optional uncertainty bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(rul_true, rul_pred, alpha=0.6, s=20, edgecolors="none")
    lims = [0, max(rul_true.max(), rul_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{title} — Scatter")
    ax.legend()

    # Bar chart per engine
    ax = axes[1]
    n = len(rul_true)
    x = np.arange(n)
    ax.bar(x - 0.15, rul_true, width=0.3, label="True RUL", alpha=0.8)
    ax.bar(x + 0.15, rul_pred, width=0.3, label="Predicted RUL", alpha=0.8)
    if uncertainties is not None:
        ax.errorbar(
            x + 0.15, rul_pred, yerr=1.96 * uncertainties,
            fmt="none", ecolor="red", alpha=0.5, capsize=2, label="95% CI"
        )
    ax.set_xlabel("Engine Unit")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{title} — Per Engine")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate RUL model on test data")
    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "cnn_lstm", "cnn_transformer"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt file")
    parser.add_argument("--fd", type=int, default=1)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--use_savgol", action="store_true")
    parser.add_argument("--rul_cap", type=int, default=125)
    parser.add_argument("--mc_samples", type=int, default=100,
                        help="MC Dropout samples (cnn_transformer only)")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()

    # Load and preprocess
    df_train = load_train(fd_number=args.fd, rul_cap=args.rul_cap)
    df_test, rul_true = load_test(fd_number=args.fd)

    data = preprocess_pipeline(
        df_train, df_test,
        window_size=args.window,
        rul_cap=args.rul_cap,
        use_savgol=args.use_savgol,
    )

    n_features = data["config"]["n_features"]

    # Load model
    if args.model == "lstm":
        model = LSTMModel(n_features=n_features)
    elif args.model == "cnn_lstm":
        model = CNNLSTMModel(n_features=n_features, seq_len=args.window)
    elif args.model == "cnn_transformer":
        model = CNNTransformerModel(n_features=n_features, seq_len=args.window)

    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # Evaluate
    if args.model == "cnn_transformer":
        results = evaluate_with_uncertainty(
            model, data["X_test"], rul_true, n_samples=args.mc_samples
        )
    else:
        results = evaluate_model(model, data["X_test"], rul_true)

    print(f"\n=== {args.model} on FD{args.fd:03d} ===")
    print(f"MAE:        {results['MAE']:.2f}")
    print(f"RMSE:       {results['RMSE']:.2f}")
    print(f"NASA Score: {results['NASA_Score']:.2f}")
    if "uncertainty_error_correlation" in results:
        print(f"Uncertainty-Error Correlation: {results['uncertainty_error_correlation']:.3f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"{args.model}_FD{args.fd:03d}_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
