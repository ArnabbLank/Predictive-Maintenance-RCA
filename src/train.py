"""
train.py — Unified training script for all PyTorch models.

Supports: MLP, LSTM, CNN+LSTM, CNN-Transformer.
Handles: early stopping, learning rate scheduling, checkpointing, logging.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_train, load_test
from preprocess import preprocess_pipeline, INFORMATIVE_SENSORS_FD001
from models.lstm import LSTMModel
from models.cnn_lstm import CNNLSTMModel
from models.cnn_transformer import CNNTransformerModel


# ── Evaluation metrics ────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and NASA scoring function."""
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # NASA scoring function (asymmetric — penalizes late predictions more)
    scores = []
    for e in errors:
        if e < 0:
            scores.append(np.exp(-e / 13.0) - 1)
        else:
            scores.append(np.exp(e / 10.0) - 1)
    nasa_score = np.sum(scores)

    return {"MAE": mae, "RMSE": rmse, "NASA_Score": nasa_score}


# ── Training loop ─────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    save_dir: str = "checkpoints",
    model_name: str = "model",
    device: str = "auto",
) -> dict:
    """
    Train a PyTorch model with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    n_epochs : int
    lr : float
    patience : int
        Early stopping patience (epochs without improvement).
    save_dir : str
        Directory to save checkpoints.
    model_name : str
        Name prefix for saved files.
    device : str
        'auto', 'cpu', 'cuda', or 'mps'.

    Returns
    -------
    dict with training history and best metrics.
    """
    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Training on: {device}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, n_epochs + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch).squeeze(-1)
                val_losses.append(criterion(pred, y_batch).item())
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        metrics = compute_metrics(all_targets, all_preds)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(metrics["MAE"])
        history["val_rmse"].append(metrics["RMSE"])

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val MAE: {metrics['MAE']:.2f} | "
                f"Val RMSE: {metrics['RMSE']:.2f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    best_path = os.path.join(save_dir, f"{model_name}_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # Save history
    hist_path = os.path.join(save_dir, f"{model_name}_history.json")
    # Convert numpy types to native Python for JSON serialization
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with open(hist_path, "w") as f:
        json.dump(history_serializable, f, indent=2)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "device": device,
    }


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "cnn_lstm", "cnn_transformer"],
                        help="Model architecture")
    parser.add_argument("--fd", type=int, default=1, help="FD subset (1-4)")
    parser.add_argument("--window", type=int, default=30, help="Window size")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stop patience")
    parser.add_argument("--use_savgol", action="store_true", help="Use SG smoothing")
    parser.add_argument("--rul_cap", type=int, default=125, help="RUL cap value")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"=== Training {args.model} on FD{args.fd:03d} ===")

    # Load data
    df_train = load_train(fd_number=args.fd, rul_cap=args.rul_cap)
    df_test, rul_true = load_test(fd_number=args.fd)

    # Preprocess
    data = preprocess_pipeline(
        df_train, df_test,
        window_size=args.window,
        rul_cap=args.rul_cap,
        use_savgol=args.use_savgol,
    )

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(data["X_train"]), torch.tensor(data["y_train"])
    )
    val_ds = TensorDataset(
        torch.tensor(data["X_val"]), torch.tensor(data["y_val"])
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    n_features = data["config"]["n_features"]
    if args.model == "lstm":
        model = LSTMModel(n_features=n_features)
    elif args.model == "cnn_lstm":
        model = CNNLSTMModel(n_features=n_features, seq_len=args.window)
    elif args.model == "cnn_transformer":
        model = CNNTransformerModel(n_features=n_features, seq_len=args.window)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        save_dir=args.save_dir,
        model_name=f"{args.model}_FD{args.fd:03d}",
        device=args.device,
    )

    print(f"\n✅ Training complete. Best val loss: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
