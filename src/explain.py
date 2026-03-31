"""
explain.py — Model explanations: saliency maps, attention, feature importance.

Provides:
  - Gradient-based saliency for deep models (LSTM, CNN-LSTM, CNN-Transformer)
  - Attention weight extraction (for Transformer)
  - SHAP / permutation importance for baseline models
  - Standardized explanation output schema
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Standardized Output Schema ────────────────────────────────────────────
def format_explanation(
    sensor_names: List[str],
    attribution_scores: np.ndarray,
    top_k: int = 5,
    model_name: str = "",
    method: str = "saliency",
    notes: str = "",
) -> dict:
    """
    Standardized explanation schema used by the Copilot.

    Returns
    -------
    dict with keys:
        top_sensors, attribution_scores, method, notes
    """
    # Ensure scores are 1D (aggregate over time if needed)
    if attribution_scores.ndim > 1:
        attribution_scores = np.mean(np.abs(attribution_scores), axis=0)

    # Rank sensors
    ranked_idx = np.argsort(-np.abs(attribution_scores))[:top_k]
    top_sensors = [
        {
            "sensor": sensor_names[i],
            "importance": float(attribution_scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(ranked_idx)
    ]

    return {
        "top_sensors": top_sensors,
        "all_attributions": {
            sensor_names[i]: float(attribution_scores[i])
            for i in range(len(sensor_names))
        },
        "method": method,
        "model": model_name,
        "notes": notes,
    }


# ── Gradient-based Saliency ──────────────────────────────────────────────
def compute_saliency(
    model: nn.Module,
    x: np.ndarray,
    sensor_names: List[str],
    device: str = "cpu",
    top_k: int = 5,
) -> dict:
    """
    Compute gradient-based saliency for a single input.

    Parameters
    ----------
    x : np.ndarray, shape (seq_len, n_features) or (1, seq_len, n_features)

    Returns
    -------
    Standardized explanation dict.
    """
    model.eval()
    model = model.to(device)

    if x.ndim == 2:
        x = x[np.newaxis, :]

    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)

    # Forward
    pred = model(x_tensor).squeeze()  # ensure scalar for .backward()
    pred.backward()

    # Gradient w.r.t. input
    grads = x_tensor.grad.detach().cpu().numpy()[0]  # (seq_len, n_features)

    # Aggregate: mean absolute gradient per feature (across time)
    feature_importance = np.mean(np.abs(grads), axis=0)

    return format_explanation(
        sensor_names=sensor_names,
        attribution_scores=feature_importance,
        top_k=top_k,
        method="gradient_saliency",
        notes="Averaged absolute gradients over the time window.",
    )


# ── Integrated Gradients (more robust) ───────────────────────────────────
def compute_integrated_gradients(
    model: nn.Module,
    x: np.ndarray,
    sensor_names: List[str],
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50,
    device: str = "cpu",
    top_k: int = 5,
) -> dict:
    """
    Integrated Gradients attribution method.

    Better than vanilla saliency — integrates gradients along a path
    from a baseline (zeros) to the actual input.
    """
    model.eval()
    model = model.to(device)

    if x.ndim == 2:
        x = x[np.newaxis, :]
    if baseline is None:
        baseline = np.zeros_like(x)

    # Interpolation
    alphas = np.linspace(0, 1, n_steps + 1)
    grads_sum = np.zeros_like(x)

    for alpha in alphas:
        interpolated = baseline + alpha * (x - baseline)
        inp = torch.tensor(interpolated, dtype=torch.float32, requires_grad=True).to(device)
        pred = model(inp).squeeze()  # ensure scalar for .backward()
        pred.backward()
        grads_sum += inp.grad.detach().cpu().numpy()

    # Average and multiply by (input - baseline)
    avg_grads = grads_sum / (n_steps + 1)
    ig = (x - baseline) * avg_grads  # (1, seq_len, n_features)
    ig = ig[0]  # (seq_len, n_features)

    feature_importance = np.mean(np.abs(ig), axis=0)

    return format_explanation(
        sensor_names=sensor_names,
        attribution_scores=feature_importance,
        top_k=top_k,
        method="integrated_gradients",
        notes=f"Integrated gradients with {n_steps} steps from zero baseline.",
    )


# ── Temporal Saliency (importance per time step) ─────────────────────────
def compute_temporal_saliency(
    model: nn.Module,
    x: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute importance of each time step (not each feature).
    Returns shape (seq_len,).
    """
    model.eval()
    if x.ndim == 2:
        x = x[np.newaxis, :]

    x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    pred = model(x_tensor).squeeze()  # ensure scalar for .backward()
    pred.backward()

    grads = x_tensor.grad.detach().cpu().numpy()[0]  # (seq_len, n_features)
    temporal_importance = np.mean(np.abs(grads), axis=1)  # (seq_len,)
    return temporal_importance


# ── SHAP Wrapper (for sklearn models) ────────────────────────────────────
def compute_shap_importance(
    model,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    top_k: int = 5,
) -> dict:
    """
    SHAP TreeExplainer for sklearn tree-based models.

    Parameters
    ----------
    model : sklearn estimator  (RandomForest, GradientBoosting, etc.)
    X_train : array for background samples
    X_explain : array of samples to explain (can be single row)
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    if X_explain.ndim == 1 or len(X_explain) == 1:
        importance = np.abs(shap_values).flatten()
    else:
        importance = np.mean(np.abs(shap_values), axis=0)

    return format_explanation(
        sensor_names=feature_names,
        attribution_scores=importance,
        top_k=top_k,
        method="shap",
        notes="SHAP TreeExplainer values.",
    )


# ── Permutation Importance (model-agnostic) ──────────────────────────────
def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    top_k: int = 5,
) -> dict:
    """
    Permutation importance (works with any sklearn model).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42)
    importance = result.importances_mean

    return format_explanation(
        sensor_names=feature_names,
        attribution_scores=importance,
        top_k=top_k,
        method="permutation_importance",
        notes=f"Permutation importance ({n_repeats} repeats).",
    )
