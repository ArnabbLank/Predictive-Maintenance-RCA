"""
copilot/tools.py — Copilot tool functions.

Each tool is a standalone function that the agent can call.
Tools follow a standardized input/output schema.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer import predict_single, load_model
from explain import compute_saliency, compute_integrated_gradients
from copilot.retriever import retrieve_kb


# ── Tool 1: Predict RUL ──────────────────────────────────────────────────
def predict_rul(
    engine_window: np.ndarray,
    model=None,
    model_type: str = "cnn_transformer",
    checkpoint_path: str = None,
    mc_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Predict Remaining Useful Life for an engine.

    Output schema:
    {
        "rul_mean": float,
        "rul_std": float or None,
        "confidence_interval_95": [float, float] or None,
        "uncertainty_level": str ("low"/"medium"/"high"/"very_high"),
    }
    """
    if model is None:
        n_features = engine_window.shape[-1]
        model = load_model(model_type, checkpoint_path, n_features, device=device)

    result = predict_single(model, engine_window, device=device, mc_samples=mc_samples)
    return result


# ── Tool 2: Explain Prediction ───────────────────────────────────────────
def explain_prediction(
    engine_window: np.ndarray,
    model=None,
    sensor_names: list = None,
    method: str = "integrated_gradients",
    device: str = "cpu",
    top_k: int = 5,
) -> dict:
    """
    Explain what drove the RUL prediction.

    Output schema:
    {
        "top_sensors": [{"sensor": str, "importance": float, "rank": int}, ...],
        "method": str,
        "notes": str,
    }
    """
    if sensor_names is None:
        from preprocess import INFORMATIVE_SENSORS_FD001
        sensor_names = INFORMATIVE_SENSORS_FD001

    if method == "integrated_gradients":
        return compute_integrated_gradients(
            model, engine_window, sensor_names, device=device, top_k=top_k
        )
    else:
        return compute_saliency(
            model, engine_window, sensor_names, device=device, top_k=top_k
        )


# ── Tool 3: Retrieve KB ──────────────────────────────────────────────────
def retrieve_knowledge(
    query: str,
    top_k: int = 3,
    kb_dir: str = None,
) -> dict:
    """
    Retrieve relevant snippets from the knowledge base.

    Output schema:
    {
        "snippets": [{"text": str, "source": str, "score": float}, ...],
        "query": str,
    }
    """
    if kb_dir is None:
        kb_dir = str(Path(__file__).resolve().parent.parent.parent / "kb")

    snippets = retrieve_kb(query, kb_dir=kb_dir, top_k=top_k)
    return {"snippets": snippets, "query": query}


# ── Tool 4: Recommend Action ─────────────────────────────────────────────
def recommend_action(
    prediction: dict,
    explanation: dict,
    kb_snippets: dict,
    escalation_threshold: float = 20.0,
) -> dict:
    """
    Generate an actionable, uncertainty-aware recommendation.

    Uses prediction, explanation, and KB to produce grounded advice.
    Abstains/escalates when uncertainty is high.

    Output schema:
    {
        "action": str ("recommend" or "escalate"),
        "recommendation": str,
        "confidence": str,
        "supporting_evidence": {
            "top_sensor": str,
            "kb_reference": str,
        },
        "escalation_reason": str or None,
    }
    """
    rul_mean = prediction.get("rul_mean", 0)
    rul_std = prediction.get("rul_std", None)
    uncertainty_level = prediction.get("uncertainty_level", "unknown")

    # Top contributing sensor
    top_sensor = "unknown"
    if explanation.get("top_sensors"):
        top_sensor = explanation["top_sensors"][0]["sensor"]

    # KB reference
    kb_ref = "No KB snippets available."
    if kb_snippets.get("snippets"):
        kb_ref = kb_snippets["snippets"][0].get("text", kb_ref)

    # Abstention logic
    should_escalate = False
    escalation_reason = None

    if rul_std is not None and rul_std > escalation_threshold:
        should_escalate = True
        escalation_reason = (
            f"High prediction uncertainty (std={rul_std:.1f} cycles). "
            f"Model is not confident enough for automated recommendation."
        )
    elif uncertainty_level in ("high", "very_high"):
        should_escalate = True
        escalation_reason = f"Uncertainty level classified as '{uncertainty_level}'."

    if should_escalate:
        recommendation = (
            f"⚠️ ESCALATE: Uncertainty too high for automated recommendation. "
            f"Predicted RUL ≈ {rul_mean:.0f} cycles (±{rul_std:.0f}). "
            f"Top contributing sensor: {top_sensor}. "
            f"Recommend manual inspection. "
            f"KB reference: {kb_ref[:200]}"
        )
        action = "escalate"
        confidence = "low"
    else:
        # Generate actionable recommendation
        if rul_mean < 30:
            urgency = "CRITICAL"
            action_text = "Schedule immediate maintenance."
        elif rul_mean < 80:
            urgency = "WARNING"
            action_text = "Plan maintenance within the next cycle window."
        else:
            urgency = "NORMAL"
            action_text = "Continue monitoring. No immediate action required."

        recommendation = (
            f"{urgency}: Predicted RUL = {rul_mean:.0f} cycles"
            + (f" (±{rul_std:.0f})" if rul_std else "")
            + f". {action_text} "
            f"Primary degradation signal: {top_sensor}. "
            f"KB reference: {kb_ref[:200]}"
        )
        action = "recommend"
        confidence = "high" if uncertainty_level == "low" else "medium"

    return {
        "action": action,
        "recommendation": recommendation,
        "confidence": confidence,
        "supporting_evidence": {
            "top_sensor": top_sensor,
            "kb_reference": kb_ref[:200],
        },
        "escalation_reason": escalation_reason,
    }
