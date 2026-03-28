"""
copilot/agent.py — Ops Copilot orchestrator.

v0: Rule-based orchestrator (calls tools in fixed order).
v1: LLM-backed orchestrator with tool calling (Week 9+).

The agent:
  1. Calls predict_rul → gets RUL + uncertainty
  2. Calls explain_prediction → gets top sensors
  3. Calls retrieve_knowledge → gets KB snippets
  4. Calls recommend_action → generates grounded recommendation
  5. Applies safety checks and abstention logic
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.tools import (
    predict_rul,
    explain_prediction,
    retrieve_knowledge,
    recommend_action,
)


class OpsCopilot:
    """
    Uncertainty-aware Ops Copilot for turbofan engine maintenance.

    Orchestrates prediction, explanation, retrieval, and recommendation
    with built-in safety guardrails.
    """

    def __init__(
        self,
        model=None,
        model_type: str = "cnn_transformer",
        checkpoint_path: str = None,
        sensor_names: list = None,
        kb_dir: str = None,
        mc_samples: int = 100,
        escalation_threshold: float = 20.0,
        device: str = "cpu",
    ):
        self.model = model
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sensor_names = sensor_names
        self.kb_dir = kb_dir
        self.mc_samples = mc_samples
        self.escalation_threshold = escalation_threshold
        self.device = device

        # Logging
        self.history = []

    def analyze_engine(
        self,
        engine_window: np.ndarray,
        engine_id: Optional[int] = None,
    ) -> dict:
        """
        Full Copilot analysis pipeline for one engine.

        Parameters
        ----------
        engine_window : np.ndarray, shape (seq_len, n_features)
        engine_id : int or None

        Returns
        -------
        dict with full analysis: prediction, explanation, kb_context, recommendation
        """
        # Step 1: Predict RUL
        prediction = predict_rul(
            engine_window=engine_window,
            model=self.model,
            model_type=self.model_type,
            checkpoint_path=self.checkpoint_path,
            mc_samples=self.mc_samples,
            device=self.device,
        )

        # Step 2: Explain prediction
        explanation = explain_prediction(
            engine_window=engine_window,
            model=self.model,
            sensor_names=self.sensor_names,
            device=self.device,
        )

        # Step 3: Retrieve KB
        # Build query from top sensors
        top_sensors_str = ", ".join(
            s["sensor"] for s in explanation.get("top_sensors", [])[:3]
        )
        kb_query = f"degradation {top_sensors_str} maintenance"
        kb_context = retrieve_knowledge(
            query=kb_query,
            kb_dir=self.kb_dir,
        )

        # Step 4: Recommend action
        recommendation = recommend_action(
            prediction=prediction,
            explanation=explanation,
            kb_snippets=kb_context,
            escalation_threshold=self.escalation_threshold,
        )

        # Assemble full report
        report = {
            "engine_id": engine_id,
            "prediction": prediction,
            "explanation": {
                "top_sensors": explanation.get("top_sensors", []),
                "method": explanation.get("method", ""),
            },
            "kb_context": kb_context,
            "recommendation": recommendation,
        }

        # Safety check: verify groundedness
        report["safety_checks"] = self._safety_checks(report)

        # Log
        self.history.append(report)

        return report

    def _safety_checks(self, report: dict) -> dict:
        """
        Verify the recommendation meets quality constraints.

        Checks:
          1. Uncertainty is reported
          2. KB citation is present
          3. Recommendation doesn't overclaim when uncertain
        """
        checks = {
            "uncertainty_reported": False,
            "kb_cited": False,
            "no_false_reassurance": True,
        }

        pred = report.get("prediction", {})
        rec = report.get("recommendation", {})

        # Check 1: Uncertainty is shown
        if pred.get("rul_std") is not None:
            checks["uncertainty_reported"] = True

        # Check 2: KB citation present
        evidence = rec.get("supporting_evidence", {})
        if evidence.get("kb_reference") and evidence["kb_reference"] != "No KB snippets available.":
            checks["kb_cited"] = True

        # Check 3: No false reassurance
        if pred.get("uncertainty_level") in ("high", "very_high"):
            if rec.get("action") == "recommend" and rec.get("confidence") == "high":
                checks["no_false_reassurance"] = False

        return checks

    def batch_analyze(
        self,
        engine_windows: np.ndarray,
        engine_ids: list = None,
    ) -> list:
        """Analyze multiple engines."""
        if engine_ids is None:
            engine_ids = list(range(1, len(engine_windows) + 1))

        results = []
        for i, window in enumerate(engine_windows):
            result = self.analyze_engine(window, engine_id=engine_ids[i])
            results.append(result)

        return results

    def get_fleet_summary(self, results: list = None) -> dict:
        """
        Summarize fleet-level status from batch analysis.
        """
        if results is None:
            results = self.history

        n_total = len(results)
        n_escalated = sum(
            1 for r in results if r["recommendation"]["action"] == "escalate"
        )
        n_critical = sum(
            1 for r in results
            if r["prediction"]["rul_mean"] < 30
            and r["recommendation"]["action"] == "recommend"
        )

        rul_means = [r["prediction"]["rul_mean"] for r in results]
        uncertainties = [
            r["prediction"]["rul_std"] for r in results
            if r["prediction"]["rul_std"] is not None
        ]

        return {
            "total_engines": n_total,
            "escalated": n_escalated,
            "critical": n_critical,
            "normal": n_total - n_escalated - n_critical,
            "avg_rul": float(np.mean(rul_means)) if rul_means else 0,
            "avg_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0,
            "fleet_risk": "HIGH" if n_critical > 0 else "MODERATE" if n_escalated > 0 else "LOW",
        }

    def format_report(self, report: dict) -> str:
        """Format a single engine report as readable text."""
        pred = report["prediction"]
        rec = report["recommendation"]
        expl = report["explanation"]

        lines = [
            f"═══ Engine {report['engine_id']} Analysis ═══",
            f"",
            f"📊 Prediction:",
            f"   RUL = {pred['rul_mean']:.0f} cycles",
        ]
        if pred.get("rul_std") is not None:
            lines.append(f"   Uncertainty (std) = {pred['rul_std']:.1f} cycles")
            lines.append(f"   95% CI = [{pred['confidence_interval_95'][0]:.0f}, {pred['confidence_interval_95'][1]:.0f}]")
            lines.append(f"   Uncertainty level: {pred['uncertainty_level']}")

        lines.extend([
            f"",
            f"🔍 Top Contributing Sensors:",
        ])
        for s in expl.get("top_sensors", [])[:3]:
            lines.append(f"   {s['rank']}. {s['sensor']} (importance: {s['importance']:.4f})")

        lines.extend([
            f"",
            f"💡 Recommendation ({rec['action'].upper()}):",
            f"   {rec['recommendation']}",
            f"   Confidence: {rec['confidence']}",
        ])

        if rec.get("escalation_reason"):
            lines.append(f"   ⚠️  Escalation reason: {rec['escalation_reason']}")

        safety = report.get("safety_checks", {})
        lines.extend([
            f"",
            f"✅ Safety Checks:",
            f"   Uncertainty reported: {'✓' if safety.get('uncertainty_reported') else '✗'}",
            f"   KB cited: {'✓' if safety.get('kb_cited') else '✗'}",
            f"   No false reassurance: {'✓' if safety.get('no_false_reassurance') else '✗'}",
        ])

        return "\n".join(lines)
