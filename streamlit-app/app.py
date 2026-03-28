"""
Agentic RUL Prediction Demo — Streamlit App

Demonstrates the Ops Copilot pipeline:
  Predict → Explain → Retrieve KB → Recommend (with uncertainty & abstention)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_train, load_test
from preprocess import (
    INFORMATIVE_SENSORS_FD001,
    fit_scaler,
    apply_scaler,
    create_test_sequences,
)
from infer import load_model, predict_single
from explain import compute_saliency, compute_integrated_gradients
from copilot.retriever import retrieve_kb
from copilot.tools import recommend_action

CHECKPOINT = str(ROOT / "checkpoints" / "cnn_transformer_FD001_best.pt")
KB_DIR = str(ROOT / "kb")
SENSOR_NAMES = INFORMATIVE_SENSORS_FD001
N_FEATURES = len(SENSOR_NAMES)
WINDOW = 30

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Ops Copilot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading (cached) ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_train = load_train(fd_number=1)
    df_test, rul_true = load_test(fd_number=1)
    scaler = fit_scaler(df_train, SENSOR_NAMES)
    df_test_scaled = apply_scaler(df_test, scaler, SENSOR_NAMES)
    X_test = create_test_sequences(df_test_scaled, SENSOR_NAMES, WINDOW)
    return df_test, rul_true, X_test


@st.cache_resource
def get_model():
    if not os.path.exists(CHECKPOINT):
        return None
    return load_model("cnn_transformer", CHECKPOINT, N_FEATURES, WINDOW, device="cpu")


# ── Sidebar ───────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.title("⚙️ Ops Copilot")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Engine Analysis", "📊 Fleet Dashboard", "ℹ️ About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Pipeline:**\n"
        "1. Predict RUL (MC Dropout)\n"
        "2. Explain (Integrated Gradients)\n"
        "3. Retrieve Knowledge Base\n"
        "4. Recommend & Safety Check"
    )
    return page


# ── Pages ─────────────────────────────────────────────────────────────────
def page_overview():
    st.title("Remaining Useful Life — Ops Copilot Demo")
    st.markdown("""
    This demo showcases an **uncertainty-aware agentic framework** for turbofan engine
    prognostics using the NASA C-MAPSS FD001 dataset.

    The Copilot follows a **4-step pipeline** for every engine it analyzes:

    | Step | Tool | Description |
    |------|------|-------------|
    | 1 | `predict_rul` | CNN-Transformer + MC Dropout → RUL mean, std, 95% CI |
    | 2 | `explain_prediction` | Integrated Gradients → top contributing sensors |
    | 3 | `retrieve_knowledge` | BM25 search over curated fault-tree & glossary |
    | 4 | `recommend_action` | Grounded recommendation with abstention logic |

    **Key capabilities:**
    - 🎯 Predicts RUL with calibrated uncertainty (MC Dropout, 100 forward passes)
    - 🔬 Explains *why* (gradient-based sensor attribution)
    - 📚 Grounds recommendations in a knowledge base
    - ⚠️ **Abstains** when uncertainty is too high → escalates to human expert
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "CNN-Transformer")
    col2.metric("Parameters", "84,417")
    col3.metric("UQ Method", "MC Dropout (N=100)")


def page_engine_analysis():
    st.title("🔍 Single Engine Analysis")

    model = get_model()
    if model is None:
        st.error("Model checkpoint not found. Please train the CNN-Transformer first.")
        return

    df_test, rul_true, X_test = load_data()
    n_engines = len(rul_true)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        engine_id = st.selectbox(
            "Select Engine",
            list(range(1, n_engines + 1)),
            format_func=lambda x: f"Engine {x}  (True RUL = {rul_true[x-1]})",
        )
    with col_ctrl2:
        mc_samples = st.slider("MC Dropout Samples", 10, 200, 100, step=10)

    escalation_threshold = st.slider(
        "Escalation Threshold (uncertainty std)", 5.0, 50.0, 20.0, step=1.0,
        help="If prediction uncertainty exceeds this, the Copilot escalates to human."
    )

    if st.button("🚀 Run Copilot Analysis", type="primary", use_container_width=True):
        idx = engine_id - 1
        window = X_test[idx]

        # Step 1: Predict
        with st.spinner("Step 1/4 — Predicting RUL with MC Dropout..."):
            prediction = predict_single(model, window, device="cpu", mc_samples=mc_samples)

        # Step 2: Explain
        with st.spinner("Step 2/4 — Computing Integrated Gradients..."):
            explanation = compute_integrated_gradients(
                model, window, SENSOR_NAMES, device="cpu", top_k=5
            )

        # Step 3: Retrieve KB
        with st.spinner("Step 3/4 — Retrieving Knowledge Base..."):
            top_sensors_str = ", ".join(
                s["sensor"] for s in explanation.get("top_sensors", [])[:3]
            )
            kb_query = f"degradation {top_sensors_str} maintenance"
            kb_context = {"snippets": retrieve_kb(kb_query, kb_dir=KB_DIR, top_k=3), "query": kb_query}

        # Step 4: Recommend
        with st.spinner("Step 4/4 — Generating Recommendation..."):
            recommendation = recommend_action(
                prediction, explanation, kb_context,
                escalation_threshold=escalation_threshold,
            )

        # ── Display results ───────────────────────────────────────────
        st.markdown("---")
        st.subheader(f"Engine {engine_id} — Copilot Report")

        # Metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Predicted RUL", f"{prediction['rul_mean']:.0f} cycles")
        c2.metric("True RUL", f"{rul_true[idx]}")
        error = abs(prediction['rul_mean'] - rul_true[idx])
        c3.metric("Absolute Error", f"{error:.1f}")
        if prediction['rul_std'] is not None:
            c4.metric("Uncertainty (σ)", f"{prediction['rul_std']:.1f}")
        else:
            c4.metric("Uncertainty", "N/A")
        c5.metric("Uncertainty Level", prediction.get('uncertainty_level', 'unknown').upper())

        # Confidence interval visualization
        if prediction.get('confidence_interval_95'):
            ci = prediction['confidence_interval_95']
            fig_ci = go.Figure()
            fig_ci.add_shape(
                type="rect", x0=-0.4, x1=0.4, y0=ci[0], y1=ci[1],
                fillcolor="rgba(70,130,180,0.3)", line=dict(color="steelblue", width=2),
            )
            fig_ci.add_trace(go.Scatter(
                x=[0], y=[prediction['rul_mean']], mode='markers',
                marker=dict(size=14, color='navy', symbol='diamond'),
                name='Predicted RUL',
            ))
            fig_ci.add_trace(go.Scatter(
                x=[0], y=[rul_true[idx]], mode='markers',
                marker=dict(size=14, color='red', symbol='x'),
                name='True RUL',
            ))
            fig_ci.update_layout(
                title="RUL Prediction with 95% Confidence Interval",
                yaxis_title="RUL (cycles)", xaxis=dict(visible=False),
                height=350, showlegend=True,
            )
            st.plotly_chart(fig_ci, use_container_width=True)

        # Recommendation box
        st.markdown("---")
        if recommendation['action'] == 'escalate':
            st.error(f"⚠️ **ESCALATED** — {recommendation['recommendation']}")
        elif prediction['rul_mean'] < 30:
            st.warning(f"🔴 **CRITICAL** — {recommendation['recommendation']}")
        elif prediction['rul_mean'] < 80:
            st.warning(f"🟡 **WARNING** — {recommendation['recommendation']}")
        else:
            st.success(f"🟢 **NORMAL** — {recommendation['recommendation']}")

        # Top sensors
        col_sensors, col_kb = st.columns(2)
        with col_sensors:
            st.subheader("🔬 Sensor Attribution")
            top_s = explanation.get("top_sensors", [])
            if top_s:
                sensor_df = pd.DataFrame(top_s)
                fig_s = px.bar(
                    sensor_df, x='importance', y='sensor', orientation='h',
                    title="Top Contributing Sensors (Integrated Gradients)",
                    color='importance', color_continuous_scale='Blues',
                )
                fig_s.update_layout(yaxis=dict(autorange="reversed"), height=300)
                st.plotly_chart(fig_s, use_container_width=True)

        with col_kb:
            st.subheader("📚 Knowledge Base Context")
            for i, snippet in enumerate(kb_context.get('snippets', [])):
                with st.expander(f"KB Snippet {i+1} (score: {snippet.get('score', 0):.3f})"):
                    st.markdown(snippet.get('text', 'N/A'))

        # Safety checks
        st.markdown("---")
        st.subheader("✅ Safety Checks")
        checks = {
            "Uncertainty Reported": prediction.get('rul_std') is not None,
            "KB Cited": bool(kb_context.get('snippets')),
            "No False Reassurance": not (
                prediction.get('uncertainty_level') in ('high', 'very_high')
                and recommendation.get('action') == 'recommend'
                and recommendation.get('confidence') == 'high'
            ),
        }
        cols = st.columns(3)
        for i, (check, passed) in enumerate(checks.items()):
            with cols[i]:
                if passed:
                    st.success(f"✓ {check}")
                else:
                    st.error(f"✗ {check}")


def page_fleet():
    st.title("📊 Fleet Dashboard")

    model = get_model()
    if model is None:
        st.error("Model checkpoint not found.")
        return

    df_test, rul_true, X_test = load_data()
    n_engines = len(rul_true)

    mc_samples_fleet = st.slider("MC Samples per engine", 10, 100, 30, step=10,
                                  help="Lower = faster. 30 is good for fleet overview.")

    if st.button("⚡ Analyze Entire Fleet", type="primary", use_container_width=True):
        progress = st.progress(0)
        results = []

        for i in range(n_engines):
            pred = predict_single(model, X_test[i], device="cpu", mc_samples=mc_samples_fleet)
            pred['engine_id'] = i + 1
            pred['true_rul'] = int(rul_true[i])
            pred['error'] = abs(pred['rul_mean'] - rul_true[i])
            results.append(pred)
            progress.progress((i + 1) / n_engines)

        df_fleet = pd.DataFrame(results)

        # Summary metrics
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fleet Size", n_engines)
        c2.metric("Mean MAE", f"{df_fleet['error'].mean():.1f}")
        n_critical = (df_fleet['rul_mean'] < 30).sum()
        c3.metric("Critical Engines", int(n_critical))
        n_escalated = (df_fleet['rul_std'] > 20).sum() if 'rul_std' in df_fleet else 0
        c4.metric("Escalated (high UQ)", int(n_escalated))

        # Scatter: predicted vs true
        fig1 = px.scatter(
            df_fleet, x='true_rul', y='rul_mean',
            error_y='rul_std',
            color='uncertainty_level',
            color_discrete_map={'low': 'green', 'medium': 'steelblue', 'high': 'orange', 'very_high': 'red'},
            hover_data=['engine_id', 'error'],
            title="Fleet RUL: Predicted vs True (colored by uncertainty)",
            labels={'true_rul': 'True RUL', 'rul_mean': 'Predicted RUL'},
        )
        fig1.add_trace(go.Scatter(
            x=[0, max(rul_true) + 10], y=[0, max(rul_true) + 10],
            mode='lines', line=dict(dash='dash', color='gray'), name='Perfect',
        ))
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)

        # Fleet risk distribution
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.histogram(
                df_fleet, x='rul_mean', nbins=20,
                title="Predicted RUL Distribution",
                labels={'rul_mean': 'Predicted RUL'},
                color_discrete_sequence=['steelblue'],
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            if df_fleet['rul_std'].notna().any():
                fig3 = px.scatter(
                    df_fleet, x='error', y='rul_std',
                    title="Uncertainty vs Prediction Error",
                    labels={'error': 'Absolute Error', 'rul_std': 'Uncertainty (σ)'},
                    color='uncertainty_level',
                    color_discrete_map={'low': 'green', 'medium': 'steelblue', 'high': 'orange', 'very_high': 'red'},
                )
                st.plotly_chart(fig3, use_container_width=True)

        # Sortable table
        st.subheader("Engine Details")
        display_df = df_fleet[['engine_id', 'rul_mean', 'true_rul', 'error', 'rul_std', 'uncertainty_level']].copy()
        display_df.columns = ['Engine', 'Pred RUL', 'True RUL', 'Error', 'Uncertainty (σ)', 'UQ Level']
        display_df = display_df.sort_values('Pred RUL')
        st.dataframe(display_df, use_container_width=True, height=400)


def page_about():
    st.title("ℹ️ About")
    st.markdown("""
    ### Architecture
    - **Model:** CNN-Transformer (84,417 params)
      - 1D-CNN → Positional Encoding → Transformer Encoder (2 layers, 4 heads) → FC
    - **Uncertainty:** MC Dropout (p=0.1) with N stochastic forward passes
    - **Explainability:** Integrated Gradients (gradient-based attribution)
    - **Knowledge Base:** BM25 retrieval over curated fault-tree and glossary

    ### Dataset
    - NASA C-MAPSS FD001: 100 train + 100 test turbofan engines
    - 14 informative sensors, window size = 30, RUL cap = 125

    ### Copilot Safety Guardrails
    1. **Uncertainty is always reported** — never hide model confidence
    2. **KB grounding** — recommendations cite knowledge base evidence
    3. **Abstention** — high uncertainty → escalate to human expert
    4. **No false reassurance** — avoid "safe" recommendations when uncertain

    ### Project
    RUL Prediction + Ops Copilot — NASA C-MAPSS Turbofan Engine Dataset
    """)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    page = sidebar()
    if page == "🏠 Overview":
        page_overview()
    elif page == "🔍 Engine Analysis":
        page_engine_analysis()
    elif page == "📊 Fleet Dashboard":
        page_fleet()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
