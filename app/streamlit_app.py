"""
Streamlit Demo — RUL Copilot Dashboard
=======================================
Run:  streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Copilot",
    page_icon="✈️",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("✈️ RUL Copilot")
st.sidebar.markdown("Remaining Useful Life prediction with uncertainty-aware AI.")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Single Engine", "📊 Fleet Dashboard", "🧪 Model Comparison", "📖 About"],
)

# ── Helper: load data & model ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remove 'CMAPSS' from here; the data_loader adds it automatically
DATA_DIR = os.path.join(BASE_DIR, 'data') 
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
KB_DIR = os.path.join(BASE_DIR, 'kb')


@st.cache_data
def load_data():
    """Load FD001 test data and true RUL."""
    from data_loader import load_test, load_train
    from preprocess import preprocess_pipeline

    train_df = load_train(1, DATA_DIR)
    test_df, rul_true = load_test(1, DATA_DIR)
    result = preprocess_pipeline(train_df, test_df, window_size=30)
    return result['X_test'], rul_true


@st.cache_resource
def load_model():
    """Load CNN-Transformer model."""
    from models.cnn_transformer import CNNTransformerModel

    device = torch.device('cpu')
    model = CNNTransformerModel(
        n_features=14, seq_len=30,
        cnn_channels=64, d_model=64, nhead=4, num_encoder_layers=2, dropout=0.2,
    )

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'cnn_transformer_FD001_best.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        key = 'model_state_dict' if 'model_state_dict' in state else None
        model.load_state_dict(state[key] if key else state)
        st.sidebar.success("Model checkpoint loaded")
    else:
        st.sidebar.warning("No checkpoint found — using random weights for demo")

    model.eval()
    return model, device


# ── Pages ──────────────────────────────────────────────────────

def page_overview():
    st.title("RUL Copilot — Overview")
    st.markdown("""
    This dashboard demonstrates a **Remaining Useful Life (RUL)** prediction system
    for turbofan engines using the **NASA C-MAPSS FD001** dataset.

    ### Features
    | Feature | Description |
    |---------|-------------|
    | **CNN-Transformer** | Hybrid architecture for temporal pattern extraction |
    | **MC Dropout** | Uncertainty quantification via stochastic forward passes |
    | **Abstention** | Refuses predictions when uncertainty is too high |
    | **OOD Detection** | Flags out-of-distribution inputs |
    | **Knowledge Base** | BM25 retrieval over curated maintenance docs |
    | **Copilot Agent** | Predict → Explain → Retrieve → Recommend pipeline |

    ### How to use
    - **Single Engine**: Select an engine and get a detailed prediction report
    - **Fleet Dashboard**: See all engines color-coded by risk level
    - **Model Comparison**: Compare architectures side-by-side
    """)


def page_single_engine():
    st.title("🔍 Single Engine Analysis")

    X_test, rul_true = load_data()
    model, device = load_model()

    n_engines = len(X_test)
    engine_id = st.slider("Select Engine ID", 0, n_engines - 1, 0)
    n_mc = st.slider("MC Dropout Samples", 10, 200, 50, step=10)

    if st.button("Analyze Engine", type="primary"):
        x = torch.tensor(X_test[engine_id:engine_id + 1], dtype=torch.float32).to(device)

        with st.spinner("Running MC Dropout inference..."):
            rul_mean, rul_std, all_preds = model.predict_with_uncertainty(x, n_samples=n_mc)

        rul_mean_val = rul_mean.item()
        rul_std_val = rul_std.item()
        true_val = rul_true[engine_id]

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted RUL", f"{rul_mean_val:.1f} cycles")
        col2.metric("True RUL", f"{true_val:.0f} cycles")
        col3.metric("Error", f"{abs(rul_mean_val - true_val):.1f}")
        col4.metric("Uncertainty (σ)", f"{rul_std_val:.2f}")

        # Uncertainty classification
        if rul_std_val < 5:
            level, color = "LOW", "green"
        elif rul_std_val < 15:
            level, color = "MEDIUM", "orange"
        elif rul_std_val < 30:
            level, color = "HIGH", "red"
        else:
            level, color = "VERY HIGH — ABSTAIN", "darkred"

        st.markdown(f"**Uncertainty Level:** :{color}[{level}]")

        # Distribution plot
        preds = all_preds.flatten()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(preds, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(x=true_val, color='red', ls='--', lw=2, label=f'True RUL = {true_val:.0f}')
        ax.axvline(x=rul_mean_val, color='navy', ls='-', lw=2, label=f'Mean = {rul_mean_val:.1f}')
        ax.set_xlabel("RUL (cycles)")
        ax.set_title(f"MC Dropout Distribution — Engine {engine_id}")
        ax.legend()
        st.pyplot(fig)

        # KB Recommendation
        st.subheader("💡 Copilot Recommendation")
        if rul_mean_val < 30:
            st.error("⚠️ **URGENT**: Schedule immediate inspection. Predicted RUL < 30 cycles.")
        elif rul_mean_val < 80:
            st.warning("🔧 **MONITOR**: Plan maintenance within next operating window.")
        else:
            st.success("✅ **HEALTHY**: No immediate action required.")


def page_fleet():
    st.title("📊 Fleet Dashboard")

    X_test, rul_true = load_data()
    model, device = load_model()

    n_mc = st.sidebar.slider("MC Samples (Fleet)", 10, 100, 30, step=10)

    if st.button("Run Fleet Analysis", type="primary"):
        with st.spinner(f"Analyzing {len(X_test)} engines with {n_mc} MC samples..."):
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            rul_mean, rul_std, _ = model.predict_with_uncertainty(X_t, n_samples=n_mc)
            rul_mean = rul_mean.flatten()
            rul_std = rul_std.flatten()

        # Build fleet DataFrame
        fleet = pd.DataFrame({
            'Engine': range(1, len(rul_mean) + 1),
            'Pred RUL': rul_mean,
            'True RUL': rul_true,
            'Error': np.abs(rul_mean - rul_true),
            'Uncertainty': rul_std,
        })

        # Risk classification
        fleet['Risk'] = pd.cut(
            fleet['Pred RUL'],
            bins=[-np.inf, 30, 80, np.inf],
            labels=['🔴 Critical', '🟡 Monitor', '🟢 Healthy'],
        )

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Engines", len(fleet))
        c2.metric("Critical (RUL<30)", (fleet['Pred RUL'] < 30).sum())
        c3.metric("Fleet MAE", f"{fleet['Error'].mean():.1f}")
        c4.metric("Avg Uncertainty", f"{fleet['Uncertainty'].mean():.2f}")

        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = fleet['Risk'].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        colors_map = {'🔴 Critical': '#d32f2f', '🟡 Monitor': '#ffa000', '🟢 Healthy': '#388e3c'}
        risk_counts.plot.bar(ax=axes[0], color=[colors_map.get(x, 'gray') for x in risk_counts.index])
        axes[0].set_title('Engines by Risk Level')
        axes[0].set_ylabel('Count')

        # Scatter
        scatter_colors = fleet['Pred RUL'].apply(
            lambda x: '#d32f2f' if x < 30 else '#ffa000' if x < 80 else '#388e3c'
        )
        axes[1].scatter(fleet['Pred RUL'], fleet['Uncertainty'], c=scatter_colors, alpha=0.6, s=50)
        axes[1].axhline(y=20, color='red', ls='--', alpha=0.5, label='Escalation threshold')
        axes[1].set_xlabel('Predicted RUL')
        axes[1].set_ylabel('Uncertainty (σ)')
        axes[1].set_title('RUL vs Uncertainty')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Table
        st.subheader("Engine Details")
        st.dataframe(
            fleet.sort_values('Pred RUL').style.format({
                'Pred RUL': '{:.1f}', 'True RUL': '{:.0f}',
                'Error': '{:.1f}', 'Uncertainty': '{:.2f}',
            }),
            use_container_width=True,
        )


def page_comparison():
    st.title("🧪 Model Comparison")
    st.markdown("Experiment results automatically loaded from `all_metrics.json`.")

    # 1. Load the data from your reports folder
    import json
    import os
    metrics_path = os.path.join(REPORTS_DIR, 'all_metrics.json')
    
    # Default values (dashes) if file is missing
    data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)

    # Helper to safely format numbers or return a dash if missing
    def get_val(val, key='MAE'):
        if val is None: return "—"
        # Handle cases where val might be a dict or a float
        num = val.get(key) if isinstance(val, dict) else val
        return f"{num:.2f}" if num is not None else "—"

    # 2. Extract specific model data from the JSON structure
    bl = data.get('baselines', [])
    pa = data.get('paper_a', [])
    ct = data.get('cnn_transformer', {})
    ls = data.get('lstm', {})

    # 3. Create the dynamic DataFrame
    comparison = pd.DataFrame([
        {'Model': 'Linear Regression', 
         'MAE': get_val(next((m for m in bl if m['Model'] == 'Linear Regression'), None), 'Test MAE'), 
         'RMSE': get_val(next((m for m in bl if m['Model'] == 'Linear Regression'), None), 'Test RMSE'), 
         'Score': get_val(next((m for m in bl if m['Model'] == 'Linear Regression'), None), 'Test NASA Score')},
        
        {'Model': 'Random Forest', 
         'MAE': get_val(next((m for m in bl if m['Model'] == 'Random Forest'), None), 'Test MAE'), 
         'RMSE': get_val(next((m for m in bl if m['Model'] == 'Random Forest'), None), 'Test RMSE'), 
         'Score': get_val(next((m for m in bl if m['Model'] == 'Random Forest'), None), 'Test NASA Score')},
        
        {'Model': 'GBM', 
         'MAE': get_val(next((m for m in bl if m['Model'] == 'Gradient Boosting'), None), 'Test MAE'), 
         'RMSE': get_val(next((m for m in bl if m['Model'] == 'Gradient Boosting'), None), 'Test RMSE'), 
         'Score': get_val(next((m for m in bl if m['Model'] == 'Gradient Boosting'), None), 'Test NASA Score')},
        
        {'Model': 'LSTM', 'MAE': get_val(ls, 'MAE'), 'RMSE': get_val(ls, 'RMSE'), 'Score': get_val(ls, 'NASA_Score')},
        
        {'Model': 'CNN+LSTM (raw)', 
         'MAE': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_raw'), None), 'MAE'), 
         'RMSE': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_raw'), None), 'RMSE'), 
         'Score': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_raw'), None), 'NASA_Score')},
        
        {'Model': 'CNN+LSTM (SG)', 
         'MAE': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_sg'), None), 'MAE'), 
         'RMSE': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_sg'), None), 'RMSE'), 
         'Score': get_val(next((m for m in pa if m['name'] == 'CNN_LSTM_sg'), None), 'NASA_Score')},
        
        {'Model': 'CNN-Transformer + UQ', 
         'MAE': get_val(ct, 'ct_mae'), 'RMSE': get_val(ct, 'ct_rmse'), 'Score': get_val(ct, 'ct_nasa')},
    ])

    st.table(comparison)
    st.success("✅ Data loaded successfully from `reports/all_metrics.json`" if data else "⚠️ No experiment data found. Run `run_weeks_1_to_6.py` first.")


def page_about():
    st.title("📖 About")
    st.markdown("""
    ### RUL Copilot

    An end-to-end system for **Remaining Useful Life prediction** of turbofan engines,
    combining deep learning, uncertainty quantification, and an intelligent copilot agent.

    **Dataset**: NASA C-MAPSS FD001 (100 train + 100 test engines)

    **Key References**:
    - Paper A: CNN+LSTM with Savitzky-Golay smoothing (Scientific Reports, 2024)
    - Paper B: CNN-Transformer with MC Dropout (IJSIMM, 2025)

    **Architecture**:
    ```
    Raw Sensor Data → Preprocessing → CNN Feature Extraction
    → Transformer Encoder → RUL Prediction
    → MC Dropout Uncertainty → Abstention / Recommendation
    → KB Retrieval → Copilot Report
    ```

    **Built with**: PyTorch, scikit-learn, Streamlit, SHAP, Captum
    """)


# ── Router ─────────────────────────────────────────────────────

if page == "🏠 Overview":
    page_overview()
elif page == "🔍 Single Engine":
    page_single_engine()
elif page == "📊 Fleet Dashboard":
    page_fleet()
elif page == "🧪 Model Comparison":
    page_comparison()
elif page == "📖 About":
    page_about()