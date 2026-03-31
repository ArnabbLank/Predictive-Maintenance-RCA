# RUL Prediction + Ops Copilot — Project Plan

> **Repository**: `rul-copilot`  
> **Dataset**: NASA C-MAPSS FD001  
> **Duration**: 12 weeks  

---

## Project Goal

Build a **Remaining Useful Life (RUL)** predictor on **NASA C-MAPSS FD001**, then wrap it in an **agentic "Ops Copilot"** that:

1. Runs inference (RUL + uncertainty via MC Dropout)
2. Explains predictions (top sensors via saliency / integrated gradients)
3. Retrieves from a curated knowledge base (BM25 over markdown)
4. Produces actionable, uncertainty-aware recommendations
5. **Abstains / escalates** when uncertainty is too high

---

## Repository Layout

```
rul-copilot/
├── data/                          # C-MAPSS dataset (not tracked in git)
│   ├── download_data.sh           # Download script (tracked)
│   └── train_FD001.txt, ...       # Data files (gitignored)
├── src/
│   ├── data_loader.py             # Parse C-MAPSS files
│   ├── preprocess.py              # Windowing, scaling, SG smoothing
│   ├── features.py                # Hand-crafted features for baselines
│   ├── train.py                   # Unified training loop
│   ├── eval.py                    # Test evaluation + uncertainty
│   ├── infer.py                   # Single/batch inference
│   ├── explain.py                 # Saliency, IG, SHAP
│   ├── models/
│   │   ├── mlp.py                 # MLP baseline
│   │   ├── lstm.py                # LSTM
│   │   ├── cnn_lstm.py            # Paper A: CNN+LSTM
│   │   └── cnn_transformer.py     # Paper B: CNN-Transformer + MC Dropout
│   └── copilot/
│       ├── tools.py               # predict / explain / retrieve / recommend
│       ├── retriever.py           # BM25 over KB markdown
│       └── agent.py               # OpsCopilot orchestrator
├── kb/
│   ├── fault_tree.md              # Curated fault tree
│   └── glossary.md                # Sensor & setting definitions
├── app/
│   ├── streamlit_app.py           # Interactive demo dashboard
│   └── generate_pptx.py           # PowerPoint generator
├── checkpoints/                   # Saved model weights (gitignored)
├── reports/                       # Generated figures & tables
├── week01-eda/                    # Weekly notebook folders
│   └── 01_eda.ipynb
├── week02-baselines/
│   └── 02_baselines.ipynb
├── ...                            # weeks 03–12
├── requirements.txt               # Pinned dependencies
├── SETUP.md                       # Virtual environment instructions
├── GIT_GUIDE.md                   # Git workflow guide
├── .gitignore
└── README.md
```

---

## Core Papers Replicated

| # | Paper | Venue | Key Idea |
|---|-------|-------|----------|
| A | *Enhanced Savitzky-Golay + Improved DL Framework* | Scientific Reports (2024) | SG smoothing + CNN+LSTM |
| B | *UQ and RUL Prediction Using Deep Learning* | IJSIMM (2025) | CNN-Transformer + MC Dropout |

---

## Novel Contributions

1. **Uncertainty-Aware Copilot with Abstention** — the agent refuses to recommend when MC Dropout std exceeds a threshold, escalating to a human instead
2. **Tool-Grounded Explanations** — every recommendation must cite (a) a model explanation and (b) a retrieved KB snippet
3. **Calibration Analysis** — we verify that X% prediction intervals actually cover X% of true values
4. **OOD Detection** — Mahalanobis distance flags inputs outside the training distribution

---

## Week-by-Week Plan

### Week 1 — Dataset Onboarding + EDA
**Notebook**: `week01-eda/01_eda.ipynb`
- Parse train/test/RUL files for FD001
- Sensor distributions, trends over cycle life, correlations
- RUL labeling (raw + capped at 125)
- Identify informative vs near-constant sensors

### Week 2 — ML Baselines + Feature Engineering
**Notebook**: `week02-baselines/02_baselines.ipynb`
- Hand-crafted features: last/mean/std/min/max/slope/ewm per sensor
- Models: Linear Regression, Ridge, Random Forest, GBM
- Metrics: MAE, RMSE, NASA asymmetric scoring

### Week 3 — Health Index Baseline
**Notebook**: `week03-health-index/03_health_index.ipynb`
- PCA-based health index from informative sensors
- Exponential degradation curve fitting
- Compare HI baseline vs ML baselines

### Week 4 — LSTM + Agent Skeleton
**Notebook**: `week04-lstm-and-agent-skeleton/04_lstm_and_agent.ipynb`
- **Part A**: Train multi-layer LSTM on windowed sequences
- **Part B**: Define copilot tool contracts, build rule-based orchestrator v0

### Week 5 — Paper A: CNN+LSTM with SG Smoothing
**Notebook**: `week05-paper-a-cnn-lstm/05_paper_a_cnn_lstm.ipynb`
- Savitzky-Golay smoothing visualization
- 4 controlled experiments: {LSTM, CNN+LSTM} × {raw, SG-smoothed}

### Week 6 — Paper B: CNN-Transformer + MC Dropout
**Notebook**: `week06-paper-b-transformer-uq/06_paper_b_transformer_uq.ipynb`
- CNN-Transformer architecture with positional encoding
- MC Dropout inference (100 stochastic passes)
- Uncertainty sanity checks

### Week 7 — Explainability
**Notebook**: `week07-explainability/07_explainability.ipynb`
- Gradient saliency, integrated gradients, temporal saliency
- SHAP for baselines
- Explanation consistency analysis

### Week 8 — Knowledge Base + Retrieval
**Notebook**: `week08-knowledge-base-rag/08_knowledge_base_rag.ipynb`
- Curate KB: fault tree + sensor glossary
- BM25 retrieval with groundedness checks
- Copilot demo with KB citations

### Week 9 — Agentic Copilot (Full Pipeline)
**Notebook**: `week09-agentic-copilot/09_agentic_copilot.ipynb`
- End-to-end: predict → explain → retrieve → recommend
- Safety guardrails (abstention, escalation)
- Fleet-level batch analysis

### Week 10 — Novelty: OOD Detection + Abstention + Calibration
**Notebook**: `week10-novelty-abstention/10_novelty_abstention.ipynb`
- Mahalanobis distance for OOD detection
- Calibration curve + Average Calibration Error
- Abstention policy: MAE vs coverage trade-off

### Week 11 — Ablation Study + Streamlit Demo
**Notebook**: `week11-ablations-streamlit/11_ablations_streamlit.ipynb`
- Ablations: window size, SG smoothing, attention heads, MC samples
- Final model comparison table
- Streamlit app: single-engine analysis, fleet dashboard, model comparison

### Week 12 — Final Report + Paper Prep
**Notebook**: `week12-final-report/12_final_report.ipynb`
- Publication-quality figures and LaTeX tables
- Paper outline (Abstract → Methods → Experiments → Results → Conclusion)
- PowerPoint presentation
- Reproducibility checklist

---

## Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | mean(\|y - ŷ\|) | Primary metric |
| RMSE | sqrt(mean((y - ŷ)²)) | Penalizes large errors |
| NASA Score | Σ sᵢ (asymmetric exponential) | Late predictions penalized more |
| ACE | Avg Calibration Error | \|expected coverage - observed coverage\| |
| Coverage | Fraction of non-abstained predictions | Higher = fewer abstentions |

---

## Definition of Done

- [ ] Reproduced Paper A (CNN+LSTM + SG smoothing) on FD001
- [ ] Reproduced Paper B (CNN-Transformer + MC Dropout) on FD001
- [ ] Built uncertainty-aware Copilot with abstention logic
- [ ] Clean, well-documented repo with weekly notebooks
- [ ] Interactive Streamlit demo
- [ ] Final report / paper draft with tables and figures
