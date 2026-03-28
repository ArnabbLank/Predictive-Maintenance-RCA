# ✈️ RUL Copilot — Remaining Useful Life Prediction with Uncertainty-Aware AI

An end-to-end system for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset, with uncertainty quantification and an intelligent copilot agent.

---

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url> rul-copilot && cd rul-copilot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data (if not already present)
bash data/download_data.sh

# 5. Run the Streamlit demo
streamlit run app/streamlit_app.py
```

See [SETUP.md](SETUP.md) for detailed environment setup (conda, GPU, troubleshooting).

---

## Project Overview

| Component | Description |
|-----------|-------------|
| **Dataset** | NASA C-MAPSS FD001 — 100 train + 100 test turbofan engines |
| **Baselines** | Linear Regression, Ridge, Random Forest, Gradient Boosting |
| **Deep Models** | LSTM, CNN+LSTM (Paper A), CNN-Transformer (Paper B) |
| **Uncertainty** | MC Dropout with calibration analysis |
| **Explainability** | Gradient saliency, Integrated Gradients, SHAP |
| **Copilot** | Predict → Explain → Retrieve KB → Recommend (with abstention) |
| **Demo** | Streamlit app with fleet dashboard |

---

## Results (FD001)

| Model | MAE | RMSE | NASA Score |
|-------|-----|------|------------|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| LSTM | — | — | — |
| CNN+LSTM (raw) | — | — | — |
| CNN+LSTM (SG smoothed) | — | — | — |
| CNN-Transformer | — | — | — |
| **CNN-Transformer + UQ + Abstention** | **—** | **—** | **—** |

> Fill in after running experiments. Lower MAE/RMSE/Score = better.

---

## Repository Structure

```
├── data/                      # C-MAPSS dataset (gitignored, use download script)
├── src/                       # All Python source code
│   ├── data_loader.py         # Data parsing
│   ├── preprocess.py          # Windowing, scaling, SG smoothing
│   ├── features.py            # Feature engineering for baselines
│   ├── train.py               # Training loop with early stopping
│   ├── eval.py                # Evaluation & uncertainty metrics
│   ├── infer.py               # Inference with uncertainty classification
│   ├── explain.py             # Saliency, integrated gradients, SHAP
│   ├── models/                # Model architectures
│   │   ├── mlp.py, lstm.py, cnn_lstm.py, cnn_transformer.py
│   └── copilot/               # Copilot agent
│       ├── tools.py           # 4 tool functions
│       ├── retriever.py       # BM25 KB retrieval
│       └── agent.py           # OpsCopilot orchestrator
├── kb/                        # Knowledge base (fault tree, glossary)
├── app/                       # Streamlit demo
├── week01-eda/ ... week12-final-report/   # Weekly notebooks
├── checkpoints/               # Model weights (gitignored)
├── reports/                   # Generated figures
├── requirements.txt           # Pinned Python dependencies
├── SETUP.md                   # Environment setup guide
├── GIT_GUIDE.md               # Git workflow for students
└── plan.md                    # Detailed 12-week plan
```

---

## Weekly Notebooks

| Week | Topic | Notebook |
|------|-------|----------|
| 1 | EDA & Data Understanding | `week01-eda/01_eda.ipynb` |
| 2 | ML Baselines | `week02-baselines/02_baselines.ipynb` |
| 3 | Health Index | `week03-health-index/03_health_index.ipynb` |
| 4 | LSTM + Agent Skeleton | `week04-lstm-and-agent-skeleton/04_lstm_and_agent.ipynb` |
| 5 | Paper A: CNN+LSTM | `week05-paper-a-cnn-lstm/05_paper_a_cnn_lstm.ipynb` |
| 6 | Paper B: CNN-Transformer + UQ | `week06-paper-b-transformer-uq/06_paper_b_transformer_uq.ipynb` |
| 7 | Explainability | `week07-explainability/07_explainability.ipynb` |
| 8 | Knowledge Base + Retrieval | `week08-knowledge-base-rag/08_knowledge_base_rag.ipynb` |
| 9 | Agentic Copilot | `week09-agentic-copilot/09_agentic_copilot.ipynb` |
| 10 | OOD + Abstention + Calibration | `week10-novelty-abstention/10_novelty_abstention.ipynb` |
| 11 | Ablations + Streamlit | `week11-ablations-streamlit/11_ablations_streamlit.ipynb` |
| 12 | Final Report | `week12-final-report/12_final_report.ipynb` |

---

## Key Commands

```bash
# Train a model
python src/train.py --model cnn_transformer --dataset FD001 --epochs 100

# Evaluate on test set
python src/eval.py --checkpoint checkpoints/cnn_transformer_best.pt

# Run inference on a single engine
python src/infer.py --checkpoint checkpoints/cnn_transformer_best.pt --engine 42

# Launch Streamlit demo
streamlit run app/streamlit_app.py

# Format code
black src/ --line-length 100
```

---

## References

1. Saxena et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. PHM08.
2. Paper A (2024). *Enhanced Savitzky-Golay + Improved DL Framework*. Scientific Reports.
3. Paper B (2025). *UQ and RUL Prediction Using Deep Learning*. IJSIMM.

---

## License

This project is for educational purposes. The C-MAPSS dataset is provided by NASA under public domain.
