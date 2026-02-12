# Predictive Maintenance with Causal Root-Cause Analysis

**Ops Copilot**: RUL prediction + anomaly detection + causal RCA + agentic decision support

## Overview

This project combines deep learning for Remaining Useful Life (RUL) prediction and anomaly detection with causal reasoning and LLM-based agents to create an intelligent maintenance copilot.

### Key Features
- **RUL Prediction**: Transformer/LSTM models on NASA C-MAPSS turbofan data
- **Anomaly Detection**: Multi-method approach on NAB benchmark
- **Causal Root-Cause Analysis**: Granger causality + SCM for explainable RCA
- **Maintenance Copilot Agent**: RAG-enhanced LLM agent with abstention-aware recommendations

## Novel Contributions
1. **Causal + GenAI decision support**: Grounded action recommendations via causal pathways + manual retrieval
2. **Counterfactual maintenance planning**: "What-if" scenarios for maintenance optimization
3. **Abstention-aware agents**: Safety-first recommendations with uncertainty quantification

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
bash scripts/download_data.sh

# Preprocess data
python scripts/preprocess_all.py

# Run baseline experiments (W1-2)
python experiments/w1_2_cmapss_baseline/train_baseline.py
```

## Project Timeline (12 weeks)

- **W1-2**: C-MAPSS preprocessing + RUL baselines
- **W3-5**: Transformer/LSTM RUL models + uncertainty
- **W6-7**: Anomaly detection on NAB
- **W8-9**: Causal RCA baseline + integration
- **W10-11**: Agentic copilot with RAG
- **W12**: Evaluation + paper report

## Datasets

- **NASA C-MAPSS**: Turbofan run-to-failure sensor data
- **Numenta Anomaly Benchmark (NAB)**: Labeled time series anomalies

## Evaluation Metrics

- **RUL**: RMSE, MAE, PHM scoring functions, calibration
- **Anomaly/RCA**: Precision, recall, time-to-detect, RCA hit-rate
- **Agent**: Action correctness, groundedness, false reassurance rate

## References

- Saxena & Goebel (2008): C-MAPSS dataset
- Lavin & Ahmad (2015): NAB benchmark
- Recent causal RCA literature (see `docs/paper_outline.md`)
