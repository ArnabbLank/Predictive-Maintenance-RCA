# Predictive Maintenance with Causal Root-Cause Analysis

### Real-world problem
Factories and fleets want to predict failures early (**Remaining Useful Life**) and explain **why** anomalies occurred (root cause), not just detect them.

### Datasets
- **NASA C-MAPSS turbofan run-to-failure** (sensor time series with operating conditions; standard RUL benchmark).  
  Hosted by NASA (DASHlink / NASA Open Data).  
- **Numenta Anomaly Benchmark (NAB)** for streaming anomaly detection (labeled time series).  
- Optional: any public multivariate industrial dataset (SKAB, SWaT/WADI if access fits your constraints).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets
python download_data.py

```

### Core methods (non-GenAI)
- **RUL prediction**:
  - Baselines: linear degradation models, random forest, LSTM.
  - Strong DL: transformer-based RUL models; temporal CNN + attention; survival modeling.
- **Anomaly detection**:
  - Forecasting-based (DeepAR-style), reconstruction-based (autoencoders), contrastive methods.
- **Causal root cause**:
  - Granger causal discovery + intervention framing; SCM-inspired anomaly RCA.

### Agentic / GenAI layer
- “**Maintenance copilot**” agent that:
  1. Ingests anomaly alerts / RUL forecasts,
  2. Calls tools for explanation (SHAP, attention rollouts, causal graph queries),
  3. Retrieves a knowledge base (maintenance manuals / fault trees) via RAG,
  4. Outputs **actionable recommendations** with confidence + when to escalate to humans.

### Novel / paper-worthy angle (choose 1–2)
- **Causal + GenAI decision support**: combine a learned causal graph with LLM-generated actions, but require each action to link to a causal pathway and retrieved manual snippet.
- **Counterfactual maintenance planning**: “If sensor X were reduced by Δ, predicted RUL increases by …” using causal modeling; LLM turns these into maintenance steps.
- **Abstention-aware agents**: teach the agent to *not* recommend actions when uncertainty is high; measure safety.

### Evaluation
- RUL: RMSE/MAE on RUL, scoring functions used in PHM literature; calibration of uncertainty.
- Anomaly/RCA: precision/recall, time-to-detect, RCA hit-rate.
- Agent: correctness of recommended actions (expert rubric), groundedness (manual citations), false reassurance rate.

### 12-week timeline
- **W1–2**: C-MAPSS preprocessing; RUL baselines.
- **W3–5**: transformer/LSTM strong model + uncertainty.
- **W6–7**: anomaly detection on NAB (+ optionally multivariate).
- **W8–9**: causal RCA baseline; integrate into pipeline.
- **W10–11**: agentic copilot with RAG + action templates; novelty experiment.
- **W12**: evaluation, ablations, paper-style report.

### Starter papers / references
- C-MAPSS dataset note (Saxena & Goebel, 2008 citation via NASA dataset page).
- NAB benchmark paper (Lavin & Ahmad, arXiv 2015) + NAB repo.
- Recent causal RCA papers (e.g., causal anomaly + root cause framing; Granger-based RCA).
