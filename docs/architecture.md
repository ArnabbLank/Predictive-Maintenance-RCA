# Project Architecture

## System Components

### 1. Data Pipeline
- **C-MAPSS Loader**: Preprocesses NASA turbofan data, adds RUL labels
- **NAB Loader**: Handles streaming anomaly benchmark data
- **Normalization**: Z-score normalization using train statistics

### 2. RUL Prediction Models
- **Baseline**: Linear regression, Random Forest
- **LSTM**: Sequence-to-one RUL prediction
- **Transformer**: Multi-head attention for temporal patterns
- **Uncertainty**: Monte Carlo dropout / ensemble methods

### 3. Anomaly Detection
- **Autoencoder**: Reconstruction error threshold
- **Forecasting**: Prediction error-based detection
- **Contrastive**: Self-supervised representation learning

### 4. Causal Root-Cause Analysis
- **Granger Causality**: Time-lagged causal discovery
- **SCM**: Structural causal model with interventions
- **Counterfactual**: "What-if" scenario generation

### 5. Maintenance Copilot Agent
```
Input: Sensor data stream
  ↓
RUL Model → Uncertainty check
  ↓
Anomaly Detector → Alert if anomaly
  ↓
Causal RCA → Identify root causes
  ↓
RAG Retriever → Fetch relevant manuals
  ↓
LLM Generator → Grounded recommendations
  ↓
Abstention Logic → Escalate if uncertain
  ↓
Output: Actionable maintenance plan
```

## Novel Contributions

### 1. Causal + GenAI Decision Support
- Each LLM-generated action must link to:
  - A causal pathway from the learned graph
  - A retrieved manual snippet (citation required)
- Prevents hallucinated recommendations

### 2. Counterfactual Maintenance Planning
- "If sensor X reduced by Δ → RUL increases by Y"
- Uses causal model for intervention simulation
- LLM translates counterfactuals into maintenance steps

### 3. Abstention-Aware Agents
- Agent refuses to recommend when:
  - RUL uncertainty > threshold
  - No causal pathway found
  - No manual citation available
- Measures: false reassurance rate, escalation accuracy

## Evaluation Framework

### RUL Metrics
- RMSE, MAE on test set
- PHM08 scoring function (asymmetric penalty)
- Calibration plots for uncertainty

### Anomaly/RCA Metrics
- Precision, recall, F1
- Time-to-detect (latency)
- RCA hit-rate (% correct root cause)

### Agent Metrics
- Action correctness (expert rubric)
- Groundedness (% with valid citations)
- False reassurance rate
- Abstention appropriateness
