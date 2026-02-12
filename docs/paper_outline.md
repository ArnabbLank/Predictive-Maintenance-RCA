# Paper Outline: Causal-Grounded Maintenance Copilot

## Abstract
- Problem: Predictive maintenance needs explainability + actionable recommendations
- Solution: Combine RUL/anomaly models with causal RCA and LLM-based copilot
- Novel: Causal grounding + abstention-aware agent + counterfactual planning
- Results: [To be filled with metrics]

## 1. Introduction
- Industrial maintenance challenges: false alarms, unexplained predictions
- Need for: early failure prediction (RUL) + root cause + safe recommendations
- Our approach: End-to-end system with causal reasoning and LLM agent

## 2. Related Work
### 2.1 RUL Prediction
- Traditional: degradation models, survival analysis
- Deep learning: LSTM, CNN, Transformers on C-MAPSS
- Gap: lack of uncertainty quantification

### 2.2 Anomaly Detection
- Forecasting-based, reconstruction-based, contrastive
- NAB benchmark for streaming evaluation

### 2.3 Causal Root-Cause Analysis
- Granger causality for time series
- SCM for intervention reasoning
- Gap: not integrated with LLM decision support

### 2.4 LLM Agents for Operations
- Tool-calling agents, RAG for domain knowledge
- Gap: no causal grounding, hallucination risk

## 3. Methodology
### 3.1 RUL Prediction with Uncertainty
- Transformer architecture
- Monte Carlo dropout for uncertainty
- Calibration techniques

### 3.2 Anomaly Detection
- Multi-method ensemble
- Threshold tuning on NAB

### 3.3 Causal Root-Cause Analysis
- Granger causal discovery
- Intervention-based RCA
- Counterfactual generation

### 3.4 Maintenance Copilot Agent
- Architecture diagram
- Tool calling: SHAP, causal queries, RAG
- Abstention logic
- Grounding requirements

## 4. Experiments
### 4.1 Datasets
- C-MAPSS (4 subsets)
- NAB (streaming anomalies)

### 4.2 RUL Results
- Baseline vs. LSTM vs. Transformer
- Uncertainty calibration

### 4.3 Anomaly Detection Results
- Precision/recall on NAB
- Time-to-detect analysis

### 4.4 Causal RCA Results
- RCA hit-rate
- Counterfactual validation

### 4.5 Agent Evaluation
- Expert rubric for action correctness
- Groundedness analysis
- Abstention appropriateness
- Ablation: with/without causal grounding

## 5. Novel Contributions
### 5.1 Causal + GenAI Decision Support
- Requirement: action → causal pathway + manual citation
- Reduces hallucination

### 5.2 Counterfactual Maintenance Planning
- "What-if" scenarios for optimization
- Case studies

### 5.3 Abstention-Aware Agents
- Safety-first design
- False reassurance rate < X%

## 6. Discussion
- Limitations: dataset scope, LLM costs
- Future work: online learning, multi-modal sensors

## 7. Conclusion
- First system combining RUL + causal RCA + grounded LLM agent
- Demonstrates safe, explainable maintenance recommendations

## References
- Saxena & Goebel (2008): C-MAPSS
- Lavin & Ahmad (2015): NAB
- Recent causal RCA papers
- LLM agent papers (ReAct, Toolformer)
