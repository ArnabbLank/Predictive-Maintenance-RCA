# Week 6 — Paper B: CNN-Transformer + Uncertainty Quantification

## What Are We Doing?
Implementing the CNN-Transformer model from "Paper B" and adding MC Dropout for uncertainty estimation.

## Why Are We Doing This?
- Transformers with attention mechanisms can look at ALL time steps simultaneously (unlike LSTM which reads sequentially).
- MC Dropout gives us uncertainty estimates — the model can say "I'm not sure" instead of just giving one number.
- Uncertainty is critical for safety-critical applications like aircraft maintenance.

## What's New Here vs Week 4/5?

### Transformer instead of LSTM
- LSTM reads data left-to-right, one step at a time.
- Transformer uses **self-attention**: for each time step, it computes how relevant every OTHER time step is. It can directly connect cycle 1 to cycle 30 without going through all intermediate steps.

### MC Dropout (Monte Carlo Dropout)
- Normal prediction: run the model once, get one number.
- MC Dropout: run the model 50-100 times with random neurons turned off each time.
- You get a distribution of predictions → mean = estimate, std = uncertainty.

### Uncertainty Levels
| Uncertainty (std) | Level | Meaning |
|---|---|---|
| < 5 cycles | Low | Model is confident |
| 5-10 cycles | Medium | Reasonable confidence |
| 10-20 cycles | High | Be cautious |
| > 20 cycles | Very High | Don't trust this prediction → escalate |

## Results
- The CNN-Transformer achieves competitive MAE/RMSE with the LSTM.
- The uncertainty-error correlation is ~0.35: higher uncertainty tends to correlate with larger errors (this is good — it means the model "knows" when it's wrong).
- 95% confidence intervals are visualized: most true RUL values fall within the predicted CI.
- The MC Dropout distribution plots show how predictions spread for different engines.

## Key Plots
- **Training curves**: Loss decreases and converges by ~10 epochs.
- **Uncertainty vs Error**: Positive correlation — model knows when it's uncertain.
- **Predictions with 95% CI**: Error bars show the range of possible RUL values.
- **MC Dropout distributions**: Histograms showing the spread of 100 forward passes per engine.
