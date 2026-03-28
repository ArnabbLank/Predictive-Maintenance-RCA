# Novel Approaches for the Paper

This document describes **simple yet effective novelty ideas** that go beyond
replicating Paper A and Paper B. Each is designed to be implementable within the
student's 12-week timeline and contribute a clear, citable contribution.

---

## Primary Novelty: Uncertainty-Aware Copilot with Abstention

**Already implemented** in `src/copilot/` and `week10-novelty-abstention/`.

| Aspect | Detail |
|--------|--------|
| What | Agent abstains from recommending when MC Dropout std > threshold |
| Why novel | Prior work reports uncertainty but doesn't integrate it into a decision system |
| Metric | Coverage vs MAE curve, False Reassurance Rate |
| Effort | Low — already coded |

---

## Secondary Novelties (pick 1–2 to strengthen the paper)

### 1. Ensemble Uncertainty Fusion

**Idea**: Combine predictions from multiple architectures (LSTM, CNN+LSTM,
CNN-Transformer) and use **disagreement** as an additional uncertainty signal.

```python
# Pseudocode
preds = [model_lstm(x), model_cnn_lstm(x), model_transformer(x)]
ensemble_mean = np.mean(preds)
ensemble_std = np.std(preds)  # model disagreement
mc_std = model_transformer.predict_with_uncertainty(x).std
combined_uncertainty = 0.5 * mc_std + 0.5 * ensemble_std
```

**Why it's novel**: Most C-MAPSS papers use a single model's uncertainty.
Combining epistemic (MC Dropout) + model disagreement gives richer signal.

**Effort**: Low — you already have all 3 models trained.

---

### 2. Adaptive RUL Capping

**Idea**: Instead of a fixed cap (125), learn the optimal cap per operating
condition or let the model learn it via a learnable saturation function.

```python
# Replace hard cap with a soft sigmoid saturation
class AdaptiveCap(nn.Module):
    def __init__(self, init_cap=125.0):
        super().__init__()
        self.cap = nn.Parameter(torch.tensor(init_cap))

    def forward(self, raw_rul):
        return self.cap * torch.sigmoid(raw_rul / self.cap)
```

**Why it's novel**: The 125 cap is an arbitrary heuristic from the literature.
Learning it could improve performance and provides a small, testable contribution.

**Effort**: Very low — one extra module.

---

### 3. Temporal Attention Weights as Explanation

**Idea**: Extract attention weights from the Transformer encoder and use them
as a built-in explanation (which time steps matter most).

```python
# During forward pass, return attention weights
attn_weights = transformer_encoder_layer.self_attn(q, k, v, need_weights=True)[1]
# attn_weights shape: (batch, nhead, seq_len, seq_len)
# Average over heads → per-timestep importance
temporal_importance = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)
```

**Why it's novel**: Paper B doesn't analyze attention patterns. Showing that the
model "looks at" late degradation cycles validates the prediction.

**Effort**: Low — modify forward pass to return weights.

---

### 4. Conformal Prediction Intervals (distribution-free UQ)

**Idea**: Instead of assuming Gaussian uncertainty (MC Dropout mean ± z·std),
use **conformal prediction** to get guaranteed coverage.

```python
# Calibration step (on validation set)
val_residuals = np.abs(y_val - y_pred_val)
q_hat = np.quantile(val_residuals, 0.95)  # 95% quantile

# At test time
lower = y_pred - q_hat
upper = y_pred + q_hat
# Guaranteed 95% coverage (no distributional assumption)
```

**Why it's novel**: MC Dropout intervals are not calibrated by default.
Conformal prediction gives **finite-sample, distribution-free** guarantees.
This is a hot topic in ML safety and very publishable.

**Effort**: Low — ~20 lines of code. Compare conformal intervals vs MC Dropout intervals.

---

### 5. Multi-Task Learning: RUL + Fault Mode Classification

**Idea**: Add a secondary classification head that predicts the fault mode
(useful for FD003/FD004 which have 2 fault modes).

```python
class MultiTaskModel(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone
        self.rul_head = nn.Linear(128, 1)
        self.fault_head = nn.Linear(128, 2)  # 2 fault modes

    def forward(self, x):
        features = self.backbone(x)
        rul = self.rul_head(features)
        fault = self.fault_head(features)
        return rul, fault
```

**Why it's novel**: Forces the model to learn representations useful for both
tasks. Could improve RUL prediction on multi-fault datasets.

**Effort**: Medium — need FD003/FD004 data + training loop changes.

---

## Recommended Combination for the Paper

For maximum impact with minimum effort:

1. **Primary**: Uncertainty-Aware Copilot with Abstention ✅ (done)
2. **Secondary A**: Conformal Prediction Intervals (novelty #4)
3. **Secondary B**: Temporal Attention Visualization (novelty #3)

This gives you:
- A **decision system** (copilot + abstention) — practical contribution
- **Distribution-free UQ** (conformal) — theoretical contribution
- **Built-in explainability** (attention) — interpretability contribution

All three are implementable in < 1 week combined.
