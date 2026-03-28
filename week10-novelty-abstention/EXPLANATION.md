# Week 10 — Novelty Detection, Abstention Logic & Calibration

## What Are We Doing?
Three advanced topics:
1. **Novelty/OOD detection**: Can we detect inputs that look "weird" compared to training data?
2. **Calibration**: When the model says "95% confidence interval", does it actually cover 95% of cases?
3. **Abstention**: Formalizing when the model should refuse to predict.

## Why Are We Doing This?
- A model trained on FD001 has never seen FD002/FD003/FD004 engines. If you feed it data from a different fault mode, it might confidently give a wrong answer.
- Calibration tells us if the uncertainty estimates are trustworthy.
- Abstention logic saves lives in safety-critical applications.

## 1. Novelty / Out-of-Distribution (OOD) Detection

### What is it?
Detecting inputs that are "too different" from what the model was trained on.

### How? — Mahalanobis Distance
- Compute the mean and covariance of the training data features.
- For a new input, calculate its Mahalanobis distance from the training distribution.
- If the distance is very large → this input is "novel" / out-of-distribution → don't trust the prediction.

**Simple analogy**: If you trained a pizza-identifying AI only on cheese and pepperoni pizzas, and someone shows it sushi, it should say "I don't know what this is" instead of "this is a pepperoni pizza."

## 2. Calibration

### What is it?
Checking if predicted confidence intervals are honest.

If we compute 95% confidence intervals for all 100 test engines, do approximately 95 of those intervals actually contain the true RUL? If only 70 do, the model is **overconfident**. If 99 do, the model is **underconfident** (intervals are too wide).

### Calibration Curve
A plot of "expected coverage" vs "actual coverage" at different confidence levels (50%, 60%, ..., 95%). A perfectly calibrated model follows the y=x diagonal.

## 3. Abstention Policy

### Formal Rule:
```
If uncertainty (std) > threshold  → ABSTAIN (escalate to human)
If Mahalanobis distance > threshold → ABSTAIN (input is OOD)
Otherwise → predict and recommend
```

### Analysis:
By varying the abstention threshold:
- Strict threshold (abstain on 50% of engines) → remaining predictions are very accurate
- Loose threshold (abstain on 5% of engines) → some bad predictions slip through
- The goal is to find the sweet spot that maximizes safety while minimizing unnecessary abstentions.

## Results
- OOD detection can identify engines with unusual sensor patterns.
- Calibration analysis shows whether 95% CIs are reliable.
- Abstention rate vs accuracy trade-off is quantified.
