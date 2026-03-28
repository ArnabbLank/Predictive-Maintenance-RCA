# Week 2 — ML Baselines + Feature Engineering

## What Are We Doing?
We're building simple, traditional machine learning models (no deep learning yet) to establish a performance baseline. Before using fancy neural networks, you should always see how far simpler methods can go.

## Why Are We Doing This?
- To have a reference point: if LSTM gets MAE=12, is that good? Well, Random Forest gets MAE=10.8, so LSTM is only slightly worse with a different approach.
- To learn feature engineering: converting time-series data into a flat table that sklearn models can understand.
- To understand the NASA scoring function and evaluation metrics.

## What's Happening in the Feature Engineering Section?

### The Problem
Scikit-learn models (Linear Regression, Random Forest, etc.) expect ONE row per sample with fixed columns. But our data has MANY rows per engine (one per cycle). Engine 1 might have 200 rows, Engine 50 might have 180 rows.

### The Solution — `extract_features()`
For each engine, look at the **last 30 cycles** of sensor data and compute summary statistics:
- **Last value** of each sensor (most recent reading)
- **Mean** of each sensor over those 30 cycles
- **Std (standard deviation)** — how much the sensor varies
- **Min, Max** — extreme values
- **Slope** — is the sensor trending up or down? (linear fit)
- **Exponentially weighted mean** — recent values weighted more than older ones
- **Rate of change** — how much the sensor changed in the last 5 cycles

This converts each engine from ~200 rows × 26 columns into **1 row × ~130 columns**. Now sklearn can work with it.

### Why "last 30 cycles"?
Because degradation signals are strongest near failure. The last 30 cycles contain the most useful information about the engine's health. This also aligns with the window size used for deep learning models later.

## Evaluation Metrics

### MAE (Mean Absolute Error)
Average of |predicted RUL - true RUL|. If MAE=14, on average we're off by 14 cycles.

### RMSE (Root Mean Squared Error)
Similar to MAE but penalizes large errors more. If one prediction is off by 50 cycles, RMSE makes that hurt more than MAE does.

### NASA Score (Scoring Function)
An **asymmetric** scoring function where:
- **Late predictions** (saying engine has MORE life than it actually does) are penalized MORE.
- **Early predictions** (saying engine has LESS life) are penalized LESS.

Why? Because in aviation, saying "the engine is fine" when it's about to fail (late prediction) is MUCH more dangerous than saying "replace it early" (early prediction → just extra cost, no safety risk).

Formula: If the error $d = \text{predicted} - \text{true}$:
- If $d < 0$ (early): $s = e^{-d/13} - 1$ (mild penalty)
- If $d \geq 0$ (late): $s = e^{d/10} - 1$ (harsh penalty)

Lower score = better.

## About the Zero-Prediction Bug (Question 9)

The screenshot shows all predicted RUL values near 0. This is a known issue that can happen when:
1. The features are not scaled/normalized properly
2. The model fails to learn and defaults to predicting the most common value
3. There's a data leakage or preprocessing error

Looking at the second screenshot (the one showing Predicted vs True scatter plots with MAE of 75.5 and RMSE of 86.2 for Linear Regression and Ridge), this appears to be from a different run or visualization section where models predicted near-zero values consistently. The **working** baselines (shown in the other plot with MAE ~14) are correct.

If you see predictions stuck at 0, check:
- Are you using the correct feature columns?
- Is the RUL column present in the training features?
- Are there NaN values in the features?

## Results
| Model | Val MAE | Test MAE | Test RMSE |
|-------|---------|----------|-----------|
| Linear Regression | — | 14.7 | 18.1 |
| Ridge (alpha=10) | — | 16.1 | 19.6 |
| Random Forest | — | 10.8 | 14.4 |
| Gradient Boosting | — | 10.4 | 14.4 |

Random Forest and Gradient Boosting significantly outperform linear models, showing that the relationship between sensor features and RUL is non-linear.
