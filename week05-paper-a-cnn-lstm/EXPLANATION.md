# Week 5 — Paper A: CNN+LSTM with Savitzky-Golay Smoothing

## What Are We Doing?
Implementing the model from "Paper A" — a CNN+LSTM architecture tested with and without Savitzky-Golay (SG) smoothing on the sensor data.

## Why Are We Doing This?
- To replicate a published approach and see if it beats our plain LSTM.
- To understand whether smoothing noisy sensor data helps deep learning models.
- To compare 4 configurations: LSTM-raw, LSTM-SG, CNN+LSTM-raw, CNN+LSTM-SG.

## What is Savitzky-Golay Smoothing?
A mathematical filter that smooths data while trying to preserve the shape of peaks and trends. Unlike a simple moving average (which tends to flatten everything), SG smoothing fits a polynomial to a local window of points and uses that polynomial to estimate the smoothed value.

**In simple terms**: It's like redrawing a noisy line with a smoother pen, while trying not to lose the important bumps and dips.

## CNN+LSTM Architecture
```
Input (30 cycles × 14 sensors)
    → 1D CNN layers (detect local patterns in each time step)
    → LSTM layers (capture how patterns evolve over time)
    → Fully Connected → RUL prediction
```

The CNN acts as a feature extractor that creates a richer representation for each time step, which the LSTM then processes sequentially.

## Results

| Model | MAE |
|-------|-----|
| LSTM (raw) | 12.1 |
| LSTM (SG smoothed) | 20.0 |
| CNN+LSTM (raw) | 19.4 |
| CNN+LSTM (SG smoothed) | 21.7 |

### Surprising finding:
The plain LSTM on raw data beat all other configurations! This could be because:
1. SG smoothing might remove useful high-frequency signal along with noise.
2. Our CNN+LSTM might need more tuning (different learning rate, kernel sizes, etc.).
3. The original paper may have used different preprocessing or hyperparameters.

This is a valid and honest finding — not all methods from papers improve things in every setting.
