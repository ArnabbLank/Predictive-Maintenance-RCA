# Week 3 — Health Index (HI) Baseline

## What Are We Doing?
Building a "health score" for each engine that goes from 1.0 (perfectly healthy) to 0.0 (about to fail). This is a physics-inspired approach — instead of directly predicting RUL, we first estimate "how healthy is the engine?" and then map that to remaining life.

## Why Are We Doing This?
- It's more interpretable than a black-box model: "engine health is at 40%" is intuitive.
- It provides a sanity check: if our Health Index doesn't decrease over time, something is wrong with our data processing.
- It's a stepping stone to understanding degradation curves.

## Three Methods

### 1. PCA-based Health Index
**PCA = Principal Component Analysis**

Think of it this way: you have 14 sensors, each telling you something slightly different about the engine. PCA finds the ONE direction in this 14-dimensional space that captures the most variation across all sensors.

**Steps:**
1. Scale all 14 sensors to [0, 1] range (so no sensor dominates by its scale).
2. Run PCA with 1 component — this gives a single number per time step that summarizes all 14 sensors.
3. Normalize this number per engine so that:
   - Start of life → HI ≈ 1.0
   - End of life → HI ≈ 0.0

**In simple terms**: PCA finds the "summary axis" that best separates healthy from unhealthy states. The first principal component usually captures the overall degradation trend since all sensors tend to drift in correlated ways as the engine degrades.

### 2. Weighted Sensor Sum
A simpler approach:
1. Pick sensors that are known to correlate with degradation (from the correlation matrix in Week 1).
2. Assign weights based on how strongly each sensor correlates with RUL.
3. Health Index = weighted sum of normalized sensor values.

**In simple terms**: It's a manual version of what PCA does automatically. You just pick the most important sensors, weight them by importance, and add them up.

### 3. Degradation Curve Fitting
1. Take the Health Index (from PCA or weighted sum) for each engine.
2. Fit a mathematical curve (like an exponential decay) to it.
3. Extrapolate: at what cycle does the curve reach the failure threshold?
4. RUL = predicted failure cycle − current cycle.

**In simple terms**: If health is dropping like a curve, extend that curve into the future to estimate when the engine will fail.

## How Are We Calculating It?

### PCA Health Index Calculation:
```python
# 1. Scale sensors to [0,1]
scaler = MinMaxScaler()
scaled_sensors = scaler.fit_transform(df[sensor_columns])

# 2. PCA: 14 sensors → 1 number
pca = PCA(n_components=1)
hi_raw = pca.fit_transform(scaled_sensors)

# 3. Normalize per engine: first cycle = 1.0, last cycle = 0.0
for each engine:
    hi = (hi - min) / (max - min)  # scale to [0, 1]
    hi = 1 - hi  # flip if needed so 1 = healthy
```

## Results
- PCA Health Index shows a clear monotonic decrease over engine lifetime — confirms the degradation signal exists.
- Health Index vs RUL plot shows a positive correlation: higher HI = more life remaining.
- Curve fitting can estimate RUL, but with lower accuracy than ML baselines.
