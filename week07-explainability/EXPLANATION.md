# Week 7 — Explainability: "Why Did the Model Say That?"

## What Are We Doing?
Making the model's predictions interpretable. Instead of just saying "RUL = 50", we now say "RUL = 50, mainly because sensor_11 (HPC outlet pressure) is dropping and sensor_4 (HPC outlet temperature) is rising."

## Why Are We Doing This?
- A maintenance engineer won't trust a prediction without understanding WHY.
- Regulations may require explainability for safety-critical AI.
- Explanations can help catch model errors: if the model says "sensor_1 is most important" but sensor_1 is constant, something is wrong.

## Methods

### 1. Gradient Saliency
- Give the model an input, get a prediction.
- Ask: "If I slightly change each input value, how much does the output change?"
- Mathematically: compute the gradient (derivative) of the output with respect to each input feature.
- Sensors where the gradient is large = the model is sensitive to those sensors = they are "important."

**Simple analogy**: You're adjusting 14 volume knobs. Saliency tells you which knobs actually change the sound.

### 2. Integrated Gradients (more robust)
- Saliency can be noisy. Integrated Gradients is a more reliable version.
- Instead of computing the gradient at one point, it computes gradients along a path from a "baseline" (zero input) to the actual input, and averages them.
- This gives a more stable, meaningful attribution.

### 3. Temporal Saliency
- Goes beyond "which sensors matter" to "which TIME STEPS matter."
- Maybe the model pays most attention to the last 5 cycles (most recent data) and ignores the first 10 cycles.
- Helps understand the model's "attention horizon."

## Output Format (for the Copilot)
```json
{
  "top_sensors": [
    {"sensor": "sensor_11", "importance": 0.34, "rank": 1},
    {"sensor": "sensor_4",  "importance": 0.22,  "rank": 2},
    {"sensor": "sensor_15", "importance": 0.18,  "rank": 3}
  ],
  "method": "integrated_gradients"
}
```

## Results
- Top sensors typically include sensor_11 (P30), sensor_4 (T30), sensor_15 (Ps30) — all related to HPC degradation, which matches the known fault mode in FD001.
- This confirms the model is learning physics-meaningful patterns, not just memorizing noise.
