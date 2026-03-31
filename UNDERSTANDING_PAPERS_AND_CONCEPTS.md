# Understanding Paper A, Paper B, Abstention, Ablation & The Final App

> A plain-English guide for someone new to this project.

---

## Paper A — CNN+LSTM with Savitzky-Golay Smoothing

### What is it about?
Paper A proposes combining **two neural network types** (CNN + LSTM) and **smoothing the data first** to better predict how many cycles an engine has left before it fails.

### The idea in simple terms:
1. **Problem**: Sensor data from engines is noisy (jumpy up and down). This noise makes it hard for a model to see the real degradation trend.
2. **Solution 1 — Savitzky-Golay (SG) Smoothing**: Before feeding data into the model, smooth out the random noise. Think of it like looking at a stock price through a moving average instead of raw second-by-second ticks. The underlying trend becomes clearer.
3. **Solution 2 — CNN+LSTM architecture**:
   - **CNN (Convolutional Neural Network)**: Good at detecting local patterns in the sensor readings — like a spike or a gradual shift in one part of the data.
   - **LSTM (Long Short-Term Memory)**: Good at remembering long-term trends — like "this sensor has been slowly increasing over the last 50 cycles."
   - Combining them: CNN extracts local features → LSTM captures how those features evolve over time → predict RUL.

### Our results:
- LSTM (raw data) achieved MAE ≈ 12.1
- LSTM (SG smoothed) achieved MAE ≈ 20.0
- CNN+LSTM (raw) achieved MAE ≈ 19.4
- CNN+LSTM (SG smoothed) achieved MAE ≈ 21.7

**Observation**: In our experiments, the plain LSTM on raw data actually performed best. Smoothing + CNN didn't always help — this is an honest finding, and a good discussion point. The original paper used different hyperparameters and datasets, so results can vary.

---

## Paper B — CNN-Transformer with MC Dropout (Uncertainty)

### What is it about?
Paper B replaces the LSTM with a **Transformer** (the same architecture behind ChatGPT, but much smaller) and adds **uncertainty estimation** so the model can say "I'm not sure" instead of giving a single number.

### The idea in simple terms:
1. **Transformer**: Unlike LSTM which reads data step-by-step (left to right), a Transformer uses **attention** — it can look at ALL time steps at once and decide which ones matter most. This is often more powerful for sequential data.
2. **MC Dropout (Monte Carlo Dropout)**: 
   - During training, dropout randomly "turns off" some neurons (to prevent memorization).
   - Normally, dropout is turned OFF during prediction. But MC Dropout **keeps it on**.
   - You run the same input through the model 50-100 times, each time with different neurons randomly turned off.
   - You get 50-100 slightly different predictions. 
   - **Mean of predictions** = your RUL estimate.
   - **Standard deviation** = how uncertain the model is. If all 100 runs agree → low uncertainty. If they're spread out → high uncertainty.
3. **Why this matters**: Telling a maintenance engineer "RUL = 50 cycles" is less useful than "RUL = 50 ± 5 cycles (confident)" or "RUL = 50 ± 30 cycles (NOT confident, check manually)."

### Architecture:
```
Input sensor window (30 cycles × 14 sensors)
    → 1D CNN (extract local patterns)
    → Positional Encoding (tell Transformer which time step is which)
    → Transformer Encoder (attention over all time steps)
    → Global Average Pooling
    → Dropout (kept ON during prediction for MC Dropout)
    → Fully Connected layer → RUL prediction
```

---

## Abstention — "I Don't Know, Ask a Human"

### What is it?
Abstention means the model **refuses to make a recommendation** when it's not confident enough, and instead **escalates to a human expert**.

### Why?
In safety-critical applications (like aircraft maintenance), a wrong prediction can be catastrophic:
- If the model says "80 cycles left" but the engine actually fails in 10 cycles → disaster.
- It's better for the model to say "I'm not sure, please have an expert check" than to give a wrong confident answer.

### How does it work in our project?
1. The CNN-Transformer makes a prediction with MC Dropout.
2. If the uncertainty (standard deviation) exceeds a threshold (default: 20 cycles) → **ESCALATE**. Don't give automated maintenance advice.
3. If uncertainty is low → give the recommendation (maintenance, continue monitoring, etc.)

### Example:
- Engine 42: Predicted RUL = 25 ± 4 cycles → **RECOMMEND**: Schedule immediate maintenance.
- Engine 17: Predicted RUL = 70 ± 25 cycles → **ESCALATE**: Uncertainty too high. Manual inspection needed.

---

## Ablation Study — "What Actually Helps?"

### What is it?
An ablation study systematically removes or changes ONE component at a time to see how much each part contributes.

### Think of it like cooking:
You made a great dish with 5 spices. To know which spices matter:
- Remove spice 1, taste it. Does it get worse? → Spice 1 was important.
- Remove spice 2, taste it. Same quality? → Spice 2 wasn't needed.

### What we test in our ablation:
| What we change | Why |
|---|---|
| **Window size** (10, 20, 30, 40, 50) | Does looking at more history help? |
| **Savitzky-Golay smoothing** (on/off) | Does smoothing the sensor data help? |
| **Number of attention heads** (2, 4, 8) | Does the Transformer need more "thinking lanes"? |
| **MC Dropout samples** (10, 50, 100, 200) | More samples = better uncertainty estimate, but slower |
| **Model type** (LSTM vs CNN+LSTM vs CNN-Transformer) | Which architecture is best? |

---

## How the Final App Looks (Vision)

The final Streamlit app (`streamlit-app/app.py`) is an interactive dashboard with these pages:

### Page 1: Overview
A summary page showing the model being used (CNN-Transformer), number of parameters, and the UQ method (MC Dropout).

### Page 2: Single Engine Analysis
- Pick an engine from a dropdown.
- Click "Run Copilot Analysis."
- **4-step pipeline runs live**:
  1. **Predict**: Shows RUL mean, uncertainty, 95% confidence interval.
  2. **Explain**: Bar chart of which sensors drove the prediction (Integrated Gradients).
  3. **Retrieve**: Snippets from the knowledge base relevant to the degradation.
  4. **Recommend**: A written recommendation — or an escalation alert if uncertainty is too high.
- Visualization: scatter plot with confidence interval, sensor importance bar chart.

### Page 3: Fleet Dashboard
- Runs the Copilot on ALL 100 test engines at once.
- Shows a heatmap/table: which engines need maintenance, which are safe, which are uncertain.
- Summary stats: how many escalated, how many critical, fleet average RUL.

### Page 4: About
Technical details, model description, dataset reference.

### What it looks like in practice:
```
┌──────────────────────────────────────────────────────────────┐
│  ⚙️ Ops Copilot                                              │
│                                                              │
│  Select Engine: [Engine 42 (True RUL = 25)]  ▾              │
│  MC Samples: [====100====]                                    │
│  Escalation Threshold: [====20.0====]                        │
│                                                              │
│  [🚀 Run Copilot Analysis]                                   │
│                                                              │
│  ─────────── Engine 42 — Copilot Report ───────────         │
│                                                              │
│  Predicted RUL │ True RUL │ Error │ Uncertainty │ Level      │
│     27 cycles  │    25    │  2.0  │    4.3      │  LOW       │
│                                                              │
│  📊 Sensor Importance:                                       │
│  ████████████ sensor_11 (0.34)                               │
│  ████████     sensor_4  (0.22)                               │
│  ██████       sensor_15 (0.18)                               │
│                                                              │
│  📚 KB Reference:                                            │
│  "RUL < 30 cycles: Immediate maintenance — compressor wash   │
│   or blade replacement recommended."                         │
│                                                              │
│  ✅ RECOMMEND: CRITICAL — Schedule immediate maintenance.    │
│  Primary degradation signal: sensor_11 (HPC outlet pressure) │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Concept | One-Line Summary |
|---|---|
| **Paper A** | CNN+LSTM + Savitzky-Golay smoothing for cleaner, more accurate RUL prediction |
| **Paper B** | CNN-Transformer + MC Dropout for RUL prediction WITH uncertainty |
| **Abstention** | Model says "I don't know" when uncertainty is too high → escalates to human |
| **Ablation** | Systematically test each component to see what actually helps |
| **Final App** | Streamlit dashboard: pick an engine → predict → explain → recommend (or escalate) |
