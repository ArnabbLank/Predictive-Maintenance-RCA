# Week 1 — Exploratory Data Analysis (EDA)

## What Are We Doing?
We're exploring the NASA C-MAPSS FD001 dataset to understand its structure before building any models. Think of this as "getting to know your data" — you wouldn't build a house without inspecting the land first.

## Why Are We Doing This?
- To understand what each column means
- To see how engines degrade over time
- To identify which sensors are useful and which are just noise
- To decide on preprocessing steps for later weeks

## Key Concepts

### The Dataset
The C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset simulates turbofan jet engines running until failure. We use **FD001**, which is the simplest sub-dataset:
- **100 training engines**: run from healthy to failure (complete lifetime)
- **100 test engines**: run for some time, then cut off BEFORE failure
- **RUL_FD001.txt**: tells us the TRUE remaining life for each test engine at the cutoff point

### Columns (26 total)
| Column | Meaning |
|--------|---------|
| `unit_id` | Engine number (1 to 100). Yes, it means engine number. |
| `cycle` | The operating cycle number — think of it as "day of operation." Cycle 1 is the first day, cycle 200 is the 200th day. |
| `op_setting_1/2/3` | Operating conditions (altitude, Mach number, throttle angle). In FD001 these are nearly constant. |
| `sensor_1` to `sensor_21` | 21 sensor readings (temperature, pressure, speed, etc.) |

### RUL (Target Variable)
**RUL = Remaining Useful Life** = "How many more cycles will this engine survive?"
- For training data: `RUL = max_cycle_of_engine - current_cycle`
- For test data: the true RUL is in `RUL_FD001.txt`

### Why rul_cap=125?
When an engine is brand new (say cycle 1 out of 200), its RUL is 199. But there's no practical difference between RUL=199 and RUL=150 — the engine is "perfectly healthy" in both cases. The cap at 125 means:
- If raw RUL > 125 → set it to 125 (treat as "still healthy")
- If raw RUL ≤ 125 → keep the actual value
- This creates a "piece-wise linear" target: flat at 125 for early cycles, then linearly decreasing to 0
- Almost all academic papers use this convention

### Train vs Test Data
- **Train**: Each engine runs from start to FAILURE. You see the complete degradation story.
- **Test**: Each engine runs from start to SOME POINT before failure. The time series is cut short. Your job is to predict how many cycles remain (the RUL).

### cycles_per_engine
Yes, this tells you the maximum cycle number each engine reached. For **training data**, it's the total lifetime (since engines run to failure). For test data, it's where the recording was cut off. The same engine number does NOT appear in both train and test — they are completely separate sets of 100 engines each.

### "Test Engines — Observed Lifetime" Plot
This shows how many cycles we OBSERVE for each test engine (i.e., how long the recording is before cutoff). Some test engines have 50 cycles of data, others have 300+. This is just how much data we get to work with.

### "Test Engines — True Remaining Life" Plot
This shows the ACTUAL RUL (from RUL_FD001.txt) for each test engine at its cutoff point. Some engines were cut off very close to failure (low RUL ≈ 10), others were cut off while still healthy (high RUL ≈ 140).

### "Window size 30 cycles"
For deep learning models (LSTM, Transformer), we feed in a "window" of recent data — the last 30 consecutive cycles of sensor readings. So the model sees a (30 × 14) matrix: 30 time steps, 14 sensor values each. This window slides along the engine's timeline to create training samples. 30 is a common starting point; we test other sizes in the ablation study.

## Key Findings from EDA
- Mean engine lifetime: ~206 cycles
- 7 out of 21 sensors have near-zero variance → useless → dropped (leaving 14 informative sensors)
- Strong correlations between certain sensors and RUL (e.g., sensor_11, sensor_4, sensor_12 show clear degradation trends)
- Sensors show clear degradation trends in later cycles (approaching failure)
