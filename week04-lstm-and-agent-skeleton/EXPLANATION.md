# Week 4 — LSTM Model + Copilot Agent Skeleton

## What Are We Doing?
**Part A**: Building our first deep learning model — an LSTM (Long Short-Term Memory) — that reads sequences of sensor data to predict RUL.

**Part B**: Defining the "Copilot" concept — a system that doesn't just predict, but also explains, retrieves knowledge, and recommends actions.

## Why Are We Doing This?
- Baseline ML models (Week 2) treated each engine as one flat row. LSTM can look at the **sequence** of 30 consecutive sensor readings to understand how values are changing over time.
- The Copilot skeleton sets up the framework that we'll build on for the next 8 weeks.

## Part A: The LSTM Model

### What is LSTM?
LSTM is a type of neural network designed for sequential data. It reads the sensor values cycle-by-cycle and maintains a "memory" of what it has seen.

**Simple analogy**: Imagine reading a book word by word. An LSTM remembers important plot points from earlier pages while reading the current page. A regular neural network would only see the current page.

### How it works here:
1. **Input**: A window of 30 cycles × 14 sensors = (30, 14) matrix.
2. **LSTM reads it step by step**: cycle 1 → update memory → cycle 2 → update memory → ... → cycle 30.
3. **After reading all 30 steps**: the LSTM's final memory state summarizes the entire sequence.
4. **Output layer**: Converts that summary into a single number → predicted RUL.

### Training:
- **Loss function**: MSE (Mean Squared Error) — punish the model for being wrong.
- **Optimizer**: Adam — adjusts model weights to reduce error.
- **Early stopping**: If validation loss doesn't improve for 10 epochs, stop training (prevents memorizing the training data).
- **Learning rate scheduler**: If loss plateaus, automatically reduce learning rate.

### Results:
- Test MAE: ~12.6
- Test RMSE: ~17.4
- This is competitive with Gradient Boosting (MAE=10.4) from Week 2, showing that the LSTM can learn from raw sequences without hand-crafted features.

## Part B: The Copilot Skeleton

### What are copilot.tools?
They are **4 Python functions** that the Copilot agent calls in sequence:

| Tool | What it does | Input | Output |
|------|-------------|-------|--------|
| `predict_rul()` | Predicts RUL with optional uncertainty | Sensor window (30×14) | `{rul_mean, rul_std, confidence_interval, uncertainty_level}` |
| `explain_prediction()` | Explains which sensors drove the prediction | Sensor window + model | `{top_sensors: [{sensor, importance, rank}], method}` |
| `retrieve_knowledge()` | Searches the knowledge base for relevant info | Text query | `{snippets: [{text, source, score}]}` |
| `recommend_action()` | Generates a maintenance recommendation | Prediction + explanation + KB | `{action: "recommend"/"escalate", recommendation, confidence}` |

### How does the agent work?
The `OpsCopilot` class calls these 4 tools in a fixed order:
1. **Predict** → Get RUL and uncertainty
2. **Explain** → Get top contributing sensors
3. **Retrieve** → Search KB using the top sensor names as query
4. **Recommend** → Generate actionable advice (or escalate if uncertain)

### Is this a "real" agent? Which LLM does it use?
**In Week 4: NO**, this is NOT using any LLM. It's a **rule-based orchestrator** — a Python class that calls functions in a fixed sequence with if/else logic for deciding when to escalate.

**In Week 9+**: The plan is to upgrade to an LLM-backed agent (using OpenAI GPT or similar) that can dynamically decide which tools to call and generate natural language recommendations. But the current version is pure Python, no LLM needed.

Think of it like:
- **Week 4**: A cooking robot that always follows the same recipe steps in order.
- **Week 9**: A chef that can decide which steps to do based on the ingredients available.
