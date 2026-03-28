# Week 9 — Agentic Copilot (Full Pipeline)

## What Are We Doing?
Wiring together all 4 tools (predict → explain → retrieve → recommend) into a complete Copilot agent that can analyze individual engines and entire fleets.

## Why Are We Doing This?
- Individual tools are useful, but the real value is the complete pipeline: from raw sensor data to a human-readable maintenance recommendation.
- Fleet-level analysis lets you prioritize: "Which engines need attention first?"
- Abstention/escalation logic makes the system safe for real-world use.

## What the Copilot Does for One Engine
```
Step 1: predict_rul(sensor_window) 
    → RUL = 42 cycles, uncertainty = 5.3 cycles (LOW)

Step 2: explain_prediction(sensor_window)
    → Top sensors: sensor_11 (0.34), sensor_4 (0.22), sensor_15 (0.18)

Step 3: retrieve_knowledge("degradation sensor_11, sensor_4, sensor_15 maintenance")
    → "RUL 30-80 cycles: Schedule borescope inspection of HPC blades."

Step 4: recommend_action(prediction, explanation, kb)
    → WARNING: Predicted RUL = 42 cycles (±5). Plan maintenance within the next cycle window.
      Primary degradation signal: sensor_11. KB reference: "Schedule borescope inspection..."
```

## Safety Checks
The agent verifies:
1. **Uncertainty is reported** — never give advice without showing confidence level.
2. **KB citation is present** — recommendations must be grounded.
3. **No false reassurance** — if uncertainty is high, don't say "everything is fine."

## Fleet Summary
When you run the Copilot on all 100 test engines:
- How many need immediate maintenance (RUL < 30)?
- How many were escalated (uncertainty too high)?
- What is the average fleet RUL?

## Is This a "Real" LLM Agent?
**Week 9 is still rule-based** — it uses if/else logic, not an LLM. However, the architecture is designed to be swappable: you could replace the rule-based orchestrator with a LangChain agent that uses OpenAI's function calling. The tools themselves stay the same.

## Results
- Full pipeline runs in ~2 seconds per engine (with MC Dropout at 100 samples).
- Fleet analysis completes in ~3-4 minutes for all 100 engines.
- Escalation rate depends on the threshold setting (default: uncertainty > 20 cycles).
