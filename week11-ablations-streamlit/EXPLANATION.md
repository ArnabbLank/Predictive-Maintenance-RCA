# Week 11 — Ablation Study + Streamlit Demo

## What Are We Doing?
1. **Ablation study**: Systematically testing which model components help and which don't.
2. **Streamlit app**: Building an interactive web demo for the final presentation.

## Why Are We Doing This?
- Ablation studies are required in academic papers to justify design choices.
- A Streamlit demo makes the project tangible — someone can click buttons and see results live.

## Part 1: Ablation Study

### What we test:

#### Window Size (10, 20, 30, 40, 50)
How many past cycles does the model look at?
- Too small (10): Not enough history to see degradation trends.
- Too large (50): More data, but also more noise and computational cost.
- Sweet spot: Usually around 30.

#### Savitzky-Golay Smoothing (on/off)
Does smoothing help? Our findings from Week 5 suggest not always.

#### Attention Heads (2, 4, 8)
The Transformer's attention mechanism can have multiple "heads" — each head looks at the data from a different perspective. More heads = more capacity, but also more parameters.

#### MC Dropout Samples (10, 50, 100, 200)
More samples = more stable uncertainty estimates, but slower inference.
- 10 samples: Fast but noisy uncertainty.
- 100 samples: Good balance.
- 200 samples: Diminishing returns.

## Part 2: Streamlit App

The interactive app at `streamlit-app/app.py` has:
- **Overview**: Model description and pipeline explanation.
- **Engine Analysis**: Select an engine, run the 4-step pipeline, see results.
- **Fleet Dashboard**: Analyze all engines at once.
- **About**: Technical details.

See `UNDERSTANDING_PAPERS_AND_CONCEPTS.md` in the project root for a detailed description of the app's layout.

## Results
- Ablation tables show the contribution of each component.
- The Streamlit app is functional and ready for demonstration.
