# Week 8 — Knowledge Base + Retrieval (RAG Basics)

## What Are We Doing?
Building a small "knowledge base" (KB) of curated documents and a search system to retrieve relevant information. This grounds the Copilot's recommendations in factual references.

## Why Are We Doing This?
- If the model says "sensor_11 is the main driver", we want the Copilot to look up what sensor_11 means (HPC outlet pressure) and what the recommended action is (borescope inspection).
- This is a simple version of **RAG (Retrieval-Augmented Generation)** — the technique used by ChatGPT-like systems to reference external documents.
- Grounding recommendations in a knowledge base makes them more trustworthy and traceable.

## What's in the Knowledge Base?
The `kb/` folder contains manually curated markdown files:

| File | Contents |
|------|----------|
| `fault_tree.md` | How HPC degradation progresses, which sensors are affected, maintenance recommendations based on RUL ranges |
| `glossary.md` | What each sensor actually measures (e.g., sensor_13 = core speed in RPM), RUL labeling explanation, NASA scoring formula |
| `novel_approaches.md` | Research ideas and advanced techniques |

## How Retrieval Works

### BM25 (Best Matching 25)
A text search algorithm that ranks documents by relevance to a query. Think of it as a smarter version of "Ctrl+F" that understands word frequency and document length.

**Steps:**
1. **Chunk**: Split each markdown file into chunks (~300 words each). Each chunk is one searchable "document."
2. **Tokenize**: Break text into words, lowercase everything.
3. **Index**: Build a BM25 index over all chunks.
4. **Search**: Given a query like "HPC degradation sensor_11 maintenance", BM25 scores each chunk by how many query words it contains, weighted by rarity (rare words matter more).
5. **Return top-k**: Return the 3 most relevant chunks.

### Example Query Flow:
```
Query: "degradation sensor_11, sensor_4, sensor_15 maintenance"
    → BM25 searches kb/fault_tree.md chunks
    → Returns: "RUL < 30 cycles: Immediate maintenance — compressor wash or blade replacement"
    → This gets included in the Copilot's recommendation
```

## Results
- BM25 reliably retrieves relevant fault tree entries when queried with sensor names.
- The retrieved text is cited in the Copilot's output as "KB reference."
- This is a simple but effective approach — no vector database or LLM embeddings needed.
