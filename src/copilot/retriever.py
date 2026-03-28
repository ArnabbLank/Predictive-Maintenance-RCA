"""
copilot/retriever.py — Simple BM25 / TF-IDF retrieval over markdown KB files.

The knowledge base is a small set of curated markdown files in kb/.
This module chunks them, indexes with BM25, and returns top-k snippets.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

# BM25 for retrieval
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def _load_kb_files(kb_dir: str) -> List[dict]:
    """Load all .md files from KB directory and split into chunks."""
    kb_dir = Path(kb_dir)
    if not kb_dir.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    documents = []
    for md_file in sorted(kb_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(content, source=md_file.name)
        documents.extend(chunks)

    if not documents:
        raise ValueError(f"No .md files found in {kb_dir}")

    return documents


def _chunk_markdown(
    text: str,
    source: str = "",
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[dict]:
    """
    Split markdown into chunks by headers first, then by size.

    Returns list of {"text": str, "source": str, "section": str}
    """
    # Split by markdown headers
    sections = re.split(r"\n(#{1,3}\s+.*)\n", text)

    chunks = []
    current_section = "Introduction"

    for part in sections:
        part = part.strip()
        if not part:
            continue
        if re.match(r"^#{1,3}\s+", part):
            current_section = part.lstrip("#").strip()
            continue

        # Split long sections into smaller chunks
        words = part.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 10:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "section": current_section,
            })

    return chunks


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return re.findall(r"\w+", text.lower())


class KBRetriever:
    """BM25-based retriever over the knowledge base."""

    def __init__(self, kb_dir: str):
        self.documents = _load_kb_files(kb_dir)
        self.corpus = [_tokenize(doc["text"]) for doc in self.documents]

        if HAS_BM25:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            # Fallback: simple TF-IDF-like scoring
            self.bm25 = None

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Search the KB and return top-k snippets."""
        query_tokens = _tokenize(query)

        if self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
        else:
            # Fallback: word overlap scoring
            scores = []
            for doc_tokens in self.corpus:
                overlap = len(set(query_tokens) & set(doc_tokens))
                scores.append(overlap / max(len(query_tokens), 1))
            scores = np.array(scores)

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append({
                    "text": doc["text"],
                    "source": f"{doc['source']} > {doc['section']}",
                    "score": float(scores[idx]),
                })

        return results


# ── Module-level convenience function ─────────────────────────────────────
_retriever_cache = {}


def retrieve_kb(
    query: str,
    kb_dir: str = None,
    top_k: int = 3,
) -> List[dict]:
    """
    Retrieve KB snippets (creates/caches retriever on first call).
    """
    if kb_dir is None:
        kb_dir = str(Path(__file__).resolve().parent.parent.parent / "kb")

    if kb_dir not in _retriever_cache:
        _retriever_cache[kb_dir] = KBRetriever(kb_dir)

    return _retriever_cache[kb_dir].search(query, top_k)
