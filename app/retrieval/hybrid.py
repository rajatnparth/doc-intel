"""Hybrid retrieval: BM25 + dense, fused by Reciprocal Rank Fusion.

    python -m app.retrieval.hybrid

Section 3.3 as running code. Two retrievers that fail in OPPOSITE directions:

    dense (embeddings) : understands MEANING, loses exact tokens
    BM25  (lexical)    : nails EXACT TOKENS, understands nothing

and RRF, which merges them using ONLY RANKS — because BM25 scores are unbounded
and corpus-dependent while cosine is bounded, so averaging them is meaningless.

THE EMBEDDINGS HERE ARE REAL. bge-small-en-v1.5 via fastembed (ONNX, no torch,
no API key). That matters: a fake dense retriever could be rigged to fail on
E-4471, which would prove nothing. This one fails on its own.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import re                               # stdlib — the tokeniser for BM25
from dataclasses import dataclass       # stdlib — Hit / the result rows
from functools import lru_cache         # stdlib — load the embedding model once

import numpy as np                      # 3rd-party: numpy — vector math for dense scoring
from rank_bm25 import BM25Okapi         # 3rd-party: rank-bm25 — the classical lexical scorer

from app.ingest import Chunk            # local — app/ingest/chunker.py (what we retrieve)

K_RRF = 60                              # the conventional RRF constant; see rrf() below


# =============================================================================
# Dense retrieval — real embeddings
# =============================================================================
@lru_cache(maxsize=1)
def _embedder():
    """Load the ONNX embedding model once per process.

    fastembed downloads bge-small-en-v1.5 (~130MB) on first use and runs it on
    CPU via onnxruntime. Chosen over sentence-transformers purely because it
    doesn't drag in 427MB of PyTorch.

    In the capstone this is where AzureLLMClient's embedding deployment goes —
    same interface, same seam, different provider.
    """
    from fastembed import TextEmbedding  # 3rd-party: fastembed — lazy import: only
                                         #   needed when you actually retrieve

    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts: list[str]) -> np.ndarray:
    """Embed and L2-normalise. After normalising, cosine == dot product (3.2)."""
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# =============================================================================
# Lexical retrieval — BM25
# =============================================================================
def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumerics, KEEPING hyphenated ids intact.

    This tokeniser is a design decision, not boilerplate. Split "E-4471" into
    ["e", "4471"] and you've thrown away the exact-token advantage that is the
    entire reason BM25 is here. The regex keeps [A-Za-z0-9-]+ together.
    """
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if t]


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    score: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        # BOTH retrievers index text_to_embed — the SAME text, including the
        # provenance prefix ("Invoice Register 2024 > INV-2024-0891 — Acme Corp:").
        #
        # This line was a bug once, and it is worth keeping the scar tissue:
        # BM25 originally indexed `c.text` (the body only) while dense indexed
        # `c.text_to_embed` (body + heading). The invoice NUMBER lives in the
        # heading. So BM25 was searching an index that did not contain the token
        # at all, and scored 0.0000 on every query for it — while looking like a
        # property of BM25 rather than a mistake in the wiring.
        #
        # RULE: if two retrievers index different text, you are not comparing
        # retrievers, you are comparing corpora. Fuse only over a shared view.
        indexed = [c.text_to_embed for c in chunks]
        self._bm25 = BM25Okapi([tokenize(t) for t in indexed])
        self._vecs = embed(indexed)

    # -- the two retrievers ----------------------------------------------------
    def dense_search(self, query: str, k: int = 10) -> list[Hit]:
        qv = embed([query])[0]
        sims = self._vecs @ qv                       # unit vectors -> dot == cosine
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], r, float(sims[i])) for r, i in enumerate(order)]

    def bm25_search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], r, float(scores[i])) for r, i in enumerate(order)]

    # -- fusion ----------------------------------------------------------------
    def rrf(self, query: str, k: int = 10, pool: int = 20) -> list[Hit]:
        """Reciprocal Rank Fusion.

            score(doc) = Σ over retrievers of  1 / (K_RRF + rank)

        RANKS ONLY. Never scores. Why:
          - BM25 is unbounded and corpus-dependent (18.4 means nothing absolute);
            cosine is bounded in [-1, 1]. The scales are unrelated.
          - Min-max normalising is arbitrary: normalise against which maximum?
          - One BM25 outlier (a rare term repeated) would bury everything dense said.
          - "You ranked 1st" IS comparable across retrievers. So use only that.

        K_RRF (60) flattens the top: rank 1 contributes 1/61, rank 2 contributes
        1/62 — barely less. So AGREEMENT between retrievers outweighs any single
        retriever's confidence. That's the design intent.
        """
        fused: dict[int, float] = {}
        for hits in (self.dense_search(query, pool), self.bm25_search(query, pool)):
            for h in hits:
                key = h.chunk.chunk_index
                fused[key] = fused.get(key, 0.0) + 1.0 / (K_RRF + h.rank)

        by_index = {c.chunk_index: c for c in self.chunks}
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(by_index[i], r, s) for r, (i, s) in enumerate(ranked)]
