"""Hybrid retrieval: BM25 + dense, fused by Reciprocal Rank Fusion.

    python -m app.retrieval.hybrid

Section 3.3 as running code. Two retrievers that fail in OPPOSITE directions:

    dense (embeddings) : understands MEANING, loses exact tokens
    BM25  (lexical)    : nails EXACT TOKENS, understands nothing

and RRF, which merges them using ONLY RANKS — because BM25 scores are unbounded
and corpus-dependent while cosine is bounded, so averaging them is meaningless.

THE EMBEDDINGS HERE ARE REAL — bge-small-en-v1.5 by default. That matters: a
fake dense retriever could be rigged to fail on E-4471, which would prove
nothing. This one fails on its own.

They arrive through the SAME SEAM as chat (EmbeddingClient, app/llm/base.py).
The first draft imported fastembed directly right here — it worked, and it
silently made "swap providers by config" a false claim for embeddings.
EMBEDDING_PROVIDER=azure now swaps the model without touching this file.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import re                               # stdlib — the tokeniser for BM25
from dataclasses import dataclass       # stdlib — Hit / the result rows

import numpy as np                      # 3rd-party: numpy — vector math for dense scoring
from rank_bm25 import BM25Okapi         # 3rd-party: rank-bm25 — the classical lexical scorer

from app.config import get_settings     # local — app/config.py (the default embedder's config)
from app.ingest import Chunk            # local — app/ingest/chunker.py (what we retrieve)
from app.llm.base import EmbeddingClient        # local — app/llm/base.py (the seam)
from app.llm.factory import build_embedding_client  # local — app/llm/factory.py

K_RRF = 60                              # the conventional RRF constant; see rrf() below


# =============================================================================
# Dense retrieval — real embeddings, through the seam
# =============================================================================
def _unit_vectors(embedder: EmbeddingClient, texts: list[str]) -> np.ndarray:
    """Embed and L2-normalise. After normalising, cosine == dot product (3.2).

    float32 and the normalise step live HERE, not in the client: what a vector
    must look like to sit in an index is a retrieval decision. The seam hands
    over plain floats and stays out of it.
    """
    vecs = np.array(embedder.embed(texts), dtype=np.float32)
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
    def __init__(self, chunks: list[Chunk], embedder: EmbeddingClient | None = None) -> None:
        self.chunks = chunks

        # Injected, like main.py's LLMClient — defaulted from config so demos
        # and tests stay one-liners. Passing a fake embedder in a test is now
        # possible for the same reason passing StubLLMClient is.
        self._embedder = embedder or build_embedding_client(get_settings())

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
        self._vecs = _unit_vectors(self._embedder, indexed)

    # -- the two retrievers ----------------------------------------------------
    def dense_search(self, query: str, k: int = 10) -> list[Hit]:
        qv = _unit_vectors(self._embedder, [query])[0]
        sims = self._vecs @ qv                       # unit vectors -> dot == cosine
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], r, float(sims[i])) for r, i in enumerate(order)]

    def bm25_search(self, query: str, k: int = 10) -> list[Hit]:
        """A zero score means the chunk matched NO query term. Returning it
        anyway would hand back the corpus in index order dressed up as a
        ranking — and rrf() below would then pay that noise real fusion credit
        (1/(60+rank) per chunk) on every query whose vocabulary misses the
        corpus. Found by measurement: a paraphrase query "ranked" the right
        chunk #1 with score 0.0000, purely because it was chunk 0."""
        scores = self._bm25.get_scores(tokenize(query))
        order = [i for i in np.argsort(-scores) if scores[i] > 0.0][:k]
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
