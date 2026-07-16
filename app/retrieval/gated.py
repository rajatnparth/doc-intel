"""Section 3.4 — the gate stack and the refusal path.

    query + principal
      → PRE-FILTER (tenant · acl · status)   ← constrains the candidate set
      → hybrid retrieve (dense + BM25 → RRF)
      → cross-encoder rerank                 ← the only ABSOLUTE score
      → score < threshold ? REFUSE : generate

Two ideas live here because they are the same idea: both are ENFORCEMENT, and
enforcement is deterministic code outside the model.

`PostFilterRetriever` is kept as a deliberate villain. It exists so
tests/test_gated.py can DEMONSTRATE the cross-tenant leak rather than assert it.
It is never wired into the app.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import math                             # stdlib — sigmoid, to turn logits into 0..1
from dataclasses import dataclass       # stdlib — Principal / Answer result types
from functools import lru_cache         # stdlib — the per-principal view cache

from app.config import get_settings     # local — app/config.py (the default reranker's config)
from app.ingest import Chunk            # local — app/ingest/chunker.py
from app.llm.base import RerankClient   # local — app/llm/base.py (the seam)
from app.llm.factory import build_reranker  # local — app/llm/factory.py
from app.retrieval.hybrid import Hit, HybridRetriever  # local — app/retrieval/hybrid.py


@dataclass(frozen=True)
class Principal:
    """Who is asking. Passed explicitly to every retrieve call.

    Deliberately NOT read from a global/contextvar. In a multi-tenant system the
    identity of the caller is an ARGUMENT to retrieval, not ambient state — if it
    can be forgotten, it will be.
    """

    tenant_id: str
    groups: frozenset[str]


# =============================================================================
# The gate
# =============================================================================
class PreFilterRetriever:
    """The correct shape: the predicate constrains the CANDIDATE SET.

    Real systems push the predicate into the ANN index itself — Azure AI Search
    calls this `vectorFilterMode: preFilter`, and it constrains the graph
    traversal. We can't do that with a plain numpy/BM25 index, so we build a
    per-principal view: the foreign chunks are not in the index being searched.

    The property that matters is the same either way: a chunk the principal
    cannot see is NEVER A CANDIDATE. There is no window between "retrieved" and
    "filtered" for anyone to insert a cache, a reranker, or a log into.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._all = chunks

    @lru_cache(maxsize=32)
    def _view_for(self, tenant_id: str, groups: frozenset[str]) -> HybridRetriever:
        """Build (and cache) an index containing ONLY what this principal may see.

        NOTE the cache key: (tenant_id, groups). Caching a per-principal view is
        safe precisely BECAUSE the principal is part of the key. Compare the
        semantic cache in the post-filter version below, whose key is the query
        embedding alone — that's the leak.
        """
        visible = [c for c in self._all if c.meta and c.meta.visible_to(tenant_id, groups)]
        if not visible:
            raise ValueError(f"no visible chunks for tenant {tenant_id!r}")
        return HybridRetriever(visible)

    def search(self, query: str, principal: Principal, k: int = 10) -> list[Hit]:
        view = self._view_for(principal.tenant_id, principal.groups)
        return view.rrf(query, k=k)


class PostFilterRetriever:
    """💀 THE VILLAIN. Do not use. Exists to be tested and fail.

    Searches everything, then drops what the principal can't see. The returned
    list IS correct — that's what makes it seductive, and why "the user never
    sees it" is the wrong objection.

    The defects, in order of how much they should worry you:
      3. result counts are unpredictable (ask for 50, keep 3)
      2. you READ data you had no right to read
      1. correctness depends on nothing ever being inserted between the two
         calls — an invariant that exists only in the author's head.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._index = HybridRetriever(chunks)          # ← every tenant, one index
        # A semantic cache, added six months later by someone optimising latency.
        # Keyed on the query — because at THIS point in the pipeline, results are
        # not yet principal-specific, so keying on the principal looks redundant.
        # That reasoning is correct and the outcome is a breach.
        self.cache: dict[str, list[Hit]] = {}

    def search_unfiltered(self, query: str, k: int = 10) -> list[Hit]:
        if query in self.cache:                        # ← the inserted component
            return self.cache[query]
        hits = self._index.rrf(query, k=k)             # ALL tenants
        self.cache[query] = hits
        return hits

    def search(self, query: str, principal: Principal, k: int = 10) -> list[Hit]:
        hits = self.search_unfiltered(query, k=k)      # 1. retrieve globally
        return [                                       # 2. then filter
            h for h in hits
            if h.chunk.meta
            and h.chunk.meta.visible_to(principal.tenant_id, principal.groups)
        ]


# =============================================================================
# The refusal path
# =============================================================================
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rerank(
    query: str,
    chunks: list[Chunk],
    *,
    reranker: RerankClient | None = None,
) -> list[tuple[Chunk, float]]:
    """Score (query, chunk) pairs jointly. Returns 0..1, best first.

    The cross-encoder arrives through the seam (RerankClient — by default
    ms-marco-MiniLM via app/llm/local.py) and hands back RAW scores. Everything
    after that line is OUR problem, and deliberately so:

    ⚠️ THE DETAIL EVERY TUTORIAL SKIPS: ms-marco cross-encoders emit LOGITS, not
    probabilities. Measured on our corpus: +5.60 for a matching pair, -11.33 for
    an unrelated one. Any advice to "threshold at 0.5" silently assumes a 0..1
    scale — apply it to raw logits and you refuse everything.

    We sigmoid into 0..1 so the threshold is interpretable. That is a presentation
    choice, not a calibration: sigmoid is monotonic, so it changes no ranking and
    creates no information. You still have to MEASURE where to cut (calibrate.py).
    """
    reranker = reranker or build_reranker(get_settings())
    raw = reranker.rerank(query, [c.text_to_embed for c in chunks])
    scored = [(c, _sigmoid(r)) for c, r in zip(chunks, raw)]
    return sorted(scored, key=lambda t: -t[1])


@dataclass
class Answer:
    """A refusal is a first-class OUTCOME, not an exception.

    Modelling it as a return value rather than a raise is the point: the caller
    cannot forget to handle it, and `refused` is right there in the type.
    """

    refused: bool
    score: float                        # the top reranker score, always reported
    chunks: list[Chunk]                 # [] when refused
    near_misses: list[Chunk]            # offered as links on a refusal
    reason: str = ""


# Where we cut. NOT a guess — see `python -m app.retrieval.calibrate`, which
# scores answerable vs unanswerable query sets and prints the tradeoff.
# The distributions overlap; this number encodes a business decision:
# in a document-intelligence product a fabricated invoice total costs far more
# than a refusal, so we FAIL CLOSED.
REFUSAL_THRESHOLD = 0.5


def answer(
    query: str,
    principal: Principal,
    retriever: PreFilterRetriever,
    *,
    threshold: float = REFUSAL_THRESHOLD,
    pool: int = 20,
    top_k: int = 5,
) -> Answer:
    """The full gated path. Returns an Answer; never calls a generator itself."""
    hits = retriever.search(query, principal, k=pool)     # pre-filtered, always
    if not hits:
        return Answer(True, 0.0, [], [], "no candidates after gates")

    ranked = rerank(query, [h.chunk for h in hits])
    best_chunk, best_score = ranked[0]

    if best_score < threshold:
        # NEVER call the generator here. Handed three confident-looking
        # irrelevant chunks and a "say I don't know if unsure" prompt, models
        # answer anyway — you'd be paying tokens for a coin flip on a control
        # you already had a NUMBER for.
        return Answer(
            refused=True,
            score=best_score,
            chunks=[],
            near_misses=[c for c, _ in ranked[:3]],   # offer them as links, not as an answer
            reason=f"best reranker score {best_score:.3f} < threshold {threshold}",
        )

    return Answer(
        refused=False,
        score=best_score,
        chunks=[c for c, _ in ranked[:top_k]],
        near_misses=[],
    )
