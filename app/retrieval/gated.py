"""Section 3.4 — the gate stack and the refusal path.

    query + principal
      → PRE-FILTER (tenant · acl · in-force window)   ← constrains the candidate set
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
from datetime import date               # stdlib — as_of, the gate's time anchor

from app.config import get_settings     # local — app/config.py (default embedder/reranker config)
from app.ingest import Chunk            # local — app/ingest/chunker.py
from app.llm.base import EmbeddingClient, RerankClient  # local — app/llm/base.py (the seam)
from app.llm.factory import build_embedding_client, build_reranker  # local — app/llm/factory.py
from app.retrieval.hybrid import (      # local — app/retrieval/hybrid.py
    Hit,
    HybridRetriever,
    _unit_vectors,
    bm25_rank,
    fuse_rrf,
)
from app.store.base import Gate, VectorStore  # local — app/store/base.py (the storage seam)


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

    Phase 5 finished this sentence. Until then we SIMULATED pre-filtering by
    building a separate index per principal and caching it — correct, but the
    cache key silently became the security boundary (the as_of time-leak had
    to be pinned by a test). Now the predicate travels WITH the query as a
    Gate and is enforced inside the store's search — Azure AI Search calls
    this `vectorFilterMode: preFilter`; Qdrant evaluates the filter during
    the HNSW traversal. There is no per-principal cache anymore, so there is
    no key to forget: that entire class of bug is structurally gone.

    The property is unchanged and the same tests still assert it: a chunk the
    principal cannot see is NEVER A CANDIDATE. There is no window between
    "retrieved" and "filtered" for anyone to insert a cache, a reranker, or a
    log into.
    """

    def __init__(self, store: VectorStore, embedder: EmbeddingClient | None = None) -> None:
        self._store = store
        self._embedder = embedder or build_embedding_client(get_settings())

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "PreFilterRetriever":
        """Demo/test convenience: embed now, into an in-memory store.

        This is the old constructor's behaviour, kept as a named alternative
        so the intent is visible at the call site: `from_chunks` says "fixture
        data, ephemeral", while `PreFilterRetriever(store)` says "the real,
        already-ingested thing".
        """
        from app.ingest.index import ingest_into    # local — app/ingest/index.py
        from app.store.memory import MemoryStore    # local — app/store/memory.py

        store = MemoryStore()
        retriever = cls(store)
        ingest_into(store, chunks, embedder=retriever._embedder)
        return retriever

    def search_many(
        self, queries: list[str], principal: Principal, k: int = 10, *,
        as_of: date | None = None, pool: int = 20,
    ) -> list[Hit]:
        """Retrieve for SEVERAL phrasings of one question and fuse the lot.

        Each query contributes a dense ranking and a lexical ranking over the
        same gate-visible set; RRF fuses all 2N rankings. Fusion by rank is
        what makes this safe to do with heterogeneous queries: a HyDE
        pseudo-answer and the user's original question produce scores on
        incomparable scales, but "you ranked 1st for this phrasing" is
        comparable across all of them (see fuse_rrf).

        A chunk that several phrasings agree on rises — which is exactly the
        agreement-over-confidence property K_RRF was chosen for.
        """
        gate = Gate(principal.tenant_id, principal.groups, as_of or date.today())
        visible = self._store.visible_chunks(gate)
        if not visible:
            raise ValueError(
                f"no visible chunks for tenant {gate.tenant_id!r} as of {gate.as_of}"
            )

        rankings: list[list[Hit]] = []
        for q in queries:
            if not q.strip():
                continue
            qv = _unit_vectors(self._embedder, [q])[0]
            rankings.append([
                Hit(c, rank, score)
                for rank, (c, score) in enumerate(self._store.search(qv.tolist(), gate, k=pool))
            ])
            rankings.append(bm25_rank(q, visible, k=pool))
        return fuse_rrf(rankings, k=k)

    def search(
        self, query: str, principal: Principal, k: int = 10, *, as_of: date | None = None,
        pool: int = 20,
    ) -> list[Hit]:
        """One phrasing — the single-query case of search_many.

        as_of defaults to today — the right anchor for "what does my policy
        say?". A claims handler passes the DATE OF LOSS instead: the wording
        that governs a claim is the one in force when the accident happened,
        not the one in force when the question got asked."""
        return self.search_many([query], principal, k, as_of=as_of, pool=pool)


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

    def search(
        self, query: str, principal: Principal, k: int = 10, *, as_of: date | None = None
    ) -> list[Hit]:
        hits = self.search_unfiltered(query, k=k)      # 1. retrieve globally
        return [                                       # 2. then filter
            h for h in hits
            if h.chunk.meta
            and h.chunk.meta.visible_to(principal.tenant_id, principal.groups, as_of or date.today())
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
    return rerank_many([query], chunks, reranker=reranker)


def rerank_many(
    queries: list[str],
    chunks: list[Chunk],
    *,
    reranker: RerankClient | None = None,
) -> list[tuple[Chunk, float]]:
    """Score each chunk against EVERY phrasing and keep its best.

    Max-over-phrasings, not mean: the question being asked is "does any
    faithful phrasing of this question match this passage?", and a passage
    that one phrasing matches perfectly is a good passage even if three
    other phrasings miss it. Averaging would punish exactly the case this
    phase exists to rescue — measured on a real PDF, the same chunk scores
    0.0886 for the user's words and 0.9998 for the document's.

    Cost is linear in phrasings: N queries x |chunks| cross-encoder pairs.
    That is the price of the fix and it is measured, not assumed —
    `python -m evals.ablation` reports it per configuration.
    """
    reranker = reranker or build_reranker(get_settings())
    texts = [c.text_to_embed for c in chunks]

    best: list[float] = [float("-inf")] * len(chunks)
    for q in queries:
        if not q.strip():
            continue
        for i, raw in enumerate(reranker.rerank(q, texts)):
            best[i] = max(best[i], float(raw))

    scored = [(c, _sigmoid(s)) for c, s in zip(chunks, best)]
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
    as_of: date | None = None,
    threshold: float = REFUSAL_THRESHOLD,
    pool: int = 20,
    top_k: int = 5,
    queries: list[str] | None = None,
) -> Answer:
    """The full gated path. Returns an Answer; never calls a generator itself.

    `queries` are alternative phrasings of `query` (app/retrieval/rewrite.py),
    used for BOTH retrieval and reranking. They are a retrieval device only:
    the gate still decides on real chunks from the principal's own corpus,
    and what reaches the prompt is unchanged. Default — and every failure
    path upstream — is `[query]`, i.e. exactly the pre-phase-11 behaviour.
    """
    phrasings = [q for q in (queries or [query]) if q.strip()] or [query]

    hits = retriever.search_many(phrasings, principal, k=pool, as_of=as_of)  # pre-filtered, always
    if not hits:
        return Answer(True, 0.0, [], [], "no candidates after gates")

    ranked = rerank_many(phrasings, [h.chunk for h in hits])
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
