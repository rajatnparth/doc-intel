"""Section 3.3 — the opposite-failures claim, as assertions.

These tests use REAL embeddings (bge-small-en-v1.5). They are slower than the
rest of the suite and they download ~130MB on first run. That cost is the point:
a stubbed dense retriever could be rigged to fail on a claim number, which
would prove nothing at all.
"""

import pytest                           # 3rd-party: pytest — fixtures, markers

from app.ingest import chunk_document   # local — app/ingest/chunker.py
from app.retrieval.hybrid import HybridRetriever  # local — app/retrieval/hybrid.py

# Two tiny corpora, inline, so the test doesn't depend on sample_docs/ layout.
POLICY = """# Asha Rao — Motor Policy Kit

## 2. Premium and Payment

The renewal premium is due within fifteen (15) days of the renewal notice.
Cover is suspended while any instalment is overdue.

## 5. Damage Assessment Codes

| Damage Code | Remedy |
|-------------|--------|
| D-4470      | Book any network garage |
| D-4471      | Hold repairs until the surveyor inspects |
"""

CLAIMS = """# Claims File

## CLM-2026-0888 — bumper scrape, parking

Reported 2026-01-12. Assessed at 8,150.00. Cashless payment released.

## CLM-2026-0891 — rear quarter panel dent

Reported 2026-03-06. Assessed at 18,400.00. Payment released after discharge.

## CLM-2026-0892 — alloy wheel kerb damage

Reported 2026-03-08. Assessed at 7,350.00. Cashless payment released.

## CLM-2026-0893 — flood water in cabin

Reported 2026-03-11. Assessed at 66,000.00. Payment pending.
"""


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    chunks = chunk_document(POLICY, doc_title="Asha Rao — Motor Policy Kit", max_chars=700)
    chunks += chunk_document(CLAIMS, doc_title="Claims File", max_chars=700)
    for i, c in enumerate(chunks):
        c.chunk_index = i               # unique across docs — RRF keys on this
    return HybridRetriever(chunks)


def rank_of(hits, needle: str) -> int | None:
    """1-based rank of the first hit containing `needle`, or None.

    Checks text_to_embed, because that is what BOTH retrievers index — the
    claim number lives in the heading, not the body.
    """
    for h in hits:
        if needle.lower() in h.chunk.text_to_embed.lower():
            return h.rank + 1
    return None


# =============================================================================
# Failure 1: dense loses exact tokens.
# =============================================================================
def test_dense_confuses_near_identical_claim_numbers(retriever) -> None:
    """The predicted failure, with a real model: ask for one claim, get another.

    Deliberately asserted LOOSELY — we assert dense does not put it first, not
    that it lands on an exact rank. Model versions shift; the phenomenon doesn't.
    """
    dense = retriever.dense_search("CLM-2026-0891", k=10)
    top = dense[0].chunk.text_to_embed
    assert "CLM-2026-0891" not in top, (
        "dense ranked a DIFFERENT claim first — 0888/0892/0893 are "
        "semantically identical to 0891; the digits carry no meaning"
    )


def test_bm25_nails_the_exact_claim_number(retriever) -> None:
    """BM25 understands nothing and gets it exactly right."""
    assert rank_of(retriever.bm25_search("CLM-2026-0891", k=10), "CLM-2026-0891") == 1


def test_rrf_rescues_the_exact_token_query(retriever) -> None:
    """Fusion recovers what dense alone lost."""
    assert rank_of(retriever.rrf("CLM-2026-0891", k=10), "CLM-2026-0891") == 1


# =============================================================================
# Failure 2 (the mirror image): BM25 has no idea what words mean.
# =============================================================================
def test_dense_handles_the_paraphrase_bm25_cannot(retriever) -> None:
    """'wait before paying' shares NO content token with 'premium is due'.

    The first draft used "settle the premium" — and BM25 ranked the right chunk
    #1 with score 2.78, because "premium" IS the section's vocabulary. A
    paraphrase test with a shared content word tests nothing. This query was
    checked against the tokeniser: zero overlapping tokens.
    """
    q = "how long can I wait before paying?"
    dense_rank = rank_of(retriever.dense_search(q, k=10), "fifteen (15) days")
    bm25_hits = retriever.bm25_search(q, k=10)

    assert dense_rank == 1, "meaning is exactly what embeddings are for"
    assert bm25_hits == [], (
        "no query token appears anywhere in the corpus — BM25 must return "
        "nothing, not the corpus in index order"
    )


def test_bm25_returns_nothing_when_no_token_matches(retriever) -> None:
    """The regression test for a measured artifact: bm25_search used to return
    ALL chunks for an out-of-vocabulary query, every score 0.0000, ordered by
    chunk index — and the 'rank 1' it reported was whichever chunk happened to
    be first. RRF then paid that noise real fusion credit. A zero-score chunk
    matched no query term; it is not a result."""
    hits = retriever.bm25_search("zzz qqq completely alien vocabulary", k=10)
    assert hits == []


def test_rrf_keeps_the_answer_in_the_pool(retriever) -> None:
    """RRF's job is the POOL, not rank 1 — and this test's history is the lesson.

    On the old contract corpus, dense ranked the answer 1 and RRF demoted it to
    3: BM25's confidently-wrong rankings earned 1/(k+rank) credit and pushed the
    true answer down. Fusion is not free — that finding stands (see README).

    On THIS corpus, after bm25_search stopped returning zero-score noise, the
    demotion measures ZERO: for a paraphrase query BM25 now returns nothing, so
    fusion degrades to dense-only and the answer stays at rank 1. Part of the
    old tax was fusing garbage rankings that should never have existed.

    So the invariants worth asserting are the ones that survive both corpora:
    fusion must never LOSE the answer, and it can never beat the better
    retriever on a query only one of them understands. Fixing the ordering is
    the cross-encoder's job — which is precisely why the architecture is
        retrieve wide + cheap (RRF, top 50)  ->  rerank narrow + expensive (top 5)
    and not "just use RRF's top 5".
    """
    q = "how long can I wait before paying?"
    dense_rank = rank_of(retriever.dense_search(q, k=10), "fifteen (15) days")
    rrf_rank = rank_of(retriever.rrf(q, k=10), "fifteen (15) days")

    assert dense_rank == 1
    assert rrf_rank is not None, "fusion must not LOSE the answer entirely"
    assert rrf_rank <= 5, "it stays in the pool a reranker would rescore"
    assert rrf_rank >= dense_rank, (
        "fusion cannot improve on a correct #1 — at best it preserves it, at "
        "worst it demotes it; the reranker after it exists for exactly that"
    )


# =============================================================================
# The wiring bug that once made BM25 look useless.
# =============================================================================
def test_both_retrievers_index_the_same_text(retriever) -> None:
    """BM25 originally indexed chunk.text (body only) while dense indexed
    text_to_embed (body + heading). Claim numbers live in the HEADING, so
    BM25's index didn't contain the token at all and it scored 0.0000 on every
    claim query — looking like a property of BM25 rather than a wiring bug.

    If two retrievers index different text, you aren't comparing retrievers.
    You're comparing corpora.
    """
    c = next(c for c in retriever.chunks if "CLM-2026-0891" in c.heading)
    assert "CLM-2026-0891" not in c.text, "the number is in the heading, not the body"
    assert "CLM-2026-0891" in c.text_to_embed, "…and text_to_embed is what we index"
    assert rank_of(retriever.bm25_search("CLM-2026-0891", k=10), "CLM-2026-0891") == 1
