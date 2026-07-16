"""Section 3.3 — the opposite-failures claim, as assertions.

These tests use REAL embeddings (bge-small-en-v1.5). They are slower than the
rest of the suite and they download ~130MB on first run. That cost is the point:
a stubbed dense retriever could be rigged to fail on an invoice number, which
would prove nothing at all.
"""

import pytest                           # 3rd-party: pytest — fixtures, markers

from app.ingest import chunk_document   # local — app/ingest/chunker.py
from app.retrieval.hybrid import HybridRetriever  # local — app/retrieval/hybrid.py

# Two tiny corpora, inline, so the test doesn't depend on sample_docs/ layout.
CONTRACT = """# Acme MSA

## 2. Payment Terms

Payment is due within thirty (30) days of receipt of a valid invoice. Late
amounts accrue interest at 1.5% per month.

## 5. Troubleshooting Reference

| Error Code | Remedy |
|------------|--------|
| E-4470     | Rotate the token |
| E-4471     | Restart the ingestion worker |
"""

INVOICES = """# Invoice Register

## INV-2024-0888 — Northwind Traders
Issued 2024-03-02. Total 84,200.00. Net 30.

## INV-2024-0891 — Acme Corp
Issued 2024-03-06. Total 412,000.00. Net 30.

## INV-2024-0892 — Adventure Works
Issued 2024-03-08. Total 7,350.00. Net 30.

## INV-2024-0893 — Tailspin Toys
Issued 2024-03-11. Total 66,000.00. Net 60.
"""


@pytest.fixture(scope="module")
def retriever() -> HybridRetriever:
    chunks = chunk_document(CONTRACT, doc_title="Acme MSA", max_chars=700)
    chunks += chunk_document(INVOICES, doc_title="Invoice Register", max_chars=700)
    for i, c in enumerate(chunks):
        c.chunk_index = i               # unique across docs — RRF keys on this
    return HybridRetriever(chunks)


def rank_of(hits, needle: str) -> int | None:
    """1-based rank of the first hit containing `needle`, or None.

    Checks text_to_embed, because that is what BOTH retrievers index — the
    invoice number lives in the heading, not the body.
    """
    for h in hits:
        if needle.lower() in h.chunk.text_to_embed.lower():
            return h.rank + 1
    return None


# =============================================================================
# Failure 1: dense loses exact tokens.
# =============================================================================
def test_dense_confuses_near_identical_invoice_numbers(retriever) -> None:
    """The predicted failure, with a real model: ask for one invoice, get another.

    Deliberately asserted LOOSELY — we assert dense does not put it first, not
    that it lands on an exact rank. Model versions shift; the phenomenon doesn't.
    """
    dense = retriever.dense_search("INV-2024-0891", k=10)
    top = dense[0].chunk.text_to_embed
    assert "INV-2024-0891" not in top, (
        "dense ranked a DIFFERENT invoice first — 0888/0892/0893 are "
        "semantically identical to 0891; the digits carry no meaning"
    )


def test_bm25_nails_the_exact_invoice_number(retriever) -> None:
    """BM25 understands nothing and gets it exactly right."""
    assert rank_of(retriever.bm25_search("INV-2024-0891", k=10), "INV-2024-0891") == 1


def test_rrf_rescues_the_exact_token_query(retriever) -> None:
    """Fusion recovers what dense alone lost."""
    assert rank_of(retriever.rrf("INV-2024-0891", k=10), "INV-2024-0891") == 1


# =============================================================================
# Failure 2 (the mirror image): BM25 has no idea what words mean.
# =============================================================================
def test_dense_handles_the_paraphrase_bm25_cannot(retriever) -> None:
    """'settle an invoice' shares almost no vocabulary with 'Payment is due'."""
    q = "how long do we have to settle an invoice?"
    dense_rank = rank_of(retriever.dense_search(q, k=10), "thirty (30) days")
    bm25_rank = rank_of(retriever.bm25_search(q, k=10), "thirty (30) days")

    assert dense_rank == 1, "meaning is exactly what embeddings are for"
    assert bm25_rank is None or bm25_rank > dense_rank, (
        "lexical overlap is near zero, so BM25 has nothing to score"
    )


def test_rrf_costs_you_the_top_slot_but_keeps_the_answer_in_the_pool(retriever) -> None:
    """FUSION IS NOT FREE — and this test exists because it caught me over-claiming.

    I first asserted rrf(...) == 1, reasoning "dense had it at rank 1, so fusion
    keeps it". Wrong. On this corpus dense ranks it 1 and RRF ranks it 3.

    Why: when one retriever is confidently RIGHT and the other is confidently
    WRONG, the wrong one's rankings still earn 1/(k+rank) credit. BM25's
    top-ranked garbage gets fused in and pushes the true answer down.

    So RRF's job is NOT to produce rank 1. Its job is to get the answer into the
    CANDIDATE POOL when either retriever finds it. Fixing the ordering is the
    cross-encoder's job — which is precisely why the architecture is
        retrieve wide + cheap (RRF, top 50)  ->  rerank narrow + expensive (top 5)
    and not "just use RRF's top 5".
    """
    q = "how long do we have to settle an invoice?"
    dense_rank = rank_of(retriever.dense_search(q, k=10), "thirty (30) days")
    rrf_rank = rank_of(retriever.rrf(q, k=10), "thirty (30) days")

    assert dense_rank == 1
    assert rrf_rank is not None, "fusion must not LOSE the answer entirely"
    assert rrf_rank <= 5, "it stays in the pool a reranker would rescore"
    # The honest, uncomfortable part, asserted so it can't be forgotten:
    assert rrf_rank >= dense_rank, (
        "fusion can DEMOTE a correct dense hit — that is the cost of pooling, "
        "and the reason a reranker follows it"
    )


# =============================================================================
# The wiring bug that once made BM25 look useless.
# =============================================================================
def test_both_retrievers_index_the_same_text(retriever) -> None:
    """BM25 originally indexed chunk.text (body only) while dense indexed
    text_to_embed (body + heading). Invoice numbers live in the HEADING, so
    BM25's index didn't contain the token at all and it scored 0.0000 on every
    invoice query — looking like a property of BM25 rather than a wiring bug.

    If two retrievers index different text, you aren't comparing retrievers.
    You're comparing corpora.
    """
    c = next(c for c in retriever.chunks if "INV-2024-0891" in c.heading)
    assert "INV-2024-0891" not in c.text, "the number is in the heading, not the body"
    assert "INV-2024-0891" in c.text_to_embed, "…and text_to_embed is what we index"
    assert rank_of(retriever.bm25_search("INV-2024-0891", k=10), "INV-2024-0891") == 1
