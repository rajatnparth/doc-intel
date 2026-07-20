"""Phase 5 — the storage seam, as executable claims.

Every gate property is asserted against BOTH stores through one parametrised
fixture. That is the point of the seam: the tests define what "a store" means,
and an implementation that filters differently — say, a Qdrant filter missing
a clause that MemoryStore's predicate has — goes red on the same assertions
the honest implementation passes. Two implementations, one contract.

The qdrant store here is REAL (embedded local mode in a tmp dir), not mocked:
a mocked store would prove our mock filters correctly, which proves nothing.
"""

from datetime import date               # stdlib — as_of anchors

import pytest                           # 3rd-party: pytest — fixtures, params

from app.ingest.index import ingest_into        # local — app/ingest/index.py
from app.retrieval.corpus import (      # local — app/retrieval/corpus.py
    ASHA_AGENT,
    ASHA_CUSTOMER,
    load_corpus,
)
from app.retrieval.gated import Principal, PreFilterRetriever  # local — app/retrieval/gated.py
from app.store.base import Gate         # local — app/store/base.py
from app.store.memory import MemoryStore  # local — app/store/memory.py
from app.store.qdrant import QdrantStore  # local — app/store/qdrant.py

# In force TODAY (the 2026 kits) vs in force at a 2025 date of loss.
NOW = date(2026, 7, 20)
LOSS_2025 = date(2025, 12, 20)


def _build(kind: str, tmp_path):
    if kind == "memory":
        return MemoryStore()
    return QdrantStore(path=str(tmp_path / "qdrant"), collection="chunks")


@pytest.fixture(params=["memory", "qdrant"])
def store(request, tmp_path):
    s = _build(request.param, tmp_path)
    ingest_into(s, load_corpus())
    yield s
    s.close()


# =============================================================================
# The gate properties — identical assertions for every implementation
# =============================================================================
def test_foreign_tenants_are_never_candidates(store) -> None:
    gate = Gate("asha", frozenset({"customer"}), NOW)
    hits = store.search([0.0] * 384, gate, k=50)   # any direction; k > corpus
    assert hits, "asha must see her own kit"
    tenants = {c.meta.tenant_id for c, _ in hits}
    assert tenants == {"asha"}, f"foreign tenant in the candidate set: {tenants}"


def test_acl_gates_inside_the_search(store) -> None:
    """The claims file is agent-only. A customer gate must not surface it —
    not 'not in the top k': not AT ALL, at any k."""
    customer = Gate(*ASHA_CUSTOMER, NOW)
    agent = Gate(*ASHA_AGENT, NOW)
    claims = "Claims File — Asha Rao"

    customer_docs = {c.doc_title for c, _ in store.search([0.0] * 384, customer, k=50)}
    agent_docs = {c.doc_title for c, _ in store.search([0.0] * 384, agent, k=50)}

    assert claims not in customer_docs
    assert claims in agent_docs


def test_the_effective_window_travels_with_the_query(store) -> None:
    """Same store, same principal, two dates → two disjoint wording sets.
    This is the old time-leak test with nowhere left for a leak to live:
    there is no cached view, only the filter that came with each query."""
    now_docs = {c.doc_title for c in store.visible_chunks(Gate(*ASHA_CUSTOMER, NOW))}
    loss_docs = {c.doc_title for c in store.visible_chunks(Gate(*ASHA_CUSTOMER, LOSS_2025))}

    assert "Asha Rao — Motor Policy Kit (2026)" in now_docs
    assert "Asha Rao — Motor Policy Kit (2025)" not in now_docs
    assert "Asha Rao — Motor Policy Kit (2025)" in loss_docs
    assert "Asha Rao — Motor Policy Kit (2026)" not in loss_docs


def test_upsert_is_idempotent(store) -> None:
    """Re-running ingestion must be a no-op, not a second corpus. Deterministic
    ids are what make 're-ingest after a doc changes' safe to automate."""
    before = store.count()
    ingest_into(store, load_corpus())
    assert store.count() == before


def test_search_returns_k_visible_results_not_k_minus_filtered(store) -> None:
    """The anti-post-filter assertion: ask for 5, get 5 gate-passing chunks
    (the visible corpus is larger than 5). A post-filtering store returns
    'however many survived', which is how result counts become unpredictable."""
    gate = Gate(*ASHA_AGENT, NOW)
    assert len(store.search([0.0] * 384, gate, k=5)) == 5


# =============================================================================
# Persistence — qdrant only, because that is the promise memory doesn't make
# =============================================================================
def test_qdrant_survives_a_restart(tmp_path) -> None:
    path = str(tmp_path / "qdrant")

    s1 = QdrantStore(path=path, collection="chunks")
    n = ingest_into(s1, load_corpus())
    s1.close()                          # ← the process ends

    s2 = QdrantStore(path=path, collection="chunks")   # ← a new process boots
    try:
        assert s2.count() == n, "the corpus must outlive the process"
        # And it is still GATED, not merely present: reopen, same filter law.
        hits = s2.search([0.0] * 384, Gate(*ASHA_CUSTOMER, NOW), k=50)
        assert hits and {c.meta.tenant_id for c, _ in hits} == {"asha"}
    finally:
        s2.close()


# =============================================================================
# End-to-end: the retriever over a persistent store
# =============================================================================
def test_full_pipeline_over_qdrant(tmp_path) -> None:
    """The December-loss scenario from test_gated, replayed against the real
    persistent store: retrieval + BM25 + RRF over the store's filtered view."""
    s = QdrantStore(path=str(tmp_path / "qdrant"), collection="chunks")
    try:
        ingest_into(s, load_corpus())
        r = PreFilterRetriever(s)
        asha = Principal(*ASHA_CUSTOMER)

        at_loss = r.search("what is my excess for an own damage claim?", asha, k=20, as_of=LOSS_2025)
        assert any("₹1,000" in h.chunk.text for h in at_loss), "2025 wording governs a 2025 loss"
        assert not any("₹2,000" in h.chunk.text for h in at_loss), "2026 wording must be out of scope"
    finally:
        s.close()
