"""Section 3.4 — the gates and the refusal path, as executable claims.

The headline test is `test_post_filter_leaks_across_tenants_via_the_cache`:
it does not ASSERT that post-filtering is dangerous, it DEMONSTRATES the leak.
"""

import pytest                           # 3rd-party: pytest — fixtures, raises

from app.retrieval.corpus import (      # local — app/retrieval/corpus.py
    ASHA_AGENT,
    ASHA_CUSTOMER,
    VIKRAM_CUSTOMER,
    load_corpus,
)
from app.retrieval.gated import (       # local — app/retrieval/gated.py
    PostFilterRetriever,
    PreFilterRetriever,
    Principal,
    answer,
)


@pytest.fixture(scope="module")
def chunks():
    return load_corpus()


@pytest.fixture(scope="module")
def pre(chunks):
    return PreFilterRetriever(chunks)


# =============================================================================
# THE GATE — tenant isolation
# =============================================================================
def test_pre_filter_never_makes_foreign_chunks_candidates(pre) -> None:
    """Not 'the user doesn't see them' — they are NEVER CANDIDATES.

    The query matches both policyholders' kits (both have an excess clause), so
    a cross-tenant hit is a genuine retrieval outcome, not a contrived one.
    """
    asha = Principal(*ASHA_CUSTOMER)
    hits = pre.search("what is my excess for an own damage claim?", asha, k=20)

    assert hits, "asha should get answers"
    tenants = {h.chunk.meta.tenant_id for h in hits}
    assert tenants == {"asha"}, f"foreign tenant entered the candidate set: {tenants}"


def test_superseded_documents_are_not_retrievable(pre) -> None:
    """'The LLM will notice the date' is not a control. Filter status=active."""
    asha = Principal(*ASHA_CUSTOMER)
    hits = pre.search("how long do I have to pay my renewal premium?", asha, k=20)

    titles = {h.chunk.doc_title for h in hits}
    assert "Asha Rao — Motor Policy Kit (2025)" not in titles, (
        "superseded policy kit must not be retrievable"
    )
    assert "Asha Rao — Motor Policy Kit (2026)" in titles, "…but the current one must be"

    # Content check, keyed on a string UNIQUE to the superseded version.
    #
    # A content assertion must key on something that exists in EXACTLY ONE
    # place, or it tests coincidence. "thirty (30) days" would be the wrong key:
    # the 2025 kit uses it for the premium due date and the 2026 kit uses it for
    # the renewal-notice lead time — same words, different fact. (The original
    # contract-domain version of this test hit the same trap with "sixty (60)
    # days".) The 2025 kit's ₹1,000 excess appears nowhere else.
    assert not any("₹1,000" in h.chunk.text for h in hits), (
        "the 2025 excess leaked — a customer would be quoted the wrong amount"
    )


def test_acl_gates_within_a_tenant(pre) -> None:
    """Tenancy is not the only boundary — the claims file carries internal
    handler notes, so it is agent-only. The customer's self-service view must
    never retrieve it, even though it is THEIR OWN claim."""
    customer = Principal(*ASHA_CUSTOMER)    # claims file is acl={"agent"}
    hits = pre.search("what is the status of claim CLM-2026-0891?", customer, k=20)
    assert not any("Claims File" in h.chunk.doc_title for h in hits)

    agent = Principal(*ASHA_AGENT)          # …but the call-centre view can see it
    hits = pre.search("what is the status of claim CLM-2026-0891?", agent, k=20)
    assert any("Claims File" in h.chunk.doc_title for h in hits)


# =============================================================================
# THE HEADLINE: post-filtering leaks. Demonstrated, not asserted.
# =============================================================================
def test_post_filter_leaks_across_tenants_via_the_cache(chunks) -> None:
    """The scenario, executed.

    A developer adds a semantic cache between retrieval and the filter. Nothing
    they did is wrong — inserting a component into a pipeline is what pipelines
    are for. But the unfiltered top-k is now cached, and the next tenant reads it.

    The vulnerability was introduced the day post-filtering was chosen. This test
    is the proof.
    """
    post = PostFilterRetriever(chunks)
    query = "what is my excess for an own damage claim?"

    # 1. Asha asks. Her filtered result is CORRECT — this is what makes
    #    post-filtering seductive, and why "the user sees it" is the wrong objection.
    asha_hits = post.search(query, Principal(*ASHA_CUSTOMER), k=20)
    assert {h.chunk.meta.tenant_id for h in asha_hits} == {"asha"}

    # 2. But the CACHE was populated BEFORE the filter ran…
    cached = post.cache[query]
    leaked = {h.chunk.meta.tenant_id for h in cached}
    assert "vikram" in leaked, (
        "the cache holds Vikram's chunks — the filter ran AFTER this line"
    )

    # 3. …so anything reading the cache sees every tenant. A reranker, a debug
    #    log, a 'related documents' sidebar, an eval harness, a trace exporter.
    #    Each is a reasonable thing to add. Each is now a breach.
    vikram_chunks = [h.chunk for h in cached if h.chunk.meta.tenant_id == "vikram"]
    assert vikram_chunks, "foreign data is sitting in a shared, query-keyed cache"
    assert any("₹5,000" in c.text for c in vikram_chunks), (
        "and it's a real policy term — another customer's excess amount"
    )


def test_pre_filter_makes_that_leak_unrepresentable(pre) -> None:
    """The contrast, and the whole design argument.

    PreFilterRetriever caches too — but its key is (tenant_id, groups), so a
    cached view CANNOT contain a foreign chunk. There is no window between
    'retrieved' and 'filtered' to insert anything into, because the foreign data
    never entered the pipeline.

    Prefer designs where the mistake is IMPOSSIBLE over designs where it is
    merely PROHIBITED.
    """
    q = "what is my excess for an own damage claim?"
    asha = pre.search(q, Principal(*ASHA_CUSTOMER), k=20)
    vikram = pre.search(q, Principal(*VIKRAM_CUSTOMER), k=20)

    assert {h.chunk.meta.tenant_id for h in asha} == {"asha"}
    assert {h.chunk.meta.tenant_id for h in vikram} == {"vikram"}

    # Same query string, two principals, two disjoint result sets — with a cache
    # in play. The principal is part of the key, so caching is safe here.
    assert not ({c.chunk.chunk_index for c in asha} & {c.chunk.chunk_index for c in vikram})


# =============================================================================
# THE REFUSAL PATH
# =============================================================================
def test_answerable_question_is_answered(pre) -> None:
    a = answer("how quickly must I report an accident?", Principal(*ASHA_CUSTOMER), pre)
    assert a.refused is False
    assert a.score > 0.9
    assert any("twenty-four (24) hours" in c.text for c in a.chunks)


def test_unanswerable_question_is_refused(pre) -> None:
    """The corpus says nothing about courtesy cars. Saying so is the CORRECT
    answer — not a failure state. The alternative is fabricating cover."""
    a = answer("is a courtesy car provided during repairs?", Principal(*ASHA_CUSTOMER), pre)
    assert a.refused is True
    assert a.score < 0.5
    assert a.chunks == [], "a refusal must carry no context onward to a generator"
    assert a.near_misses, "but we still offer the closest documents as links"
    assert "threshold" in a.reason


def test_refusal_reports_its_score(pre) -> None:
    """The score is always reported, answered or refused. You cannot debug, tune
    or defend a gate whose number you never see."""
    for q in ["how quickly must I report an accident?", "who is my claims handler?"]:
        a = answer(q, Principal(*ASHA_CUSTOMER), pre)
        assert 0.0 <= a.score <= 1.0


# =============================================================================
# The uncomfortable one, kept because it is true.
# =============================================================================
def test_reranker_ranks_correctly_but_scores_a_lie(pre) -> None:
    """CALIBRATION != RANKING, and the refusal gate depends on calibration.

    'is there an upper limit on what a claim pays out?' IS answerable — section
    7 covers it. The pipeline retrieves it and the cross-encoder RANKS IT #1.
    Then scores it 0.0006, so the gate refuses a question it could answer.

    The domain conversion SHARPENED this finding. Ask the same section the same
    thing in ITS OWN words — "what is the limit of liability?", echoing the
    heading — and the same chunk scores 0.9987. Same chunk, same meaning:
    0.9987 anchored, 0.0006 paraphrased. The score does not measure whether the
    chunk answers the question; it collapses the moment the customer stops
    using the document's vocabulary — and customers never use the document's
    vocabulary. ms-marco-MiniLM was trained on MS MARCO web passages; policy
    wording is out-of-distribution. The ordering survives; the number does not.

    This test asserts the DEFECT, on purpose. If a future reranker fixes it, this
    test fails loudly and tells you the model got better — which is exactly the
    signal you want. Deleting it would hide the one thing a reviewer should know.
    """
    a = answer("is there an upper limit on what a claim pays out?", Principal(*ASHA_CUSTOMER), pre)

    assert a.refused is True, "the gate refuses it — a FALSE refusal"
    assert a.score < 0.1, "scored ~0 despite being genuinely answerable"

    # And here is the proof it is calibration, not retrieval: the right chunk is
    # the top near-miss. Retrieval and ranking both did their job.
    assert "Limit of Liability" in a.near_misses[0].heading, (
        "ranked #1 and still refused — no threshold fixes this; a better "
        "reranker does"
    )
