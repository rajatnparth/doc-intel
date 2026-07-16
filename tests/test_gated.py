"""Section 3.4 — the gates and the refusal path, as executable claims.

The headline test is `test_post_filter_leaks_across_tenants_via_the_cache`:
it does not ASSERT that post-filtering is dangerous, it DEMONSTRATES the leak.
"""

import pytest                           # 3rd-party: pytest — fixtures, raises

from app.retrieval.corpus import (      # local — app/retrieval/corpus.py
    ACME_FINANCE,
    ACME_LEGAL,
    CONTOSO_LEGAL,
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

    The query matches both tenants' contracts (both have payment terms), so a
    cross-tenant hit is a genuine retrieval outcome, not a contrived one.
    """
    acme = Principal(*ACME_FINANCE)
    hits = pre.search("what are our payment terms?", acme, k=20)

    assert hits, "acme should get answers"
    tenants = {h.chunk.meta.tenant_id for h in hits}
    assert tenants == {"acme"}, f"foreign tenant entered the candidate set: {tenants}"


def test_superseded_documents_are_not_retrievable(pre) -> None:
    """'The LLM will notice the date' is not a control. Filter status=active."""
    acme = Principal(*ACME_LEGAL)
    hits = pre.search("how long do we have to pay an invoice?", acme, k=20)

    titles = {h.chunk.doc_title for h in hits}
    assert "Acme MSA (2022)" not in titles, "superseded contract must not be retrievable"
    assert "Acme MSA (2024)" in titles, "…but the current one must be"

    # Content check, keyed on a string UNIQUE to the superseded version.
    #
    # The first draft asserted `"sixty (60) days" not in ...` — reasoning that the
    # 2022 doc says 60-day payment terms and the 2024 doc says 30. It failed, and
    # the CODE was right: the 2024 contract also says "sixty (60) days", for the
    # non-renewal notice period. The phrase wasn't unique to the doc I was testing.
    #
    # Lesson worth keeping: a content assertion must key on something that exists
    # in EXACTLY ONE place, or it tests coincidence. 0.5% interest is 2022-only
    # (2024 charges 1.5%).
    assert not any("0.5% per month" in h.chunk.text for h in hits), (
        "the 2022 rate leaked — a user would be quoted the wrong interest"
    )


def test_acl_gates_within_a_tenant(pre) -> None:
    """Tenancy is not the only boundary — not everyone in a tenant sees everything."""
    legal = Principal(*ACME_LEGAL)          # invoices.md is acl={"finance"}
    hits = pre.search("what is the total on invoice INV-2024-0891?", legal, k=20)
    assert not any("Invoice Register" in h.chunk.doc_title for h in hits)

    finance = Principal(*ACME_FINANCE)      # …but finance can see them
    hits = pre.search("what is the total on invoice INV-2024-0891?", finance, k=20)
    assert any("Invoice Register" in h.chunk.doc_title for h in hits)


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
    query = "what are our payment terms?"

    # 1. Acme asks. Their filtered result is CORRECT — this is what makes
    #    post-filtering seductive, and why "the user sees it" is the wrong objection.
    acme_hits = post.search(query, Principal(*ACME_FINANCE), k=20)
    assert {h.chunk.meta.tenant_id for h in acme_hits} == {"acme"}

    # 2. But the CACHE was populated BEFORE the filter ran…
    cached = post.cache[query]
    leaked = {h.chunk.meta.tenant_id for h in cached}
    assert "contoso" in leaked, (
        "the cache holds Contoso's chunks — the filter ran AFTER this line"
    )

    # 3. …so anything reading the cache sees every tenant. A reranker, a debug
    #    log, a 'related documents' sidebar, an eval harness, a trace exporter.
    #    Each is a reasonable thing to add. Each is now a breach.
    contoso_chunks = [h.chunk for h in cached if h.chunk.meta.tenant_id == "contoso"]
    assert contoso_chunks, "foreign data is sitting in a shared, query-keyed cache"
    assert any("forty-five (45) days" in c.text for c in contoso_chunks), (
        "and it's real contract terms, not a stub"
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
    q = "what are our payment terms?"
    acme = pre.search(q, Principal(*ACME_FINANCE), k=20)
    contoso = pre.search(q, Principal(*CONTOSO_LEGAL), k=20)

    assert {h.chunk.meta.tenant_id for h in acme} == {"acme"}
    assert {h.chunk.meta.tenant_id for h in contoso} == {"contoso"}

    # Same query string, two principals, two disjoint result sets — with a cache
    # in play. The principal is part of the key, so caching is safe here.
    assert not ({c.chunk.chunk_index for c in acme} & {c.chunk.chunk_index for c in contoso})


# =============================================================================
# THE REFUSAL PATH
# =============================================================================
def test_answerable_question_is_answered(pre) -> None:
    a = answer("how long do we have to pay an invoice?", Principal(*ACME_FINANCE), pre)
    assert a.refused is False
    assert a.score > 0.9
    assert any("thirty (30) days" in c.text for c in a.chunks)


def test_unanswerable_question_is_refused(pre) -> None:
    """The corpus has no parental leave policy. Saying so is the CORRECT answer —
    not a failure state. The alternative is fabricating one."""
    a = answer("what is the parental leave policy?", Principal(*ACME_FINANCE), pre)
    assert a.refused is True
    assert a.score < 0.5
    assert a.chunks == [], "a refusal must carry no context onward to a generator"
    assert a.near_misses, "but we still offer the closest documents as links"
    assert "threshold" in a.reason


def test_refusal_reports_its_score(pre) -> None:
    """The score is always reported, answered or refused. You cannot debug, tune
    or defend a gate whose number you never see."""
    for q in ["how long do we have to pay an invoice?", "who is the CEO?"]:
        a = answer(q, Principal(*ACME_FINANCE), pre)
        assert 0.0 <= a.score <= 1.0


# =============================================================================
# The uncomfortable one, kept because it is true.
# =============================================================================
def test_reranker_ranks_correctly_but_scores_a_lie(pre) -> None:
    """CALIBRATION != RANKING, and the refusal gate depends on calibration.

    'what is the cap on liability?' IS answerable — section 7 covers it. The
    pipeline retrieves it and the cross-encoder RANKS IT #1. Then scores it
    0.009, so the gate refuses a question we could have answered.

    ms-marco-MiniLM was trained on MS MARCO web passages; contract prose is
    out-of-distribution. The ordering survives; the absolute number does not.

    This test asserts the DEFECT, on purpose. If a future reranker fixes it, this
    test fails loudly and tells you the model got better — which is exactly the
    signal you want. Deleting it would hide the one thing a reviewer should know.
    """
    a = answer("what is the cap on liability?", Principal(*ACME_FINANCE), pre)

    assert a.refused is True, "the gate refuses it — a FALSE refusal"
    assert a.score < 0.1, "scored ~0 despite being genuinely answerable"

    # And here is the proof it is calibration, not retrieval: the right chunk is
    # the top near-miss. Retrieval and ranking both did their job.
    assert "Limitation of Liability" in a.near_misses[0].heading, (
        "ranked #1 and still refused — no threshold fixes this; a better "
        "reranker does"
    )
