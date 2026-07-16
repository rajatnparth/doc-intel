"""Watch the gates hold, the post-filter leak, and the system refuse.

    python gated_demo.py

Two tenants with genuinely similar contracts, one superseded document, and a
reranker that decides whether we answer at all.
"""

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


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> None:
    chunks = load_corpus()
    pre = PreFilterRetriever(chunks)
    q = "what are our payment terms?"

    banner("THE GATE — same query, two tenants, disjoint answers")
    for label, prin in [("acme", ACME_FINANCE), ("contoso", CONTOSO_LEGAL)]:
        hits = pre.search(q, Principal(*prin), k=5)
        tenants = {h.chunk.meta.tenant_id for h in hits}
        print(f"\n  {label:8} → tenants in candidate set: {tenants}")
        print(f"           top hit: {hits[0].chunk.heading[:44]}")
        print(f"           says   : {hits[0].chunk.text[:64].strip()}...")
    print("\n  Foreign chunks were never CANDIDATES — not 'filtered later'.")

    banner("💀 THE LEAK — post-filter + a cache someone added later")
    post = PostFilterRetriever(chunks)
    acme_view = post.search(q, Principal(*ACME_FINANCE), k=20)
    returned_tenants = {h.chunk.meta.tenant_id for h in acme_view}
    print(f"\n  Acme's RETURNED result is correct: {returned_tenants}")
    print("  ...which is exactly why 'the user never sees it' is the wrong objection.\n")
    cached = post.cache[q]
    print("  But the CACHE (populated before the filter ran) holds:")
    for t in sorted({h.chunk.meta.tenant_id for h in cached}):
        n = sum(1 for h in cached if h.chunk.meta.tenant_id == t)
        mark = "  ← FOREIGN" if t != "acme" else ""
        print(f"    {t:10} {n} chunks{mark}")
    foreign = [h.chunk for h in cached if h.chunk.meta.tenant_id == "contoso"]
    print(f"\n  Contoso's actual contract text, sitting in a query-keyed cache:")
    print(f"    {foreign[0].text[:70].strip()}...")
    print("\n  Anything reading that cache — a reranker, a log, a 'related docs'")
    print("  sidebar, a trace exporter — is now a breach. The cache dev did")
    print("  nothing wrong. The vulnerability arrived with post-filtering.")

    banner("THE FRESHNESS GATE — superseded is not retrievable")
    hits = pre.search("how long do we have to pay an invoice?", Principal(*ACME_LEGAL), k=10)
    print(f"\n  docs reachable: {sorted({h.chunk.doc_title for h in hits})}")
    print("  Acme MSA (2022) says 60 days / 0.5% interest. It is in the store and")
    print("  unreachable. 'The LLM will notice the date' is not a control.")

    banner("THE REFUSAL PATH")
    for q2 in [
        "how long do we have to pay an invoice?",
        "what is the parental leave policy?",
        "what is the cap on liability?",
    ]:
        a = answer(q2, Principal(*ACME_FINANCE), pre)
        verdict = "REFUSED" if a.refused else "answered"
        print(f"\n  {q2}")
        print(f"    {verdict:8}  score={a.score:.4f}")
        if a.refused:
            print(f"    near miss: {a.near_misses[0].heading[:46]}")

    print("\n  ⚠️  The third one is a FALSE refusal. Section 7 answers it, retrieval")
    print("     found it, and the reranker RANKED IT #1 — then scored it 0.009.")
    print("     ms-marco-MiniLM was trained on web search passages; contract prose")
    print("     is out-of-domain. Its RANKING is fine; its CALIBRATION is not.")
    print("     No threshold fixes that. See: python -m app.retrieval.calibrate")


if __name__ == "__main__":
    main()
