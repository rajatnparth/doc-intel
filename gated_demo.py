"""Watch the gates hold, the post-filter leak, and the system refuse.

    python gated_demo.py

Two policyholders on the same motor product, one superseded policy kit, and a
reranker that decides whether we answer at all.
"""

from datetime import date              # stdlib — the time-gate demo

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


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> None:
    chunks = load_corpus()
    pre = PreFilterRetriever.from_chunks(chunks)
    q = "what is my excess for an own damage claim?"

    banner("THE GATE — same query, two policyholders, disjoint answers")
    for label, prin in [("asha", ASHA_CUSTOMER), ("vikram", VIKRAM_CUSTOMER)]:
        hits = pre.search(q, Principal(*prin), k=5)
        tenants = {h.chunk.meta.tenant_id for h in hits}
        print(f"\n  {label:8} → tenants in candidate set: {tenants}")
        print(f"           top hit: {hits[0].chunk.heading[:44]}")
        print(f"           says   : {hits[0].chunk.text[:64].strip()}...")
    print("\n  Foreign chunks were never CANDIDATES — not 'filtered later'.")

    banner("💀 THE LEAK — post-filter + a cache someone added later")
    post = PostFilterRetriever(chunks)
    asha_view = post.search(q, Principal(*ASHA_CUSTOMER), k=20)
    returned_tenants = {h.chunk.meta.tenant_id for h in asha_view}
    print(f"\n  Asha's RETURNED result is correct: {returned_tenants}")
    print("  ...which is exactly why 'the user never sees it' is the wrong objection.\n")
    cached = post.cache[q]
    print("  But the CACHE (populated before the filter ran) holds:")
    for t in sorted({h.chunk.meta.tenant_id for h in cached}):
        n = sum(1 for h in cached if h.chunk.meta.tenant_id == t)
        mark = "  ← FOREIGN" if t != "asha" else ""
        print(f"    {t:10} {n} chunks{mark}")
    foreign = [h.chunk for h in cached if h.chunk.meta.tenant_id == "vikram"]
    print(f"\n  Vikram's actual policy terms, sitting in a query-keyed cache:")
    print(f"    {foreign[0].text[:70].strip()}...")
    print("\n  Anything reading that cache — a reranker, a log, a 'related docs'")
    print("  sidebar, a trace exporter — is now a breach. The cache dev did")
    print("  nothing wrong. The vulnerability arrived with post-filtering.")

    banner("THE TIME GATE — 'active' is the wrong question")
    q3 = "what is my excess for an own damage claim?"
    for label, as_of in [("today", None), ("2025-12-20 (date of loss)", date(2025, 12, 20))]:
        hits = pre.search(q3, Principal(*ASHA_CUSTOMER), k=10, as_of=as_of)
        docs = sorted({h.chunk.doc_title for h in hits})
        quoted = next(
            (amt for amt in ("₹1,000", "₹2,000") if any(amt in h.chunk.text for h in hits)), "?"
        )
        print(f"\n  as of {label}")
        print(f"    kit in force : {docs}")
        print(f"    excess quoted: {quoted}")
    print("\n  A December accident reported late is assessed under DECEMBER's")
    print("  wording — same question, two dates, two answers, both CORRECT.")
    print("  A status flag cannot represent that question. 'Superseded' is now")
    print("  a DERIVED fact: the window closed; nobody had to remember a flag.")

    banner("THE REFUSAL PATH — and the phrasing cliff")
    for q2 in [
        "how quickly must I report an accident?",
        "is a courtesy car provided during repairs?",
        "what is the limit of liability?",
        "is there an upper limit on what a claim pays out?",
    ]:
        a = answer(q2, Principal(*ASHA_CUSTOMER), pre)
        verdict = "REFUSED" if a.refused else "answered"
        print(f"\n  {q2}")
        print(f"    {verdict:8}  score={a.score:.4f}")
        if a.refused:
            print(f"    near miss: {a.near_misses[0].heading[:46]}")

    print("\n  ⚠️  The last two are THE SAME QUESTION, and section 7 answers both.")
    print("     Phrased in the document's own words ('limit of liability') the")
    print("     reranker scores ~0.999. Phrased the way a customer would say it,")
    print("     the SAME chunk is still RANKED #1 — and scored near zero, so the")
    print("     gate refuses. The score tracks lexical anchoring, not relevance:")
    print("     ms-marco was trained on web passages, policy wording is out-of-")
    print("     domain, and customers never use the document's vocabulary.")
    print("     No threshold fixes that. See: python -m app.retrieval.calibrate")


if __name__ == "__main__":
    main()
