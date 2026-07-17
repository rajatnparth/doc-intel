"""Where the refusal threshold comes from. This is the module's real artifact.

    python -m app.retrieval.calibrate

You cannot pick a threshold from a vibe. You build two labelled query sets —
questions the corpus ANSWERS, and plausible questions it DOESN'T — score both
through the real pipeline, look at the two distributions, and choose from the
tradeoff you can defend to a product owner.

WHAT THIS ACTUALLY MEASURED (on the motor policy corpus)
------------------------------------------------------------------
The received wisdom — and the first draft of this file — says: "the answerable
and unanswerable distributions overlap in the middle, so pick a threshold from
the business tradeoff." Run it. Three things happen instead, sharing one cause.

1. A FALSE REFUSAL no threshold can fix. "is there an upper limit on what a
   claim pays out?" is answered by section 7. Retrieval surfaces it, the
   cross-encoder RANKS IT #1 — then scores it 0.0006. Ask the SAME section the
   same thing in its own words ("what is the limit of liability?") and the same
   chunk scores 0.9987. Anchored 0.9987, paraphrased 0.0006: the score tracks
   LEXICAL ANCHORING, not relevance — and customers never phrase questions in
   the document's vocabulary.

2. A FALSE ANSWER the threshold can't fix either. "what will next year's
   renewal premium be?" scores 0.7785 — section 2 is ABOUT the renewal premium,
   so topical proximity scores high. The amount is not in the corpus.
   Topicality is not answerability.

3. And the two errors sit on OPPOSITE WRONG SIDES of every threshold: from
   0.10 to 0.75 the sweep table is flat at 1 false answer + 2 false refusals.
   Moving the knob trades nothing. The threshold is not the interesting
   control; the RERANKER is.

Bonus defect: "what should I do if the car is not driveable?" scores 0.0010
even though "Vehicle not driveable" appears VERBATIM in the corpus — inside a
table. A cross-encoder trained on web prose does not read tables.

Why: ms-marco-MiniLM was trained on MS MARCO — web search passages. Policy
wording ("aggregate liability shall not exceed the Insured's Declared Value")
is out-of-distribution. So the finding stands, sharper than the textbook one:
A CROSS-ENCODER'S RANKING CAN BE TRUSTWORTHY WHILE ITS CALIBRATION IS NOT. The
refusal gate depends on the calibration, not the ranking — so an out-of-domain
reranker breaks refusal even when retrieval is perfect. No threshold rescues
that; you need a domain-suitable reranker, or a threshold fit on YOUR data, or
a different signal entirely.

Which is the "MTEB is not your corpus" lesson, arriving where it actually hurts.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from app.retrieval.corpus import ASHA_AGENT, load_corpus  # local — app/retrieval/corpus.py
from app.retrieval.gated import (       # local — app/retrieval/gated.py
    PreFilterRetriever,
    Principal,
    rerank,
)

# -----------------------------------------------------------------------------
# The labelled sets. Small, hand-built, honest about it.
#
# Real calibration wants ~100 of each, drawn from PRODUCTION QUERY LOGS — not
# from your imagination, which systematically produces questions your corpus
# happens to answer. These are enough to show the method and the overlap.
# -----------------------------------------------------------------------------
ANSWERABLE = [
    "how long do I have to pay my renewal premium?",
    "what happens if a premium instalment is overdue?",
    "is windscreen glass replacement covered?",
    "how quickly must I report an accident?",
    "what does damage code D-4471 mean?",
    "what is the status of claim CLM-2026-0891?",
    "how much notice is required to cancel the policy?",
    "is there an upper limit on what a claim pays out?",
    "when is a claim deemed settled?",
    "what should I do if the car is not driveable?",
]

UNANSWERABLE = [
    # Plausible. Same domain, same vocabulary. Simply not in the corpus.
    "is a courtesy car provided during repairs?",
    "who is my claims handler?",
    "what is the no-claims bonus after five years?",
    "am I covered when driving outside the country?",
    "can I add a second driver to the policy?",
    "is there a discount for installing a dashcam?",
    "how do I transfer the policy when I sell the car?",
    "does the policy cover engine damage from waterlogging?",
    "what will next year's renewal premium be?",
    "is my laptop covered if it is stolen from the car?",
]


def score_queries(queries: list[str], retriever, principal) -> list[tuple[str, float]]:
    """Top reranker score per query — one number each, the full pipeline."""
    out = []
    for q in queries:
        hits = retriever.search(q, principal, k=20)
        if not hits:
            out.append((q, 0.0))
            continue
        _, best = rerank(q, [h.chunk for h in hits])[0]
        out.append((q, best))
    return out


# Which chunk heading SHOULD win, for each answerable query. This is the label
# that turns "the score was low" into "and here is WHY" — without it, a false
# refusal is an undiagnosable aggregate.
EXPECTED_HEADING = {
    "how long do I have to pay my renewal premium?": "2. Premium and Payment",
    "what happens if a premium instalment is overdue?": "2. Premium and Payment",
    "is windscreen glass replacement covered?": "3. Cover and Benefits",
    "how quickly must I report an accident?": "4. Claims Process",
    "what does damage code D-4471 mean?": "5. Damage Assessment Codes",
    "what is the status of claim CLM-2026-0891?": "CLM-2026-0891",
    "how much notice is required to cancel the policy?": "1. Policy and Period of Insurance",
    "is there an upper limit on what a claim pays out?": "7. Limit of Liability",
    "when is a claim deemed settled?": "4. Claims Process",
    "what should I do if the car is not driveable?": "5. Damage Assessment Codes",
}


def diagnose_refusals(queries, retriever, principal, threshold: float) -> None:
    """DECOMPOSE every false refusal. Two causes, two completely different fixes.

    A false refusal is an aggregate, and aggregates hide structure:

      RETRIEVAL FAILED   — the right chunk never made the candidate set.
                           Fix: chunking, hybrid weighting, pool size, filters.
      RERANKER MISCALIBRATED — the right chunk was retrieved AND ranked #1,
                           and still scored below the threshold.
                           Fix: a domain-suitable reranker. NOT a lower threshold —
                           lowering it to admit a 0.009 admits everything.

    Reporting "2 false refusals" tells you nothing actionable. Reporting
    "0 retrieval failures, 2 calibration failures" tells you exactly what to change.
    """
    print("\n" + "=" * 78)
    print(f"FALSE REFUSAL DECOMPOSITION  (threshold = {threshold})")
    print("=" * 78)

    retrieval_failed = calibration_failed = 0
    for q in queries:
        hits = retriever.search(q, principal, k=20)
        ranked = rerank(q, [h.chunk for h in hits]) if hits else []
        if not ranked or ranked[0][1] >= threshold:
            continue                                   # answered — not a refusal

        want = EXPECTED_HEADING.get(q, "")
        found_at = next(
            (i for i, (c, _) in enumerate(ranked) if want and want in c.heading), None
        )
        top_chunk, top_score = ranked[0]

        if found_at is None:
            retrieval_failed += 1
            verdict = "RETRIEVAL FAILED — right chunk not in the pool at all"
        elif found_at == 0:
            calibration_failed += 1
            verdict = (
                f"CALIBRATION FAILED — right chunk RANKED #1, scored {top_score:.4f}. "
                f"Ordering correct, number is a lie."
            )
        else:
            calibration_failed += 1
            verdict = f"RANKED #{found_at + 1}, not #1 — reranker ordering is off"

        print(f"\n  {q}")
        print(f"    expected : {want}")
        print(f"    got #1   : {top_chunk.heading[:48]}  ({top_score:.4f})")
        print(f"    → {verdict}")

    print(f"\n  retrieval failures  : {retrieval_failed}")
    print(f"  calibration failures: {calibration_failed}")
    if calibration_failed and not retrieval_failed:
        print("\n  Every false refusal is the RERANKER, not retrieval. Lowering the")
        print("  threshold cannot help: to admit a 0.0006 you must admit ~everything.")
        print("  The fix is a reranker that understands policy wording — ms-marco was")
        print("  trained on web search passages, and this corpus is out-of-domain.")


def histogram(scores: list[float], width: int = 40) -> str:
    """A 10-bucket ASCII histogram over 0..1. Crude and sufficient."""
    buckets = [0] * 10
    for s in scores:
        buckets[min(9, int(s * 10))] += 1
    peak = max(buckets) or 1
    lines = []
    for i, n in enumerate(buckets):
        bar = "█" * int(n / peak * width)
        lines.append(f"  {i/10:.1f}-{(i+1)/10:.1f} |{bar:<{width}} {n}")
    return "\n".join(lines)


def sweep(answerable: list[float], unanswerable: list[float]) -> None:
    """The tradeoff table. This is the output you show a product owner."""
    print(f"\n{'threshold':>10} | {'false answers':>14} | {'false refusals':>15} | note")
    print("-" * 78)
    for t in [i / 20 for i in range(1, 20)]:
        # We ANSWER when score >= t.
        false_answers = sum(1 for s in unanswerable if s >= t)   # fabrication risk
        false_refusals = sum(1 for s in answerable if s < t)     # unhelpful "I don't know"
        if false_answers == 0 and false_refusals == 0:
            note = "← perfect on THIS set (small sample; don't believe it)"
        elif false_answers == 0:
            note = "fails closed"
        elif false_refusals == 0:
            note = "fails open"
        else:
            note = ""
        print(f"{t:>10.2f} | {false_answers:>14} | {false_refusals:>15} | {note}")


def main() -> None:
    print("Loading corpus, embedding, scoring 20 queries through the real pipeline...\n")
    retriever = PreFilterRetriever(load_corpus())
    # The agent view: sees the policy kit AND the claims file, like ACME_FINANCE
    # used to see the contract and the invoice register.
    principal = Principal(*ASHA_AGENT)

    ans = score_queries(ANSWERABLE, retriever, principal)
    una = score_queries(UNANSWERABLE, retriever, principal)

    a_scores = [s for _, s in ans]
    u_scores = [s for _, s in una]

    print("=" * 78)
    print("ANSWERABLE — questions the corpus genuinely answers")
    print("=" * 78)
    for q, s in sorted(ans, key=lambda t: -t[1]):
        print(f"  {s:.4f}  {q}")
    print(f"\n  range: {min(a_scores):.4f} .. {max(a_scores):.4f}")
    print(histogram(a_scores))

    print("\n" + "=" * 78)
    print("UNANSWERABLE — plausible, same domain, not in the corpus")
    print("=" * 78)
    for q, s in sorted(una, key=lambda t: -t[1]):
        print(f"  {s:.4f}  {q}")
    print(f"\n  range: {min(u_scores):.4f} .. {max(u_scores):.4f}")
    print(histogram(u_scores))

    print("\n" + "=" * 78)
    print("THE TRADEOFF — every threshold buys one mistake with the other")
    print("=" * 78)
    sweep(a_scores, u_scores)

    diagnose_refusals(ANSWERABLE, retriever, principal, threshold=0.5)

    # ---- the conclusion, COMPUTED from the data, never hardcoded --------------
    # (An earlier draft printed a confident "overlaps 0.35-0.55, ~8% false
    #  refusals" regardless of what was measured. It was wrong, and it was the
    #  exact failure this whole module warns about: asserting the tidy story
    #  instead of reporting the number. If the text below and the table above
    #  ever disagree, the table wins.)
    mid_band = [s for s in a_scores + u_scores if 0.1 < s < 0.9]
    fr_at_50 = sum(1 for s in a_scores if s < 0.5)
    fa_at_50 = sum(1 for s in u_scores if s >= 0.5)

    print("\n" + "=" * 78)
    print("WHAT THIS SAMPLE ACTUALLY SHOWS")
    print("=" * 78)
    print(f"  answerable   : {min(a_scores):.4f} .. {max(a_scores):.4f}")
    print(f"  unanswerable : {min(u_scores):.4f} .. {max(u_scores):.4f}")
    print(f"  scores in the 0.1-0.9 middle band: {len(mid_band)} / {len(a_scores) + len(u_scores)}")

    if not mid_band:
        print("\n  → BIMODAL, not overlapping. The textbook 'pick a threshold from the")
        print("    overlap' story does NOT apply: the middle is empty, so every")
        print("    threshold in 0.1-0.9 behaves identically (see the flat table above).")
        print("    The threshold is not the interesting knob here. The RERANKER is.")
    else:
        print(f"\n  → {len(mid_band)} scores land in the ambiguous middle. THERE the score is")
        print("    uninformative, and the threshold is a real business choice.")

    print(f"\n  at threshold 0.5: {fa_at_50} false answers, {fr_at_50} false refusals")
    print("\n  HONEST CAVEAT: 10+10 hand-written queries is not an eval set. The")
    print("  unanswerable ones were invented by the same person who wrote the corpus,")
    print("  so they are easier than real users'. Draw both sets from production query")
    print("  logs before you quote any of these numbers to anyone.")
    print("=" * 78)


if __name__ == "__main__":
    main()
