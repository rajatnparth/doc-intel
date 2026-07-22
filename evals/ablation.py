"""What does each retrieval stage actually BUY? Measured, per configuration.

    python -m evals.ablation

A feature announcement is not evidence. This runs the labelled set through
four configurations and reports the three metrics plus the cost, so the
decision to ship query transformation — or not — rests on numbers:

    baseline    the user's question alone (pre-phase-11 behaviour)
    +rewrites   the question plus paraphrases in policy vocabulary
    +hyde       the question plus a fabricated one-sentence ANSWER
    all         everything

WHERE THE REWRITES COME FROM (read this before trusting the table)
-------------------------------------------------------------------
`evals/rewrites.jsonl` is FROZEN. In production the rewriter is a model
call; freezing its output makes this measurement reproducible and keyless,
which is the same reason the eval suite runs on a stub provider.

The frozen file was hand-written to stand in for that model, which is a real
threat to the result: an author who writes the rewrites AND reads the scores
can tune one against the other until the feature looks good. Two guards were
applied, and they are stated so a reviewer can judge them rather than take
them on faith:

  1. every rewrite was written BEFORE any measurement was run
  2. answerable and unanswerable cases were treated IDENTICALLY — the
     rewriter in production cannot know which is which, so neither did the
     author. If transformation helps answerable cases by also dragging
     unanswerable ones over the threshold, this table shows it as false
     answers rather than hiding it.

With a real provider configured, regenerate the file rather than trusting
the hand-written one:  python -m evals.ablation --regenerate
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import argparse                         # stdlib — --regenerate
import asyncio                          # stdlib — the rewriter is async
import json                             # stdlib — the jsonl artifacts
import time                             # stdlib — the cost column
from datetime import date               # stdlib — dated cases
from pathlib import Path                # stdlib — locate the artifacts

from app.retrieval.corpus import ASHA_AGENT, load_corpus  # local — app/retrieval/corpus.py
from app.retrieval.gated import (       # local — app/retrieval/gated.py
    REFUSAL_THRESHOLD,
    Principal,
    PreFilterRetriever,
    rerank_many,
)

_DIR = Path(__file__).resolve().parent
TOP_K = 5
POOL = 20


def _load(name: str) -> list[dict]:
    path = _DIR / name
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _queries_for(case: dict, rw: dict, config: str) -> list[str]:
    q = [case["question"]]                      # the original is never dropped
    if config in ("+rewrites", "all"):
        q += rw.get("rewrites", [])
    if config in ("+hyde", "all"):
        hypo = rw.get("hypothetical_answer", "")
        if hypo:
            q.append(hypo)
    return q


def _score_config(config: str, cases: list[dict], rewrites: dict, retriever, principal) -> dict:
    hits = false_refusals = false_answers = 0
    n_answerable = n_unanswerable = 0
    started = time.monotonic()
    rerank_pairs = 0

    for case in cases:
        rw = rewrites.get(case["id"], {})
        queries = _queries_for(case, rw, config)
        as_of = date.fromisoformat(case["as_of"]) if "as_of" in case else None

        candidates = retriever.search_many(queries, principal, k=POOL, as_of=as_of)
        chunks = [h.chunk for h in candidates]
        if not chunks:
            continue
        ranked = rerank_many(queries, chunks)
        rerank_pairs += len(queries) * len(chunks)
        best_score = ranked[0][1]
        refused = best_score < REFUSAL_THRESHOLD

        if case["kind"] == "unanswerable":
            n_unanswerable += 1
            if not refused:
                false_answers += 1
            continue

        n_answerable += 1
        if refused:
            false_refusals += 1
            continue
        expected = case["expected"]
        if any(expected in c.heading or expected in c.text for c, _ in ranked[:TOP_K]):
            hits += 1

    return {
        "config": config,
        "hit_at_5": hits,
        "n_answerable": n_answerable,
        "false_refusals": false_refusals,
        "false_answers": false_answers,
        "n_unanswerable": n_unanswerable,
        "seconds": time.monotonic() - started,
        "rerank_pairs": rerank_pairs,
    }


async def _regenerate() -> None:
    """Rebuild rewrites.jsonl from the CONFIGURED provider (needs a real one:
    the stub cannot produce a valid verdict, by design)."""
    from app.config import get_settings          # local — app/config.py
    from app.llm.factory import build_llm_client  # local — app/llm/factory.py
    from app.retrieval.rewrite import transform   # local — app/retrieval/rewrite.py

    settings = get_settings()
    llm = build_llm_client(settings)
    out = []
    try:
        for case in _load("cases.jsonl"):
            t = await transform(case["question"], llm, max_variants=2)
            if t.degraded:
                raise SystemExit(
                    "the configured provider produced no usable rewrites "
                    f"(LLM_PROVIDER={settings.llm_provider}). Set a real provider."
                )
            out.append({
                "id": case["id"],
                "rewrites": t.variants,
                "hypothetical_answer": t.hypothetical,
            })
    finally:
        await llm.aclose()

    (_DIR / "rewrites.jsonl").write_text(
        "\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8"
    )
    print(f"regenerated {len(out)} rewrites from the live provider")


def run() -> None:
    cases = _load("cases.jsonl")
    rewrites = {r["id"]: r for r in _load("rewrites.jsonl")}
    retriever = PreFilterRetriever.from_chunks(load_corpus())
    principal = Principal(*ASHA_AGENT)

    print("Frozen rewrites from evals/rewrites.jsonl — see this module's docstring")
    print("for how they were produced and why that matters.\n")
    print(f"{'config':12} {'hit@5':>7} {'false ref':>10} {'false ans':>10} {'pairs':>7} {'secs':>7}")
    print("-" * 60)

    rows = []
    for config in ("baseline", "+rewrites", "+hyde", "all"):
        r = _score_config(config, cases, rewrites, retriever, principal)
        rows.append(r)
        print(
            f"{r['config']:12} {r['hit_at_5']:>3}/{r['n_answerable']:<3} "
            f"{r['false_refusals']:>10} {r['false_answers']:>10} "
            f"{r['rerank_pairs']:>7} {r['seconds']:>7.1f}"
        )

    base = rows[0]
    # Tie-break on COST. Configurations that score identically are not
    # equally good: one of them bills 4x the cross-encoder work on every
    # question forever. Ranking by quality alone recommended the expensive
    # option over an identical cheap one — a bug in the analysis, which is
    # its own small lesson about reading ablation tables.
    best = min(
        rows,
        key=lambda r: (r["false_refusals"] + r["false_answers"], -r["hit_at_5"], r["rerank_pairs"]),
    )
    print()
    print("pairs   cross-encoder (query, chunk) evaluations — the latency and CPU bill")
    print("        for transformation, which is linear in the number of phrasings")
    print()
    if best["config"] == "baseline":
        print("VERDICT: transformation did not beat the baseline on this set. Ship the")
        print("         configuration that wins, not the one that was interesting to build.")
    else:
        print(f"VERDICT: '{best['config']}' wins — "
              f"false refusals {base['false_refusals']} -> {best['false_refusals']}, "
              f"false answers {base['false_answers']} -> {best['false_answers']}, "
              f"at {best['rerank_pairs'] / max(base['rerank_pairs'], 1):.1f}x the rerank cost.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regenerate", action="store_true",
                   help="rebuild rewrites.jsonl from the configured provider")
    args = p.parse_args()
    if args.regenerate:
        asyncio.run(_regenerate())
    else:
        run()
