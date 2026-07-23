"""Is the refusal gate's number a CONFIDENCE, or just a ranking score?

    python -m evals.reranker_ablation

WHY THIS EXPERIMENT EXISTS
--------------------------
A real third-party policy PDF was uploaded and asked "how many times can I
avail the policy in a year?" — a question that document answers in plain
text ("...up to 4 times in a year"). Retrieval was perfect: the right chunk
ranked #1. The gate refused it at 0.089.

The obvious hypothesis was "our reranker is too small — ms-marco-MiniLM is a
22M-parameter 2020 model". This script tests that hypothesis, and REFUTES
it: every reranker tried ranks the right chunk first and none produces a
calibrated absolute score. Swapping in a bigger model changes the numbers
and not the outcome.

WHAT IT ACTUALLY MEASURES — the distinction that matters
--------------------------------------------------------
  RANKING     within one query, does the right chunk beat the wrong ones?
              (top-1 accuracy on answerable cases)
  CALIBRATION across DIFFERENT queries, does one fixed threshold separate
              answerable from unanswerable? (best achievable error over
              every possible threshold — not our 0.5, the BEST one)

Cross-encoders are trained with a per-query objective: rank these candidates
for THIS query. Nothing in that objective makes score 0.4 for query A mean
the same thing as 0.4 for query B. Our refusal gate compares every query's
score against one global constant — which is a type error, and the reason
an unanswerable question can score 0.778 while an answerable one scores
0.089.

Downloads models on first use (ms-marco 80MB, jina-turbo 150MB,
bge-reranker-base 1.04GB). Manual experiment, like calibrate.py and
ann_bench.py — not part of the CI suite.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import argparse                         # stdlib — --models
import json                             # stdlib — cases.jsonl
import math                             # stdlib — the sigmoid gated.py applies
from pathlib import Path                # stdlib — locate the cases

from app.retrieval.corpus import ASHA_AGENT, load_corpus  # local — app/retrieval/corpus.py
from app.retrieval.gated import Principal, PreFilterRetriever  # local — gates

_DIR = Path(__file__).resolve().parent

# Small → large. The point of the list is that SIZE DOES NOT RESCUE THE
# PROPERTY — bge-reranker-base is 13× ms-marco and calibrates no better.
DEFAULT_MODELS = [
    "Xenova/ms-marco-MiniLM-L-6-v2",      # what the engine ships with
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
]

POOL = 20                               # candidates the gate reranks, as in production


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cases() -> list[dict]:
    path = _DIR / "cases.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _best_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float, int, int]:
    """The kindest possible threshold, found by exhaustive sweep.

    Reporting errors at OUR 0.5 would only show that 0.5 is a bad constant.
    Reporting them at the BEST achievable threshold shows something much
    stronger: that no constant works, because the two distributions
    interleave. If the minimum is zero, the score IS separable and the fix
    is simply to move the threshold.
    """
    candidates = sorted(set(answerable + unanswerable + [0.0, 1.0]))
    best = (0.5, len(answerable), len(unanswerable))
    best_errors = len(answerable) + len(unanswerable)
    for t in candidates:
        fr = sum(s < t for s in answerable)      # answerable, refused
        fa = sum(s >= t for s in unanswerable)   # unanswerable, answered
        if fr + fa < best_errors:
            best_errors, best = fr + fa, (t, fr, fa)
    return best


def run(models: list[str]) -> None:
    retriever = PreFilterRetriever.from_chunks(load_corpus())
    principal = Principal(*ASHA_AGENT)
    cases = _cases()

    # Retrieval is reranker-independent, so do it ONCE. Every model then
    # sees exactly the same candidates — the comparison isolates reranking.
    print(f"Retrieving candidates for {len(cases)} cases (once, shared)...\n")
    from datetime import date            # stdlib — dated cases carry as_of

    prepared = []
    for c in cases:
        as_of = date.fromisoformat(c["as_of"]) if "as_of" in c else None
        hits = retriever.search(c["question"], principal, k=POOL, as_of=as_of)
        prepared.append((c, [h.chunk for h in hits]))

    print(f"{'model':44} {'rank@1':>7} {'best thr':>9} {'errors at best':>15}")
    print("-" * 80)

    for name in models:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # 3rd-party: fastembed

        model = TextCrossEncoder(name)

        top1_correct = top1_total = 0
        ans_scores: list[float] = []
        una_scores: list[float] = []

        for case, chunks in prepared:
            if not chunks:
                continue
            raw = list(model.rerank(case["question"], [c.text_to_embed for c in chunks]))
            scored = sorted(zip(chunks, (_sigmoid(float(r)) for r in raw)), key=lambda t: -t[1])
            best_chunk, best_score = scored[0]

            if case["kind"] == "answerable":
                ans_scores.append(best_score)
                top1_total += 1
                exp = case["expected"]
                if exp in best_chunk.heading or exp in best_chunk.text:
                    top1_correct += 1
            else:
                una_scores.append(best_score)

        thr, fr, fa = _best_threshold(ans_scores, una_scores)
        rank = f"{top1_correct}/{top1_total}"
        print(f"{name:44} {rank:>7} {thr:>9.3f} {f'{fr} FR + {fa} FA':>15}")

    print()
    print("rank@1        answerable cases whose EXPECTED section the reranker put first")
    print("best thr      the kindest threshold that exists for this model, found by sweep")
    print("errors        false refusals + false answers REMAINING at that best threshold")
    print()
    print("Read the last column. If it is non-zero, no threshold separates answerable")
    print("from unanswerable for that model — the score orders candidates within a")
    print("query and means nothing across queries. That is a property of the training")
    print("objective, not of model size, which is why a 13x larger model does not fix")
    print("it. The fixes that DO work: transform the query so the phrasing gap closes")
    print("(evals/ablation.py measures it), fine-tune so scores mean something on this")
    print("domain, or replace the score gate with a groundedness verdict.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    run(p.parse_args().models)
