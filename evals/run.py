"""The eval runner: labelled cases -> scorecard -> compared against baseline.

    python -m evals.run

WHAT IS MEASURED (and what deliberately is not)
-----------------------------------------------
Each case runs the real wording pipeline — gates -> hybrid retrieval -> RRF
-> cross-encoder rerank -> refusal decision (gated.answer). No generation:
the stub LLM would make "answer quality" a measurement of canned text.
Generation/citation evals need a real model as judge — that is the noted
extension, not smuggled in here. The numbers-vs-wording router is upstream
of this pipeline and has its own deterministic tests.

THE THREE NUMBERS
-----------------
  hit@5           answerable questions where the expected label appears in
                  the top-5 chunks handed to generation
  false_refusals  answerable questions the gate refused — the ANNOYING
                  failure (a customer sent to a human unnecessarily)
  false_answers   unanswerable questions that got past the gate — the
                  EXPENSIVE failure (context for a fabricated fact)

Tracked separately because they have different costs and different fixes;
one blended "accuracy" is how a false answer hides behind ten easy hits.

THE BASELINE IS MEASURED, NOT ASPIRED TO
----------------------------------------
baseline.json holds what the pipeline ACTUALLY does today, known warts
included (calibrate.py: an out-of-domain reranker causes false refusals no
threshold can fix). The gate in tests/test_evals.py is a ratchet against
WORSE. When a better reranker lands and the warts vanish, the run prints
IMPROVED, and the baseline gets tightened in the same PR — never loosened
without a written reason.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import json                             # stdlib — cases.jsonl + baseline.json
import sys                              # stdlib — exit code for CI/manual use
from dataclasses import dataclass       # stdlib — one row of the scorecard
from datetime import date               # stdlib — dated cases (as_of)
from pathlib import Path                # stdlib — locate the eval artifacts

from app.retrieval.corpus import ASHA_AGENT, load_corpus  # local — app/retrieval/corpus.py
from app.retrieval.gated import Principal, PreFilterRetriever, answer  # local — gates+refusal

_DIR = Path(__file__).resolve().parent
CASES = _DIR / "cases.jsonl"
BASELINE = _DIR / "baseline.json"

TOP_K = 5


@dataclass(frozen=True)
class Verdict:
    case_id: str
    kind: str
    ok: bool
    detail: str


def _load_cases() -> list[dict]:
    return [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run_case(case: dict, retriever: PreFilterRetriever, principal: Principal) -> Verdict:
    as_of = date.fromisoformat(case["as_of"]) if "as_of" in case else None
    a = answer(case["question"], principal, retriever, as_of=as_of, top_k=TOP_K)

    if case["kind"] == "unanswerable":
        # The only correct outcome is a refusal. An answer here means the
        # gate handed confident-looking context downstream for a question
        # the corpus cannot answer.
        if a.refused:
            return Verdict(case["id"], "unanswerable", True, f"refused at {a.score:.3f}")
        return Verdict(case["id"], "unanswerable", False, f"FALSE ANSWER at {a.score:.3f}")

    # answerable
    if a.refused:
        return Verdict(case["id"], "answerable", False, f"FALSE REFUSAL at {a.score:.3f}")
    expected = case["expected"]
    hit = any(expected in c.heading or expected in c.text for c in a.chunks)
    detail = f"hit@{TOP_K}" if hit else f"answered but '{expected}' not in top-{TOP_K}"
    return Verdict(case["id"], "answerable", hit, detail)


def run() -> dict:
    """Run every case; return the metrics dict (the scorecard)."""
    retriever = PreFilterRetriever.from_chunks(load_corpus())
    principal = Principal(*ASHA_AGENT)   # the widest legitimate view — the
                                         # claims-file cases need `agent`

    verdicts = [_run_case(c, retriever, principal) for c in _load_cases()]

    answerable = [v for v in verdicts if v.kind == "answerable"]
    unanswerable = [v for v in verdicts if v.kind == "unanswerable"]
    false_refusals = sum("FALSE REFUSAL" in v.detail for v in answerable)
    false_answers = sum(not v.ok for v in unanswerable)
    hits = sum(v.ok for v in answerable)

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "hit_at_5": hits,
        "false_refusals": false_refusals,
        "false_answers": false_answers,
        "verdicts": [v.__dict__ for v in verdicts],
    }


def compare(metrics: dict, baseline: dict) -> list[str]:
    """Regressions vs the measured floor. Direction matters per metric."""
    problems = []
    if metrics["hit_at_5"] < baseline["hit_at_5"]:
        problems.append(f"hit@5 fell: {baseline['hit_at_5']} -> {metrics['hit_at_5']}")
    if metrics["false_refusals"] > baseline["false_refusals"]:
        problems.append(
            f"false refusals grew: {baseline['false_refusals']} -> {metrics['false_refusals']}"
        )
    if metrics["false_answers"] > baseline["false_answers"]:
        problems.append(
            f"false answers grew: {baseline['false_answers']} -> {metrics['false_answers']}"
        )
    return problems


def main() -> None:
    print("Running the wording-pipeline evals (real embedder + reranker)…\n")
    m = run()

    for v in m["verdicts"]:
        mark = "✓" if v["ok"] else "✗"
        print(f"  {mark} {v['case_id']:<8} {v['detail']}")

    print(
        f"\n  hit@5           {m['hit_at_5']}/{m['n_answerable']}"
        f"\n  false refusals  {m['false_refusals']}"
        f"\n  false answers   {m['false_answers']}"
    )

    if not BASELINE.exists():
        print("\nNo baseline.json — writing one from this run. Commit it: the")
        print("numbers are the measured floor the CI gate ratchets against.")
        BASELINE.write_text(
            json.dumps({k: m[k] for k in
                        ("n_answerable", "n_unanswerable", "hit_at_5",
                         "false_refusals", "false_answers")}, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = compare(m, baseline)
    if problems:
        print("\nREGRESSION vs baseline:")
        for p in problems:
            print(f"  ✗ {p}")
        sys.exit(1)

    better = (
        m["hit_at_5"] > baseline["hit_at_5"]
        or m["false_refusals"] < baseline["false_refusals"]
        or m["false_answers"] < baseline["false_answers"]
    )
    print("\nOK vs baseline" + (" — IMPROVED: tighten baseline.json in this PR." if better else "."))


if __name__ == "__main__":
    main()
