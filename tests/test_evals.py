"""Phase 7 — the eval gate. Quality regressions fail CI like broken code.

This file is thin on purpose: the measurement lives in evals/run.py, the
labelled set in evals/cases.jsonl, the measured floor in evals/baseline.json.
Here we only assert the RATCHET — no metric may be worse than the baseline.

The baseline encodes reality, warts included: 2 false refusals (calibrate.py
proved they are reranker-calibration failures no threshold fixes) and 1
false answer in the wording pipeline (0.778 on "next year's premium" — the
ROUTER closes it in /v1/ask; this eval deliberately measures the layer
beneath). A gate demanding zero would fail on day one and be ignored by
day three. When a better reranker lands, these numbers improve, the run
prints IMPROVED, and baseline.json gets TIGHTENED in that same PR.
"""

import json                             # stdlib — read the measured floor
from pathlib import Path                # stdlib — locate baseline.json

import pytest                           # 3rd-party: pytest — the module-scoped fixture

from evals.run import compare, run      # local — evals/run.py (the measurement)

BASELINE = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    """One measurement, shared by every assertion — the eval is the slow
    part (embeds the corpus, cross-encodes every case), the checks are free."""
    return run()


def test_baseline_is_committed() -> None:
    """The floor must be in the repo: an eval gated against a file CI
    regenerates on the fly gates nothing."""
    assert BASELINE.exists(), "run `python -m evals.run` once and commit baseline.json"


def test_no_metric_is_worse_than_the_measured_baseline(metrics) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    problems = compare(metrics, baseline)
    assert not problems, f"quality regression vs evals/baseline.json: {problems}"


def test_the_eval_set_did_not_silently_shrink(metrics) -> None:
    """Deleting hard cases is the oldest way to 'fix' an eval. The counts are
    part of the baseline so removing a case is a visible, reviewable act."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert metrics["n_answerable"] >= baseline["n_answerable"]
    assert metrics["n_unanswerable"] >= baseline["n_unanswerable"]
