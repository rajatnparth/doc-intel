"""Phase 7 — evals: measured quality, gated in CI.

Tests (tests/) assert code contracts; evals measure SYSTEM quality on a
labelled set. The 117 tests stayed green through a major embedder upgrade —
which is exactly the blind spot: every contract can hold while the ranking
quietly gets worse. cases.jsonl is the labelled set, run.py the scorecard,
baseline.json the measured floor, tests/test_evals.py the CI ratchet.
"""
