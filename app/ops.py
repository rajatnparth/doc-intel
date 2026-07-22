"""Phase 9 — the operability surfaces: metrics and dependency health.

Three audiences talk to a production service, and only one of them is a
user. The orchestrator asks /health ("alive?" -> restart on no) and /ready
("serve traffic?" -> rotate out on no). The metrics scraper reads /metrics
every ~15s and turns counters into time series. This module is the second
and third contract; the routes live in main.py.

THE CARDINALITY RULE
--------------------
Every distinct label value is a separate time series, stored forever by the
scraper. Outcomes are a closed set of five — fine. tenant_id would multiply
every metric by the customer count AND copy identifying data into a third
storage system, undoing phase 8's minimization in a new place. So: no
tenant labels, ever. Aggregate product health, not per-customer surveillance.

AUDIT HEALTH (the phase-6 deferral, closed)
-------------------------------------------
"No record, no answer" cannot be checked by writing before answering — the
record's defining field is what was DELIVERED, which doesn't exist until the
stream ends. So health is a FLAG: flipped false by a real failed write,
flipped true by the next success. Admission control reads the flag for free;
/ready exposes it; AUDIT_STRICT decides whether it blocks. The flag makes
the check cost nothing; strictness stays a business decision in config.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import threading                        # stdlib — the flag is written from request tasks

from prometheus_client import (          # 3rd-party: prometheus-client — the boring
                                        #   industry standard; maintains counters and
                                        #   renders the /metrics text format
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# A dedicated registry, not the library's process-global default: what this
# app exports is exactly what this module declares — no accidental
# platform/python metrics, and tests can read it deterministically.
REGISTRY = CollectorRegistry()

# RED — rate by outcome (the refusal rate is THE product-health signal in a
# refusal-first system: a spike means a corpus gap, a degraded reranker, or
# someone probing), errors are an outcome not a separate counter, duration
# as a histogram so p50/p99 are queryable.
ASK_REQUESTS = Counter(
    "ask_requests_total",
    "Completed /v1/ask exchanges by outcome",
    labelnames=["outcome"],
    registry=REGISTRY,
)
ASK_DURATION = Histogram(
    "ask_duration_seconds",
    "End-to-end /v1/ask duration, including streaming",
    registry=REGISTRY,
)
# Cost as a graph instead of a monthly surprise.
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Tokens billed by the provider",
    labelnames=["kind"],                # prompt | completion
    registry=REGISTRY,
)
AUDIT_WRITE_FAILURES = Counter(
    "audit_write_failures_total",
    "Audit records that could not be written — each one is a compliance gap",
    registry=REGISTRY,
)
HANDOFF_TICKETS = Counter(
    "handoff_tickets_total",
    "Human-handoff tickets created",
    registry=REGISTRY,
)
DOCUMENTS_INGESTED = Counter(
    "documents_ingested_total",
    "Documents accepted through /v1/documents",
    registry=REGISTRY,
)


def observe_ask(outcome: str, duration_ms: int, prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """One call per exchange, from the same finally that writes the audit
    record — the metric and the record must agree about what happened."""
    ASK_REQUESTS.labels(outcome=outcome).inc()
    ASK_DURATION.observe(duration_ms / 1000.0)
    if prompt_tokens:
        LLM_TOKENS.labels(kind="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(kind="completion").inc(completion_tokens)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


class AuditHealth:
    """The sink's health, as observed from actual writes.

    Not a probe that touches the disk — the last real write already told us.
    Thread-safe because stream tasks finish concurrently; a bool assignment
    is atomic in CPython, but the lock documents the intent and survives a
    future move to counters-with-thresholds.
    """

    def __init__(self) -> None:
        self._ok = True
        self._lock = threading.Lock()

    @property
    def ok(self) -> bool:
        return self._ok

    def mark_ok(self) -> None:
        with self._lock:
            self._ok = True

    def mark_failed(self) -> None:
        with self._lock:
            self._ok = False
