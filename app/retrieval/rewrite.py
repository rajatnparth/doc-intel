"""Phase 11 — query transformation: close the phrasing gap before retrieval.

THE PROBLEM, MEASURED
---------------------
A real third-party policy PDF says "...up to 4 times in a year". Asked "how
many times can I avail the policy in a year?" the pipeline REFUSED: retrieval
ranked the right chunk #1, and the cross-encoder scored it 0.0886. Asked in
the document's own vocabulary — "how many times is complimentary road side
assistance available in a year?" — the SAME chunk scored 0.9998.

An 11,000x swing on wording alone. Users do not speak in their documents'
vocabulary, so the system must close that gap itself.

WHY NOT JUST A BETTER RERANKER
-------------------------------
Tested and largely refuted — `python -m evals.reranker_ablation`. Three
rerankers spanning 80MB to 1.04GB all rank correctly and none calibrates;
the best achievable threshold still leaves errors, and those thresholds
scatter across 0.14-0.77. The score is a per-query RANKING signal. Rather
than keep hunting for a model whose scores happen to be comparable across
queries, we make the query match the document's language.

THE THREE TRANSFORMS
--------------------
  original  always kept — a rewrite can drift; the user's words cannot
  rewrites  N paraphrases in the corpus's likely vocabulary
  hypo      HyDE: a fabricated one-sentence ANSWER. Embedding an answer to
            match answers beats embedding a question to match answers,
            because the vector lands in answer-space. The fabrication is
            never shown to anyone and never enters the prompt — it is a
            retrieval probe, and the gate still decides on real chunks.

FAIL-SAFE BY CONSTRUCTION
-------------------------
Transformation adds a model call BEFORE retrieval — new latency, new cost,
new failure mode on the critical path. So: bounded variants, a small token
cap, every exception swallowed, and any failure degrades to [original].
The worst case is exactly today's behaviour, which is what makes this safe
to switch on by default.

Nothing here imports FastAPI. The LLM arrives through the seam.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import logging                          # stdlib — one warning when a rewrite fails
from dataclasses import dataclass, field  # stdlib — the result type
from typing import Literal              # stdlib — the closed shape below

from pydantic import BaseModel, ConfigDict, Field, ValidationError  # 3rd-party: pydantic —
                                        #   Gate 2 for the rewriter's output

from app.llm.base import LLMClient, LLMError  # local — app/llm/base.py (the seam)

log = logging.getLogger("doc_intel")

MAX_VARIANTS = 4                        # a hard ceiling independent of config:
                                        #   every variant is a full retrieval pass


class _Rewrites(BaseModel):
    """Gate 2 for the rewriter, in the same discipline as RouteDecision.

    The model is asked for data, not instructions, and the shape is closed:
    a bounded list of bounded strings and one hypothetical answer. There is
    no field here through which a hostile document — or a hostile question —
    could widen what the pipeline does. The worst a bad verdict achieves is
    a wasted retrieval pass over the SAME gated corpus.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    rewrites: list[str] = Field(default_factory=list, max_length=MAX_VARIANTS)
    hypothetical_answer: str = Field("", max_length=400)


_INSTRUCTION = """Rewrite one customer question so it matches the wording of an
insurance policy document, which uses formal contractual language.

Return JSON only:
{"rewrites": ["...", "..."], "hypothetical_answer": "..."}

rewrites             up to %(n)d alternative phrasings. Keep the MEANING exactly;
                     vary the vocabulary toward policy-document language
                     (e.g. "avail" -> "available", "how many times" ->
                     "usage limit", "get money back" -> "reimbursement").
hypothetical_answer  one sentence that would ANSWER the question if it appeared
                     in the policy. Invent plausible specifics; it is used only
                     to search, never shown to anyone.

Question: """


@dataclass(frozen=True)
class TransformedQuery:
    """What retrieval will actually run. `original` is always variants[0]."""

    original: str
    variants: list[str] = field(default_factory=list)
    hypothetical: str = ""
    # Cost is reported, not hidden: transformation moves tokens onto the
    # critical path of every wording question, and "what does one request
    # cost you?" is the follow-up question every time (Usage, section 1.1).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degraded: bool = False              # True when the rewriter failed and we
                                        #   fell back to the original alone

    @property
    def all_queries(self) -> list[str]:
        """Everything retrieval should search for, original first."""
        out = [self.original, *self.variants]
        if self.hypothetical:
            out.append(self.hypothetical)
        return out


def _parse(raw: str, *, max_variants: int) -> tuple[list[str], str]:
    """Gate 2. A malformed verdict yields NO variants — never an exception,
    never a partially-trusted parse."""
    try:
        parsed = _Rewrites.model_validate_json(raw)
    except ValidationError:
        return [], ""
    # Deduplicate and drop empties: a rewriter that echoes the question back
    # would otherwise buy a duplicate retrieval pass for nothing.
    seen: set[str] = set()
    rewrites: list[str] = []
    for r in parsed.rewrites:
        r = r.strip()
        if r and r.lower() not in seen:
            seen.add(r.lower())
            rewrites.append(r)
    return rewrites[:max_variants], parsed.hypothetical_answer.strip()


async def transform(
    question: str,
    llm: LLMClient,
    *,
    enabled: bool = True,
    max_variants: int = 3,
) -> TransformedQuery:
    """Question -> the set of queries retrieval will run. Never raises.

    Returns `degraded=True` when the rewriter was unavailable or unusable,
    so the caller can record it: a silent degradation is a quality incident
    nobody notices — /metrics counts these, and the audit record shows which
    exchanges ran on the original query alone.
    """
    if not enabled or not question.strip():
        return TransformedQuery(original=question, degraded=not enabled)

    max_variants = max(0, min(max_variants, MAX_VARIANTS))

    try:
        raw = await llm.extract(
            (_INSTRUCTION % {"n": max_variants}) + question,
            _Rewrites.model_json_schema(),
            # Rewrites are short by construction. A generous cap here would
            # be a quota decision made by accident (Azure reserves
            # prompt + max_tokens at admission — AZURE_SETUP.md).
            max_tokens=200,
        )
    except LLMError as exc:
        # The rewriter is an ENHANCEMENT. If the provider is rate-limited or
        # down, the question still gets answered the old way rather than not
        # at all — degradation, not failure.
        log.warning("query rewrite unavailable (%s); using the original query", exc.code)
        return TransformedQuery(original=question, degraded=True)

    rewrites, hypo = _parse(raw, max_variants=max_variants)
    if not rewrites and not hypo:
        # The stub provider lands here by construction, so the keyless
        # quickstart behaves exactly as it did before this phase existed.
        return TransformedQuery(original=question, degraded=True)

    return TransformedQuery(original=question, variants=rewrites, hypothetical=hypo)
