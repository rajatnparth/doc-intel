"""Phase 4 (+ the tier-2 upgrade) — the numbers-vs-wording router.

A router is a CLASSIFIER, and this module now holds two of them, TIERED:

  tier 1  deterministic keyword + question-shape match. Free, instant, and —
          the property that matters in a regulated domain — EXPLAINABLE:
          "why did the bot answer this from the record?" has a greppable
          answer, not a model's mood.
  tier 2  LLM intent classification, for the phrasings tier 1 cannot see:
          "how much do I pay from my own pocket when I claim?" names no fact
          noun, but it IS an excess question — and letting it fall to RAG
          means the model quotes the excess FROM PROSE, which is right today
          and stale the day an endorsement changes the record mid-term. The
          paraphrase gap is a hole in numbers-never-from-RAG itself, not a
          UX nit. (This corrected the phase-4 claim that fall-through always
          fails safe: refusal is safe; prose numbers are not.)

TIER 2 IS GATE 2 ALL OVER AGAIN
-------------------------------
The model's routing verdict is UNTRUSTED INPUT: it arrives through the same
extract() seam as invoice JSON and is validated by the same discipline —
strict Pydantic against a CLOSED decision space. That closure is also the
injection containment: a hostile question can at worst flip which subsystem
answers; it cannot name a tenant, widen an ACL, or invent a fact field.
Every failure (LLMError, junk JSON, "wording") falls to RAG, where the
refusal gate stands — the router may ADD confidence, never remove a layer.

Costs, stated: tier 2 spends one small LLM call on tier-1 misses (production
points it at a cheap model — a router does not need the flagship). Tier-1
false positives are NOT rescued by tier 2, because tier 1 short-circuits —
that is the price of cost-ordering, and the known wart stays documented in
the tests. And the fakes in test_router.py pin OUR WIRING, not the model's
judgment: router QUALITY needs a labelled eval set (phase 7).

THE HARD CASE, WHICH IS THE DESIGN
----------------------------------
    "what is my excess?"                        -> FACTS   (a value question)
    "does the excess apply to windscreen claims?" -> WORDING (a terms question)

Same noun, different question. Tier-1 intent requires BOTH a fact noun AND a
value-asking shape; tier 2 is told the same rule in its instruction.

Nothing here imports FastAPI or openai — tier 2 speaks through LLMClient,
the same seam everything else uses. Text in, enum out.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from enum import Enum                   # stdlib — the closed set of routable facts
from typing import Literal              # stdlib — RouteDecision's closed value spaces

from pydantic import BaseModel, ConfigDict, ValidationError  # 3rd-party: pydantic —
                                        #   Gate-2 validation of the model's verdict

from app.llm.base import LLMClient, LLMError  # local — app/llm/base.py (the seam)


class FactField(str, Enum):
    OWN_DAMAGE_EXCESS = "own_damage_excess"
    ANNUAL_PREMIUM = "annual_premium"
    IDV = "idv"
    NCB = "ncb_percent"
    POLICY_NUMBER = "policy_number"
    RENEWAL_DATE = "renewal_date"


# A value-asking shape. Word stems, matched against the lowercased question.
_VALUE_SHAPES = (
    "what is",
    "what's",
    "what will",
    "how much",
    "how big",
    "when is",
    "when does",
    "when do i renew",
    "tell me my",
)

# Fact nouns -> field. Ordered: first match wins, so the more specific phrases
# sit above the generic ones ("renewal premium" must beat "renewal").
_FIELD_NOUNS: list[tuple[str, FactField]] = [
    ("excess", FactField.OWN_DAMAGE_EXCESS),
    ("renewal premium", FactField.ANNUAL_PREMIUM),
    ("premium", FactField.ANNUAL_PREMIUM),
    ("idv", FactField.IDV),
    ("declared value", FactField.IDV),
    ("no-claim", FactField.NCB),
    ("no claim", FactField.NCB),
    ("ncb", FactField.NCB),
    ("policy number", FactField.POLICY_NUMBER),
    ("renewal date", FactField.RENEWAL_DATE),
    ("renew", FactField.RENEWAL_DATE),
]


# Enum values double as PolicyRecord attribute names, so rendering is a
# getattr away; these are the human labels the client shows.
FIELD_LABELS: dict[FactField, str] = {
    FactField.OWN_DAMAGE_EXCESS: "Own damage excess",
    FactField.ANNUAL_PREMIUM: "Annual premium",
    FactField.IDV: "Insured's Declared Value",
    FactField.NCB: "No-claim bonus",
    FactField.POLICY_NUMBER: "Policy number",
    FactField.RENEWAL_DATE: "Renewal date",
}


def classify(question: str) -> FactField | None:
    """FACTS if the question ASKS FOR A VALUE of a known account field.

    Anything else — including wording questions that merely MENTION a fact
    noun — returns None and flows to retrieval. The gate there refuses what it
    cannot ground, so the fall-through direction fails safe.
    """
    q = question.lower()
    if not any(shape in q for shape in _VALUE_SHAPES):
        return None
    for noun, field in _FIELD_NOUNS:
        if noun in q:
            return field
    return None


# =============================================================================
# Tier 2 — the LLM classifier, behind Gate-2 discipline.
# =============================================================================
class RouteDecision(BaseModel):
    """The model's verdict, as a CLOSED contract.

    strict + extra="forbid" + Literal everywhere: the decision space is an
    enum, not a text field. This is what bounds both hallucination and
    injection — there is no string the model can emit that widens what the
    facts path can do."""

    model_config = ConfigDict(strict=True, extra="forbid")

    route: Literal["facts", "wording"]
    field: (
        Literal[
            "own_damage_excess",
            "annual_premium",
            "idv",
            "ncb_percent",
            "policy_number",
            "renewal_date",
        ]
        | None
    ) = None


_ROUTER_INSTRUCTION = """Classify one customer question for a motor insurance assistant.

Decide between:
  "facts"   — the question asks for the CURRENT VALUE of one account field:
              own_damage_excess, annual_premium, idv, ncb_percent,
              policy_number, renewal_date.
  "wording" — anything else: cover terms, processes, conditions, claims —
              including questions that merely MENTION a fact field
              ("does the excess apply to windscreen claims?" is wording).

Return JSON only: {"route": "facts", "field": "<field>"} or {"route": "wording", "field": null}.

Question: """


def _parse_decision(raw: str) -> FactField | None:
    """Gate 2 for the router. Anything that is not a clean, complete 'facts'
    verdict routes to wording — the direction with a refusal gate behind it."""
    try:
        decision = RouteDecision.model_validate_json(raw)
    except ValidationError:
        return None
    if decision.route != "facts" or decision.field is None:
        return None
    return FactField(decision.field)


async def route(question: str, llm: LLMClient) -> FactField | None:
    """Tier 1, then tier 2, then RAG. The tiers may only ADD a facts route.

    With the stub provider, tier 2 is INERT by construction: the stub's canned
    extract() output fails RouteDecision validation and falls to wording — the
    keyless quickstart behaves exactly like tier-1-only, and a real provider
    lights tier 2 up without a code change."""
    field = classify(question)
    if field is not None:
        return field                     # free, explainable, short-circuits

    try:
        raw = await llm.extract(
            _ROUTER_INSTRUCTION + question,
            RouteDecision.model_json_schema(),
            max_tokens=60,               # a verdict, not an essay — quota honesty
        )
    except LLMError:
        return None                     # the router never breaks the ask; RAG absorbs
    return _parse_decision(raw)
