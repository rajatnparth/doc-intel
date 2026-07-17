"""Phase 4 — the numbers-vs-wording router.

A router is a CLASSIFIER, and this is the cheapest classifier that meets the
bar: keyword + question-shape matching. Deterministic, zero-token, and — the
property that matters in a regulated domain — EXPLAINABLE: "why did the bot
answer this from the record?" has a greppable answer, not a model's mood.

The production upgrade is LLM function-calling (the router IS tool selection —
this module is the agents lesson with the model removed). When that upgrade
comes, only the classifier body changes: `classify(question) -> FactField |
None` is the interface both versions satisfy, exactly like swapping the stub
LLM for Azure behind LLMClient.

THE HARD CASE, WHICH IS THE DESIGN
----------------------------------
    "what is my excess?"                        -> FACTS   (a value question)
    "does the excess apply to windscreen claims?" -> WORDING (a terms question)

Same noun, different question. A noun match alone would route the second one
to the record and answer a question nobody asked. So an intent requires BOTH:
a fact noun AND a value-asking shape ("what is", "how much", "when is"...).
False negatives fall through to RAG, which refuses when it should — the safe
direction. False positives would put the WRONG SUBSYSTEM's authority behind
an answer, so the patterns stay tight on purpose.

Nothing here imports FastAPI, openai, or retrieval. Pure text in, enum out.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from enum import Enum                   # stdlib — the closed set of routable facts


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
