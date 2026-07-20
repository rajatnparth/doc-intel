"""Phase 8 — PII redaction: what you don't store can't breach.

The audit trail (phase 6) is a compliance asset with a liability inside it:
questions and answers are stored VERBATIM, and customers type phone numbers,
emails and vehicle registrations into chat boxes. The dispute handler needs
the EXCHANGE — what was asked, retrieved, scored, delivered — not the
identifiers. So identifiers are replaced with type-tagged placeholders
BEFORE the record is written: "call me on 98765 43210 about MH12AB1234"
is stored as "call me on [PHONE] about [VEHICLE-REG]". The narrative stays
readable; the raw value is simply never on disk. Identity, when a dispute
needs it, is already in the record as tenant_id — from the VERIFIED JWT,
not from whatever the customer typed.

Deliberately NOT reversible: a reversible redaction (tokenisation vault) is
an enterprise variant with its own key-management burden. The engine default
is the safe one. And deliberately deterministic regexes, not a model: the
redactor runs on every exchange, must never hallucinate, and must be
testable to exact strings. Azure AI Language's PII detection is the
enterprise connector behind this same Protocol (private side of the split).

Nothing here imports FastAPI. Pure functions over strings.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

import re                               # stdlib — the deterministic patterns
from typing import Protocol, runtime_checkable  # stdlib — the seam, as everywhere


@runtime_checkable
class Redactor(Protocol):
    def redact(self, text: str) -> str:
        """Replace identifiers with type-tagged placeholders. Idempotent:
        redacting already-redacted text changes nothing."""
        ...


# Ordered: EMAIL first (it contains digits and letters that other patterns
# could nibble at), then the vehicle registration, then phones. Each pattern
# is deliberately NARROW — the corpus's own reference numbers (policy
# MTR-2026-1147, claim CLM-2026-0891, damage code D-4471) must survive,
# because redacting the numbers the DISPUTE is about defeats the audit.
# The vehicle pattern anchors on exactly TWO letters (state code) followed
# by digits: three-letter prefixes like MTR-/CLM- can never match.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    # Indian vehicle registration: MH12AB1234 / MH-12-AB-1234 / mh 12 ab 1234
    (
        re.compile(r"\b[A-Za-z]{2}[\s-]?\d{1,2}[\s-]?[A-Za-z]{1,3}[\s-]?\d{3,4}\b"),
        "[VEHICLE-REG]",
    ),
    # Indian mobile: optional +91, then 10 digits, optionally split 5+5.
    (re.compile(r"(?:\+91[\s-]?)?\b(?:\d{5}[\s-]\d{5}|\d{10})\b"), "[PHONE]"),
]


class RegexRedactor:
    """The engine default. Conforms to Redactor by shape (Protocol)."""

    def redact(self, text: str) -> str:
        for pattern, placeholder in _PATTERNS:
            text = pattern.sub(placeholder, text)
        return text


class NullRedactor:
    """AUDIT_REDACT_PII=false — an explicit operator decision (e.g. a
    jurisdiction whose retention rules demand verbatim records). The default
    is redaction: fail safe, opt INTO storing identifiers."""

    def redact(self, text: str) -> str:
        return text


def build_redactor(redact_pii: bool) -> Redactor:
    return RegexRedactor() if redact_pii else NullRedactor()
