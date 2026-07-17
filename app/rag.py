"""Phase 1 — where retrieval finally meets generation.

This module owns exactly one decision: WHAT THE MODEL GETS TO READ. Everything
upstream (gates, hybrid retrieval, rerank, refusal) decided whether we answer;
everything downstream (SSE, the route) decides how the answer travels. Between
them sits the prompt — and the prompt is a BUDGET, not a template.

WHY PARENTS, AND WHY DEDUPED
----------------------------
We index small chunks (precise to search) but generate from their PARENTS —
the whole section (search small, generate large; section 3.1). Two retrieved
chunks often share one parent, and pasting the same section twice doesn't make
the model twice as informed — it spends the budget twice and teaches the model
that repetition is signal. So sources are deduped BY PARENT, in reranked order:
source [1] is the parent of the best-scored chunk.

WHY A CHARACTER BUDGET
----------------------
"Stuff the context window" fails three ways: cost (you pay per token, per
request, forever), latency (prefill scales with input), and attention (the
model reads the middle of a long context worst — 'lost in the middle').
The budget is in CHARS with the ~4 chars/token heuristic, because this module
must not import a tokenizer: counting tokens exactly is the provider's business
and would drag a provider dependency across the seam. A budget that is ±10%
wrong but dependency-free beats an exact one that couples this file to a model.

Nothing here imports FastAPI, openai, or fastembed. Pure functions over Chunks.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from dataclasses import dataclass       # stdlib — Source is plain data

from app.ingest import Chunk            # local — app/ingest/ (the retrieval unit)


@dataclass(frozen=True)
class Source:
    """One deduped parent, as it will be cited: [n]."""

    n: int
    doc_title: str
    heading: str
    text: str                           # the parent text the model actually reads


# The instruction block. Three rules, each earned:
#   - "only from the extracts": the model's own knowledge about insurance is a
#     liability here — plausible generic answers about SOME policy are exactly
#     the failure mode a policy-specific product exists to prevent.
#   - "cite [n]": an uncited claim is unverifiable; the client renders [n]
#     against the sources frame it already holds.
#   - "say so": the reranker gate refuses BEFORE generation, but the extracts
#     can still be on-topic without containing the asked fact (topicality is
#     not answerability — measured in calibrate.py). This is the second net.
_INSTRUCTIONS = (
    "You answer questions about policy documents.\n"
    "Use ONLY the numbered extracts below. After each claim, cite its extract "
    "like [1] or [2].\n"
    "If the extracts do not contain the answer, say exactly that — do not use "
    "outside knowledge, do not guess.\n"
)


def select_sources(chunks: list[Chunk], *, budget_chars: int) -> list[Source]:
    """Dedupe chunks to parents, keep reranked order, stop at the budget.

    Always admits the FIRST parent even if it alone exceeds the budget: an
    empty context is not a smaller context, it is a different (broken) request.
    The gate already decided this question deserves an answer.
    """
    sources: list[Source] = []
    seen: set[tuple[str, str]] = set()
    spent = 0

    for c in chunks:
        parent = c.parent_text or c.text
        key = (c.doc_title, parent)
        if key in seen:
            continue
        if sources and spent + len(parent) > budget_chars:
            break
        seen.add(key)
        sources.append(Source(n=len(sources) + 1, doc_title=c.doc_title, heading=c.heading, text=parent))
        spent += len(parent)

    return sources


def build_prompt(question: str, sources: list[Source]) -> str:
    """One string, because the seam's stream_chat takes one string.

    The protocol has no role separation (system vs user) yet — widening
    LLMClient to messages is a deliberate, separate change with its own tests,
    not something to smuggle in here. Until then: instructions first, extracts
    second, question LAST, because models weight the end of the prompt and the
    question is what must survive.
    """
    extracts = "\n\n".join(
        f"[{s.n}] {s.doc_title} — {s.heading}\n{s.text}" for s in sources
    )
    return f"{_INSTRUCTIONS}\nEXTRACTS:\n\n{extracts}\n\nQUESTION: {question}"
