"""Phase 1 — /v1/ask, the RAG loop, as executable claims.

The headline test is `test_refusal_never_calls_the_generator`: the refusal
gate's whole value is that the model never sees bad context, and that is a
claim about a CALL NOT HAPPENING — which only a counting fake can prove.
"""

import json                             # stdlib — parse SSE frame payloads

import pytest                           # 3rd-party: pytest — fixtures, raises

from fastapi.testclient import TestClient  # 3rd-party: fastapi (submodule) — drives the app

from app.ingest import chunk_document   # local — app/ingest/chunker.py
from app.llm.base import TokenChunk, Usage  # local — app/llm/base.py (wire types)
from app.main import app, get_llm       # local — app/main.py (app + the DI seam)
from app.rag import build_prompt, select_sources  # local — app/rag.py


# -----------------------------------------------------------------------------
# SSE plumbing for assertions: "data: {json}" blocks -> parsed events,
# with the [DONE] sentinel kept as a plain string.
# -----------------------------------------------------------------------------
def events_of(body: str) -> list:
    out = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def types_of(events: list) -> list[str]:
    return [e["type"] if isinstance(e, dict) else e for e in events]


class CountingLLM:
    """Conforms to the LLMClient Protocol; counts and records every call.

    A stub that streams a plausible cited answer — and, crucially, remembers
    whether it was called at all. `tests can assert stream_chat_calls == 0`,
    which no amount of output-inspection can establish.
    """

    def __init__(self) -> None:
        self.stream_chat_calls = 0
        self.prompts: list[str] = []

    async def stream_chat(self, prompt, *, temperature=0.0, max_tokens=512):
        self.stream_chat_calls += 1
        self.prompts.append(prompt)
        for word in ["Accidents", "must", "be", "reported", "within", "24", "hours", "[1]."]:
            yield TokenChunk(text=word + " ")
        yield TokenChunk(text="", usage=Usage(prompt_tokens=200, completion_tokens=8))

    async def extract(self, text, schema, *, max_tokens=512):
        raise AssertionError("extract must not be called by /v1/ask")

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_llm():
    fake = CountingLLM()
    app.dependency_overrides[get_llm] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


# =============================================================================
# The happy path: sources frame FIRST, then tokens, then done, then [DONE].
# =============================================================================
def test_ask_streams_sources_then_tokens_then_done(fake_llm) -> None:
    with TestClient(app) as client:
        r = client.post("/v1/ask", json={"question": "how quickly must I report an accident?"})

    assert r.status_code == 200
    ev = events_of(r.text)
    kinds = types_of(ev)

    # Ordering IS the contract: the client renders citations before the first
    # token arrives, because the sources came from the retriever, not the model.
    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds.index("sources") < kinds.index("token")
    assert kinds[-2:] == ["done", "[DONE]"]

    sources = ev[0]["sources"]
    assert sources, "an answered question must cite at least one source"
    assert any("Claims Process" in s["heading"] for s in sources)
    # The tenant gate held all the way to the wire: asha's question can only
    # ever cite asha's documents.
    assert all("Asha Rao" in s["doc_title"] for s in sources)
    # Citation numbers are 1-based and contiguous — the client renders "[1]"
    # by looking up n; a gap means a dangling citation in the answer text.
    assert [s["n"] for s in sources] == list(range(1, len(sources) + 1))


def test_ask_prompt_contains_extracts_and_question_last(fake_llm) -> None:
    q = "how quickly must I report an accident?"
    with TestClient(app) as client:
        client.post("/v1/ask", json={"question": q})

    assert fake_llm.stream_chat_calls == 1
    prompt = fake_llm.prompts[0]
    # The extracts made it in — the model reads the PARENT section, and only
    # documents the principal may see.
    assert "twenty-four (24) hours" in prompt
    assert "Vikram" not in prompt
    # The question is the LAST line: models weight the end of the prompt, and
    # the question is what must survive.
    assert prompt.rstrip().endswith(q)


# =============================================================================
# THE HEADLINE: a refusal never reaches the model.
# =============================================================================
def test_refusal_never_calls_the_generator(fake_llm) -> None:
    """The gate's promise is not "we return refused: true". It is "the model
    never saw the question". Only the call counter can prove a non-event."""
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask", json={"question": "is a courtesy car provided during repairs?"}
        )

    ev = events_of(r.text)
    kinds = types_of(ev)

    assert "refusal" in kinds
    assert "token" not in kinds, "a refusal must not stream an answer"
    assert "sources" not in kinds, "refusals cite nothing — near-misses are links, not sources"
    assert kinds[-1] == "[DONE]", "every exit path ends the protocol honestly"

    refusal = ev[kinds.index("refusal")]
    assert 0.0 <= refusal["score"] < 0.5
    assert refusal["near_misses"], "offer the closest documents as links"

    # The claim that matters, provable only this way:
    assert fake_llm.stream_chat_calls == 0, (
        "the generator was called on a refusal — handed confident-looking "
        "irrelevant chunks, models answer anyway; that is the exact failure "
        "the gate exists to prevent"
    )


# =============================================================================
# Tenant scoping at the route level (the body-principal is phase-1 scaffolding,
# but the GATE behind it is real either way).
# =============================================================================
def test_ask_is_tenant_scoped(fake_llm) -> None:
    with TestClient(app) as client:
        r = client.post(
            "/v1/ask",
            json={
                "question": "what is my excess for an own damage claim?",
                "tenant_id": "vikram",
            },
        )

    ev = events_of(r.text)
    sources = ev[0]["sources"]
    assert sources
    assert all("Vikram" in s["doc_title"] for s in sources)
    # And the prompt the model saw contains Vikram's excess, not Asha's.
    assert "₹5,000" in fake_llm.prompts[0]
    assert "₹2,000" not in fake_llm.prompts[0]


# =============================================================================
# The prompt builder, unit-level: dedupe, budget, numbering.
# =============================================================================
# Section 1 is deliberately TWO paragraphs: the chunker cuts on paragraph
# boundaries, so a single long paragraph never splits — measured before this
# fixture settled (the first draft was one paragraph, and never produced
# siblings at any max_chars).
_DOC = """# Kit

## 1. Alpha

Alpha first paragraph, padded with enough words that this paragraph together
with the next cannot fit in one small chunk at the size used below.

Alpha second paragraph, also padded with enough words to force the chunker to
emit it separately once the first paragraph has used up the budget.

## 2. Beta

Beta body.
"""


def _chunks():
    return chunk_document(_DOC, doc_title="Kit", max_chars=160)


def test_select_sources_dedupes_siblings_to_one_parent() -> None:
    chunks = _chunks()
    alpha = [c for c in chunks if c.heading.startswith("1.")]
    assert len(alpha) >= 2, "setup: section 1 must have split into siblings"

    # Two siblings + one other section in, TWO parents out — the model never
    # reads the same section twice, however many of its chunks retrieval liked.
    picked = select_sources([alpha[0], alpha[1], next(c for c in chunks if c.heading.startswith("2."))],
                            budget_chars=10_000)
    assert [s.heading for s in picked] == [alpha[0].heading, "2. Beta"]
    assert [s.n for s in picked] == [1, 2]


def test_select_sources_respects_the_budget_but_never_returns_empty() -> None:
    chunks = _chunks()
    # A budget smaller than the first parent: it is admitted anyway. An empty
    # context is not a smaller context; the gate already said "answer this".
    picked = select_sources(chunks, budget_chars=1)
    assert len(picked) == 1

    # A budget that fits one parent but not two: the second is dropped.
    first_len = len(picked[0].text)
    picked2 = select_sources(chunks, budget_chars=first_len + 5)
    assert len(picked2) == 1


def test_build_prompt_numbers_extracts_and_ends_with_question() -> None:
    picked = select_sources(_chunks(), budget_chars=10_000)
    prompt = build_prompt("what is alpha?", picked)
    for s in picked:
        assert f"[{s.n}] Kit — {s.heading}" in prompt
    assert prompt.rstrip().endswith("what is alpha?")
    assert "ONLY the numbered extracts" in prompt
