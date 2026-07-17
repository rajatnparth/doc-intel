"""Section 1.3 — proof that the stream tells the truth.

Each test corresponds to one of the four production problems. If you cannot
write these tests, you have not built streaming; you have built a demo.
"""

import asyncio                          # stdlib — drive the async generator directly + sleep
import json                             # stdlib — parse SSE frame payloads

import pytest                           # 3rd-party: pytest — @pytest.mark.asyncio, raises
from fastapi.testclient import TestClient  # 3rd-party: fastapi — in-process HTTP client

from app.llm.stub import FaultMode, StubLLMClient  # local — app/llm/stub.py
from app.main import _llm_frames, app, get_llm     # local — app/main.py (generator, app, DI dep)
from app.schemas import ChatStreamRequest          # local — app/schemas.py


def parse_frames(body: str) -> list[str]:
    """Split an SSE body into payloads.

    Note what this asserts implicitly: frames are separated by a BLANK LINE.
    Send one `\\n` instead of two and this returns garbage — which is exactly
    what a real client would experience (it waits forever).
    """
    out = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            out.append(block[len("data: ") :])
    return out


def _client_with(stub: StubLLMClient) -> TestClient:
    """Inject a stub via FastAPI's dependency_overrides.

    This is why `get_llm` is a dependency rather than a module global: we can
    swap the provider per-test without monkeypatching imports.
    """
    app.dependency_overrides[get_llm] = lambda: stub
    return TestClient(app)


# =============================================================================
# Happy path: token frames, then done + [DONE], with usage.
# =============================================================================
def test_stream_emits_tokens_then_done_with_usage() -> None:
    stub = StubLLMClient(token_delay=0.0)
    with _client_with(stub) as client:
        r = client.post("/v1/chat/stream", json={"prompt": "hi", "max_tokens": 4})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    # Problem 1: the proxy-defeating headers are actually on the response.
    assert r.headers["x-accel-buffering"] == "no"
    assert r.headers["cache-control"] == "no-cache"

    payloads = parse_frames(r.text)
    events = [json.loads(p) for p in payloads if p != "[DONE]"]

    assert [e["type"] for e in events] == ["token"] * 4 + ["done"]

    # Problem 5: usage arrived. Without stream_options={"include_usage": True}
    # on the real client, this is None and you cannot bill.
    assert events[-1]["usage"]["completion_tokens"] == 4
    assert events[-1]["usage"]["prompt_tokens"] > 0

    # Problem 3: the sentinel is the LAST thing on the wire.
    assert payloads[-1] == "[DONE]"

    app.dependency_overrides.clear()


# =============================================================================
# Problem 2 — the 200 is already spent. The error must arrive in-band.
# =============================================================================
def test_mid_stream_error_arrives_as_a_frame_not_a_500() -> None:
    stub = StubLLMClient(token_delay=0.0, default_fault=FaultMode.MID_STREAM_ERROR)
    with _client_with(stub) as client:
        r = client.post("/v1/chat/stream", json={"prompt": "hi", "max_tokens": 8})

    # The status code is 200. It was sent before the model wrote a word.
    # There is no universe in which this is a 500.
    assert r.status_code == 200

    payloads = parse_frames(r.text)
    events = [json.loads(p) for p in payloads if p != "[DONE]"]
    types = [e["type"] for e in events]

    # Tokens flowed BEFORE it died. That's what makes this hard.
    assert types.count("token") > 0
    assert "error" in types

    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "provider_unavailable"
    assert err["retryable"] is True          # 5xx: later may differ. Retry.
    assert err["request_id"]

    # Problem 3: even after an error, the client is TOLD the stream ended.
    assert payloads[-1] == "[DONE]"

    app.dependency_overrides.clear()


def test_content_filter_is_marked_not_retryable() -> None:
    """The distinction the HTTP status code cannot express."""
    stub = StubLLMClient(token_delay=0.0, default_fault=FaultMode.CONTENT_FILTER)
    with _client_with(stub) as client:
        r = client.post("/v1/chat/stream", json={"prompt": "hi"})

    events = [json.loads(p) for p in parse_frames(r.text) if p != "[DONE]"]
    err = next(e for e in events if e["type"] == "error")

    assert err["code"] == "content_filtered"
    # Deterministic refusal. Retrying burns quota to be told no again, forever.
    assert err["retryable"] is False

    app.dependency_overrides.clear()


# =============================================================================
# Problem 4 — the user closed the tab.
#
# This test does NOT go through TestClient, because TestClient cannot simulate a
# mid-response disconnect. We drive the generator directly with a fake Request.
#
# The assertion that matters is NOT "we stopped reading". It is "the upstream
# stopped generating" — i.e. we stopped paying.
# =============================================================================
class _FakeRequest:
    """Reports connected for the first N polls, then disconnected."""

    def __init__(self, disconnect_after: int) -> None:
        self._polls = 0
        self._after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self._after


@pytest.mark.asyncio
async def test_disconnect_cancels_the_upstream_call() -> None:
    stub = StubLLMClient(token_delay=0.0)
    req = ChatStreamRequest(prompt="hi", max_tokens=20)
    fake_request = _FakeRequest(disconnect_after=3)

    frames = [
        f
        async for f in _llm_frames(
            stub,
            prompt=req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            request=fake_request,  # type: ignore[arg-type]
            request_id="rid-1",
        )
    ]

    # We stopped early. Necessary, but NOT sufficient.
    assert 0 < stub.tokens_generated < 20

    # -------------------------------------------------------------------------
    # THE ASSERTION THAT MATTERS.
    #
    # Naively you'd assert `tokens_generated` stops growing. That assertion is
    # WORTHLESS: a suspended generator produces nothing whether you cancelled it
    # or merely walked away. The test would pass with `upstream.aclose()` deleted.
    #
    # Against real Azure those two states are wildly different — one of them is
    # still generating tokens and still billing you. So we assert on the thing
    # that actually distinguishes them: did the generator receive GeneratorExit
    # and unwind?
    #
    # Delete `await upstream.aclose()` from main.py and this line fails. That is
    # the definition of a test worth having.
    # -------------------------------------------------------------------------
    assert stub.upstream_closed is True, (
        "upstream was never torn down — against real Azure you would still be "
        "generating, and paying for, tokens nobody will ever read"
    )

    # Even on the disconnect path, the protocol stays honest.
    assert frames[-1].strip() == "data: [DONE]"


# =============================================================================
# The framing itself. Boring, and the bug everyone writes once.
# =============================================================================
def test_every_frame_ends_with_a_blank_line() -> None:
    stub = StubLLMClient(token_delay=0.0)
    with _client_with(stub) as client:
        r = client.post("/v1/chat/stream", json={"prompt": "hi", "max_tokens": 2})

    # A single \n means the client blocks forever waiting for the frame to end.
    assert r.text.endswith("\n\n")
    assert "\n\n\n" not in r.text, "no accidental triple newline"
    for line in [l for l in r.text.split("\n") if l]:
        assert line.startswith("data: ") or line.startswith(":")

    app.dependency_overrides.clear()
