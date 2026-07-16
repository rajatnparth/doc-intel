"""The FastAPI app.

RIGHT NOW: lifespan, dependency injection, error envelope, /health.
COMING:    /v1/chat/stream (section 1.3), /v1/extract (1.2 + 1.4).

Read the route handlers as they arrive and notice what is NOT in them:
no retry loops, no semaphores, no `openai` imports. A route handler should read
like a paragraph of business logic.
"""

import logging                          # stdlib — structured logs
import uuid                             # stdlib — request_id generation
from contextlib import asynccontextmanager  # stdlib — turns a generator into the
                                        #   lifespan context manager (yield splits it)
from typing import Annotated, AsyncIterator  # stdlib — DI annotation + the stream's type

from fastapi import Depends, FastAPI, Request  # 3rd-party: fastapi — app, DI, raw request
from fastapi.responses import JSONResponse, StreamingResponse  # 3rd-party: fastapi
                                        #   (submodule) — JSON errors + the SSE stream

from app.config import Settings, get_settings          # local — app/config.py
from app.llm.base import LLMClient, LLMError, Usage     # local — app/llm/base.py (the seam)
from app.llm.factory import build_llm_client            # local — app/llm/factory.py
from app.schemas import ChatStreamRequest, ErrorBody, ErrorEnvelope  # local — app/schemas.py
from app.sse import DoneEvent, ErrorEvent, TokenEvent, done_frame, frame  # local — app/sse.py

log = logging.getLogger("doc_intel")


# =============================================================================
# Lifespan — startup and shutdown, as one function split by a `yield`.
#
# Everything before `yield` runs once at startup. Everything after runs at
# shutdown. This replaced the old @app.on_event("startup") decorators, which
# are deprecated and had no way to share state between the two halves.
#
# `@asynccontextmanager` turns a generator into an async context manager, which
# is what Starlette expects. Same `yield`-splits-the-function trick as pytest
# fixtures — recognise the pattern once and you see it everywhere in Python.
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.validate_for_provider()  # fail at boot, not at 3am on the first request

    # ONE client for the whole process, not one per request.
    # It owns a connection pool; rebuilding it per request would mean a fresh TLS
    # handshake every time and would defeat the semaphore that's coming in 1.4
    # (a per-request semaphore caps nothing).
    app.state.llm = build_llm_client(settings)
    try:
        yield
    finally:
        await app.state.llm.aclose()


app = FastAPI(
    title="doc-intel",
    version="0.1.0",
    lifespan=lifespan,
)


# =============================================================================
# Dependency injection.
#
# `Depends(...)` tells FastAPI: before calling this handler, call THIS function
# and pass me the result. Two payoffs:
#   1. handlers never reach into globals
#   2. tests override it — app.dependency_overrides[get_llm] = lambda: FakeThing()
#      — so you can inject a failing client without monkeypatching imports.
# =============================================================================
def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


# `Annotated[X, Depends(f)]` is the modern spelling of `x: X = Depends(f)`.
# It's preferred because the dependency lives in the TYPE, so the parameter can
# still have a real default, and the annotation is reusable.
LLMDep = Annotated[LLMClient, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# =============================================================================
# One error shape for the whole service.
#
# Without this, every endpoint invents its own failure JSON and every client
# writes a different guess about which errors are worth retrying.
# =============================================================================
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # 429 if the caller should back off; 502 if the upstream is simply refusing.
    status = 429 if exc.code == "rate_limited" else 502

    headers = {"x-request-id": request_id}
    if exc.retry_after is not None:
        # Pass Azure's own advice straight through. It knows more than we do.
        headers["Retry-After"] = str(int(exc.retry_after))

    return JSONResponse(
        status_code=status,
        headers=headers,
        content=ErrorEnvelope(
            error=ErrorBody(
                code=exc.code,
                message=str(exc),
                request_id=request_id,
                # The field that earns its place: 429 and content_filter are both
                # "the LLM call failed", and the status code cannot tell them apart.
                retryable=exc.retryable,
            )
        ).model_dump(),
    )


# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
async def health(settings: SettingsDep) -> dict[str, str]:
    """Liveness. Deliberately does NOT call the LLM.

    A health check that hits your provider means a provider blip takes your pods
    out of rotation — you amplify their outage into yours. Liveness answers
    "is this process alive?", not "is the world well?".
    """
    return {"status": "ok", "provider": settings.llm_provider}


# -----------------------------------------------------------------------------
# Section 1.3 — SSE streaming.
#
# Headers that stop a reverse proxy from silently defeating the whole feature.
# nginx (and the nginx-derived edges: Azure Front Door, App Gateway) collect a
# response before forwarding it, because for 99% of traffic that is faster.
# For SSE it turns a stream into a 12-second blank screen followed by a burst.
#
# Not IANA-registered, not universal. Send them, then verify against your edge.
# -----------------------------------------------------------------------------
SSE_HEADERS = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


async def _sse_events(
    llm: LLMClient,
    req: ChatStreamRequest,
    request: Request,
    request_id: str,
) -> AsyncIterator[str]:
    """Yield SSE frames as tokens arrive.

    THE CONTRACT THIS FUNCTION UPHOLDS
    ----------------------------------
    1. Never collect. Each token becomes a socket write immediately.
    2. Every exit path emits [DONE]. A stream that just stops is
       indistinguishable from a stream that died.
    3. Errors travel IN-BAND, because the 200 OK is long gone.
    4. A client that vanishes stops costing us money.
    """
    usage: Usage | None = None

    # We hold a reference to the upstream async generator, rather than iterating
    # the call expression directly, so that we can explicitly CLOSE it.
    # That distinction is the whole of problem 4 — see the `finally` block.
    upstream = llm.stream_chat(
        req.prompt, temperature=req.temperature, max_tokens=req.max_tokens
    )

    try:
        async for chunk in upstream:
            # Problem 4: the user closed the tab. Breaking out of this loop stops
            # us READING. It does not stop Azure GENERATING — or billing.
            # The cancellation happens in `finally`, via upstream.aclose().
            if await request.is_disconnected():
                log.info("client disconnected, cancelling upstream", extra={"request_id": request_id})
                break

            if chunk.usage is not None:
                # The final chunk: usage, no text. Only arrives because the
                # client asked for stream_options={"include_usage": True}.
                # Without it you cannot answer "what does one request cost?"
                usage = chunk.usage
                continue

            if chunk.text:
                yield frame(TokenEvent(text=chunk.text))

    except LLMError as exc:
        # Problem 2. We already sent 200 OK — the status code was spent before
        # the model wrote a word. We cannot send a 500 now. So the error becomes
        # a MESSAGE in our protocol, tagged with a `type` the client can switch on.
        log.warning("stream failed mid-flight: %s", exc.code, extra={"request_id": request_id})
        yield frame(
            ErrorEvent(
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,   # 429 -> True. content_filter -> False, forever.
                request_id=request_id,
            )
        )

    finally:
        # `aclose()` throws GeneratorExit INTO the suspended upstream generator
        # at its current `yield`. That unwinds it, which closes the HTTP
        # connection to Azure, which stops generation. This is what "cancel the
        # upstream call" concretely means: stop the meter, not just the display.
        #
        # It runs on EVERY path — normal completion, error, disconnect, and even
        # if the ASGI server throws CancelledError into us. That is what `finally`
        # is for, and it is why we bound `upstream` to a name.
        await upstream.aclose()

        # Problem 3. Always. Even after an error. Even after a disconnect (nobody
        # reads it then, but the code path stays honest).
        #
        # `done` carries the payload; `[DONE]` is the marker clients look for.
        # Together they turn "the stream finished" from something the client
        # INFERS FROM SILENCE into something it is TOLD.
        yield frame(DoneEvent(usage=usage))
        yield done_frame()


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatStreamRequest, request: Request, llm: LLMDep) -> StreamingResponse:
    """Note what this handler does NOT contain: no retry loop, no semaphore, no
    `openai` import, no token-counting. It reads like a paragraph."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Calling _sse_events(...) runs NOTHING. It returns a paused async generator.
    # Uvicorn sends the status line + headers first, THEN pulls the first frame.
    # That ordering is exactly why the 200 is unrecallable, and why problem 2 exists.
    return StreamingResponse(
        _sse_events(llm, req, request, request_id),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "x-request-id": request_id},
    )


# TODO(1.2 + 1.4): @app.post("/v1/extract", response_model=InvoiceExtract)
#   - client.extract(text, InvoiceExtract.model_json_schema())
#   - InvoiceExtract.model_validate_json(raw)
#   - except ValidationError -> exactly ONE repair retry, feeding e.errors() back
#   - second failure -> SchemaRepairFailed. Fail closed. No answer beats a
#     fabricated invoice total.
