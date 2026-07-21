"""The FastAPI app.

RIGHT NOW: lifespan, DI, error envelope, /health, /v1/chat/stream, /v1/ask.
COMING:    /v1/extract (1.2 + 1.4).

Read the route handlers and notice what is NOT in them: no retry loops, no
semaphores, no `openai` imports, no prompt templates. A route handler should
read like a paragraph of business logic.
"""

import logging                          # stdlib — structured logs
import time                             # stdlib — audit duration (monotonic, not wall)
from datetime import date              # stdlib — upload effective-window fields
import uuid                             # stdlib — request_id generation
from contextlib import asynccontextmanager  # stdlib — turns a generator into the
                                        #   lifespan context manager (yield splits it)
from dataclasses import dataclass, field  # stdlib — StreamCapture is plain mutable state
from typing import Annotated, AsyncIterator  # stdlib — DI annotation + the stream's type

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile  # 3rd-party: fastapi —
                                        #   app, DI, raw request, multipart upload pieces
from fastapi.concurrency import run_in_threadpool  # 3rd-party: fastapi (submodule) —
                                        #   push CPU-bound work off the event loop
from fastapi.responses import JSONResponse, Response, StreamingResponse  # 3rd-party: fastapi
                                        #   (submodule) — JSON errors + the SSE stream

from app.audit import (                 # local — app/audit.py (the exchange record)
    AuditRecord,
    AuditSink,
    FactRef,
    JsonlAuditSink,
    RetrievedRef,
    now_utc,
)
from app.auth import PrincipalDep                       # local — app/auth.py (verified identity)
from app.config import Settings, get_settings          # local — app/config.py
from app.handoff import StubTicketStore, TicketStore    # local — app/handoff.py (refusal -> ticket)
from app.llm.base import LLMClient, LLMError, Usage     # local — app/llm/base.py (the seam)
from app.llm.factory import build_llm_client            # local — app/llm/factory.py
from app.policy_admin import PolicyAdmin, StubPolicyAdmin  # local — app/policy_admin.py
                                        #   (the system of record — numbers live here)
from app.ingest import ChunkMeta, chunk_sections        # local — app/ingest/ (upload -> chunks)
from app.ingest.index import ingest_into                # local — app/ingest/index.py (memory-mode boot + uploads)
from app.ingest.loaders import SUPPORTED_SUFFIXES, parse_upload  # local — app/ingest/loaders.py
                                        #   (the format seam: md/pdf/docx today,
                                        #   Azure Document Intelligence tomorrow)
from app.ops import (                   # local — app/ops.py (metrics + dependency health)
    AUDIT_WRITE_FAILURES,
    DOCUMENTS_INGESTED,
    HANDOFF_TICKETS,
    AuditHealth,
    observe_ask,
    render_metrics,
)
from app.rag import build_prompt, select_sources        # local — app/rag.py (the context budget)
from app.safety import Redactor, build_redactor         # local — app/safety.py (PII stays out of storage)
from app.store.factory import build_vector_store        # local — app/store/factory.py (the storage seam)
from app.router import FIELD_LABELS, FactField, route   # local — app/router.py
                                        #   (numbers-vs-wording: tier 1 deterministic,
                                        #   tier 2 LLM-classified, both behind Gate 2)
from app.retrieval.corpus import load_corpus            # local — app/retrieval/corpus.py (fixture)
from app.retrieval.gated import (       # local — app/retrieval/gated.py (gates + refusal)
    REFUSAL_THRESHOLD,
    Principal,
    PreFilterRetriever,
    answer,
)
from app.schemas import (               # local — app/schemas.py
    AskRequest,
    ChatStreamRequest,
    DocumentIngested,
    ErrorBody,
    ErrorEnvelope,
    HandoffRequest,
    HandoffResponse,
)
from app.sse import (                   # local — app/sse.py (the wire protocol)
    DoneEvent,
    ErrorEvent,
    FactItem,
    FactsEvent,
    RefusalEvent,
    SourceRef,
    SourcesEvent,
    TokenEvent,
    done_frame,
    frame,
)

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
    # Fail at boot, not at 3am on the first request. This includes refusing to
    # serve without AUTH_JWT_SECRET: a warning would be a log line someone
    # greps for after the incident; a crash at boot is a deploy that never
    # went out wrong. The only default secret is no secret.
    settings.validate_for_serving()

    # ONE client for the whole process, not one per request.
    # It owns a connection pool; rebuilding it per request would mean a fresh TLS
    # handshake every time and would defeat the semaphore that's coming in 1.4
    # (a per-request semaphore caps nothing).
    app.state.llm = build_llm_client(settings)

    # The retrieval stack, over the vector store (phase 5). memory mode keeps
    # the old behaviour — embed the fixture corpus at boot; qdrant mode opens
    # the persisted collection READ-ONLY and refuses to serve if ingestion
    # never ran: an empty index doesn't error, it just refuses every question,
    # which is the worse failure because it looks like a model problem.
    store = build_vector_store(settings)
    if store.count() == 0:
        if settings.vector_store == "memory":
            ingest_into(store, load_corpus())
        else:
            store.close()
            raise RuntimeError(
                "vector store 'qdrant' is empty — run `python -m app.ingest.index` "
                "once, then start the server."
            )
    app.state.store = store
    app.state.retriever = PreFilterRetriever(store)

    # The system of record — the stub against the fixture corpus. A real
    # deployment swaps in a connector to the insurer's core system; the
    # PolicyAdmin Protocol is the plug, and this line is the socket.
    app.state.policy_admin = StubPolicyAdmin()

    # Phase 6: the audit trail and the handoff queue. Same socket pattern —
    # JSONL file and in-memory stub here; WORM store and the insurer's
    # ticketing connector on the private side of the split.
    app.state.audit = JsonlAuditSink(settings.audit_path)
    app.state.tickets = StubTicketStore()

    # Phase 8: identifiers are removed BEFORE anything reaches storage.
    # Default is the RegexRedactor; NullRedactor is an explicit opt-out.
    app.state.redactor = build_redactor(settings.audit_redact_pii)

    # Phase 9: the audit sink's health, observed from real writes — the flag
    # admission control and /ready both read. Starts optimistic; the first
    # failed write flips it, the next success restores it.
    app.state.audit_health = AuditHealth()
    try:
        yield
    finally:
        await app.state.llm.aclose()
        # Local-mode qdrant holds a folder lock; release it so the ingest CLI
        # (or the next boot) can open the store. No-op for memory.
        app.state.store.close()


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


def get_retriever(request: Request) -> PreFilterRetriever:
    return request.app.state.retriever


def get_policy_admin(request: Request) -> PolicyAdmin:
    return request.app.state.policy_admin


def get_audit(request: Request) -> AuditSink:
    return request.app.state.audit


def get_tickets(request: Request) -> TicketStore:
    return request.app.state.tickets


def get_redactor(request: Request) -> Redactor:
    return request.app.state.redactor


def get_audit_health(request: Request) -> AuditHealth:
    return request.app.state.audit_health


# `Annotated[X, Depends(f)]` is the modern spelling of `x: X = Depends(f)`.
# It's preferred because the dependency lives in the TYPE, so the parameter can
# still have a real default, and the annotation is reusable.
LLMDep = Annotated[LLMClient, Depends(get_llm)]
RetrieverDep = Annotated[PreFilterRetriever, Depends(get_retriever)]
PolicyAdminDep = Annotated[PolicyAdmin, Depends(get_policy_admin)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
AuditDep = Annotated[AuditSink, Depends(get_audit)]
TicketsDep = Annotated[TicketStore, Depends(get_tickets)]
RedactorDep = Annotated[Redactor, Depends(get_redactor)]
AuditHealthDep = Annotated[AuditHealth, Depends(get_audit_health)]


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
    """Liveness. Deliberately does NOT call the LLM — or the store, or the
    sink. A liveness failure means RESTART THE POD; put a dependency check
    here and a store blip makes the orchestrator restart-loop every healthy
    pod, amplifying a hiccup into a fleet outage. Dependencies belong in
    /ready, whose failure just means "no traffic for now".
    """
    return {"status": "ok", "provider": settings.llm_provider}


@app.get("/ready")
async def ready(
    request: Request,
    settings: SettingsDep,
    audit: AuditDep,
    audit_health: AuditHealthDep,
) -> JSONResponse:
    """Readiness: should THIS instance receive traffic right now?

    Checks what serving actually needs: an open, non-empty vector store (the
    phase-5 fail-closed boot check, made continuous) and an audit sink that
    can accept a write. The sink check is an ACTIVE probe, and that is the
    recovery mechanism, not an optimisation: under strict admission the flag
    blocks all exchanges, so no exchange can ever discover the disk came
    back — the orchestrator's own readiness polling becomes the retry loop,
    and rotation back in is automatic. (The deadlock this design replaces
    was caught by test_ops.py, not foresight — recorded honestly.)
    """
    checks: dict[str, str] = {}

    try:
        # count() is file/lock IO on the qdrant path — off the event loop,
        # like every other blocking call in this file.
        n = await run_in_threadpool(request.app.state.store.count)
        checks["store"] = "ok" if n > 0 else "empty"
    except Exception as exc:  # noqa: BLE001 — a probe reports, never raises
        checks["store"] = f"unreachable: {type(exc).__name__}"

    try:
        await run_in_threadpool(audit.probe)
        audit_health.mark_ok()          # the probe IS the recovery path
        checks["audit"] = "ok"
    except Exception:  # noqa: BLE001 — a probe reports, never raises
        audit_health.mark_failed()
        checks["audit"] = "failing writes"

    ok = checks["store"] == "ok" and checks["audit"] == "ok"
    return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, **checks})


@app.get("/metrics")
async def metrics() -> Response:
    """The scraper's contract: current counter values, Prometheus text
    format. Aggregate numbers only — outcomes, durations, tokens; never a
    tenant label (cardinality + the phase-8 minimization argument, again).
    """
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


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


@dataclass
class StreamCapture:
    """What actually crossed the wire — collected FOR THE AUDIT RECORD while
    the stream runs. The stream itself never buffers (contract rule 1); this
    is a second, passive accumulation whose consumer is the dispute handler,
    not the client. `answer_text` is unknowable until the stream ends, which
    is why the audit write lives in a `finally` and not at request time."""

    parts: list[str] = field(default_factory=list)
    usage: Usage | None = None
    error_code: str = ""
    disconnected: bool = False

    @property
    def text(self) -> str:
        return "".join(self.parts)


async def _llm_frames(
    llm: LLMClient,
    *,
    prompt: str,
    temperature: float,
    max_tokens: int,
    request: Request,
    request_id: str,
    capture: StreamCapture | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames as tokens arrive. Shared by /v1/chat/stream and /v1/ask
    — the disconnect-cancel machinery must not be duplicated, because the copy
    that drifts is the copy that leaks money.

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
    upstream = llm.stream_chat(prompt, temperature=temperature, max_tokens=max_tokens)

    try:
        async for chunk in upstream:
            # Problem 4: the user closed the tab. Breaking out of this loop stops
            # us READING. It does not stop Azure GENERATING — or billing.
            # The cancellation happens in `finally`, via upstream.aclose().
            if await request.is_disconnected():
                log.info("client disconnected, cancelling upstream", extra={"request_id": request_id})
                if capture is not None:
                    capture.disconnected = True
                break

            if chunk.usage is not None:
                # The final chunk: usage, no text. Only arrives because the
                # client asked for stream_options={"include_usage": True}.
                # Without it you cannot answer "what does one request cost?"
                usage = chunk.usage
                if capture is not None:
                    capture.usage = chunk.usage
                continue

            if chunk.text:
                if capture is not None:
                    capture.parts.append(chunk.text)
                yield frame(TokenEvent(text=chunk.text))

    except LLMError as exc:
        # Problem 2. We already sent 200 OK — the status code was spent before
        # the model wrote a word. We cannot send a 500 now. So the error becomes
        # a MESSAGE in our protocol, tagged with a `type` the client can switch on.
        log.warning("stream failed mid-flight: %s", exc.code, extra={"request_id": request_id})
        if capture is not None:
            capture.error_code = exc.code
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
async def chat_stream(
    req: ChatStreamRequest,
    request: Request,
    principal: PrincipalDep,
    llm: LLMDep,
) -> StreamingResponse:
    """Note what this handler does NOT contain: no retry loop, no semaphore, no
    `openai` import, no token-counting. It reads like a paragraph.

    Authenticated even though tenancy doesn't apply to a raw prompt: an
    unmetered passthrough to a paid model is a COST hole, not a data hole —
    and "who spent this?" (the principal) is the first question after any
    bill spike. The principal is required here for the same reason Usage is
    a first-class type."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Calling _llm_frames(...) runs NOTHING. It returns a paused async generator.
    # Uvicorn sends the status line + headers first, THEN pulls the first frame.
    # That ordering is exactly why the 200 is unrecallable, and why problem 2 exists.
    return StreamingResponse(
        _llm_frames(
            llm,
            prompt=req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            request=request,
            request_id=request_id,
        ),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "x-request-id": request_id},
    )


# -----------------------------------------------------------------------------
# Phase 1 — /v1/ask: the RAG loop. Gates -> rerank -> refuse OR cite + generate.
# -----------------------------------------------------------------------------
def _refs(chunks) -> list[RetrievedRef]:
    return [
        RetrievedRef(doc_title=c.doc_title, heading=c.heading, chunk_index=c.chunk_index)
        for c in chunks
    ]


async def _ask_events(
    llm: LLMClient,
    retriever: PreFilterRetriever,
    policy_admin: PolicyAdmin,
    settings: Settings,
    principal: Principal,
    req: AskRequest,
    request: Request,
    request_id: str,
    audit: AuditSink,
    redactor: Redactor,
    audit_health: AuditHealth,
) -> AsyncIterator[str]:
    # The principal arrived VERIFIED — built in app/auth.py from signed claims,
    # the only request-path place a Principal is constructed. The body-field
    # version of this line is gone, and schemas.py rejects clients still
    # sending it.

    # Phase 6: everything below feeds this dict; the AuditRecord is
    # constructed and WRITTEN in the finally block, because its most
    # important field — what the customer actually saw — does not exist
    # until the stream ends. `outcome` starts pessimistic ("error") and is
    # upgraded by whichever path completes.
    started = time.monotonic()
    rec: dict = {
        "request_id": request_id,
        "at": now_utc(),
        "tenant_id": principal.tenant_id,
        "groups": sorted(principal.groups),
        "question": req.question,
        "as_of": req.as_of,
        "outcome": "error",
    }
    capture = StreamCapture()
    try:
        async for f in _ask_events_inner(
            llm, retriever, policy_admin, settings, principal, req, request,
            request_id, rec, capture,
        ):
            yield f
    finally:
        rec["duration_ms"] = int((time.monotonic() - started) * 1000)
        # Merged HERE, not in the happy path: if the generator was torn down
        # mid-stream, the post-stream update never ran, but the capture still
        # holds exactly what was delivered before the teardown.
        if capture.parts:
            rec["answer_text"] = capture.text
        if capture.usage is not None:
            rec["prompt_tokens"] = capture.usage.prompt_tokens
            rec["completion_tokens"] = capture.usage.completion_tokens
        # Phase 8: identifiers leave the record BEFORE it touches disk. The
        # customer-authored fields only — reference numbers the dispute is
        # ABOUT (policy, claim ids) are preserved by the patterns' design.
        rec["question"] = redactor.redact(rec["question"])
        if rec.get("answer_text"):
            rec["answer_text"] = redactor.redact(rec["answer_text"])
        try:
            audit.write(AuditRecord(**rec))
            audit_health.mark_ok()
        except Exception:  # noqa: BLE001 — the record must never kill the stream…
            # …but a swallowed audit failure is a compliance hole, so it is
            # the loudest thing this module logs — and (phase 9) it flips the
            # health flag that admission control and /ready read: under
            # AUDIT_STRICT, the NEXT exchange is refused at the door.
            audit_health.mark_failed()
            AUDIT_WRITE_FAILURES.inc()
            log.exception("AUDIT WRITE FAILED — exchange not recorded", extra={"request_id": request_id})
        # The metric and the record must agree about what happened — same
        # finally, same fields. (No tenant labels: see app/ops.py.)
        observe_ask(
            rec["outcome"], rec["duration_ms"], rec.get("prompt_tokens"), rec.get("completion_tokens")
        )


async def _ask_events_inner(
    llm: LLMClient,
    retriever: PreFilterRetriever,
    policy_admin: PolicyAdmin,
    settings: Settings,
    principal: Principal,
    req: AskRequest,
    request: Request,
    request_id: str,
    rec: dict,
    capture: StreamCapture,
) -> AsyncIterator[str]:
    # ROUTE FIRST: is this a question about a VALUE the system of record holds?
    # Facts never come from prose — calibrate.py measured why (a premium
    # question scored 0.7785 against a section that merely DISCUSSES premiums).
    # The record lookup is keyed by the VERIFIED tenant: isolation extends to
    # structured data, not just documents.
    # …and only for PRESENT-TENSE questions. The stub record holds the CURRENT
    # term; a question anchored to a past date (as_of) needs the record as of
    # that date, which this connector cannot serve — but the effective-dated
    # wording archive can. Dated questions skip the router entirely: no point
    # spending a tier-2 LLM call deciding a route that is already decided.
    field = None
    if req.as_of is None:
        field = await route(req.question, llm)
    if field is not None:
        record = policy_admin.get_record(principal.tenant_id)
        if record is None:
            # No record, and falling through to RAG would answer a numbers
            # question from prose — the exact thing this router forbids.
            rec.update(outcome="refusal", refusal_reason="no policy record on file")
            yield frame(RefusalEvent(score=0.0, reason="no policy record on file", near_misses=[]))
        else:
            facts = [FactItem(name=FIELD_LABELS[field], value=str(getattr(record, field.value)))]
            if field is FactField.ANNUAL_PREMIUM:
                # "What will next year's premium be?" — the honest answer is
                # the current premium plus WHEN it changes. Next year's number
                # does not exist yet, in any subsystem.
                facts.append(FactItem(name=FIELD_LABELS[FactField.RENEWAL_DATE], value=str(record.renewal_date)))
            log.info("ask routed to policy_admin: %s", field.value, extra={"request_id": request_id})
            rec.update(
                outcome="facts",
                facts=[FactRef(name=f.name, value=f.value) for f in facts],
            )
            yield frame(FactsEvent(policy_number=record.policy_number, facts=facts))
        yield frame(DoneEvent(usage=None))   # no usage: NO MODEL WAS CALLED
        yield done_frame()
        return

    # Embedding the query, building a first-use index view, and cross-encoding
    # 20 candidates are all CPU-bound. Run them inline and they block the event
    # loop — every OTHER live stream stalls while this one thinks. Async buys
    # occupancy only if the loop stays free (1.1); the threadpool keeps it free.
    a = await run_in_threadpool(
        answer, req.question, principal, retriever, as_of=req.as_of
    )

    if a.refused:
        # The generator is NEVER called on a refusal (gated.py explains why:
        # handed confident-looking irrelevant chunks, models answer anyway).
        # tests/test_ask.py proves it with a counting fake.
        log.info(
            "ask refused: score=%.4f", a.score, extra={"request_id": request_id}
        )
        rec.update(
            outcome="refusal",
            rerank_score=a.score,
            threshold=REFUSAL_THRESHOLD,
            refusal_reason=a.reason,
            retrieved=_refs(a.near_misses),
        )
        yield frame(
            RefusalEvent(
                score=a.score,
                reason=a.reason,
                near_misses=[
                    SourceRef(n=i + 1, doc_title=c.doc_title, heading=c.heading)
                    for i, c in enumerate(a.near_misses)
                ],
            )
        )
        yield frame(DoneEvent(usage=None))
        yield done_frame()
        return

    # Sources are known NOW, before the model says a word — they came from the
    # retriever, not from the model. Streaming them first lets the client render
    # the citations panel during generation, and keeps the provenance honest.
    sources = select_sources(a.chunks, budget_chars=settings.ask_context_chars)
    yield frame(
        SourcesEvent(
            sources=[
                SourceRef(n=s.n, doc_title=s.doc_title, heading=s.heading)
                for s in sources
            ]
        )
    )

    # Pessimistic until the stream completes: if the client (or the ASGI
    # server) tears this generator down mid-answer, the finally in
    # _ask_events records "disconnected" with whatever text had been sent.
    rec.update(
        outcome="disconnected",
        rerank_score=a.score,
        threshold=REFUSAL_THRESHOLD,
        retrieved=_refs(a.chunks),
        sources=[
            RetrievedRef(doc_title=s.doc_title, heading=s.heading)
            for s in sources
        ],
    )

    prompt = build_prompt(req.question, sources)
    async for f in _llm_frames(
        llm,
        prompt=prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        request=request,
        request_id=request_id,
        capture=capture,
    ):
        yield f

    # The stream ran to completion — resolve the outcome from what the
    # capture saw. answer_text/usage are filled either way: a mid-stream
    # error still delivered a prefix, and the record must say which prefix.
    rec.update(
        outcome=(
            "error" if capture.error_code
            else "disconnected" if capture.disconnected
            else "answer"
        ),
        error_code=capture.error_code,
    )


@app.post("/v1/ask")
async def ask(
    req: AskRequest,
    request: Request,
    principal: PrincipalDep,
    llm: LLMDep,
    retriever: RetrieverDep,
    policy_admin: PolicyAdminDep,
    settings: SettingsDep,
    audit: AuditDep,
    redactor: RedactorDep,
    audit_health: AuditHealthDep,
):
    """Route -> facts from the system of record, OR retrieve -> gate ->
    refuse or cite + generate. Streamed either way.

    Still no retry loop, no semaphore, no `openai` import, no prompt template —
    and now also no retrieval logic, no threshold, no token parsing, and no
    intent patterns. Each lives where a second consumer can reach it. Note the
    401 happens BEFORE this body runs: an unauthenticated request never
    touches the retriever, the record store, the models, or the corpus."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # Phase 9 — "no record, no answer", enforced AT ADMISSION. The flag was
    # flipped by a real failed write, so this check costs nothing per
    # request; and refusing HERE — before the 200 is spent — means we can
    # still say no cleanly, with a Retry-After, instead of erroring
    # mid-stream. One exchange (the one that discovered the failure) always
    # slips through un-audited; that is the price of not pre-writing a fake
    # record, and the metric counts it.
    if settings.audit_strict and not audit_health.ok:
        return JSONResponse(
            status_code=503,
            headers={"x-request-id": request_id, "Retry-After": "30"},
            content=ErrorEnvelope(
                error=ErrorBody(
                    code="audit_unavailable",
                    message="exchanges cannot be recorded right now; refusing to serve unrecorded answers",
                    request_id=request_id,
                    retryable=True,
                )
            ).model_dump(),
        )

    return StreamingResponse(
        _ask_events(llm, retriever, policy_admin, settings, principal, req, request, request_id, audit, redactor, audit_health),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, "x-request-id": request_id},
    )


# -----------------------------------------------------------------------------
# Phase 6 — /v1/handoff: a refusal must not be a dead end.
# -----------------------------------------------------------------------------
@app.post("/v1/handoff", response_model=HandoffResponse, status_code=201)
async def handoff(
    req: HandoffRequest,
    request: Request,
    principal: PrincipalDep,
    audit: AuditDep,
    tickets: TicketsDep,
    redactor: RedactorDep,
):
    """Turn an audited exchange into a ticket for a human.

    The ticket REFERENCES the exchange (request_id) rather than copying the
    conversation: the agent who picks it up reads the audit record — what the
    customer saw AND what the system retrieved and scored. One source of truth.

    THE CHECK THAT MATTERS: the record must belong to the caller's verified
    tenant. Request ids leak — headers, logs, support screenshots — and
    without this line any authenticated tenant could pull another tenant's
    exchange into a ticket. And it answers 404, not 403: a 403 on a foreign
    id would CONFIRM the id exists, which is itself a leak.
    """
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    # The JSONL lookup is file I/O — off the event loop, like every other
    # blocking call in this file.
    rec = await run_in_threadpool(audit.get, req.request_id)
    if rec is None or rec.tenant_id != principal.tenant_id:
        return JSONResponse(
            status_code=404,
            headers={"x-request-id": request_id},
            content=ErrorEnvelope(
                error=ErrorBody(
                    code="not_found",
                    message="no such exchange",
                    request_id=request_id,
                    retryable=False,
                )
            ).model_dump(),
        )

    t = tickets.create(
        request_id=rec.request_id,
        tenant_id=principal.tenant_id,
        question=rec.question,
        reason=rec.refusal_reason or f"customer requested a human after outcome={rec.outcome}",
        # The note is customer free text headed into a ticketing system whose
        # retention we don't control — redacted like everything else stored.
        note=redactor.redact(req.note),
    )
    HANDOFF_TICKETS.inc()
    log.info("handoff created: %s -> %s", rec.request_id, t.ticket_id, extra={"request_id": request_id})
    return HandoffResponse(
        ticket_id=t.ticket_id,
        request_id=t.request_id,
        status=t.status,
        created_at=t.created_at,
    )


# TODO(1.2 + 1.4): @app.post("/v1/extract", response_model=InvoiceExtract)
#   - client.extract(text, InvoiceExtract.model_json_schema())
#   - InvoiceExtract.model_validate_json(raw)
#   - except ValidationError -> exactly ONE repair retry, feeding e.errors() back
#   - second failure -> SchemaRepairFailed. Fail closed. No answer beats a
#     fabricated invoice total.


# -----------------------------------------------------------------------------
# Phase 10 — /v1/documents: the fixture becomes a feature.
# -----------------------------------------------------------------------------
def _doc_error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        headers={"x-request-id": request_id},
        content=ErrorEnvelope(
            error=ErrorBody(code=code, message=message, request_id=request_id, retryable=False)
        ).model_dump(),
    )


@app.post("/v1/documents", response_model=DocumentIngested, status_code=201)
async def upload_document(
    request: Request,
    principal: PrincipalDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    title: str = Form(..., min_length=3, max_length=120),
    acl: str = Form("customer,agent"),
    effective_from: date | None = Form(None),
    effective_to: date | None = Form(None),
):
    """Upload -> parse -> chunk -> embed -> upsert -> immediately answerable.

    The most hostile input this API accepts: attacker-controlled bytes, fed
    to parser libraries, destined for prompts. Hence the order of the gates
    below — group, size, type — each failing CLOSED with a client error
    before the next layer spends any work.

    Note the field split (the phase-3 lesson, third appearance): acl and the
    effective window arrive from the FORM — they describe the document,
    inside the uploader's own corpus, where the worst a lie can do is
    mislabel their own data. tenant_id arrives ONLY from the verified token,
    because it decides WHOSE corpus changes. And ingestion is a back-office
    act: the `agent` group is required — a customer never writes the corpus.
    """
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

    if "agent" not in principal.groups:
        return _doc_error(403, "forbidden", "document ingestion requires the agent role", request_id)

    # Size gate BEFORE parsing: read one byte past the cap so "too big" is
    # detectable without buffering an unbounded body.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        return _doc_error(413, "payload_too_large",
                          f"upload exceeds {settings.max_upload_bytes} bytes", request_id)

    acl_groups = frozenset(g.strip() for g in acl.split(",") if g.strip())
    if not acl_groups or not acl_groups <= {"customer", "agent"}:
        return _doc_error(422, "invalid_acl", "acl must be a subset of: customer, agent", request_id)
    if effective_from and effective_to and effective_to <= effective_from:
        return _doc_error(422, "invalid_window", "effective_to must be after effective_from", request_id)

    try:
        # Parsing is CPU-bound library code over untrusted bytes — off the
        # event loop, and every failure is the CLIENT's 4xx, never our 500.
        sections = await run_in_threadpool(
            parse_upload, file.filename or "", data, doc_title=title
        )
    except ValueError as exc:
        return _doc_error(415, "unsupported_type", str(exc), request_id)
    except Exception as exc:  # noqa: BLE001 — corrupt uploads are client errors
        return _doc_error(422, "unparseable", f"could not parse file: {type(exc).__name__}", request_id)

    meta = ChunkMeta(
        tenant_id=principal.tenant_id,          # the verified token, never the form
        acl=acl_groups,
        effective_from=effective_from or date.min,
        effective_to=effective_to,
    )
    chunks = chunk_sections(sections, doc_title=title, meta=meta)
    if not chunks:
        return _doc_error(422, "empty_document", "no extractable text in the upload", request_id)

    store = request.app.state.store

    def _replace() -> tuple[int, int]:
        # REPLACE = delete-then-upsert. Upsert alone would let a shorter
        # revision orphan the old tail — stale wording, retrievable forever.
        removed = store.delete_doc(title, principal.tenant_id)
        written = ingest_into(store, chunks)
        return removed, written

    removed, written = await run_in_threadpool(_replace)

    DOCUMENTS_INGESTED.inc()
    log.info(
        "document ingested: %r (%d chunks, %d replaced)", title, written, removed,
        extra={"request_id": request_id},
    )
    return DocumentIngested(
        doc_title=title, chunks=written, replaced_chunks=removed, tenant_id=principal.tenant_id
    )
