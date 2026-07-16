# doc-intel

**A multi-tenant document intelligence API** — upload contracts and invoices, ask
questions, get cited answers. Built to be *defended*, not demoed: every design
decision here has a failure mode attached, and most of them have a test that
fails when you remove the fix.

Python · FastAPI · async · Azure OpenAI · embeddings · hybrid vector search · RAG

```bash
git clone https://github.com/rajatnparth/doc-intel && cd doc-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # defaults to a stub provider — no Azure key needed
pytest -q                       # 27 tests, no network
```

**No API key required to run any of it.** `LLM_PROVIDER=stub` swaps in a fake
provider that streams tokens and fails on command. That isn't a shortcut: you
cannot ask the real Azure to return a 429 on demand, or to die exactly halfway
through a stream — and untested resilience is decorative.

---

## Three things you can watch happen

```bash
python hybrid_demo.py              # retrieval failures, with real embeddings
python -m app.retrieval.ann_bench  # the recall you sell for latency, measured
python chunk_demo.py               # why chunk size is not a number you pick
uvicorn app.main:app --reload      # then: curl localhost:8000/health
```

**`hybrid_demo.py`** — ask for invoice `INV-2024-0891`. Dense retrieval confidently
returns `0888`, `0892`, `0893` at ranks 1–3; the right one lands at rank 5.
Embeddings encode *meaning*, and an invoice number doesn't have any. BM25 nails it
at rank 1. Then ask *"how long do we have to settle an invoice?"* and it inverts:
dense finds it, BM25 can't. RRF fuses both by **rank** (never score — BM25 is
unbounded and corpus-dependent). Real `bge-small-en-v1.5` embeddings, nothing staged.

**`ann_bench.py`** — exact k-NN as ground truth, then HNSW sweeping `efSearch`
(recall 0.585 → 0.999, latency rising with it) and IVF-Flat vs IVF-PQ. The gap
between those last two isolates the *product-quantisation tax*: IVF-Flat recovers
to recall 1.000 as you probe more clusters; IVF-PQ plateaus at 0.595 forever,
because compression is a second loss `nprobe` can't undo — for 16× less RAM.

**`chunk_demo.py`** — naive fixed-size chunking keeps a table row with its header
at sizes 600 and 1000, and orphans it at 300/400/500/700/800. It isn't reliably
bad, it's *arbitrary* — which is worse, because you can't reason about it.
Structure-aware chunking is invariant at every size.

---

## Layout

```
app/
  config.py          Settings from env, validated at BOOT — not at 3am
  schemas.py         The two contracts (Gate 1: clients. Gate 2: the model.)
  sse.py             The streaming protocol: discriminated frames + [DONE]
  main.py            FastAPI app, DI, error envelope, routes
  llm/
    base.py          LLMClient Protocol + the error taxonomy. The seam.
    stub.py          Fault-injecting fake provider (429, mid-stream death, hang)
    azure.py         Real AsyncAzureOpenAI client
    factory.py       One `if`. The entire provider swap.
  ingest/
    loaders.py       bytes -> Sections (structure kept, page furniture stripped)
    chunker.py       Sections -> Chunks (tables atomic, context prepended, parents)
  retrieval/
    ann_bench.py     exact vs HNSW vs IVF-Flat vs IVF-PQ, measured
    hybrid.py        BM25 + dense, fused by RRF
tests/               executable proof of each claim — 27 tests, no network
```

**The seam rule:** nothing under `app/llm/` imports FastAPI. Nothing outside it
imports `openai`. Retry, timeout, semaphore and circuit breaker are properties of
the *provider relationship*, not of an endpoint — so they live in the client
wrapper, and a route handler reads like a paragraph of business logic.

That's not aesthetics. It's the answer to *"where does this live, and why does
that matter when we add a second provider next quarter?"*

**Every import is tagged with its origin** (`# stdlib`, `# 3rd-party: <pkg>`,
`# local — <path>`). See [CONVENTIONS.md](CONVENTIONS.md).

---

## Decisions that are load-bearing, not decoration

| | Why |
|---|---|
| `async def` + `AsyncAzureOpenAI`, never a sync client inside `async def` | An LLM call is ~100% socket wait. Async buys **occupancy, not latency**. The mixed case is the uniquely bad one: flat CPU, detonating p99. |
| `extract()` returns a raw `str`, never a parsed model | Structured outputs guarantee **shape, never truth**. The type makes the unvalidated path unwriteable — you can't read `.invoice_total` off a string. |
| `currency: Literal[...]`, not `str` | Type validity is not semantic validity. A `str` accepts `"₹"`. |
| `strict=True` on money | A value you had to coerce is a value you don't understand. |
| `response_model=` on every route | A data-leak control in a multi-tenant system, not a docs feature. |
| SSE errors are **in-band frames**, not status codes | The `200 OK` was spent before the model wrote a word. A dead stream and a finished one are identical at the TCP layer — hence `[DONE]`. |
| Disconnect cancels the **upstream** call | Breaking your loop stops you *reading*; Azure keeps *generating*, and billing. |
| `/health` never calls the LLM | Liveness answers "is this process alive?", not "is the world well?" A provider blip shouldn't take your pods out of rotation. |
| `Usage` is a first-class type | *"What does one request cost you?"* is the follow-up question every time. Streaming sends no usage unless you ask. |
| `_translate()` in `azure.py` | Where every `openai.*` exception dies. If one reaches a handler, the seam leaked. |

---

## Status

| | Concept | |
|---|---|---|
| 1.1 | Async is occupancy, not speed | ✅ |
| 1.2 | Pydantic contracts at both boundaries | ✅ |
| 1.3 | Streaming without lying about it | ✅ `POST /v1/chat/stream` |
| 1.4 | Resilience: semaphore, jittered retry, breaker | ⬜ TODOs in `azure.py` |
| 3.1 | Chunking as information architecture | ✅ |
| 3.2 | ANN indexes: recall/latency/memory | ✅ |
| 3.3 | Hybrid search + RRF | ✅ |
| 3.4 | Metadata gates + the refusal path | ⬜ |

---

## Notes

Built while working through Ripostiq's *Ship It: Python, FastAPI & Azure OpenAI
RAG Agents*. The tests are the interesting part — several exist because a first
draft **passed while the code it covered was deleted**, which is the most
dangerous kind of green. See `test_streaming.py::test_disconnect_cancels_the_upstream_call`
and `test_hybrid.py::test_rrf_costs_you_the_top_slot_but_keeps_the_answer_in_the_pool`.

Azure setup (resource vs deployment, TPM/RPM quota, the admission-time token
reservation that causes 429s at 40% utilisation): [AZURE_SETUP.md](AZURE_SETUP.md).

MIT licensed.
