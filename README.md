# doc-intel

[![CI](https://github.com/rajatnparth/doc-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatnparth/doc-intel/actions/workflows/ci.yml)

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
pytest -q                       # 39 tests. First run downloads ~210MB of local
                                # models (embedder + cross-encoder); after that,
                                # no network.
```

**No API key required to run any of it.** `LLM_PROVIDER=stub` swaps in a fake
provider that streams tokens and fails on command. That isn't a shortcut: you
cannot ask the real Azure to return a 429 on demand, or to die exactly halfway
through a stream — and untested resilience is decorative.

---

## Things you can watch happen

```bash
python gated_demo.py               # tenant isolation, a live cross-tenant leak, refusal
python -m app.retrieval.calibrate  # where the refusal threshold comes from
python hybrid_demo.py              # retrieval failures, with real embeddings
python -m app.retrieval.ann_bench  # the recall you sell for latency, measured
python chunk_demo.py               # why chunk size is not a number you pick
uvicorn app.main:app --reload      # then: curl localhost:8000/health
```

**`gated_demo.py`** — two tenants with near-identical contracts. Pre-filtering keeps
their candidate sets disjoint. Then the villain: `PostFilterRetriever` returns the
*correct* answer to Acme — and its query-keyed cache is holding four of Contoso's
chunks, because the cache was populated before the filter ran. The cache developer
did nothing wrong. **The vulnerability arrived the day post-filtering was chosen.**

**`calibrate.py`** — the module's real artifact, and the one that proved me wrong.
The textbook says "the answerable and unanswerable distributions overlap, so pick a
threshold from the tradeoff." Measured: **0 of 20 scores land in the 0.1–0.9 middle**.
It's bimodal, the tradeoff table is flat, and the threshold is not the interesting knob.
The actual defect is a false refusal *no threshold can fix* — see below.

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
    base.py          LLMClient + EmbeddingClient + RerankClient Protocols. The seam.
    stub.py          Fault-injecting fake provider (429, mid-stream death, hang)
    azure.py         Real Azure OpenAI clients (chat + embeddings)
    local.py         bge-small embedder + ms-marco reranker — the ONLY fastembed import
    factory.py       Three functions, one `if` each. The entire provider swap.
  ingest/
    loaders.py       bytes -> Sections (structure kept, page furniture stripped)
    chunker.py       Sections -> Chunks (tables atomic, context prepended, parents)
  retrieval/
    ann_bench.py     exact vs HNSW vs IVF-Flat vs IVF-PQ, measured
    hybrid.py        BM25 + dense, fused by RRF
    gated.py         pre-filter gates, cross-encoder rerank, the refusal path
    calibrate.py     where the threshold comes from + why a refusal happened
    corpus.py        2 tenants, 1 superseded doc — the fixture
tests/               executable proof of each claim — 39 tests
```

**The seam rule:** nothing under `app/llm/` imports FastAPI. Nothing outside it
imports `openai` — or `fastembed`, because embeddings and reranking are provider
calls too. The first draft got this wrong: chat crossed the seam while `hybrid.py`
imported its embedder directly, which quietly made "swap providers by config" a
false claim. `tests/test_seam.py` now walks the AST of every module (lazy imports
included) and fails the build on a violation. Retry, timeout, semaphore and
circuit breaker are properties of the *provider relationship*, not of an endpoint
— so they live in the client wrapper, and a route handler reads like a paragraph
of business logic.

That's not aesthetics. It's the answer to *"where does this live, and why does
that matter when we add a second provider next quarter?"* For embeddings, that
quarter already came: `EMBEDDING_PROVIDER=local|azure` is the whole swap.

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
| **Pre**-filter on `tenant_id`, never post-filter | `tenant_id` is a security boundary, not a relevance signal. Post-filtering returns the right answer *and* leaks — see `gated_demo.py`. A control that depends on the ordering of two function calls is not a control. |
| A refusal is a return value, not an exception | `Answer(refused=True, score=...)` — the caller can't forget to handle it, and the score is always reported. On a refusal the generator is **never called**: handed confident-looking irrelevant chunks, models answer anyway. |
| The seam is a **test**, not a convention | A rule that lives in a README gets violated by the author within the month — measured: it did. `test_seam.py` parses every module's AST, so a lazy `import fastembed` inside a helper function fails CI the same as a top-level one. |

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
| 3.4 | Metadata gates + the refusal path | ✅ `gated.py`, `calibrate.py` |

---

## The finding I didn't expect

The refusal gate has a **false refusal that no threshold can fix**, and finding it
is the most useful thing in this repo.

*"What is the cap on liability?"* is answerable — section 7 covers it. Retrieval
surfaces the right chunk. The cross-encoder **ranks it #1**. Then scores it
**0.009**, and the gate refuses.

`ms-marco-MiniLM` was trained on MS MARCO — web search passages. Contract prose
(*"Neither party's aggregate liability will exceed the fees paid in the twelve
months preceding the claim"*) is out-of-distribution. So:

> **A cross-encoder's ranking can be trustworthy while its calibration is not.**
> The refusal gate depends on the calibration, not the ranking — so an
> out-of-domain reranker breaks refusal *even when retrieval is perfect*.

Lowering the threshold can't help: to admit a 0.009 you must admit everything.
The fix is a domain-suitable reranker. `calibrate.py` decomposes every false
refusal into **retrieval failure** vs **calibration failure**, because they have
completely different fixes and "2 false refusals" tells you neither.
Current measurement: **0 retrieval failures, 2 calibration failures.**

`test_gated.py::test_reranker_ranks_correctly_but_scores_a_lie` asserts the defect
on purpose. If a better reranker fixes it, that test fails loudly — which is
exactly the signal you want.

---

## Notes

Built while working through Ripostiq's *Ship It: Python, FastAPI & Azure OpenAI
RAG Agents*. The tests are the interesting part — several exist because a first
draft **passed while the code it covered was deleted**, which is the most
dangerous kind of green. See `test_streaming.py::test_disconnect_cancels_the_upstream_call`
and `test_hybrid.py::test_rrf_costs_you_the_top_slot_but_keeps_the_answer_in_the_pool`.

Both gates in `gated.py` are sabotage-verified: delete the `tenant_id` check and
three tests go red; delete the `status == "active"` check and one does.

Azure setup (resource vs deployment, TPM/RPM quota, the admission-time token
reservation that causes 429s at 40% utilisation): [AZURE_SETUP.md](AZURE_SETUP.md).

MIT licensed.
