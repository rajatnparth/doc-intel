# doc-intel

The Module 1 lab for *Ship It: Python, FastAPI & Azure OpenAI RAG Agents*.

A multi-tenant document intelligence API: upload contracts and invoices, ask
questions, get **cited** answers, and let an agent perform bounded actions.

Module 1 builds only the **service boundary** — the part you have to defend when
a principal engineer asks what happens to your event loop.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # LLM_PROVIDER=stub needs nothing else

pytest -q                     # 8 tests, no network, no Azure
uvicorn app.main:app --reload
curl localhost:8000/health
```

No Azure account required. `LLM_PROVIDER=stub` gives you a fake provider that
streams tokens and fails on command (429, mid-stream death, hang, bad JSON).
That's not a shortcut — you cannot ask the real Azure to 429 you on demand, and
untested resilience is decorative. See `AZURE_SETUP.md` for the real thing.

---

## Layout

```
app/
  config.py          Settings from env, validated at BOOT not at 3am
  schemas.py         The two contracts (Gate 1: clients. Gate 2: the model.)
  main.py            FastAPI app, DI, error envelope, routes
  llm/
    base.py          LLMClient Protocol + the error taxonomy. The seam.
    stub.py          Fault-injecting fake provider
    azure.py         Real AsyncAzureOpenAI client
    factory.py       One `if`. The entire provider swap.
  ingest/            (Module 3.1) document -> chunks worth embedding
    loaders.py       bytes -> Sections (structure preserved, furniture stripped)
    chunker.py       Sections -> Chunks (structure-aware, tables atomic, context, parents)
  retrieval/         (Module 3.2, 3.3) query -> nearest chunks
    ann_bench.py     run: python -m app.retrieval.ann_bench
                     exact vs HNSW (efSearch) vs IVF-Flat/IVF-PQ (nprobe + compression),
                     measuring recall@10 / p95 latency / bytes-per-vector
    hybrid.py        BM25 + REAL dense embeddings, fused by RRF (ranks, never scores)
chunk_demo.py        run this: watch naive coin-flip vs structure-aware invariance
hybrid_demo.py       run this: watch dense return the WRONG invoice, BM25 nail it,
                     and RRF rescue — with real bge-small-en-v1.5 embeddings
sample_docs/         acme_msa.md — a contract with prose + a troubleshooting table
tests/
  test_scaffold.py   Executable proof of section 1.2's claims
  test_chunking.py   Executable proof of section 3.1's claims
```

**Conventions:** every import is tagged with its origin (`# stdlib`,
`# 3rd-party: <pkg>`, or `# local — <path>`) so you always know what's free,
what's a dependency, and what's ours. Full list in `CONVENTIONS.md`.

**The seam rule:** nothing under `app/llm/` imports FastAPI. Nothing outside it
imports `openai`. Retry, timeout, semaphore and circuit breaker are properties
of the *provider relationship*, not of an endpoint — so they live in the client
wrapper. Route handlers should read like a paragraph of business logic.

That rule is not aesthetics. It is the answer to *"where does this live, and why
does that matter when we add a second provider next quarter?"*

---

## Build status

| Section | Concept | Status |
|---|---|---|
| 1.1 | Async is occupancy, not speed | ✅ `AsyncAzureOpenAI`, async generators, no `time.sleep` anywhere |
| 1.2 | Pydantic contracts at both boundaries | ✅ `schemas.py`, `extra="forbid"`, `strict=True`, error envelope |
| 1.3 | Streaming without lying about it | ✅ `POST /v1/chat/stream` — `sse.py`, discriminated frames, `[DONE]`, disconnect → upstream cancel |
| 1.4 | Resilience at the boundary | ⬜ semaphore, jittered retry, breaker — see TODOs in `azure.py` |
| lab | `loadtest.py` | ⬜ 30 concurrent + faults → p50/p99, 429s absorbed, requests shed, tokens billed |

---

## Things in here that are load-bearing, not decoration

- `Usage` is a first-class type, because *"what does one request cost you?"* is
  the follow-up question roughly every time.
- `InvoiceExtract.currency` is a `Literal`, not a `str` — **type validity is not
  semantic validity**. A `str` accepts `"₹"`.
- `extract()` returns a raw `str`, never a `dict`. Structured outputs guarantee
  **shape, never truth**. Validation is a deterministic gate that runs *after*
  the model, in code the model cannot influence.
- `/health` does not call the LLM. A health check that hits your provider turns
  their blip into your outage.
- `_translate()` in `azure.py` is where every `openai.*` exception dies. If one
  reaches a route handler, the seam has leaked.
