# doc-intel

[![CI](https://github.com/rajatnparth/doc-intel/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatnparth/doc-intel/actions/workflows/ci.yml)

**A multi-tenant document intelligence API** — upload policy documents and
claims records, ask questions, get cited answers. The demo corpus is motor
insurance (an invented insurer, two policyholders, an effective-dated prior
policy year);
the engine is domain-agnostic. Built to be *defended*, not demoed: every design
decision here has a failure mode attached, and most of them have a test that
fails when you remove the fix.

Python · FastAPI · async · Azure OpenAI · embeddings · hybrid vector search · RAG

```bash
git clone https://github.com/rajatnparth/doc-intel && cd doc-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # defaults to a stub provider — no Azure key needed
python -c "import secrets; print(f'AUTH_JWT_SECRET={secrets.token_hex(32)}')" >> .env
                                # the API has NO default secret and refuses to
                                # boot without one — so you generate a real one
pytest -q                       # 134 tests + a CI-gated eval scorecard. First run downloads ~210MB of local
                                # models (embedder + cross-encoder); after that,
                                # no network. (Tests mint their own ephemeral
                                # secret — the suite depends on no fixed value.)
```

Optional — persist the index instead of re-embedding at every boot:

```bash
echo "VECTOR_STORE=qdrant" >> .env
python -m app.ingest.index      # embed once -> var/qdrant (run it twice: the
                                # count doesn't change — upserts are idempotent)
uvicorn app.main:app            # boots read-only; fails closed if you skipped
                                # the ingest (an empty index would look exactly
                                # like "every question refused")
```

**No API key required to run any of it.** `LLM_PROVIDER=stub` swaps in a fake
provider that streams tokens and fails on command. That isn't a shortcut: you
cannot ask the real Azure to return a 429 on demand, or to die exactly halfway
through a stream — and untested resilience is decorative.

---

## Things you can watch happen

```bash
python loadtest.py                 # 30 concurrent + 429s: what the boundary buys, measured
python gated_demo.py               # tenant isolation, a live cross-tenant leak, refusal
python -m app.retrieval.calibrate  # where the refusal threshold comes from
python hybrid_demo.py              # retrieval failures, with real embeddings
python -m app.retrieval.ann_bench  # the recall you sell for latency, measured
python chunk_demo.py               # why chunk size is not a number you pick
uvicorn app.main:app --reload      # then: curl localhost:8000/health
```

And the loop itself, end to end (stub provider — no Azure key needed; the
bearer token is minted locally with the dev secret):

```bash
TOKEN=$(python -m app.auth --tenant asha --groups customer)
curl -N localhost:8000/v1/ask -X POST \
  -H 'content-type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "how quickly must I report an accident?"}'   # wording -> RAG
curl -N localhost:8000/v1/ask -X POST \
  -H 'content-type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "what is my excess?"}'                        # value -> the record, 0 tokens
```

**`gated_demo.py`** — two policyholders on the same motor product. Pre-filtering
keeps their candidate sets disjoint. Then the villain: `PostFilterRetriever` returns
the *correct* answer to Asha — and its query-keyed cache is holding Vikram's policy
chunks, his ₹5,000 excess included, because the cache was populated before the
filter ran. The cache developer did nothing wrong. **The vulnerability arrived the
day post-filtering was chosen.** Then the *time gate*: the same excess question
answered ₹2,000 as of today and ₹1,000 as of a December date of loss — same
customer, two dates, two answers, both correct, because a claim is assessed
under the wording in force when the accident happened. Closes with the
*phrasing cliff*: the same liability question answered at 0.9987 or refused at
0.0006, depending on whether you use the document's own words.

**`calibrate.py`** — the module's real artifact, and it keeps proving the textbook
wrong. The threshold sweep is **flat from 0.10 to 0.75**: 1 false answer + 2 false
refusals at every setting, because the errors sit on *opposite wrong sides* of any
threshold. Moving the knob trades nothing; the reranker is the knob — see below.

**`hybrid_demo.py`** — ask for claim `CLM-2026-0891` among near-identical siblings:
dense "wins" by a margin of **0.012** — luck, not signal (BM25's margin is
structural: no other chunk contains the token). Then ask *"how long do we have to
settle the premium?"* and BM25 confidently puts **6. Personal Data** first —
"settle" appears exactly once in the corpus, in *"settle a claim"*, and rare
tokens score big. Dense reads the meaning; RRF fuses both by **rank** (never score
— BM25 is unbounded and corpus-dependent). Real `bge-small-en-v1.5` embeddings,
nothing staged.

**`ann_bench.py`** — exact k-NN as ground truth, then HNSW sweeping `efSearch`
(recall 0.585 → 0.999, latency rising with it) and IVF-Flat vs IVF-PQ. The gap
between those last two isolates the *product-quantisation tax*: IVF-Flat recovers
to recall 1.000 as you probe more clusters; IVF-PQ plateaus at 0.595 forever,
because compression is a second loss `nprobe` can't undo — for 16× less RAM.

**`chunk_demo.py`** — naive fixed-size chunking keeps a table row with its header
at sizes 700 and 800, and orphans it at 300/400/500/600/1000. It isn't reliably
bad, it's *arbitrary* — which is worse, because you can't reason about it.
Structure-aware chunking is invariant at every size.

---

## Layout

```
app/
  config.py          Settings from env, validated at BOOT — not at 3am
  schemas.py         The two contracts (Gate 1: clients. Gate 2: the model.)
  sse.py             The streaming protocol: discriminated frames + [DONE]
  auth.py            JWT -> Principal. The only request-path place one is built.
  router.py          Numbers vs wording: tier 1 deterministic, tier 2 an LLM verdict behind Gate 2.
  policy_admin.py    The system of record (Protocol + stub). Numbers live HERE.
  rag.py             The context budget: parents deduped, chars capped, [n] cited
  safety.py          PII -> typed placeholders BEFORE storage. Dispute refs survive.
  ops.py             /metrics counters (refusal rate is the star) + audit health.
  audit.py           One record per exchange: what was asked, retrieved, scored, DELIVERED
  handoff.py         Refusal -> ticket. References the audit record; copies nothing.
  main.py            FastAPI app, DI, error envelope, routes (incl. /v1/ask)
  llm/
    base.py          LLMClient + EmbeddingClient + RerankClient Protocols. The seam.
    stub.py          Fault-injecting fake provider (429, mid-stream death, hang)
    azure.py         Real Azure OpenAI clients (chat + embeddings)
    local.py         bge-small embedder + ms-marco reranker — the ONLY fastembed import
    resilience.py    breaker -> semaphore -> retry+jitter. Provider-agnostic.
    factory.py       Three functions, one `if` each. The entire provider swap.
  ingest/
    loaders.py       bytes -> Sections (structure kept, page furniture stripped)
    chunker.py       Sections -> Chunks (tables atomic, context prepended, parents)
    index.py         chunk -> embed -> upsert. Run once, not at every boot.
  store/
    base.py          VectorStore Protocol + Gate. The third seam: storage.
    memory.py        embeds at boot, dies with the process (the default)
    qdrant.py        persistent; filter runs INSIDE the search — the ONLY qdrant import
    factory.py       one `if`. local folder -> server -> cloud is config, not code.
  retrieval/
    ann_bench.py     exact vs HNSW vs IVF-Flat vs IVF-PQ, measured
    hybrid.py        BM25 + dense, fused by RRF
    gated.py         pre-filter gates, cross-encoder rerank, the refusal path
    calibrate.py     where the threshold comes from + why a refusal happened
    corpus.py        2 policyholders, 1 effective-dated prior-year kit — the fixture
evals/               labelled cases + scorecard + measured baseline — the CI ratchet
tests/               executable proof of each claim — 134 tests
ui/                  reference client (React SPA): renders the SSE contract —
                     facts vs cited answers vs refusals. See ui/README.md.
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
| **Numbers never come from RAG** | `calibrate.py` measured why: "what will next year's renewal premium be?" scored 0.7785 against a section that merely *discusses* premiums — topicality isn't truth, and the number isn't in the corpus at all. Value questions route to the policy-admin record: exact, 0 tokens, and the LLM is provably never called (`ExplodingLLM` in the tests). The refusal wasn't a dead end either: the NCB question RAG correctly refused is answered by the record. |
| The router is **tiered**: deterministic first, LLM second | Tier 1 is free, instant, and explainable to a regulator ("why did the bot answer from the record?" greps); it needs a fact-noun AND a value shape. Tier 2 catches what keywords can't: *"how much do I pay from my own pocket when I claim?"* names no fact noun but IS an excess question — and unrouted it would get its number **from prose**, right today and stale the day an endorsement changes the record. The paraphrase gap was a hole in numbers-never-from-RAG itself. |
| Tier 2's verdict is **untrusted input** | The LLM's routing decision goes through the same Gate-2 discipline as invoice JSON: strict Pydantic against a CLOSED decision space (`route: Literal`, `field: Literal`). That closure is also the injection containment — a hostile question can at worst flip which subsystem answers; it cannot name a tenant or invent a field. Every failure (LLMError, junk, "wording") falls to RAG, where the refusal gate stands. With the stub provider tier 2 is inert *by construction*: canned output fails validation. Known wart, still tested: tier-1 false positives short-circuit past tier 2. |
| Effective **windows**, not a status flag | "Active" asks the wrong question. Whether a wording applies is relative to a date — and not today's: a claim is assessed under the wording in force on the **date of loss**. A flag cannot represent that question; `effective_from/to` + `as_of` answer it, and "superseded" becomes a derived fact nobody has to remember to flip. |
| `as_of` rides in the request body — and `tenant_id` may not | The contrast IS the rule: `tenant_id` *expands* what you may see, so it must arrive signed (JWT). `as_of` only *selects among versions you already own* — a time cursor inside your authorization scope. Which knobs need a signature is a per-knob decision. |
| A refusal is a return value, not an exception | `Answer(refused=True, score=...)` — the caller can't forget to handle it, and the score is always reported. On a refusal the generator is **never called**: handed confident-looking irrelevant chunks, models answer anyway. |
| The seam is a **test**, not a convention | A rule that lives in a README gets violated by the author within the month — measured: it did. `test_seam.py` parses every module's AST, so a lazy `import fastembed` inside a helper function fails CI the same as a top-level one. |
| The gate is an **argument**, not a cache key | `store.search(vector, gate, k)` — tenant, ACL and as-of travel WITH the query and filter inside the ANN traversal. The old per-principal view cache was correct only while its key listed every predicate input (the `as_of` time-leak had to be test-pinned). A parameter can't be forgotten; a cache key can. Both stores pass the same parametrised gate tests — delete one clause from the Qdrant filter and two go red. |
| The audit record is written when the stream **ends** | Its most important field is what the customer actually SAW — unknowable until the last token. A disconnect after 40 tokens produces a record saying exactly that. Audit is not logging: different reader (a dispute handler, months later), different artifact. |
| A handoff ticket **references** the exchange, never copies it | `POST /v1/handoff {request_id}` — the agent reads the audit record: what the customer saw plus what the system retrieved and scored. The lookup is tenant-gated and 404s identically for foreign and nonexistent ids (a 403 would confirm existence). Sabotage-verified: drop the tenant check and the cross-tenant test goes red. |
| The eval baseline is **measured**, not aspired to | `evals/baseline.json` records what the pipeline actually does — 2 false refusals and 1 wording-layer false answer included, because they're real (calibrate.py). The CI gate is a ratchet against WORSE; a gate demanding zero fails on day one and is ignored by day three. All 117 behavior tests stayed green through a major embedder upgrade — the eval is the only thing that would have noticed a ranking regression. |
| **Containment** beats injection detection | A detector must win every adversarial round; containment bounds the blast radius when it loses. Gates, `as_of`, the refusal threshold, the record lookup and routing are deterministic code the model never controls — a hostile document can be retrieved and cited, but it cannot cross tenants, flip a route, or pick another customer's record (`test_safety.py` asserts each). What a real model would *obey* inside extracts is a model property — that requires real-model adversarial evals, and this repo says so instead of pretending a stub can test it. |
| PII becomes **typed placeholders** before storage | `[EMAIL]` / `[PHONE]` / `[VEHICLE-REG]` in audit records and handoff notes — the dispute keeps its narrative, the identifier is never on disk, and identity is already present as the *verified* `tenant_id`. The patterns are deliberately narrow: policy/claim/damage-code references survive, because redacting the numbers the dispute is ABOUT defeats the audit. Default ON — storing identifiers is what needs the documented reason. |
| Liveness and readiness are **different questions** | `/health` = "alive?" → restart; it checks no dependencies, ever — a store blip must not restart-loop healthy pods. `/ready` = "traffic?" → rotation; it checks the store AND actively probes the audit sink. No `tenant_id` metric labels: every label value is a time series stored forever, and it would re-leak what phase 8 minimized. |
| "No record, no answer" recovers via the **probe**, not traffic | Strict admission refuses at the door (503 + Retry-After) while the sink is failing — and that created a deadlock the tests caught, not foresight: with all exchanges blocked, no write remained to discover the disk came back. So `/ready`'s active probe is the retry loop — the orchestrator's own polling flips the flag and rotation back in is automatic. One exchange always slips through un-audited (the one that discovers the failure); pre-writing a fake record to prevent it would defeat the record's defining field: what was DELIVERED. |
| `AUTH_JWT_SECRET` has **no default** — boot fails without it | A service that *can* start in an unsafe state *will* be run in an unsafe state. A boot warning is a log line you grep for after the incident; a boot failure is a deploy that never went out wrong. The only default secret is no secret. |

---

## Status

| | Concept | |
|---|---|---|
| 1.1 | Async is occupancy, not speed | ✅ |
| 1.2 | Pydantic contracts at both boundaries | ✅ |
| 1.3 | Streaming without lying about it | ✅ `POST /v1/chat/stream` |
| 1.4 | Resilience: semaphore, jittered retry, breaker | ✅ `llm/resilience.py`, `loadtest.py` |
| 3.1 | Chunking as information architecture | ✅ |
| 3.2 | ANN indexes: recall/latency/memory | ✅ |
| 3.3 | Hybrid search + RRF | ✅ |
| 3.4 | Metadata gates + the refusal path | ✅ `gated.py`, `calibrate.py` |
| P0 | Domain conversion: motor insurance corpus | ✅ `sample_docs/` |
| P1 | `/v1/ask` — retrieval meets generation | ✅ `rag.py`, `test_ask.py` |
| P2 | Identity: JWT claims → Principal | ✅ `auth.py`, `test_auth.py` |
| P3 | Effective-dated version gate (`as_of`) | ✅ date-of-loss retrieval |
| P4 | Numbers-vs-wording router + system of record | ✅ `router.py`, `policy_admin.py` |
| P5 | Vector persistence — the gate travels with the query | ✅ `store/`, `test_store.py` |
| P6 | Audit trail + human handoff | ✅ `audit.py`, `handoff.py`, `POST /v1/handoff` |
| P7 | Evals in CI — quality regressions fail the build | ✅ `evals/`, `test_evals.py` |
| P8 | Injection containment + PII redaction | ✅ `safety.py`, `test_safety.py` |
| P9 | Ops: /metrics, /ready, strict audit admission | ✅ `ops.py`, `test_ops.py` |

---

## /v1/ask — the loop, wired

One route, two pipelines: a deterministic router sends **value questions to the
system of record** and **wording questions through RAG** (gates → hybrid →
rerank → refuse or cite + generate). A fact answer is one frame, no model:

```
data: {"type":"facts","policy_number":"MTR-2026-1147","facts":[{"name":"Own damage excess","value":"₹2,000"}],"source":"policy_admin"}
data: {"type":"done","usage":null}
data: [DONE]
```

The `source` field is not decoration — a client (and an auditor) must be able
to tell a record lookup from a generated sentence at a glance. Routing is
tiered: a deterministic classifier takes the explicit phrasings for free, and
an LLM classifier (through the same `extract()` seam, validated like any other
model output) catches paraphrases like *"how much do I pay from my own pocket
when I claim?"*. A wording question looks like this instead:

```
data: {"type":"sources","sources":[{"n":1,"doc_title":"Asha Rao — Motor Policy Kit (2026)","heading":"4. Claims Process"}]}
data: {"type":"token","text":"Accidents "}
data: {"type":"token","text":"must "}
...
data: {"type":"done","usage":{"prompt_tokens":412,"completion_tokens":58}}
data: [DONE]
```

A refused one replaces all of that with a single `refusal` frame carrying the
score, the reason, and near-misses as links. Four decisions worth defending:

- **Sources stream before the first token.** They're known the moment retrieval
  ends — they come from the retriever, not the model's mouth. The client renders
  the citations panel while the model is still thinking, and provenance stays
  honest.
- **On a refusal the generator is never called** — and that's a claim about a
  call *not happening*, so `test_ask.py` proves it with a counting fake, and the
  test goes red if the short-circuit is deleted (sabotage-verified).
- **Retrieval runs in a threadpool.** Embedding + cross-encoding are CPU-bound;
  inline in an `async def` they block the event loop and stall every other live
  stream. Async buys occupancy only if the loop stays free — section 1.1's
  lesson, biting from the other side.
- **The prompt is a budget, not a template** (`rag.py`): parents deduped (two
  chunks from one section must not spend the budget twice), capped in chars,
  question last. Nothing in `rag.py` imports FastAPI, openai, or a tokenizer.

**Identity is a property of the transport, not the payload.** The principal is
built in exactly one request-path place — `auth.py`, from *verified* JWT claims
(signature, mandatory `exp`, audience — each check blocks a named attack, each
attack has a test). Phase 1's `tenant_id` body field didn't just stop working;
with `extra="forbid"` it now 422s, so no client keeps believing it controls its
tenant. Sabotage note: flipping `verify_signature: False` silently disabled
three defenses at once (PyJWT couples them) — and three tests went red, which
is why forged, expired, and cross-audience tokens each have their own.

**There is no default secret.** `AUTH_JWT_SECRET` has no fallback value:
`validate_for_serving()` refuses to boot the API without one — fail closed at
boot, with the fix in the error message, not at 3am when `jwt.decode(token,
None)` verifies nothing. Tests generate an ephemeral secret per run, so the
suite provably depends on no fixed value. `/v1/chat/stream` is authenticated
too: it has no tenancy, but an unmetered passthrough to a paid model is a cost
hole, and "who spent this?" needs an answer on every request. The enterprise
swap (IdP-signed RS256 + JWKS) lands entirely inside `get_principal`.

---

## The semaphore made throughput *worse*, and that's the interesting part

`loadtest.py`, 30 concurrent requests against a 429-injecting provider:

| config | ok | 429 | shed | p99 ms | calls that reached the provider |
|---|---|---|---|---|---|
| no controls | 0 | 30 | 0 | 56 | 30 |
| retry only | **30** | 0 | 0 | 160 | **60** |
| retry + semaphore | 8 | 8 | 14 | 537 | 48 |

Retry-only serves everyone — by putting **60 calls** into a provider that is
already saying *"over quota"*. Adding the semaphore served **fewer**. Two reasons,
both measured, neither a bug:

1. **A retry holds its slot while it sleeps.** A request backing off for 0.4s
   occupies its slot for the full 0.4s (`test_a_retry_holds_its_semaphore_slot_while_backing_off`).
   That's correct — the cap counts calls that *will* hit the provider, and a
   backing-off request certainly will. Release the slot and the cap is a lie:
   100 requests could wake at once. So **concurrency and retry interact
   multiplicatively**: `cap × (1 + retries × backoff)` bounds throughput.
2. **Under sustained overload you cannot serve everyone.** The real choice is
   *everyone waits and some time out* vs *some are served and the rest are told
   "come back in 2s" in half a second*. That's the trade, stated.

Related: `CapacityShed` exists because the first loadtest reported the provider's
429s as "shed" — a metric that says *"raise your concurrency cap"* when the answer
is *"the provider is out of quota"*. **A metric that can't tell you which system
said no is not a metric.**

---

## The finding I didn't expect

The refusal gate has a **false refusal that no threshold can fix** — and since the
motor-insurance conversion, a matching **false answer** on the other side. Finding
the pair is the most useful thing in this repo.

*"Is there an upper limit on what a claim pays out?"* is answerable — section 7
covers it. Retrieval surfaces the right chunk. The cross-encoder **ranks it #1**.
Then scores it **0.0006**, and the gate refuses.

Ask the same section the same thing *in its own words* — "what is the limit of
liability?" — and the same chunk scores **0.9987**.

> **The score doesn't measure whether the chunk answers the question. It measures
> whether the question uses the document's vocabulary — and customers never use
> the document's vocabulary.**

The mirror image: *"what will next year's renewal premium be?"* scores **0.7785**.
Section 2 is *about* the renewal premium, so topical proximity scores high — but
the amount isn't in the corpus. A false answer, sitting on the *opposite wrong
side* of every threshold from the false refusals: the sweep table is flat at
1 + 2 from 0.10 to 0.75. Moving the knob trades nothing.

`ms-marco-MiniLM` was trained on MS MARCO — web search passages. Policy wording
(*"aggregate liability shall not exceed the Insured's Declared Value"*) is
out-of-distribution — and so are **tables**: *"what should I do if the car is not
driveable?"* scores 0.0010 even though "Vehicle not driveable" appears verbatim
in the damage-codes table. So:

> **A cross-encoder's ranking can be trustworthy while its calibration is not.**
> The refusal gate depends on the calibration, not the ranking — so an
> out-of-domain reranker breaks refusal *even when retrieval is perfect*.

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
and `test_hybrid.py::test_rrf_keeps_the_answer_in_the_pool` — whose own story had
to change during the domain conversion: the RRF demotion measured on the old
corpus was partly fed by `bm25_search` returning **zero-score chunks in index
order**, which fusion then paid real credit. The artifact is fixed (a chunk that
matches no query term is not a result), the demotion on this corpus now measures
zero, and the invariant the test asserts is the one that survives both corpora:
fusion must never *lose* the answer.

Both gates in `gated.py` are sabotage-verified against this corpus: delete the
`tenant_id` check and three tests go red; delete the effective-window check and
five do — including the cache-key test, because when the date joined the
predicate it had to join the per-view cache key in the same commit (leave it
out and a December query is served January's cached view: the post-filter
cache leak again, across time instead of across tenants).

Azure setup (resource vs deployment, TPM/RPM quota, the admission-time token
reservation that causes 429s at 40% utilisation): [AZURE_SETUP.md](AZURE_SETUP.md).

**Every document in `sample_docs/` is invented** — the insurer, both customers,
every policy number, amount and claim. Structure modelled on publicly available
motor policy wordings; content written for this repo. This is a demonstration
system, not an insurance product and not insurance advice.

MIT licensed.
