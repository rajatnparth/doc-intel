# Getting an Azure OpenAI deployment

You do **not** need this to do the lab — `LLM_PROVIDER=stub` runs everything,
including the load test, locally. Read this anyway: the vocabulary here (resource
vs deployment, TPM vs RPM, api-version) is *directly* interview material, and
"what does a request cost you?" is asked nearly every time.

> ⚠️ This information changes. Azure moves fast. Treat the steps as a map, verify
> against the current portal, and check current pricing before you spend anything.

---

## The mental model first

Three nested things, and people conflate them constantly:

```
Azure subscription
└── Azure OpenAI RESOURCE          ← lives in a REGION, has an endpoint + keys
    └── DEPLOYMENT                 ← a NAME YOU CHOOSE, pointing at a model
        └── model (gpt-4o, text-embedding-3-small, …)
```

**The single most common first-time error:** the `model=` parameter in the SDK
does not take a model name. It takes **your deployment name**.

```python
# If you named your deployment "prod-chat" in the portal:
await client.chat.completions.create(model="prod-chat", ...)   # ✅
await client.chat.completions.create(model="gpt-4o", ...)      # ❌ 404 DeploymentNotFound
```

The `model=` kwarg is a lie inherited from the OpenAI SDK. Say this out loud in
an interview and you've demonstrated you've actually deployed something.

---

## Steps

### 1. Get access
Azure OpenAI historically required an access request per subscription. This has
loosened over time — as of writing, most subscriptions can create the resource
directly. If you hit a gate, the portal tells you.

You need an Azure subscription. A free trial gives you credit; **Azure OpenAI is
not free**, and there is no free tier for the models themselves.

### 2. Create the resource
Portal → *Create a resource* → search **Azure OpenAI** → Create.

- **Region** matters enormously. Model availability and quota differ per region.
  If `gpt-4o` isn't offered in your region's dropdown at deployment time, that's
  why. Check the model-availability table in Microsoft's docs before choosing.
- **Pricing tier**: Standard S0.

Note the **resource name**. Your endpoint is `https://<resource-name>.openai.azure.com/`.

### 3. Deploy a model
Go to **Azure AI Foundry** (formerly Azure OpenAI Studio) → *Deployments* →
*Create new deployment*.

| Field | What to pick | Why |
|---|---|---|
| Model | `gpt-4o` or `gpt-4o-mini` | mini is ~20x cheaper; fine for this lab |
| Deployment name | `gpt-4o` (match the model, to save yourself pain) | this is what goes in `model=` |
| Deployment type | Standard (or Global Standard) | Provisioned = you pre-buy capacity |
| Tokens per Minute | start at 10K–30K | ← **this is your quota. See below.** |

Do the same for `text-embedding-3-small` — you'll need it in Module 2.

### 4. Get endpoint + key
Resource → *Keys and Endpoint*. Copy **KEY 1** and the **Endpoint**.

```bash
cp .env.example .env
# then fill in:
#   LLM_PROVIDER=azure
#   AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
#   AZURE_OPENAI_API_KEY=<key 1>
#   AZURE_OPENAI_CHAT_DEPLOYMENT=<your deployment name>
```

`.env` is gitignored. Keys in git is a resume-generating event.

### 5. Pick an api-version
`AZURE_OPENAI_API_VERSION` is a **date string** that pins the shape of the REST
contract — it is versioning-in-the-path, exactly as in section 1.2, applied to
you as the client.

- Structured outputs (`response_format={"type": "json_schema"}`) need
  **`2024-08-01-preview`** or later.
- `2024-10-21` is a stable GA version that supports it.

Pin it. Never leave it floating. A silently-changed api-version is a silently-
changed contract.

---

## Quota — the part that shows up in the interview

Your deployment has two budgets, refreshed every minute:

- **TPM** — tokens per minute
- **RPM** — requests per minute (Azure derives this from TPM, roughly 6 RPM per 1K TPM)

Exceed either → **HTTP 429**, with a `Retry-After` header.

**The insight worth stating unprompted:**

> Azure charges quota at **admission time**, against `prompt_tokens + max_tokens`
> — not against the tokens actually generated.

So if you set `max_tokens=4096` on every call, but your answers average 180 tokens,
each request *reserves* 4096 tokens of your TPM and gives back nothing. You will
eat 429s while your billing dashboard shows you at 40% real utilisation.

**The fix costs nothing and needs no quota increase:** set `max_tokens` honestly,
per route. A summarisation route and a classification route should not share a
number.

```python
# app/llm/azure.py — this parameter is a quota decision, not a length cap.
max_tokens=max_tokens,
```

Raising quota: Azure AI Foundry → *Quotas* → request an increase. It is per
region, per model, per subscription.

---

## Cost sanity

Before you run the load test against real Azure, know the shape of the bill.
Pricing changes; **check the current Azure OpenAI pricing page** rather than
trusting a number in a markdown file. But the structure is always:

```
cost = (prompt_tokens × input_rate) + (completion_tokens × output_rate)
```

- Output tokens cost several times more than input tokens.
- Embeddings are ~2 orders of magnitude cheaper than chat. This is why Module 2
  spends time on embedding economics — it changes what architectures are viable.
- `gpt-4o-mini` is dramatically cheaper than `gpt-4o`. Use it for the lab.

This is why `Usage` is a first-class type in `app/llm/base.py` and why the
streaming path passes `stream_options={"include_usage": True}`. Without it,
streaming responses carry **no usage block at all**, and you cannot bill, cannot
alert on cost, and cannot answer the follow-up question.

---

## Verify it works

```bash
LLM_PROVIDER=azure python -c "
import asyncio
from app.config import get_settings
from app.llm.factory import build_llm_client

async def main():
    s = get_settings(); s.validate_for_provider()
    c = build_llm_client(s)
    async for chunk in c.stream_chat('Say hello in five words.', max_tokens=20):
        if chunk.usage: print('\n[usage]', chunk.usage)
        else: print(chunk.text, end='', flush=True)
    await c.aclose()

asyncio.run(main())
"
```

If you see `DeploymentNotFound`, re-read step 3: you passed a model name where a
deployment name belongs.

If you see 401, the key is for a different resource than the endpoint.

If tokens arrive in one lump rather than trickling — that's section 1.3, and it
is almost certainly not your Python.
