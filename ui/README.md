# doc-intel reference client

A deliberately small React SPA that is a **client of the engine's contract** —
it renders the SSE frame protocol (`app/sse.py`) exactly as a product client
would have to: streamed tokens with `[n]` citations, `facts` frames with their
system-of-record badge, refusals as first-class outcomes with near-misses, and
in-band errors after the `200 OK` was already spent.

Not the product UI. The branded, channel-integrated product client lives on
the private side of the open-core split; this one exists so anyone cloning the
repo can *see* the engine behave — and so the wire contract has a second,
independently written implementation (TypeScript discriminated unions
mirroring pydantic's `Literal` discriminators; the frame `switch` is
exhaustive, so a new server frame type fails to compile here).

## Run it

```bash
# 1. engine up (repo root; see main README for .env setup)
uvicorn app.main:app

# 2. mint a dev token (operator tool — same signing secret the API verifies)
python -m app.auth --tenant asha --groups customer

# 3. client up
cd ui && npm install && npm run dev
```

Open http://localhost:5173, paste the token, ask. The Vite dev server proxies
`/v1` to `localhost:8000`, so there is no CORS configuration to add (and none
shipped by accident).

Things worth trying (each verified against the running stack):

- *"what is my excess for an own damage claim?"* — the router catches the
  VALUE question: a `facts` card from the system of record. **No tokens
  stream and no usage is reported**, because no model was called.
- *"what documents do I need to submit for an own damage claim?"* — a
  WORDING question: sources arrive first (Claims Process ranked [1]), then
  the streamed answer, then token usage.
- *"does my policy cover veterinary bills for my dog?"* — an honest refusal:
  the reranker score, the threshold it missed, and near-misses as leads.
- the excess question **as of** `2025-12-20` — dated questions bypass the
  router (the record holds only the CURRENT term) and answer from the
  wording in force on the date of loss.
- press **Stop** mid-answer — the server observes the disconnect and cancels
  its upstream call (the tested stop-the-meter path)

## Notes

- The JWT lives in component state only — never localStorage, never a cookie.
  Refresh = re-paste. A reference client should model the safe default.
- All data is invented (Northwind Motor Insurance). Not insurance advice.
