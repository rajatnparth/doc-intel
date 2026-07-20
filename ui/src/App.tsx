/**
 * The reference client. One screen, three outcomes, honestly rendered:
 *
 *   facts    — from the system of record (badge says so; no model ran)
 *   answer   — streamed tokens + numbered citations from the sources frame
 *   refusal  — score, reason, near-misses as links. An OUTCOME, not an error.
 *
 * The JWT is pasted in and held in COMPONENT STATE only — not localStorage,
 * not a cookie. A reference client that quietly persists bearer tokens
 * teaches the wrong default; a page refresh costs one paste.
 */

import { useRef, useState } from "react";
import { askStream, ApiError } from "./api";
import { FactsCard, RefusalCard, SourcesPanel, StreamedAnswer } from "./components";
import type { FactsEvent, RefusalEvent, SourceRef, Usage } from "./types";

interface Exchange {
  id: number;
  question: string;
  asOf: string;
  text: string;
  sources: SourceRef[];
  facts: FactsEvent | null;
  refusal: RefusalEvent | null;
  error: { code: string; message: string; retryable: boolean } | null;
  usage: Usage | null;
  done: boolean;
}

let nextId = 1;

export default function App() {
  const [token, setToken] = useState("");
  const [question, setQuestion] = useState("");
  const [asOf, setAsOf] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const patch = (id: number, up: (e: Exchange) => Exchange) =>
    setExchanges((xs) => xs.map((e) => (e.id === id ? up(e) : e)));

  async function ask() {
    const q = question.trim();
    if (!q || !token.trim() || busy) return;

    const id = nextId++;
    setExchanges((xs) => [
      ...xs,
      { id, question: q, asOf, text: "", sources: [], facts: null, refusal: null, error: null, usage: null, done: false },
    ]);
    setQuestion("");
    setBusy(true);

    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const req = { question: q, ...(asOf ? { as_of: asOf } : {}) };
      for await (const frame of askStream(req, token.trim(), ac.signal)) {
        switch (frame.type) {
          case "token":
            patch(id, (e) => ({ ...e, text: e.text + frame.text }));
            break;
          case "sources":
            patch(id, (e) => ({ ...e, sources: frame.sources }));
            break;
          case "facts":
            patch(id, (e) => ({ ...e, facts: frame }));
            break;
          case "refusal":
            patch(id, (e) => ({ ...e, refusal: frame }));
            break;
          case "error":
            patch(id, (e) => ({ ...e, error: frame }));
            break;
          case "done":
            patch(id, (e) => ({ ...e, usage: frame.usage, done: true }));
            break;
          default: {
            // Exhaustiveness: a new server frame type fails to COMPILE here.
            const unknown: never = frame;
            patch(id, (e) => ({
              ...e,
              error: { code: "unknown_frame", message: `unrecognised frame: ${JSON.stringify(unknown)}`, retryable: false },
            }));
          }
        }
      }
    } catch (err) {
      if (!ac.signal.aborted) {
        const e =
          err instanceof ApiError
            ? { code: e2s(err.code), message: err.message, retryable: err.retryable }
            : { code: "network", message: String(err), retryable: true };
        patch(id, (x) => ({ ...x, error: e }));
      }
    } finally {
      patch(id, (e) => ({ ...e, done: true }));
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    // Aborting the fetch closes the connection; the server sees the
    // disconnect and cancels ITS upstream call. This button is the UI end of
    // test_disconnect_cancels_the_upstream_call.
    abortRef.current?.abort();
  }

  return (
    <div className="shell">
      <header>
        <h1>doc-intel</h1>
        <span className="sub">reference client — invented data, not insurance advice</span>
      </header>

      <section className="auth">
        <input
          type="password"
          placeholder="Paste a JWT — mint one: python -m app.auth --tenant asha --groups customer"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          aria-label="Bearer token"
        />
        <label className="asof">
          as of
          <input
            type="date"
            value={asOf}
            onChange={(e) => setAsOf(e.target.value)}
            title="Date of loss — retrieves the wording in force ON this date"
          />
        </label>
      </section>

      <main>
        {exchanges.length === 0 && (
          <p className="empty">
            Ask about the wording — <em>“what documents do I need to submit for a claim?”</em> —
            or a value question — <em>“what is my excess for an own damage claim?”</em> — and
            watch which pipeline answers.
          </p>
        )}
        {exchanges.map((e) => (
          <article key={e.id} className="exchange">
            <div className="q">
              {e.question}
              {e.asOf && <span className="pill">as of {e.asOf}</span>}
            </div>

            {e.facts && <FactsCard facts={e.facts} />}
            {e.refusal && <RefusalCard refusal={e.refusal} />}
            {(e.text || (!e.facts && !e.refusal && !e.error && !e.done)) && (
              <StreamedAnswer text={e.text} streaming={!e.done} />
            )}
            {e.sources.length > 0 && <SourcesPanel sources={e.sources} />}

            {e.error && (
              <div className="error">
                <strong>{e.error.code}</strong> — {e.error.message}
                {e.error.retryable && <span className="pill">retryable</span>}
              </div>
            )}
            {e.done && e.usage && (
              <div className="usage">
                {e.usage.prompt_tokens} prompt + {e.usage.completion_tokens} completion tokens
              </div>
            )}
          </article>
        ))}
      </main>

      <footer>
        <textarea
          rows={2}
          placeholder="Ask a question…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void ask();
            }
          }}
        />
        {busy ? (
          <button className="stop" onClick={stop} title="Closes the stream — the server cancels its upstream call">
            Stop
          </button>
        ) : (
          <button onClick={() => void ask()} disabled={!question.trim() || !token.trim()}>
            Ask
          </button>
        )}
      </footer>
    </div>
  );
}

function e2s(code: string): string {
  return code || "error";
}
