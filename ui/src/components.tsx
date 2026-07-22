/**
 * The three outcome renderings. Each one makes a server-side design decision
 * VISIBLE — that is the entire job of a reference client.
 */

import { useState } from "react";
import { uploadDocument, ApiError } from "./api";
import type { DocumentIngested, FactsEvent, HandoffResponse, RefusalEvent, SourceRef } from "./types";

/** Facts: provenance badge front and centre. The number came from the system
 * of record, not from prose — and no tokens streamed because no model ran. */
export function FactsCard({ facts }: { facts: FactsEvent }) {
  return (
    <div className="facts">
      <div className="facts-head">
        <span className="badge">system of record</span>
        <span className="policy">policy {facts.policy_number}</span>
      </div>
      <dl>
        {facts.facts.map((f) => (
          <div key={f.name}>
            <dt>{f.name}</dt>
            <dd>{f.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** A refusal is an outcome, not an error: neutral styling, the score that
 * drove it, the near-misses offered as leads — and phase 6's addition, the
 * way OUT: a handoff button, because a refusal must not be a dead end. The
 * ticket carries only the request_id; the human agent reads the audit
 * record and sees what the customer saw plus what the system scored. */
export function RefusalCard({
  refusal,
  ticket,
  onHandoff,
  handoffBusy,
}: {
  refusal: RefusalEvent;
  ticket: HandoffResponse | null;
  onHandoff: (() => void) | null;
  handoffBusy: boolean;
}) {
  return (
    <div className="refusal">
      <div className="refusal-head">
        <span className="badge subtle">refused</span>
        <span className="score">confidence {refusal.score.toFixed(3)}</span>
      </div>
      <p>{refusal.reason}</p>
      {refusal.near_misses.length > 0 && (
        <>
          <p className="nm-label">Closest sections — none scored as an answer:</p>
          <ul className="near-misses">
            {refusal.near_misses.map((s) => (
              <li key={`${s.doc_title}-${s.heading}`}>
                {s.doc_title} — {s.heading}
              </li>
            ))}
          </ul>
        </>
      )}
      {ticket ? (
        <div className="ticket">
          <span className="badge">{ticket.ticket_id}</span> A human will pick this
          up with the full context of this exchange.
        </div>
      ) : (
        onHandoff && (
          <button className="handoff" onClick={onHandoff} disabled={handoffBusy}>
            {handoffBusy ? "Creating ticket…" : "Talk to a human"}
          </button>
        )
      )}
    </div>
  );
}

/** Sources arrive BEFORE the first token (they came from the retriever, not
 * the model), so this panel renders while the answer is still streaming. */
export function SourcesPanel({ sources }: { sources: SourceRef[] }) {
  return (
    <div className="sources">
      {sources.map((s) => (
        <div key={s.n} className="source">
          <span className="cite">[{s.n}]</span> {s.doc_title} — {s.heading}
        </div>
      ))}
    </div>
  );
}

/** The ingestion panel: upload -> parse -> chunk -> embed -> upsert, with the
 * chunk count as the visible proof the embedding pipeline ran. Collapsed by
 * default — asking is the common act, ingesting is the back-office one.
 *
 * The role split is visible here too: this needs an `agent` token, and a 403
 * is explained rather than shown raw, because "customers never write the
 * corpus" is a design decision worth reading, not a mystery to debug. */
export function UploadPanel({ token }: { token: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [acl, setAcl] = useState<string[]>(["customer", "agent"]);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<DocumentIngested | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);

  const toggle = (g: string) =>
    setAcl((cur) => (cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]));

  async function submit() {
    if (!file || !title.trim() || !token.trim() || busy) return;
    setBusy(true);
    setResult(null);
    setError(null);
    try {
      setResult(await uploadDocument(file, title.trim(), acl, token.trim(), from || undefined, to || undefined));
    } catch (err) {
      const code = err instanceof ApiError ? err.code : "network";
      const message =
        err instanceof ApiError && err.status === 403
          ? "This token lacks the agent role. Ingestion is a back-office act — mint an agent token: python -m app.auth --tenant asha --groups agent"
          : err instanceof ApiError
            ? err.message
            : String(err);
      setError({ code, message });
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="upload">
      <summary>Add a document to the corpus</summary>

      <div className="upload-body">
        <label className="field">
          <span>File</span>
          <input
            type="file"
            accept=".md,.pdf,.docx"
            onChange={(e) => {
              const f = e.target.files?.[0] ?? null;
              setFile(f);
              // A sensible default title beats an empty required field — the
              // filename is what the uploader was already thinking of.
              if (f && !title.trim()) setTitle(f.name.replace(/\.[^.]+$/, ""));
            }}
          />
        </label>

        <label className="field">
          <span>Title</span>
          <input
            type="text"
            placeholder="Asha Rao — Roadside Addendum"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </label>

        <div className="field">
          <span>Visible to</span>
          <div className="acl">
            {["customer", "agent"].map((g) => (
              <label key={g}>
                <input type="checkbox" checked={acl.includes(g)} onChange={() => toggle(g)} />
                {g}
              </label>
            ))}
          </div>
        </div>

        <div className="field">
          <span>In force</span>
          <div className="window">
            <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} title="effective_from" />
            <span>→</span>
            <input type="date" value={to} onChange={(e) => setTo(e.target.value)} title="effective_to (exclusive)" />
          </div>
        </div>

        <button onClick={() => void submit()} disabled={!file || !title.trim() || !token.trim() || busy}>
          {busy ? "Parsing, embedding…" : "Upload"}
        </button>

        {result && (
          <div className="ingested">
            <span className="badge">{result.chunks} chunks embedded</span>
            <div>
              <strong>{result.doc_title}</strong> is now queryable for tenant{" "}
              <strong>{result.tenant_id}</strong>
              {result.replaced_chunks > 0 && (
                <> — replaced {result.replaced_chunks} chunks from the previous version</>
              )}
              .
            </div>
            <div className="hint">
              Ask something in the document's own words to see it cited — paraphrases
              score lower (the phrasing cliff applies to your uploads too).
            </div>
          </div>
        )}

        {error && (
          <div className="error">
            <strong>{error.code}</strong> — {error.message}
          </div>
        )}
      </div>
    </details>
  );
}

/** Streamed text with [n] citations highlighted to match the sources panel.
 * Split-render, not dangerouslySetInnerHTML: model output is UNTRUSTED text
 * and must never be interpreted as markup. */
export function StreamedAnswer({ text, streaming }: { text: string; streaming: boolean }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <div className="answer">
      {parts.map((p, i) =>
        /^\[\d+\]$/.test(p) ? (
          <span key={i} className="cite">
            {p}
          </span>
        ) : (
          <span key={i}>{p}</span>
        ),
      )}
      {streaming && <span className="cursor">▍</span>}
    </div>
  );
}
