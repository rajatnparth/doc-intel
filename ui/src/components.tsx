/**
 * The three outcome renderings. Each one makes a server-side design decision
 * VISIBLE — that is the entire job of a reference client.
 */

import type { FactsEvent, RefusalEvent, SourceRef } from "./types";

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
 * drove it, and the near-misses offered as leads — never as an answer. */
export function RefusalCard({ refusal }: { refusal: RefusalEvent }) {
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
