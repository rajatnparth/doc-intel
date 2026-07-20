/**
 * The wire contract — a hand-written mirror of app/sse.py.
 *
 * The server discriminates frames with a pydantic `Literal["…"]` type field;
 * TypeScript's discriminated unions are the same idea on the other end of the
 * wire. The payoff is in App.tsx: the switch over `frame.type` is exhaustive
 * (see the `never` check), so when the engine grows a new frame type, this
 * client FAILS TO COMPILE until someone decides how to render it — instead of
 * silently dropping frames on the floor.
 *
 * Mirrored by hand, deliberately: generating types from the OpenAPI schema is
 * the enterprise move, but the SSE frames live inside a text/event-stream body
 * that OpenAPI does not describe. The contract test for this file is using the
 * app: an unknown frame renders as a visible "unknown frame" error, not nothing.
 */

export interface TokenEvent {
  type: "token";
  text: string;
}

export interface SourceRef {
  n: number;
  doc_title: string;
  heading: string;
}

export interface SourcesEvent {
  type: "sources";
  sources: SourceRef[];
}

export interface FactItem {
  name: string;
  value: string;
}

/** Facts come from the SYSTEM OF RECORD, never from prose — `source` says so
 * on the wire, and the UI renders that provenance as a badge. Note there is
 * no usage on this path's done frame: no model was called. */
export interface FactsEvent {
  type: "facts";
  policy_number: string;
  facts: FactItem[];
  source: "policy_admin";
}

/** A refusal is a first-class outcome (gated.py). The client renders the
 * score, the reason, and the near-misses as links — not an error state. */
export interface RefusalEvent {
  type: "refusal";
  score: number;
  reason: string;
  near_misses: SourceRef[];
}

export interface ErrorEvent {
  type: "error";
  code: string;
  message: string;
  retryable: boolean;
  request_id: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface DoneEvent {
  type: "done";
  usage: Usage | null;
}

export type Frame =
  | TokenEvent
  | SourcesEvent
  | FactsEvent
  | RefusalEvent
  | ErrorEvent
  | DoneEvent;

/** The request body for POST /v1/ask (app/schemas.py: AskRequest).
 * Note what is ABSENT: tenant_id / groups. Identity travels ONLY in the
 * Authorization header; the server 422s a body that still includes it. */
export interface AskRequest {
  question: string;
  as_of?: string; // ISO date — the date-of-loss time anchor
  temperature?: number;
  max_tokens?: number;
}
