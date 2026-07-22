/**
 * SSE over fetch — because EventSource cannot do the two things this API
 * requires: POST a JSON body, and send an Authorization header. So we read
 * the response body as a stream and split frames ourselves. ~30 lines, and
 * you can see the protocol instead of trusting a wrapper.
 */

import type { AskRequest, DocumentIngested, Frame, HandoffResponse } from "./types";

/** HTTP headers may only carry ISO-8859-1, and a JWT is narrower still —
 * base64url segments joined by dots, pure printable ASCII. Anything outside
 * that means the paste grabbed something other than the token: a chat UI's
 * masking bullets (••••), smart quotes, a zero-width space. Without this
 * check, fetch() throws `TypeError: … non ISO-8859-1 code point` — true,
 * useless, and rendered as a scary "network" error. Validate BEFORE the
 * header is built and say what actually happened. */
export function tokenProblem(token: string): string | null {
  if (!token) return null;
  if (/[^\x21-\x7e]/.test(token)) {
    return (
      "The pasted token contains characters a JWT can't contain — this " +
      "usually means the paste grabbed masking dots or formatting instead " +
      "of the raw token. Mint one in your terminal and copy it from there: " +
      "python -m app.auth --tenant asha --groups customer"
    );
  }
  return null;
}

/** A non-stream failure (401 bad token, 422 bad body, 429 shed load): the
 * engine's JSON error envelope, surfaced before any SSE began. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public retryable: boolean,
  ) {
    super(message);
  }
}

export async function* askStream(
  req: AskRequest,
  token: string,
  signal: AbortSignal,
  onRequestId?: (id: string) => void,
): AsyncGenerator<Frame> {
  const res = await fetch("/v1/ask", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(req),
    signal,
  });

  // The exchange's identity, minted by the server before the first frame.
  // It is the key into the AUDIT TRAIL — /v1/handoff references it, which is
  // how a "talk to a human" ticket carries full context without copying any.
  const rid = res.headers.get("x-request-id");
  if (rid && onRequestId) onRequestId(rid);

  if (!res.ok) {
    // Before the stream starts, errors are ordinary HTTP + JSON envelope.
    // AFTER it starts they arrive in-band as {type:"error"} frames — the 200
    // was spent before the model wrote a word (see app/main.py).
    const body = await res.json().catch(() => null);
    const e = body?.error ?? body?.detail ?? {};
    throw new ApiError(
      res.status,
      e.code ?? String(res.status),
      typeof e === "string" ? e : (e.message ?? `HTTP ${res.status}`),
      e.retryable ?? false,
    );
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return; // transport closed without [DONE]: the dead-stream case
      buf += decoder.decode(value, { stream: true });

      // SSE framing: events are separated by a blank line; each data line is
      // prefixed "data: ". We only ever emit single-line data frames.
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        for (const line of rawEvent.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return; // told, not inferred from silence
          yield JSON.parse(data) as Frame;
        }
      }
    }
  } finally {
    // Cancelling the reader closes the connection, which the server observes
    // as a disconnect — and cancels ITS upstream call (the tested path that
    // stops the provider's meter). The Stop button is wired to real money.
    reader.cancel().catch(() => {});
  }
}

/** POST /v1/documents — multipart upload: parse -> chunk -> embed -> upsert.
 *
 * FormData, not JSON: the body carries bytes. Note what is NOT sent —
 * tenant_id. It is stamped server-side from the verified token, because it
 * decides WHOSE corpus changes; the fields below only DESCRIBE the document
 * inside the uploader's own corpus. The browser sets its own multipart
 * content-type boundary, so we must not set that header ourselves. */
export async function uploadDocument(
  file: File,
  title: string,
  acl: string[],
  token: string,
  effectiveFrom?: string,
  effectiveTo?: string,
): Promise<DocumentIngested> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);
  form.append("acl", acl.join(","));
  if (effectiveFrom) form.append("effective_from", effectiveFrom);
  if (effectiveTo) form.append("effective_to", effectiveTo);

  const res = await fetch("/v1/documents", {
    method: "POST",
    headers: { authorization: `Bearer ${token}` },
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const e = body?.error ?? {};
    throw new ApiError(res.status, e.code ?? String(res.status), e.message ?? `HTTP ${res.status}`, false);
  }
  return (await res.json()) as DocumentIngested;
}

/** POST /v1/handoff — turn a refused (or any audited) exchange into a ticket.
 * Sends the request_id, not the conversation: the server-side agent reads
 * the audit record, the single source of truth. */
export async function createHandoff(
  requestId: string,
  note: string,
  token: string,
): Promise<HandoffResponse> {
  const res = await fetch("/v1/handoff", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ request_id: requestId, note }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const e = body?.error ?? {};
    throw new ApiError(res.status, e.code ?? String(res.status), e.message ?? `HTTP ${res.status}`, false);
  }
  return (await res.json()) as HandoffResponse;
}
