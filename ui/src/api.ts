/**
 * SSE over fetch — because EventSource cannot do the two things this API
 * requires: POST a JSON body, and send an Authorization header. So we read
 * the response body as a stream and split frames ourselves. ~30 lines, and
 * you can see the protocol instead of trusting a wrapper.
 */

import type { AskRequest, Frame } from "./types";

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
