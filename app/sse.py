"""The wire format for streaming: event models + frame encoding.

WHY THIS IS ITS OWN FILE
------------------------
Once you have sent `200 OK`, HTTP's error channel is gone. So the stream needs
its own error channel — and that means the stream has a PROTOCOL, not just a
payload. A protocol deserves a file.

Every frame carries a `type` discriminator. The client switches on it. Adding a
new frame type later (Module 3 adds "citation") is safe: old clients ignore
types they don't recognise. That is the additive-is-safe rule from section 1.2,
now applied to a stream instead of a JSON body.
"""

from typing import Literal              # stdlib — the "type" discriminator on each event

from pydantic import BaseModel          # 3rd-party: pydantic — event models + .model_dump_json()

from app.llm.base import Usage          # local — app/llm/base.py (the Usage dataclass)

# The terminal sentinel. Not JSON — a literal marker, matching the convention
# OpenAI's SSE uses, which clients already look for.
#
# Why it exists: a stream that ENDS and a stream that DIED look identical at the
# TCP layer. Both are just a closed socket. Without a positive "I finished"
# signal, every truncation is silently treated as a success.
DONE_SENTINEL = "[DONE]"


class TokenEvent(BaseModel):
    # `Literal["token"] = "token"` does two jobs at once:
    #   - the default means you never have to pass it
    #   - the Literal type means Pydantic will REJECT any other value
    # So the field is both automatic and enforced.
    type: Literal["token"] = "token"
    text: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    # Same field, same reason, as the HTTP error envelope in schemas.py:
    #   rate_limited   -> True   (back off and retry)
    #   content_filter -> False  (retrying is guaranteed to fail, forever)
    # The client needs this whether the failure arrived as a status code or as
    # a frame. The failure is the same; only the delivery channel changed.
    retryable: bool
    request_id: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    # Optional because a stream that died mid-flight has no usage to report.
    # Modelling it as Optional forces the client to handle that case.
    usage: Usage | None = None


def frame(event: BaseModel) -> str:
    """Serialise one Pydantic event into one SSE frame.

    The SSE format, in full:

        data: <payload>\\n\\n

    The DOUBLE newline is the frame delimiter — it is what tells the client
    "this message is complete". Send one `\\n` and the client waits forever for
    an ending that never comes. This is the bug everyone writes once, and it is
    why frame construction lives in one function instead of being inlined at
    four call sites.
    """
    return f"data: {event.model_dump_json()}\n\n"


def done_frame() -> str:
    return f"data: {DONE_SENTINEL}\n\n"


def heartbeat() -> str:
    """An SSE comment frame. Any line starting with `:` is ignored by clients.

    Purpose: a model that thinks for 40 seconds before its first token is
    indistinguishable, to an idle-timeout, from a dead connection. This is a
    byte that says "still here" without saying anything.
    """
    return ": keep-alive\n\n"
