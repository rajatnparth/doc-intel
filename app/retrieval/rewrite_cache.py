"""A cache in front of the rewriter — and the argument for why it is safe.

WHY THIS FILE HAS A DOCSTRING LONGER THAN ITS CODE
--------------------------------------------------
This codebase has already been burned by a cache. `PostFilterRetriever` is
kept as a permanent villain because a semantic cache added between retrieval
and filtering leaked one tenant's chunks to another — and the developer who
added it did nothing wrong locally. Then the `as_of` gate arrived and the
per-principal view cache had to grow a third key component in the same
commit, or December's question would have been served January's documents.

The lesson stuck: **a cache key is a security boundary.** So any new cache in
this system owes an explicit argument, not a shrug.

THE ARGUMENT FOR THIS ONE
-------------------------
The rewriter is a function of the QUESTION TEXT ALONE:

  - it never reads the corpus, so no tenant's documents can enter its output
  - it never receives a Principal, so it cannot vary by tenant, and there is
    nothing tenant-shaped to leak into a shared entry
  - its output is a set of PHRASINGS, which are then run through the normal
    gated retrieval path — the gate is applied AFTER this cache, per request,
    with the caller's own verified principal

So Asha's rewrite of "what is my excess?" is identical to Vikram's, and
sharing it between them reveals nothing about either: what each of them can
SEE is decided downstream, by the gate, from the JWT. Contrast the villain,
whose cache held retrieved CONTENT, keyed on the query alone — content is
tenant-shaped and phrasings are not.

The one thing that would break this argument is a rewriter that saw
corpus text (e.g. few-shot examples drawn from the tenant's own documents,
a tempting future optimisation). If that day comes, the tenant joins the key
in the SAME commit — the phase-3 rule, restated.

PII: questions can contain identifiers, and this cache holds question text in
memory. It is process-local, never persisted, and bounded; the redactor still
governs everything that reaches STORAGE (app/safety.py). Set
QUERY_REWRITE_CACHE_SIZE=0 to disable it in an environment where even
in-memory retention of question text is unacceptable.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from collections import OrderedDict     # stdlib — LRU without a dependency
from typing import TYPE_CHECKING        # stdlib — type-only import, no cycle

if TYPE_CHECKING:                       # pragma: no cover
    from app.retrieval.rewrite import TransformedQuery  # local — app/retrieval/rewrite.py


class RewriteCache:
    """Bounded LRU over (question, variant count) -> TransformedQuery.

    The variant count is part of the key because it changes the RESULT: a
    cached 3-variant answer must not be served to a request configured for 1.
    Small, obvious, and exactly the kind of key component that gets forgotten.
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[tuple[str, int], TransformedQuery] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, question: str, variants: int) -> "TransformedQuery | None":
        if self._maxsize <= 0:
            return None
        key = (question.strip(), variants)
        if key not in self._data:
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return self._data[key]

    def put(self, question: str, variants: int, value: "TransformedQuery") -> None:
        if self._maxsize <= 0:
            return
        # A degraded transform (the provider was down) must NOT be cached:
        # caching it would turn a transient outage into a persistent quality
        # regression that outlives the incident.
        if value.degraded:
            return
        key = (question.strip(), variants)
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
