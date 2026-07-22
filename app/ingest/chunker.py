"""Step 2: turn Sections into Chunks worth embedding.

This file implements, as real code, the five techniques from section 3.1:

    1. structure-aware splitting   -> we cut on Sections, never on char count
    2. overlap                     -> add_overlap()
    3. contextual retrieval        -> Chunk.text_to_embed prepends provenance
    4. parent-document retrieval   -> Chunk.parent_text, searched small / returned large
    5. tables as atomic units      -> emitted whole, never split

And it keeps the one you must never do — fixed-size character slicing — only as
`naive_chunks`, so the demo can show you exactly what it destroys.
"""

from __future__ import annotations      # stdlib (special) — lazy annotations; first line

from dataclasses import dataclass, field  # stdlib — @dataclass for Chunk; field() for the
                                        #   mutable metadata default
from datetime import date              # stdlib — the effective window on ChunkMeta

from app.ingest.loaders import Section, load_markdown  # local — app/ingest/loaders.py


@dataclass(frozen=True)
class ChunkMeta:
    """The GATE fields (section 3.4). Not relevance signals — enforcement.

    These are separated from `metadata: dict` deliberately. A dict says "any
    string key might be here"; this says "every chunk HAS a tenant". You cannot
    forget to set tenant_id, because the dataclass won't construct without it.
    Same move as `extract() -> str`: make the unsafe state unrepresentable.
    """

    tenant_id: str                  # a SECURITY BOUNDARY, not a hint
    acl: frozenset[str]             # group ids permitted to see this chunk

    # The validity window. This replaced `status: "active" | "superseded"`,
    # because "active" asks the wrong question. Whether a policy wording
    # applies is RELATIVE TO A DATE — and not necessarily today's: a claim is
    # assessed under the wording in force on the DATE OF LOSS, so a December
    # accident reported in July is answered from December's kit. A status flag
    # cannot even represent that question. "Superseded" is now a DERIVED fact
    # (the window closed), not a stored one somebody must remember to flip.
    effective_from: date = date.min # first day this version is in force
    effective_to: date | None = None  # exclusive end; None = still in force

    def visible_to(self, tenant_id: str, groups: frozenset[str], as_of: date) -> bool:
        """The predicate. Kept next to the data it guards, so there is exactly
        one definition of 'visible' in the codebase."""
        return (
            self.tenant_id == tenant_id
            and self.effective_from <= as_of
            and (self.effective_to is None or as_of < self.effective_to)
            and bool(self.acl & groups)
        )


@dataclass
class Chunk:
    doc_title: str
    heading: str
    text: str                       # the chunk's own content
    parent_text: str                # the larger block to RETURN on a hit (technique 4)
    meta: ChunkMeta | None = None   # None only for the pre-3.4 demos
    is_table: bool = False
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def text_to_embed(self) -> str:
        """Technique 3 — contextual retrieval.

        We do NOT embed `self.text` alone. We embed the text WITH its provenance
        glued to the front, so the vector carries context the fragment lacks.

        "Payment is due within 30 days" -> whose payment? which contract?
        Prepending "Acme MSA (2024) > 2. Payment Terms:" answers that, and the
        query "what are Acme's payment terms?" now lands on this chunk.

        This is the highest-return, least-glamorous line in the module.
        """
        return f"{self.doc_title} > {self.heading}:\n{self.text}"


def naive_chunks(text: str, *, size: int = 1000, doc_title: str = "doc") -> list[Chunk]:
    """The tutorial version. Fixed-size character slices. Here to be a villain.

    It cannot see headings, so it cuts mid-sentence. It cannot see tables, so it
    guillotines them. Every downstream component then works perfectly on garbage.
    """
    out: list[Chunk] = []
    for i in range(0, len(text), size):
        piece = text[i : i + size]
        out.append(
            Chunk(
                doc_title=doc_title,
                heading="(unknown — naive splitter has no structure)",
                text=piece,
                parent_text=piece,
                chunk_index=len(out),
            )
        )
    return out


def _split_long_body(body: str, max_chars: int) -> list[str]:
    """When a single section is too big for one chunk, split it — but on
    PARAGRAPH boundaries (blank lines), never mid-sentence. Structure-aware even
    at the sub-section level."""
    if len(body) <= max_chars:
        return [body]

    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for p in paras:
        if current and len(current) + len(p) + 2 > max_chars:
            pieces.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        pieces.append(current)
    return pieces


def add_overlap(pieces: list[str], *, overlap_chars: int = 120) -> list[str]:
    """Technique 2 — overlap.

    Prepend the tail of each piece to the next, so a sentence straddling a
    boundary survives whole in the SECOND piece.

    What you buy: boundary-straddling sentences survive.
    What you pay: index grows; near-duplicate text appears at retrieval time and
                  must be de-duplicated AFTER retrieval, not before (throwing it
                  away before defeats the purpose).
    """
    if len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap_chars:]
        out.append(f"...{tail}\n\n{cur}")
    return out


def chunk_document(
    text: str,
    *,
    doc_title: str,
    max_chars: int = 700,
    overlap_chars: int = 120,
    meta: "ChunkMeta | None" = None,
) -> list[Chunk]:
    """Markdown in, chunks out — load_markdown + chunk_sections.

    Kept as the convenience entrypoint; the WORK lives in chunk_sections,
    because phase 10 made the input format a seam: PDF and DOCX loaders
    produce the same Sections, and the chunker must not care who did.
    """
    return chunk_sections(
        load_markdown(text, doc_title=doc_title),
        doc_title=doc_title,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        meta=meta,
    )


def chunk_sections(
    sections: list[Section],
    *,
    doc_title: str,
    max_chars: int = 700,
    overlap_chars: int = 120,
    meta: "ChunkMeta | None" = None,
) -> list[Chunk]:
    """The real thing. Structure-aware, table-safe, context-enriched,
    parent-aware. This is what you would defend in the interview.

    `meta` is stamped onto every chunk this document produces. It's per-DOCUMENT,
    not per-chunk, because tenancy and permissions are properties of the document
    you ingested — the chunker has no business deciding them.
    """
    chunks: list[Chunk] = []

    for section in sections:
        # --- tables first: emit each as ONE atomic chunk (technique 5) --------
        for table in section.atomic_blocks:
            chunks.append(
                Chunk(
                    doc_title=doc_title,
                    heading=section.heading,
                    # A table alone embeds poorly, so we give it a natural-language
                    # handle AND keep the full markdown table as its own text.
                    text=(
                        f"Table in section '{section.heading}'. "
                        f"Columns and rows follow.\n{table}"
                    ),
                    parent_text=table,
                    meta=meta,
                    is_table=True,
                    chunk_index=len(chunks),
                    metadata={"kind": "table"},
                )
            )

        if not section.body.strip():
            continue

        # --- prose: split on paragraphs if long, then overlap -----------------
        # The PARENT is the whole section. We index the small pieces but return
        # this on a hit (technique 4): search small, generate large.
        parent = f"{section.heading}\n\n{section.body}"
        pieces = _split_long_body(section.body, max_chars)
        pieces = add_overlap(pieces, overlap_chars=overlap_chars)

        for piece in pieces:
            chunks.append(
                Chunk(
                    doc_title=doc_title,
                    heading=section.heading,
                    text=piece,
                    parent_text=parent,
                    meta=meta,
                    chunk_index=len(chunks),
                    metadata={"kind": "prose", "section": section.heading},
                )
            )

    return chunks
