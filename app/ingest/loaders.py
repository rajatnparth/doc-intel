"""Step 1: read a document into TEXT + STRUCTURE.

The most important idea in this file: a loader's job is not "get the text out".
It is "get the text out WITHOUT throwing away the structure the author put in".

Naive loaders flatten a document to a wall of characters. Then the chunker has
nothing to cut on except character count — and character-count cutting is the
failure the whole section is about. Structure preserved here is structure the
chunker can use later. Structure lost here is lost forever.
"""

from __future__ import annotations      # stdlib (special) — lets annotations be lazy strings,
                                        #   so `list[Section]` works and forward refs are free.
                                        #   Must be the first statement in the file.

import re                               # stdlib — regex: split on headings, find tables
from dataclasses import dataclass, field  # stdlib — @dataclass for Section; field() for the
                                        #   mutable default (default_factory=list)


@dataclass
class Section:
    """One coherent unit of a document — a heading and the body beneath it.

    This is the SEAM between loading and chunking. The loader produces Sections;
    the chunker consumes them. Notice it carries `heading` separately from
    `body`: that heading is the context a fragment needs to stay meaningful
    (section 3.1, 'contextual retrieval').
    """

    doc_title: str
    heading: str            # e.g. "2. Payment Terms"
    level: int              # 1 for #, 2 for ##, ...
    body: str
    # Blocks we must NOT let the chunker split by character count — tables, code.
    # Extracted whole, here, before any splitting can touch them.
    atomic_blocks: list[str] = field(default_factory=list)


# A markdown table is a run of consecutive lines that all contain a pipe.
# This regex finds such runs so we can pull them out as atomic units.
_TABLE_RE = re.compile(r"(?:^\|.*\|\s*$\n?)+", re.MULTILINE)


def load_markdown(text: str, *, doc_title: str) -> list[Section]:
    """Split markdown into Sections on headings, keeping tables intact.

    Why markdown first: contracts, invoices, wikis and exported PDFs all reduce
    cleanly to markdown, and markdown makes structure explicit (`##`, `|`). Real
    PDFs are messier — see load_pdf below — but the SHAPE of the answer is the
    same, so we learn it on the clean case.
    """
    sections: list[Section] = []

    # Split on headings but KEEP them (the capturing group in re.split keeps the
    # delimiter). Every odd element is a heading line, every even one its body.
    parts = re.split(r"^(#{1,6}\s+.*)$", text, flags=re.MULTILINE)

    # parts[0] is any preamble before the first heading; usually empty here.
    i = 1
    while i < len(parts):
        heading_line = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        level = len(heading_line) - len(heading_line.lstrip("#"))
        heading_text = heading_line.lstrip("#").strip()

        # Pull tables OUT of the body before anyone can character-chunk them.
        # We replace each table with a placeholder so the prose flows, and store
        # the raw table in atomic_blocks to be emitted as its own chunk.
        atomic: list[str] = []

        def _extract(match: re.Match[str]) -> str:
            atomic.append(match.group(0).strip())
            return f"\n[TABLE {len(atomic)} EXTRACTED]\n"

        prose_body = _TABLE_RE.sub(_extract, body).strip()

        sections.append(
            Section(
                doc_title=doc_title,
                heading=heading_text,
                level=level,
                body=prose_body,
                atomic_blocks=atomic,
            )
        )

    return sections


def load_pdf(path: str, *, doc_title: str) -> list[Section]:
    """Read a PDF into Sections.

    Real PDFs do NOT hand you headings — they hand you positioned text runs, page
    headers/footers repeated on every page, and tables as scattered cells. A
    production loader uses layout analysis (heading detection by font size,
    header/footer stripping, table reconstruction). That is a module of its own.

    For the lab we keep the interface identical to load_markdown so the chunker
    doesn't care which loader produced the Sections — the seam again — and we
    strip the one thing that poisons every chunk if left in: repeated
    page furniture.
    """
    try:
        from pypdf import PdfReader     # 3rd-party: pypdf — pure-Python PDF reader.
                                        #   Imported LAZILY (inside the function, not at top)
                                        #   so markdown-only users never need it installed.
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pypdf to read PDFs") from e

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]

    # Strip lines that appear on nearly every page — they are headers/footers.
    # Left in, every chunk contains "CONFIDENTIAL - Page 12 of 40", and every
    # vector drifts slightly toward every other. This is not optional cleanup.
    line_counts: dict[str, int] = {}
    for pg in pages:
        for line in {ln.strip() for ln in pg.splitlines() if ln.strip()}:
            line_counts[line] = line_counts.get(line, 0) + 1
    threshold = max(2, int(len(pages) * 0.6))
    furniture = {ln for ln, c in line_counts.items() if c >= threshold}

    cleaned = "\n".join(
        ln for pg in pages for ln in pg.splitlines() if ln.strip() not in furniture
    )
    # Without real heading detection, treat the whole doc as one section. The
    # point of this stub is the furniture stripping and the shared interface.
    return [Section(doc_title=doc_title, heading=doc_title, level=1, body=cleaned)]
