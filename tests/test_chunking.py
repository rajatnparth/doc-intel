"""Section 3.1 — proof that structure-aware chunking preserves what naive destroys.

The assertions ARE the lesson: each one names a technique and shows it working.
"""

from pathlib import Path                # stdlib — locate + read the sample doc

import pytest                           # 3rd-party: pytest

from app.ingest.chunker import _split_long_body  # local — the cap under test


# =============================================================================
# The cap is a GUARANTEE, not an aspiration (found by a real PDF upload)
# =============================================================================
# PDF text extraction hard-wraps lines and emits NO blank lines. Measured on a
# real 4-page policy PDF: 0 "\n\n", 131 "\n". The old splitter only knew
# "\n\n", so it returned the section whole — a 6,613-char chunk against a
# 700-char budget, which the embedder (512 tokens ≈ 2,000 chars) then
# silently truncated. Every sample document is markdown WITH blank lines, so
# nothing in the suite had ever exercised the other shape.
PDF_SHAPED = "\n".join(
    f"Line {i} of a hard-wrapped policy paragraph that never sees a blank line."
    for i in range(40)
)


def test_splitter_honours_the_cap_without_blank_lines() -> None:
    pieces = _split_long_body(PDF_SHAPED, 300)
    assert len(pieces) > 1, "PDF-shaped text must still split"
    assert all(len(p) <= 300 for p in pieces), [len(p) for p in pieces]


def test_splitter_honours_the_cap_with_no_separators_at_all() -> None:
    """A wall of characters — no blank lines, no newlines, no sentence ends.
    The hard cut is what turns the maximum into a promise that always holds."""
    pieces = _split_long_body("x" * 2000, 300)
    assert all(len(p) <= 300 for p in pieces)
    assert "".join(pieces) == "x" * 2000, "a cut must not lose text"


def test_splitter_prefers_the_most_structural_boundary() -> None:
    """Given blank lines, use them — the cascade must not skip to a finer
    separator and shred paragraphs that would have fit."""
    body = "\n\n".join(["A" * 200, "B" * 200, "C" * 200])
    pieces = _split_long_body(body, 450)
    assert pieces[0] == "A" * 200 + "\n\n" + "B" * 200
    assert pieces[1] == "C" * 200


def test_finished_chunks_respect_max_chars_including_overlap() -> None:
    """max_chars means the size of the CHUNK. The overlap prefix is part of
    the chunk, so the split budget must leave room for it — measuring the cap
    before the last thing that grows the text produced 825-char chunks
    against a 700 budget."""
    from app.ingest import chunk_document

    doc = "# Kit\n\n## 1. Cover\n\n" + PDF_SHAPED
    for c in chunk_document(doc, doc_title="Kit", max_chars=400, overlap_chars=80):
        assert len(c.text) <= 400, f"{len(c.text)} > 400: {c.text[:80]!r}"


from app.ingest import chunk_document, load_markdown, naive_chunks  # local — app/ingest/

DOC = Path(__file__).parent.parent / "sample_docs" / "asha_policy_kit.md"


@pytest.fixture
def text() -> str:
    return DOC.read_text()


# =============================================================================
# The headline failure: naive splitting's correctness is ARBITRARY.
#
# It is tempting to assert "naive always orphans the row". That is false and
# would be a dishonest test — at some sizes it gets lucky. The real, worse,
# truth is that the outcome depends on a size you chose for unrelated reasons.
# =============================================================================
def _header_survives(chunks) -> bool:
    row = next((c for c in chunks if "D-4471" in c.text), None)
    return row is not None and "Damage Code" in row.text and "Remedy" in row.text


def test_naive_chunking_correctness_is_a_coin_flip(text: str) -> None:
    outcomes = {
        size: _header_survives(naive_chunks(text, size=size, doc_title="Asha"))
        for size in (300, 400, 500, 600, 700, 800, 1000)
    }
    # The point: it is NOT uniformly True and NOT uniformly False. It flips.
    # A component whose correctness depends on an unrelated knob is unreasonable-
    # about, which is worse than one that is reliably wrong.
    assert any(outcomes.values()), "at some sizes it gets lucky"
    assert not all(outcomes.values()), "at other sizes it orphans the row"


def test_structure_aware_keeps_the_table_whole_at_every_size(text: str) -> None:
    # The contrast with the coin-flip test above: this holds at ALL sizes,
    # because the table is extracted before any size logic runs.
    for size in (200, 300, 500, 700, 1000):
        chunks = chunk_document(text, doc_title="Asha", max_chars=size)
        d4471 = [c for c in chunks if "D-4471" in c.text]
        assert len(d4471) == 1, f"size {size}: table is one atomic chunk"
        c = d4471[0]
        assert c.is_table is True
        assert "Damage Code" in c.text and "Remedy" in c.text
        assert "Hold repairs until the surveyor inspects" in c.text
        assert "D-4470" in c.text and "D-4472" in c.text, "whole table, not a fragment"


# =============================================================================
# Technique 1 — cuts land on headings, not character counts.
# =============================================================================
def test_boundaries_fall_on_sections(text: str) -> None:
    sections = load_markdown(text, doc_title="Asha")
    headings = {s.heading for s in sections}
    assert "2. Premium and Payment" in headings
    assert "5. Damage Assessment Codes" in headings
    # The table was pulled OUT of its section's prose body.
    damage_codes = next(s for s in sections if s.heading.startswith("5."))
    assert damage_codes.atomic_blocks, "table extracted as an atomic block"
    assert "[TABLE 1 EXTRACTED]" in damage_codes.body


# =============================================================================
# Technique 3 — contextual retrieval: we embed provenance + text, not text alone.
# =============================================================================
def test_embedded_text_carries_provenance(text: str) -> None:
    chunks = chunk_document(text, doc_title="Asha Rao — Motor Policy Kit (2026)")
    premium = next(c for c in chunks if "fifteen (15) days" in c.text)

    # The chunk's own text does not say whose policy it's from.
    assert "Asha" not in premium.text
    # But the text we EMBED does — that's what makes the query match.
    assert premium.text_to_embed.startswith(
        "Asha Rao — Motor Policy Kit (2026) > 2. Premium and Payment:"
    )


# =============================================================================
# Technique 4 — parent is larger than the indexed chunk.
# =============================================================================
def test_parent_is_larger_than_chunk(text: str) -> None:
    chunks = chunk_document(text, doc_title="Asha", max_chars=300)
    prose = [c for c in chunks if not c.is_table]
    # At least one section had to be split, so its pieces are smaller than the
    # parent we'd return on a hit.
    assert any(len(c.text) < len(c.parent_text) for c in prose), (
        "search small (chunk), generate large (parent)"
    )
