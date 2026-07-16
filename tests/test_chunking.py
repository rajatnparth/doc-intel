"""Section 3.1 — proof that structure-aware chunking preserves what naive destroys.

The assertions ARE the lesson: each one names a technique and shows it working.
"""

from pathlib import Path                # stdlib — locate + read the sample doc

import pytest                           # 3rd-party: pytest — the @pytest.fixture below

from app.ingest import chunk_document, load_markdown, naive_chunks  # local — app/ingest/

DOC = Path(__file__).parent.parent / "sample_docs" / "acme_msa.md"


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
    row = next((c for c in chunks if "E-4471" in c.text), None)
    return row is not None and "Error Code" in row.text and "Remedy" in row.text


def test_naive_chunking_correctness_is_a_coin_flip(text: str) -> None:
    outcomes = {
        size: _header_survives(naive_chunks(text, size=size, doc_title="Acme"))
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
        chunks = chunk_document(text, doc_title="Acme", max_chars=size)
        e4471 = [c for c in chunks if "E-4471" in c.text]
        assert len(e4471) == 1, f"size {size}: table is one atomic chunk"
        c = e4471[0]
        assert c.is_table is True
        assert "Error Code" in c.text and "Remedy" in c.text
        assert "Restart the ingestion worker" in c.text
        assert "E-4470" in c.text and "E-4472" in c.text, "whole table, not a fragment"


# =============================================================================
# Technique 1 — cuts land on headings, not character counts.
# =============================================================================
def test_boundaries_fall_on_sections(text: str) -> None:
    sections = load_markdown(text, doc_title="Acme")
    headings = {s.heading for s in sections}
    assert "2. Payment Terms" in headings
    assert "5. Troubleshooting Reference" in headings
    # The table was pulled OUT of its section's prose body.
    troubleshooting = next(s for s in sections if s.heading.startswith("5."))
    assert troubleshooting.atomic_blocks, "table extracted as an atomic block"
    assert "[TABLE 1 EXTRACTED]" in troubleshooting.body


# =============================================================================
# Technique 3 — contextual retrieval: we embed provenance + text, not text alone.
# =============================================================================
def test_embedded_text_carries_provenance(text: str) -> None:
    chunks = chunk_document(text, doc_title="Acme MSA (2024)")
    payment = next(c for c in chunks if "thirty (30) days" in c.text)

    # The chunk's own text does not say which contract it's from.
    assert "Acme" not in payment.text
    # But the text we EMBED does — that's what makes the query match.
    assert payment.text_to_embed.startswith("Acme MSA (2024) > 2. Payment Terms:")


# =============================================================================
# Technique 4 — parent is larger than the indexed chunk.
# =============================================================================
def test_parent_is_larger_than_chunk(text: str) -> None:
    chunks = chunk_document(text, doc_title="Acme", max_chars=300)
    prose = [c for c in chunks if not c.is_table]
    # At least one section had to be split, so its pieces are smaller than the
    # parent we'd return on a hit.
    assert any(len(c.text) < len(c.parent_text) for c in prose), (
        "search small (chunk), generate large (parent)"
    )
