"""Watch a real document get chunked two ways.

    python chunk_demo.py

Left: the naive 1000-char splitter guillotines the troubleshooting table, so the
E-4471 row loses its header. Right: the structure-aware chunker keeps the table
whole and prepends provenance to every prose chunk.

This is section 3.1, made visible. Read the output, not just the code.
"""

from pathlib import Path                # stdlib — read the sample doc (Path.read_text())

from app.ingest import chunk_document, naive_chunks  # local — app/ingest/ (the two strategies)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def header_survives(chunks) -> bool:
    """True iff E-4471 appears in the SAME chunk as the table header."""
    row = next((c for c in chunks if "E-4471" in c.text), None)
    return row is not None and "Error Code" in row.text and "Remedy" in row.text


def main() -> None:
    doc = Path("sample_docs/acme_msa.md").read_text()
    title = "Acme MSA (2024)"

    # ---- NAIVE: the failure is that the outcome is ARBITRARY -----------------
    banner("STRATEGY 1 — naive fixed-size: does E-4471 keep its header?")
    print("The naive splitter cuts at char N. Whether the E-4471 row stays with")
    print("its column header is pure luck — it depends on a size you picked for")
    print("unrelated reasons. Watch it flip:\n")
    for size in (300, 400, 500, 600, 700, 800, 1000):
        naive = naive_chunks(doc, size=size, doc_title=title)
        ok = header_survives(naive)
        mark = "✅ survived (lucky)" if ok else "❌ orphaned — row is noise"
        print(f"  size={size:4}: {mark}")
    print("\n  You cannot reason about a component whose correctness is a coin flip.")

    # ---- STRUCTURE-AWARE: invariant across sizes ----------------------------
    banner("STRATEGY 2 — structure-aware: table is atomic at EVERY size")
    for size in (300, 500, 700, 1000):
        smart = chunk_document(doc, doc_title=title, max_chars=size)
        ok = header_survives(smart)
        n_tables = sum(1 for c in smart if c.is_table)
        print(f"  max_chars={size:4}: header kept? {'✅' if ok else '❌'}   "
              f"tables emitted whole: {n_tables}")
    print("\n  Invariant. The table is pulled out before any size logic runs.")

    smart = chunk_document(doc, doc_title=title, max_chars=700, overlap_chars=120)
    print(f"\n{len(smart)} chunks total. Boundaries fall on section headings.")

    # The payoff of technique 3: show text vs text_to_embed for a prose chunk.
    prose = next(c for c in smart if not c.is_table and "30" in c.text)
    banner("TECHNIQUE 3 — what we actually embed (contextual retrieval)")
    print("Chunk's own text:")
    print("  " + repr(prose.text[:120]))
    print("\nText we EMBED (provenance prepended — this is what makes")
    print("'what are Acme's payment terms?' match this chunk):")
    print("  " + repr(prose.text_to_embed[:160]))

    # Technique 4: search small, return large.
    banner("TECHNIQUE 4 — parent-document (search small, return large)")
    print(f"Indexed chunk text is {len(prose.text)} chars (precise).")
    print(f"Parent returned to the LLM is {len(prose.parent_text)} chars (context).")


if __name__ == "__main__":
    main()
