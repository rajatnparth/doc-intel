"""Watch dense and BM25 fail in opposite directions, then RRF rescue both.

    python hybrid_demo.py

The embeddings are REAL (bge-small-en-v1.5). Nothing here is staged — if dense
returns the wrong invoice, that's the model actually doing it.
"""

from pathlib import Path                # stdlib — read the sample docs

from app.ingest import chunk_document   # local — app/ingest/chunker.py
from app.retrieval.hybrid import HybridRetriever  # local — app/retrieval/hybrid.py


def load_corpus():
    chunks = []
    for name, title in [
        ("acme_msa.md", "Acme MSA (2024)"),
        ("invoices.md", "Invoice Register 2024"),
    ]:
        text = Path("sample_docs") / name
        chunks += chunk_document(text.read_text(), doc_title=title, max_chars=700)
    # chunk_index must be unique across documents for RRF's dict keying
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def show(label: str, hits, want: str, n: int = 3) -> None:
    print(f"  {label}")
    for h in hits[:n]:
        # Check text_to_embed, NOT text — the invoice number lives in the heading,
        # and text_to_embed is what both retrievers actually index. Checking
        # .text here was the same bug as indexing .text: it reported misses on
        # chunks that genuinely contained the answer.
        got = want.lower() in h.chunk.text_to_embed.lower()
        mark = "✅" if got else "  "
        head = h.chunk.heading[:44]
        print(f"    {mark} #{h.rank + 1} [{head:<44}] score={h.score:.4f}")
    rank = next(
        (h.rank + 1 for h in hits if want.lower() in h.chunk.text_to_embed.lower()), None
    )
    verdict = f"target at rank {rank}" if rank else "TARGET NOT IN TOP 10 ❌"
    print(f"    -> {verdict}\n")


def probe(r: HybridRetriever, query: str, want: str, why: str) -> None:
    print("=" * 78)
    print(f"QUERY: {query!r}")
    print(f"looking for a chunk containing: {want!r}   ({why})")
    print("=" * 78)
    show("DENSE (embeddings — understands meaning)", r.dense_search(query), want)
    show("BM25  (lexical — exact tokens only)     ", r.bm25_search(query), want)
    show("RRF   (fused by RANK, k=60)             ", r.rrf(query), want)


def main() -> None:
    print("Loading corpus + embedding chunks with bge-small-en-v1.5 (real model)...\n")
    chunks = load_corpus()
    r = HybridRetriever(chunks)
    print(f"{len(chunks)} chunks indexed.\n")

    # 1. EXACT TOKEN — an error code. Dense should struggle; BM25 should nail it.
    probe(r, "error E-4471", "E-4471", "an exact token with no semantic meaning")

    # 2. EXACT TOKEN — an invoice number among near-identical siblings.
    probe(r, "INV-2024-0891", "INV-2024-0891", "0888..0893 are all nearly identical")

    # 3. PARAPHRASE — no shared vocabulary with the source. BM25 should struggle.
    probe(r, "what happens if we pay late?", "1.5%", "source says 'interest', not 'late'")

    # 4. PARAPHRASE — 'money-back period' never appears in the corpus.
    probe(r, "how long do we have to settle an invoice?", "thirty (30) days",
          "source says 'Payment is due within thirty (30) days'")

    print("=" * 78)
    print("THE POINT")
    print("=" * 78)
    print("  Neither retriever wins both. RRF needs neither to win — only to")
    print("  RANK the answer highly, and it fuses on rank, never on score.")


if __name__ == "__main__":
    main()
