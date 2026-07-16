"""Ingestion: turning a document into chunks worth embedding.

The order of operations, and why:

    load()          bytes on disk  ->  text + structure     (loaders.py)
    chunk()         structured text -> list[Chunk]           (chunker.py)
    enrich()        Chunk           -> text you actually embed (chunker.py)

The whole module exists to defend one claim from section 3.1:

    You cannot fix a bad chunk with a better model, a better index, or a better
    prompt. The information was destroyed at ingestion time.

So this is where the care goes.
"""

from app.ingest.chunker import Chunk, chunk_document, naive_chunks  # local — app/ingest/chunker.py
from app.ingest.loaders import Section, load_markdown               # local — app/ingest/loaders.py
                                        #   both re-exported so callers write
                                        #   `from app.ingest import chunk_document`

__all__ = ["Chunk", "Section", "chunk_document", "naive_chunks", "load_markdown"]
