"""One function, one `if` — the storage swap, same shape as app/llm/factory.py.

memory : embeds at boot, dies with the process. The default; keeps
         `git clone && pytest` true with zero infrastructure.
qdrant : persistent. A local folder by default; QDRANT_URL flips the SAME
         client to a server — deployment is config, not code.
"""

from app.config import Settings         # local — app/config.py
from app.store.base import VectorStore  # local — app/store/base.py (the return TYPE;
                                        #   concrete stores imported lazily so memory
                                        #   mode never needs qdrant_client installed)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "qdrant":
        from app.store.qdrant import QdrantStore

        return QdrantStore(
            path=settings.qdrant_path,
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )

    from app.store.memory import MemoryStore

    return MemoryStore()
