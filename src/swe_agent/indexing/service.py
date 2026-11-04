"""index_chunks() writes to Qdrant. vector_search() reads from it."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from swe_agent.indexing.embedder import Embedder
from swe_agent.indexing.store import QdrantStore
from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Indexing or searching could not be done."""


@dataclass
class IndexStats:
    collection: str
    chunk_count: int
    vector_dim: int
    embedding_model: str
    seconds: float


def collection_for(run_id: str) -> str:
    """One collection per run keeps runs fully separate."""
    return f"repo_{run_id}"


def index_chunks(
    chunks: list[CodeChunk],
    *,
    collection: str,
    embedder: Embedder,
    store: QdrantStore,
    batch_size: int = 32,
) -> IndexStats:
    if not chunks:
        raise IndexingError("There are no chunks to index. Did the scan find any code?")

    started = time.perf_counter()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embedder.embed_documents([c.embed_text for c in batch]))
        logger.info("embedded %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))

    dim = len(vectors[0])
    store.recreate_collection(collection, dim)
    store.upsert_chunks(collection, chunks, vectors)

    stored = store.count(collection)
    if stored != len(chunks):
        raise IndexingError(f"Expected {len(chunks)} points but Qdrant has {stored}.")

    return IndexStats(
        collection=collection,
        chunk_count=stored,
        vector_dim=dim,
        embedding_model=embedder.model_name,
        seconds=round(time.perf_counter() - started, 2),
    )


def vector_search(
    query: str,
    *,
    collection: str,
    embedder: Embedder,
    store: QdrantStore,
    k: int = 8,
    language: str | None = None,
    kind: str | None = None,
) -> list[SearchHit]:
    if not query.strip():
        raise ValueError("The search query is empty.")
    if not store.collection_exists(collection):
        raise IndexingError(f"{collection} is not indexed yet. Run: swe-agent index <run_id>")
    vector = embedder.embed_query(query)
    return store.search(collection, vector, k, language=language, kind=kind)