"""Code search for the tools. It opens Qdrant only for the moment it needs it."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from swe_agent.config import Settings, get_settings
from swe_agent.indexing import (
    IndexingError,
    QdrantStore,
    collection_for,
    get_embedder,
    load_chunks,
)
from swe_agent.retrieval import HybridRetriever, get_reranker
from swe_agent.schemas.retrieval import RetrievalMode, RetrievalResult

logger = logging.getLogger(__name__)
class CodeSearch(Protocol):
    def search(self, query: str, k: int, mode: RetrievalMode) -> RetrievalResult: ...


class CodeSearcher:
    """Heavy parts (models, chunks) are cached. The Qdrant folder is opened per search
    and closed right after, because local Qdrant allows only one process at a time.
    """

    def __init__(self, run_id: str, run_root: Path, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._run_id = run_id
        self._run_root = Path(run_root)
        self._chunks = None
        self._embedder = None
        self._reranker = None

    def _prepare(self) -> None:
        if self._chunks is not None:
            return
        chunks_file = self._run_root / "chunks.jsonl"
        if not chunks_file.exists():
            raise IndexingError(f"No chunks.jsonl found. Run: swe-agent index {self._run_id}")
        self._chunks = load_chunks(chunks_file)
        self._embedder = get_embedder(self._settings)
        self._reranker = get_reranker(self._settings)

    def warm_up(self) -> None:
        """Load chunks and models now, so the first real search is fast."""
        logger.info("search warm-up: loading chunks")
        self._prepare()
        logger.info("search warm-up: loading embedding model")
        self._embedder.embed_query("warm up")
        load = getattr(self._reranker, "_load", None)
        if load is not None:
            logger.info("search warm-up: loading reranker")
            load()
        logger.info("search warm-up: done")

    def search(self, query: str, k: int = 5, mode: RetrievalMode = "hybrid") -> RetrievalResult:
        self._prepare()
        retriever = HybridRetriever(
            collection=collection_for(self._run_id),
            embedder=self._embedder,
            store=QdrantStore(self._settings),
            chunks=self._chunks,
            reranker=self._reranker,
            settings=self._settings,
        )
        try:
            return retriever.retrieve(query, k, mode=mode)
        finally:
            retriever.close()