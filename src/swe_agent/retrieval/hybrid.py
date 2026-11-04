"""HybridRetriever: vector + BM25 -> RRF -> dedupe -> rerank."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from swe_agent.config import Settings, get_settings
from swe_agent.indexing import (
    Embedder,
    IndexingError,
    QdrantStore,
    collection_for,
    get_embedder,
    load_chunks,
    vector_search,
)
from swe_agent.retrieval.bm25 import Bm25Index
from swe_agent.retrieval.fusion import drop_overlapping, reciprocal_rank_fusion
from swe_agent.retrieval.reranker import Reranker, get_reranker
from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import RetrievalMode, RetrievalResult, SearchHit

logger = logging.getLogger(__name__)

_MODES = ("vector", "bm25", "hybrid")


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


class HybridRetriever:
    def __init__(
        self,
        *,
        collection: str,
        embedder: Embedder,
        store: QdrantStore,
        chunks: list[CodeChunk],
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._collection = collection
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._bm25 = Bm25Index(chunks)

    def close(self) -> None:
        self._store.close()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValueError("The search query is empty.")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, not {mode!r}")
        k = k or self._settings.default_top_k
        pool_size = self._settings.retrieval_candidates
        timings: dict[str, float] = {}
        vector_hits: list[SearchHit] = []
        bm25_hits: list[SearchHit] = []

        if mode in ("vector", "hybrid"):
            started = time.perf_counter()
            vector_hits = vector_search(
                query,
                collection=self._collection,
                embedder=self._embedder,
                store=self._store,
                k=pool_size,
            )
            timings["vector_ms"] = _ms(started)
        if mode in ("bm25", "hybrid"):
            started = time.perf_counter()
            bm25_hits = self._bm25.search(query, pool_size)
            timings["bm25_ms"] = _ms(started)

        if mode == "vector":
            candidates = vector_hits
        elif mode == "bm25":
            candidates = bm25_hits
        else:
            started = time.perf_counter()
            candidates = reciprocal_rank_fusion(
                {"vector": vector_hits, "bm25": bm25_hits}, k=self._settings.rrf_k
            )
            timings["fusion_ms"] = _ms(started)
        candidates = drop_overlapping(candidates)

        did_rerank = False
        if rerank and self._reranker is not None and candidates:
            started = time.perf_counter()
            pool = candidates[: max(k, self._settings.rerank_top_n)]
            final = self._reranker.rerank(query, pool, k)
            timings["rerank_ms"] = _ms(started)
            did_rerank = True
        else:
            final = candidates[:k]

        logger.info(
            "retrieve mode=%s reranked=%s hits=%d timings=%s", mode, did_rerank, len(final), timings
        )
        return RetrievalResult(
            query=query, mode=mode, reranked=did_rerank, hits=final, timings_ms=timings
        )


def build_retriever(
    run_id: str, run_root: Path, settings: Settings | None = None
) -> HybridRetriever:
    """Wire everything for one prepared and indexed run."""
    settings = settings or get_settings()
    chunks_file = run_root / "chunks.jsonl"
    if not chunks_file.exists():
        raise IndexingError(f"No chunks.jsonl found. Run: swe-agent index {run_id}")
    chunks = load_chunks(chunks_file)
    return HybridRetriever(
        collection=collection_for(run_id),
        embedder=get_embedder(settings),
        store=QdrantStore(settings),
        chunks=chunks,
        reranker=get_reranker(settings),
        settings=settings,
    )