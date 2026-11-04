"""Second-stage ranking: read the query and each candidate together.

Backends:
  - CrossEncoderReranker: real and local (default)
  - OverlapReranker: tiny and offline, for unit tests and CI
"""

from __future__ import annotations

import logging
from typing import Protocol

from swe_agent.config import Settings, get_settings
from swe_agent.retrieval.tokenizer import tokenize_code
from swe_agent.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)

RERANK_MAX_CHARS = 2000  # the model reads at most 512 tokens anyway


class Reranker(Protocol):
    model_name: str

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class OverlapReranker:
    """Score = share of the query's words found in the chunk. Not smart, but deterministic."""

    model_name = "token-overlap"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        wanted = set(tokenize_code(query))
        if not wanted:
            return hits[:top_k]
        scored = []
        for hit in hits:
            have = set(tokenize_code(hit.chunk.embed_text))
            score = len(wanted & have) / len(wanted)
            scored.append(hit.model_copy(update={"score": score, "source": "reranked"}))
        scored.sort(key=lambda h: -h.score)  # sort is stable: ties keep the old order
        return scored[:top_k]


class CrossEncoderReranker:
    """A cross-encoder gives each (query, chunk) pair a relevance score."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.model_name = settings.reranker_model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy: keeps startup fast

            logger.info("loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
        return self._model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.chunk.embed_text[:RERANK_MAX_CHARS]) for hit in hits]
        scores = self._load().predict(pairs, show_progress_bar=False)
        scored = [
            # model_copy keeps vector_rank and bm25_rank, so we can still explain the hit
            hit.model_copy(update={"score": float(score), "source": "reranked"})
            for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda h: -h.score)
        return scored[:top_k]


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    settings = settings or get_settings()
    if settings.reranker_backend == "none":
        return None
    if settings.reranker_backend == "overlap":
        return OverlapReranker()
    return CrossEncoderReranker(settings)