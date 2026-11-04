"""Typed search results."""

from typing import Literal

from pydantic import BaseModel, Field

from swe_agent.schemas.code import CodeChunk

RetrievalMode = Literal["vector", "bm25", "hybrid"]


class SearchHit(BaseModel):
    chunk: CodeChunk
    score: float
    source: str = "vector"  # vector | bm25 | hybrid | reranked
    vector_rank: int | None = None  # position in the vector list (1 = best)
    bm25_rank: int | None = None  # position in the BM25 list (1 = best)


class RetrievalResult(BaseModel):
    query: str
    mode: RetrievalMode
    reranked: bool
    hits: list[SearchHit] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)