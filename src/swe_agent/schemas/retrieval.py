"""Typed search results. Phase 6 will add keyword and hybrid sources."""

from pydantic import BaseModel

from swe_agent.schemas.code import CodeChunk


class SearchHit(BaseModel):
    chunk: CodeChunk
    score: float
    source: str = "vector"  # later: "bm25", "hybrid", "reranked"