"""Merge several ranked lists into one."""

from __future__ import annotations

from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import SearchHit


def reciprocal_rank_fusion(rankings: dict[str, list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """RRF: each list gives a chunk 1/(k+rank) points. Points are added up.

    `rankings` maps a source name ("vector", "bm25") to its ranked hits.
    Only ranks are used, so the different score scales do not matter.
    """
    points: dict[str, float] = {}
    chunks: dict[str, CodeChunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    for source, hits in rankings.items():
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            points[chunk_id] = points.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = hit.chunk
            ranks.setdefault(chunk_id, {})[source] = rank

    order = sorted(points, key=lambda chunk_id: (-points[chunk_id], chunk_id))
    return [
        SearchHit(
            chunk=chunks[chunk_id],
            score=points[chunk_id],
            source="hybrid",
            vector_rank=ranks[chunk_id].get("vector"),
            bm25_rank=ranks[chunk_id].get("bm25"),
        )
        for chunk_id in order
    ]


def drop_overlapping(hits: list[SearchHit], max_overlap: float = 0.5) -> list[SearchHit]:
    """Keep the better-ranked hit when two chunks of one file share most of their lines.

    Our default window overlap is only 10 of 60 lines, so this is a safety net
    that matters when CHUNK_OVERLAP_LINES is set high.
    """
    kept: list[SearchHit] = []
    for hit in hits:
        c = hit.chunk
        is_duplicate = False
        for other in kept:
            o = other.chunk
            if o.file_path != c.file_path:
                continue
            shared = min(c.end_line, o.end_line) - max(c.start_line, o.start_line) + 1
            if shared <= 0:
                continue
            smaller = min(c.end_line - c.start_line, o.end_line - o.start_line) + 1
            if shared / smaller > max_overlap:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(hit)
    return kept