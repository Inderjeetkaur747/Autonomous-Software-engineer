"""Qdrant wrapper. Local mode by default (a folder on disk), or a server via QDRANT_URL."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from swe_agent.config import Settings, get_settings
from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import SearchHit

logger = logging.getLogger(__name__)


def point_id(chunk_id: str) -> str:
    """Qdrant IDs must be integers or UUIDs. Same chunk_id always gives the same UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        settings = settings or get_settings()
        if client is not None:
            self._client = client  # tests pass QdrantClient(":memory:")
        elif settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            path = Path(settings.qdrant_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------ collections
    def collection_exists(self, name: str) -> bool:
        return bool(self._client.collection_exists(name))

    def recreate_collection(self, name: str, dim: int) -> None:
        """Start clean, so indexing the same run twice never makes duplicates."""
        if self.collection_exists(name):
            self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        logger.info("collection created name=%s dim=%d", name, dim)

    def delete_collection(self, name: str) -> None:
        if self.collection_exists(name):
            self._client.delete_collection(name)
            logger.info("collection deleted name=%s", name)

    def count(self, name: str) -> int:
        return int(self._client.count(collection_name=name, exact=True).count)

    # ----------------------------------------------------------------- write
    def upsert_chunks(
        self,
        name: str,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for start in range(0, len(chunks), batch_size):
            batch = [
                models.PointStruct(
                    id=point_id(chunk.chunk_id),
                    vector=vector,
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(
                    chunks[start : start + batch_size],
                    vectors[start : start + batch_size],
                    strict=True,
                )
            ]
            self._client.upsert(collection_name=name, points=batch, wait=True)

    # ------------------------------------------------------------------ read
    def search(
        self,
        name: str,
        vector: list[float],
        limit: int = 8,
        *,
        language: str | None = None,
        kind: str | None = None,
    ) -> list[SearchHit]:
        conditions = []
        if language:
            conditions.append(
                models.FieldCondition(key="language", match=models.MatchValue(value=language))
            )
        if kind:
            conditions.append(
                models.FieldCondition(key="kind", match=models.MatchValue(value=kind))
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        response = self._client.query_points(
            collection_name=name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk=CodeChunk.model_validate(point.payload),
                score=float(point.score),
                source="vector",
            )
            for point in response.points
            if point.payload
        ]