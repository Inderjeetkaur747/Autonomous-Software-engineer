import pytest
from qdrant_client import QdrantClient

from swe_agent.config import Settings
from swe_agent.indexing import (
    HashingEmbedder,
    IndexingError,
    QdrantStore,
    chunk_file,
    collection_for,
    index_chunks,
    vector_search,
)


def make_store():
    return QdrantStore(Settings(_env_file=None), client=QdrantClient(":memory:"))


SOURCE = "def average(numbers):\n    return sum(numbers) / len(numbers)\n"


def test_index_then_search_returns_citation():
    store, embedder = make_store(), HashingEmbedder()
    chunks = chunk_file("mathkit/stats.py", SOURCE, "python")
    name = collection_for("abc123abc123")
    assert name == "repo_abc123abc123"

    stats = index_chunks(chunks, collection=name, embedder=embedder, store=store)
    assert stats.chunk_count == len(chunks)
    assert stats.vector_dim == embedder.dim

    hits = vector_search("average of numbers", collection=name, embedder=embedder, store=store)
    assert hits[0].chunk.citation.label == "mathkit/stats.py:1-2"


def test_empty_chunk_list_is_an_error():
    with pytest.raises(IndexingError):
        index_chunks([], collection="c", embedder=HashingEmbedder(), store=make_store())


def test_search_before_index_is_a_clear_error():
    with pytest.raises(IndexingError, match="not indexed"):
        vector_search("x", collection="nope", embedder=HashingEmbedder(), store=make_store())


def test_blank_query_is_rejected():
    with pytest.raises(ValueError):
        vector_search("  ", collection="c", embedder=HashingEmbedder(), store=make_store())