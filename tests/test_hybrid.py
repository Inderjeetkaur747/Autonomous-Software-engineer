import pytest
from qdrant_client import QdrantClient

from swe_agent.config import Settings
from swe_agent.indexing import (
    HashingEmbedder,
    IndexingError,
    QdrantStore,
    chunk_file,
    index_chunks,
)
from swe_agent.retrieval import HybridRetriever, OverlapReranker

FILES = {
    "mathkit/stats.py": (
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n\n"
        "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n"
    ),
    "app/users.py": "def get_user_by_email(email):\n    return db.find(email)\n",
    "app/net.py": "def connect_database(url):\n    return open_socket(url)\n",
    "README.md": "# Toolkit\nSmall helpers.\n",
}


def make_retriever(reranker=None, collection="repo_test"):
    settings = Settings(
        _env_file=None, embedding_backend="hashing", reranker_backend="overlap", default_top_k=3
    )
    chunks = []
    for path, source in FILES.items():
        chunks += chunk_file(path, source, "python" if path.endswith(".py") else "markdown")
    embedder = HashingEmbedder()
    store = QdrantStore(settings, client=QdrantClient(":memory:"))
    index_chunks(chunks, collection="repo_test", embedder=embedder, store=store)
    return HybridRetriever(
        collection=collection, embedder=embedder, store=store, chunks=chunks,
        reranker=reranker, settings=settings,
    )


def test_hybrid_finds_the_exact_identifier_and_explains_why():
    result = make_retriever().retrieve("get_user_by_email", rerank=False)
    top = result.hits[0]
    assert top.chunk.name == "get_user_by_email"
    assert top.source == "hybrid"
    assert top.bm25_rank == 1
    assert top.vector_rank is not None
    assert len(result.hits) <= 3


def test_rerank_changes_source_and_records_timings():
    result = make_retriever(OverlapReranker()).retrieve("average of numbers")
    assert result.reranked is True
    assert result.hits[0].chunk.name == "average"
    assert all(h.source == "reranked" for h in result.hits)
    assert {"vector_ms", "bm25_ms", "fusion_ms", "rerank_ms"} <= result.timings_ms.keys()


def test_single_modes_only_run_what_they_need():
    retriever = make_retriever()
    vector = retriever.retrieve("average numbers", mode="vector", rerank=False)
    assert vector.hits[0].source == "vector"
    assert "bm25_ms" not in vector.timings_ms
    keyword = retriever.retrieve("average numbers", mode="bm25", rerank=False)
    assert keyword.hits[0].source == "bm25"
    assert "vector_ms" not in keyword.timings_ms


def test_no_reranker_means_no_reranking():
    result = make_retriever(reranker=None).retrieve("average numbers", rerank=True)
    assert result.reranked is False


def test_bad_input_is_rejected():
    retriever = make_retriever()
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")  # type: ignore[arg-type]


def test_unindexed_collection_is_a_clear_error_but_bm25_still_works():
    retriever = make_retriever(collection="nope")
    with pytest.raises(IndexingError, match="not indexed"):
        retriever.retrieve("average")
    assert retriever.retrieve("average", mode="bm25", rerank=False).hits