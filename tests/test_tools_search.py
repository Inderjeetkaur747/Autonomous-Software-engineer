import pytest

from swe_agent.config import Settings
from swe_agent.indexing import (
    HashingEmbedder,
    IndexingError,
    QdrantStore,
    chunk_file,
    collection_for,
    index_chunks,
    save_chunks,
)
from swe_agent.tools import CodeSearcher

SOURCE = "def average(numbers):\n    return sum(numbers) / len(numbers)\n"


def make_settings(tmp_path):
    return Settings(
        _env_file=None,
        embedding_backend="hashing",
        reranker_backend="overlap",
        qdrant_path=tmp_path / "qdrant",
        qdrant_url="",
    )


def test_searcher_finds_code_and_can_search_twice(tmp_path):
    settings = make_settings(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    chunks = chunk_file("mathkit/stats.py", SOURCE, "python")
    save_chunks(chunks, run_root / "chunks.jsonl")

    store = QdrantStore(settings)
    index_chunks(chunks, collection=collection_for("abc"), embedder=HashingEmbedder(), store=store)
    store.close()  # local Qdrant: only one open client at a time

    searcher = CodeSearcher("abc", run_root, settings)
    first = searcher.search("average numbers", 3, "hybrid")
    second = searcher.search("average numbers", 3, "bm25")  # Qdrant was closed after the first
    assert first.hits[0].chunk.name == "average"
    assert second.hits[0].chunk.citation.label == "mathkit/stats.py:1-2"


def test_searcher_without_an_index_is_a_clear_error(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    with pytest.raises(IndexingError, match="chunks.jsonl"):
        CodeSearcher("abc", run_root, make_settings(tmp_path)).search("x", 3, "hybrid")