import pytest
from qdrant_client import QdrantClient

from swe_agent.config import Settings
from swe_agent.indexing.chunker import chunk_file
from swe_agent.indexing.embedder import HashingEmbedder
from swe_agent.indexing.store import QdrantStore

FILES = {
    "stats.py": (
        "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n\n"
        "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n"
    ),
    "db.py": "def connect_database(url):\n    return open_socket(url)\n",
    "README.md": "# Project\nA tiny toolkit for numbers.\n",
}


def make_chunks():
    chunks = []
    for path, source in FILES.items():
        language = "python" if path.endswith(".py") else "markdown"
        chunks += chunk_file(path, source, language)
    return chunks


def make_store():
    return QdrantStore(Settings(_env_file=None), client=QdrantClient(":memory:"))


def fill(store, name="repo_test"):
    embedder = HashingEmbedder()
    chunks = make_chunks()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store.recreate_collection(name, embedder.dim)
    store.upsert_chunks(name, chunks, vectors)
    return embedder, chunks


def test_search_finds_the_relevant_chunk_and_keeps_payload():
    store = make_store()
    embedder, chunks = fill(store)
    hits = store.search("repo_test", embedder.embed_query("average numbers"), limit=3)
    assert hits[0].chunk.name == "average"
    assert hits[0].chunk in chunks  # payload round trip is lossless
    assert hits[0].chunk.citation.label == "stats.py:1-2"
    assert hits[0].score >= hits[-1].score


def test_filter_by_language():
    store = make_store()
    embedder, _ = fill(store)
    hits = store.search("repo_test", embedder.embed_query("numbers"), 10, language="markdown")
    assert hits
    assert {h.chunk.language for h in hits} == {"markdown"}


def test_indexing_twice_does_not_duplicate():
    store = make_store()
    fill(store)
    first = store.count("repo_test")
    fill(store)
    assert store.count("repo_test") == first == len(make_chunks())


def test_length_mismatch_is_rejected():
    store = make_store()
    store.recreate_collection("c", 8)
    with pytest.raises(ValueError):
        store.upsert_chunks("c", make_chunks(), [[0.0] * 8])


def test_delete_collection():
    store = make_store()
    fill(store)
    assert store.collection_exists("repo_test")
    store.delete_collection("repo_test")
    assert not store.collection_exists("repo_test")
    store.delete_collection("repo_test")  # deleting again is harmless