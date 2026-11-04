import os

import pytest

from swe_agent.config import Settings
from swe_agent.indexing.embedder import (
    HashingEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_is_deterministic_and_normalized():
    e = HashingEmbedder(64)
    first = e.embed_documents(["def average(numbers): pass"])[0]
    second = e.embed_query("def average(numbers): pass")
    assert first == second
    assert len(first) == 64 == e.dim
    assert abs(dot(first, first) - 1.0) < 1e-9


def test_texts_with_shared_words_score_higher():
    e = HashingEmbedder()
    query = e.embed_query("average of numbers")
    related, unrelated = e.embed_documents(
        ["def average(numbers): return sum(numbers)", "open socket connect database url"]
    )
    assert dot(query, related) > dot(query, unrelated)


def test_empty_text_still_gives_a_valid_vector():
    vector = HashingEmbedder(32).embed_query("   ")
    assert abs(dot(vector, vector) - 1.0) < 1e-9


def test_factory_respects_backend_setting():
    settings = Settings(_env_file=None, embedding_backend="hashing")
    assert isinstance(get_embedder(settings), HashingEmbedder)
    settings = Settings(_env_file=None, embedding_backend="sentence-transformers")
    assert isinstance(get_embedder(settings), SentenceTransformerEmbedder)  # lazy: no download


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", 
                    reason="set RUN_SLOW=1 to use the real model")
def test_real_model_ranks_relevant_code_first():
    embedder = SentenceTransformerEmbedder(Settings(_env_file=None))
    docs = [
        "def average(numbers):\n    return sum(numbers) / len(numbers)",
        "def connect_database(url):\n    open a network socket to the server",
    ]
    query = embedder.embed_query("compute the mean of a list")
    scores = [dot(query, d) for d in embedder.embed_documents(docs)]
    assert scores[0] > scores[1]
    assert embedder.dim == 384