import os

import pytest

from swe_agent.config import Settings
from swe_agent.indexing import chunk_file
from swe_agent.retrieval import CrossEncoderReranker, OverlapReranker, get_reranker
from swe_agent.schemas.retrieval import SearchHit

SOURCE = (
    "def average(numbers):\n    return sum(numbers) / len(numbers)\n\n\n"
    "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered) // 2]\n"
)


def hits_for_source():
    return [SearchHit(chunk=c, score=0.0, vector_rank=i + 1) for i, c in
            enumerate(c for c in chunk_file("stats.py", SOURCE, "python") if c.name)]


def test_overlap_reranker_moves_best_match_up_and_keeps_ranks():
    hits = hits_for_source()
    assert hits[0].chunk.name == "average"  # starts first
    result = OverlapReranker().rerank("median sorted values", hits, top_k=2)
    assert result[0].chunk.name == "median"
    assert result[0].source == "reranked"
    assert result[0].vector_rank == 2  # the earlier rank is not lost


def test_top_k_and_empty_input():
    reranker = OverlapReranker()
    assert len(reranker.rerank("average", hits_for_source(), top_k=1)) == 1
    assert reranker.rerank("average", [], top_k=3) == []


def test_factory_follows_settings():
    assert get_reranker(Settings(_env_file=None, reranker_backend="none")) is None
    assert isinstance(
        get_reranker(Settings(_env_file=None, reranker_backend="overlap")), OverlapReranker
    )
    assert isinstance(  # lazy: nothing is downloaded here
        get_reranker(Settings(_env_file=None, reranker_backend="cross-encoder")),
        CrossEncoderReranker,
    )


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                     reason="set RUN_SLOW=1 to use the real model")
def test_real_cross_encoder_prefers_the_relevant_chunk():
    reranker = CrossEncoderReranker(Settings(_env_file=None))
    result = reranker.rerank("compute the mean of a list", hits_for_source(), top_k=2)
    assert result[0].chunk.name == "average"