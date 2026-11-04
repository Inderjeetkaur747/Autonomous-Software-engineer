import pytest

from swe_agent.retrieval import drop_overlapping, reciprocal_rank_fusion
from swe_agent.schemas.code import ChunkKind, CodeChunk, make_chunk_id
from swe_agent.schemas.retrieval import SearchHit


def mk(path, start=1, end=3):
    text = f"code {path} {start}"
    return CodeChunk(
        chunk_id=make_chunk_id(path, start, end, text),
        file_path=path,
        start_line=start,
        end_line=end,
        language="python",
        kind=ChunkKind.BLOCK,
        text=text,
    )


def hit(chunk):
    return SearchHit(chunk=chunk, score=1.0)


def test_rrf_rewards_chunks_found_by_both_lists():
    x, y, z, w = (mk(f"{n}.py") for n in "xyzw")
    fused = reciprocal_rank_fusion(
        {"vector": [hit(x), hit(y), hit(z)], "bm25": [hit(y), hit(w)]}, k=60
    )
    assert [h.chunk.file_path for h in fused] == ["y.py", "x.py", "w.py", "z.py"]
    best = fused[0]
    assert best.score == pytest.approx(1 / 62 + 1 / 61)
    assert (best.vector_rank, best.bm25_rank) == (2, 1)
    assert fused[1].bm25_rank is None  # x was not found by BM25
    assert all(h.source == "hybrid" for h in fused)


def test_rrf_with_one_empty_list_keeps_the_other_order():
    a, b = mk("a.py"), mk("b.py")
    fused = reciprocal_rank_fusion({"vector": [hit(a), hit(b)], "bm25": []})
    assert [h.chunk.file_path for h in fused] == ["a.py", "b.py"]


def test_drop_overlapping_keeps_better_hit_only():
    first = hit(mk("a.py", 1, 60))
    heavy_overlap = hit(mk("a.py", 11, 70))  # shares 50 of 60 lines -> dropped
    other_file = hit(mk("b.py", 1, 60))  # different file -> kept
    no_overlap = hit(mk("a.py", 61, 120))  # touches no shared line -> kept
    kept = drop_overlapping([first, heavy_overlap, other_file, no_overlap])
    assert kept == [first, other_file, no_overlap]


def test_small_overlap_is_allowed():
    kept = drop_overlapping([hit(mk("a.py", 1, 60)), hit(mk("a.py", 51, 110))])
    assert len(kept) == 2  # shares only 10 of 60 lines