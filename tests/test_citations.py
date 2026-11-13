from swe_agent.indexing import chunk_file
from swe_agent.retrieval import (
    check_answer_citations,
    check_citations_in_lines,
    citation_correctness,
    format_context,
    verify_chunk,
)
from swe_agent.schemas.code import ChunkKind, CodeChunk
from swe_agent.schemas.retrieval import SearchHit


def test_check_citations_in_lines_uses_a_plain_line_map():
    check = check_citations_in_lines("a.py:2-3 and a.py:9", {"a.py": {1, 2, 3}})
    assert check.valid == ["a.py:2-3"]
    assert check.invalid == ["a.py:9-9"]

def hit_for(path, start, end, text="x = 1"):
    chunk = CodeChunk(
        chunk_id=f"{path}{start}", file_path=path, start_line=start, end_line=end,
        language="python", kind=ChunkKind.BLOCK, text=text,
    )
    return SearchHit(chunk=chunk, score=1.0)


# ---------------------------------------------------------- answer citations
def test_valid_and_invalid_citations_are_separated():
    hits = [hit_for("mathkit/stats.py", 1, 10)]
    text = (
        "The bug is in mathkit/stats.py:4-6. See also mathkit/stats.py:50-60, "
        "other/file.py:1-2 and mathkit/stats.py:9-3. Visit localhost.com:8080."
    )
    check = check_answer_citations(text, hits)
    assert check.valid == ["mathkit/stats.py:4-6"]
    assert check.invalid == ["mathkit/stats.py:50-60", "other/file.py:1-2", "mathkit/stats.py:9-3"]
    assert not check.all_valid


def test_citation_may_span_two_retrieved_chunks():
    hits = [hit_for("a.py", 1, 10), hit_for("a.py", 11, 20)]
    assert check_answer_citations("see a.py:8-12", hits).valid == ["a.py:8-12"]


def test_single_line_backslash_and_dot_slash_forms():
    hits = [hit_for("tests/test_stats.py", 1, 10)]
    check = check_answer_citations("x tests\\test_stats.py:5 and ./tests/test_stats.py:6", hits)
    assert check.valid == ["tests/test_stats.py:5-5", "tests/test_stats.py:6-6"]


def test_text_without_citations_is_trivially_valid():
    check = check_answer_citations("No citations here.", [])
    assert check.all_valid and check.valid == []


# --------------------------------------------------------------- disk checks
def test_verify_chunk_detects_changes_missing_files_and_escapes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "def a():\n    return 1\n"
    (repo / "a.py").write_text(source)
    chunk = chunk_file("a.py", source, "python")[0]
    assert verify_chunk(repo, chunk)

    (repo / "a.py").write_text("def a():\n    return 2\n")  # file changed after indexing
    assert not verify_chunk(repo, chunk)

    missing = chunk.model_copy(update={"file_path": "nope.py"})
    assert not verify_chunk(repo, missing)

    (tmp_path / "x.py").write_text("x = 1\n")
    escape = hit_for("../x.py", 1, 1, text="x = 1").chunk
    assert not verify_chunk(repo, escape)  # the real file matches, but it is outside the repo


def test_citation_correctness_is_a_ratio(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "def a():\n    return 1\n"
    (repo / "a.py").write_text(source)
    good = chunk_file("a.py", source, "python")[0]
    stale = good.model_copy(update={"chunk_id": "z", "text": "def a():\n    return 999"})
    hits = [SearchHit(chunk=good, score=1), SearchHit(chunk=stale, score=1)]
    assert citation_correctness(repo, hits) == 0.5
    assert citation_correctness(repo, []) == 0.0


# ------------------------------------------------------------------- context
def test_format_context_numbers_cites_and_redacts_secrets():
    secret = "gsk_" + "a" * 30
    chunk = chunk_file("a.py", f"def a():\n    key = '{secret}'\n", "python")[0]
    text = format_context([SearchHit(chunk=chunk, score=1.0)])
    assert text.startswith("[1] a.py:1-2 (function: a)")
    assert secret not in text
    assert "[REDACTED]" in text


def test_format_context_respects_max_chars():
    hits = [hit_for(f"f{i}.py", 1, 20, text="line\n" * 20) for i in range(10)]
    out = format_context(hits, max_chars=250)
    assert len(out) <= 250
    assert out.startswith("[1] f0.py:1-20")
    assert len(format_context(hits[:1], max_chars=40)) <= 40  # first block alone is too big