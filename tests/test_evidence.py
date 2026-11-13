from swe_agent.agent.evidence import Evidence
from swe_agent.indexing import chunk_file
from swe_agent.retrieval import format_context
from swe_agent.schemas.retrieval import SearchHit
from swe_agent.schemas.tools import ToolResult

SRC = "def average(numbers):\n    return sum(numbers) / len(numbers)\n"
GITHUB = "ghp_" + "b" * 30


def hits_for(source=SRC, path="mathkit/stats.py"):
    return [SearchHit(chunk=c, score=1.0) for c in chunk_file(path, source, "python") if c.name]


def read_result(path, numbered):
    output = f"{path} (lines 1-{len(numbered)})\n" + "\n".join(numbered)
    return ToolResult(ok=True, output=output, meta={"path": path})


def test_hits_become_cited_blocks_and_shown_lines():
    ev = Evidence()
    ev.add_hits(hits_for())
    text = ev.render()
    assert text.startswith("### mathkit/stats.py:1-2 (function: average)")
    assert ev.shown == {"mathkit/stats.py": {1, 2}}


def test_read_file_lines_come_from_the_printed_numbers():
    ev = Evidence()
    ev.add_result("read_file", read_result("a.py", ["4 | def a():", "5 |     return 1", "6 | x"]))
    assert "### a.py:4-6 (read_file)" in ev.render()
    assert ev.shown == {"a.py": {4, 5, 6}}


def test_a_long_read_is_cut_and_only_kept_lines_count_as_shown():
    ev = Evidence(block_chars=300)
    numbered = [f"{n:>3} | {'x' * 50}" for n in range(1, 101)]
    ev.add_result("read_file", read_result("big.py", numbered))
    ev.render()
    assert 1 in ev.shown["big.py"]
    assert 100 not in ev.shown["big.py"]
    assert len(ev.shown["big.py"]) < 10


def test_search_results_and_hits_are_not_duplicated():
    hits = hits_for()
    ev = Evidence()
    ev.add_hits(hits)
    output = format_context(hits)
    ev.add_result("search_code", ToolResult(ok=True, output=output, meta={}))
    assert ev.render().count("### mathkit/stats.py:1-2") == 1


def test_failed_results_and_unknown_tools_add_nothing():
    ev = Evidence()
    ev.add_result("read_file", ToolResult(ok=False, error="nope"))
    ev.add_result("edit_file", ToolResult(ok=True, output="x"))
    assert ev.render() == ""
    assert ev.shown == {}


def test_blocks_that_do_not_fit_are_not_counted_as_shown():
    ev = Evidence(max_chars=200)
    ev.add_result("read_file", read_result("a.py", ["1 | " + "y" * 100]))
    ev.add_result("read_file", read_result("b.py", ["1 | " + "y" * 100]))
    ev.render()
    assert "a.py" in ev.shown
    assert "b.py" not in ev.shown


def test_secrets_are_redacted():
    source = f"def a():\n    k = '{GITHUB}'\n"
    ev = Evidence()
    ev.add_hits(hits_for(source, "a.py"))
    assert GITHUB not in ev.render()