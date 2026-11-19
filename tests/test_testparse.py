from swe_agent.agent import describe_outcome, find_weakened_tests, parse_test_run
from swe_agent.schemas.tools import ToolResult


def result(stdout, exit_code=1, timed_out=False):
    header = f"exit_code={exit_code} passed={exit_code == 0} timed_out={timed_out} duration_ms=5.0"
    return ToolResult(
        ok=True,
        output=f"{header}\n--- stdout (end) ---\n{stdout}",
        meta={
            "exit_code": exit_code,
            "passed": exit_code == 0 and not timed_out,
            "timed_out": timed_out,
        },
    )


FAIL = (
    "F..\n=== FAILURES ===\nstuff\n"
    "FAILED tests/test_a.py::test_x - assert 1.0 == 0.0\n1 failed, 2 passed in 0.10s\n"
)


def test_failures_and_counts_are_parsed():
    o = parse_test_run(result(FAIL))
    assert o.failed_tests == ["tests/test_a.py::test_x"]
    assert o.failure_messages["tests/test_a.py::test_x"] == "assert 1.0 == 0.0"
    assert o.counts == {"failed": 1, "passed": 2}
    assert not o.passed and o.exit_code == 1
    assert "exit_code=" not in o.output_tail


def test_passing_run():
    o = parse_test_run(result("...\n3 passed in 0.02s\n", exit_code=0))
    assert o.passed and o.counts == {"passed": 3} and o.failed_tests == []


def test_collection_error_and_no_tests():
    err = parse_test_run(
        result("ERROR tests/test_b.py - ImportError: cannot import x\n1 error in 0.05s", 2)
    )
    assert err.failed_tests == ["tests/test_b.py"]
    assert err.counts == {"errors": 1}
    none = parse_test_run(result("no tests ran in 0.01s", 5))
    assert none.counts == {} and not none.passed and "no tests ran" in none.summary_line


def test_timeout():
    o = parse_test_run(result("partial", exit_code=None, timed_out=True))
    assert o.timed_out and not o.passed and o.exit_code is None


def test_signature_ignores_timing_but_not_the_message():
    a = parse_test_run(result(FAIL))
    b = parse_test_run(result(FAIL.replace("0.10s", "9.99s")))
    c = parse_test_run(result(FAIL.replace("0.0", "2.0")))
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()


def test_long_output_keeps_the_end():
    o = parse_test_run(result("x" * 5000 + "\n1 failed in 0.1s"), tail_chars=500)
    assert len(o.output_tail) <= 520
    assert o.output_tail.endswith("1 failed in 0.1s")


def test_describe_outcome_lists_failures():
    text = describe_outcome(parse_test_run(result(FAIL)))
    assert "tests/test_a.py::test_x: assert 1.0 == 0.0" in text


DIFF = """diff --git a/tests/test_a.py b/tests/test_a.py
--- a/tests/test_a.py
+++ b/tests/test_a.py
@@ -1,4 +1,3 @@
 def test_a():
-    assert average([1]) == 1
-    assert average([2]) == 2
+    assert average([1]) == 1
diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-    assert x
+    assert y
"""


def test_removed_assertions_are_flagged_only_in_test_files():
    warnings = find_weakened_tests(DIFF)
    assert warnings == ["tests/test_a.py: 2 assertion line(s) removed, 1 added"]


def test_skip_markers_and_deleted_test_files_are_flagged():
    skip = "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n"
    skip += "@@ -1 +1,2 @@\n+@pytest.mark.skip\n def test_a(): pass\n"
    assert find_weakened_tests(skip) == ["tests/t.py: a skip or xfail marker was added"]
    gone = "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ /dev/null\n"
    gone += "@@ -1,2 +0,0 @@\n-def test_a():\n-    assert 1\n"
    assert len(find_weakened_tests(gone)) == 1


def test_a_clean_diff_has_no_warnings():
    clean = "diff --git a/tests/t.py b/tests/t.py\n--- a/tests/t.py\n+++ b/tests/t.py\n"
    clean += "@@ -1 +1,3 @@\n def test_a(): pass\n+def test_b():\n+    assert 1\n"
    assert find_weakened_tests(clean) == []
    assert find_weakened_tests("") == []