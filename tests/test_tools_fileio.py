import pytest

from swe_agent.tools.fileio import (
    EditError,
    apply_replace,
    atomic_write,
    load_text_strict,
    short_diff,
    to_bytes,
)


def test_lf_and_crlf_are_detected():
    lf = load_text_strict(b"a\nb\n")
    crlf = load_text_strict(b"a\r\nb\r\n")
    assert (lf.text, lf.newline) == ("a\nb\n", "\n")
    assert (crlf.text, crlf.newline) == ("a\nb\n", "\r\n")
    assert load_text_strict(b"no newline").newline == "\n"


@pytest.mark.parametrize("data", [b"a\r\nb\n", b"a\rb\n", b"\x00\x01", b"\xff\xfe\xfa"])
def test_files_we_could_corrupt_are_refused(data):
    with pytest.raises(EditError):
        load_text_strict(data)


def test_replace_exactly_once():
    assert apply_replace("a\nb\n", "b", "c") == "a\nc\n"
    assert apply_replace("a\nb\n", "a\r\nb", "x") == "x\n"  # CRLF in old_str is normalized


@pytest.mark.parametrize(
    ("text", "old", "new", "message"),
    [
        ("a\n", "zzz", "y", "not found"),
        ("x\nx\n", "x", "y", "2 times"),
        ("a\n", "a", "a", "identical"),
        ("a\n", "", "a", "empty"),
    ],
)
def test_bad_replacements_explain_themselves(text, old, new, message):
    with pytest.raises(EditError, match=message):
        apply_replace(text, old, new)


def test_line_endings_roundtrip():
    assert to_bytes("a\nb\n", "\r\n") == b"a\r\nb\r\n"
    assert to_bytes("a\nb\n", "\n") == b"a\nb\n"


def test_atomic_write_creates_folders_replaces_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "x" / "y.py"
    atomic_write(target, b"one")
    atomic_write(target, b"two")
    assert target.read_bytes() == b"two"
    assert [p.name for p in target.parent.iterdir()] == ["y.py"]


def test_short_diff_and_its_limit():
    diff = short_diff("a.py", "x = 1\ny = 2\n", "x = 1\ny = 3\n")
    assert diff.startswith("--- a/a.py")
    assert "-y = 2" in diff
    assert "+y = 3" in diff
    old = "\n".join(f"a{i}" for i in range(200))
    new = "\n".join(f"b{i}" for i in range(200))
    assert "more diff lines" in short_diff("big.py", old, new)