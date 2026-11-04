from swe_agent.indexing.chunker import chunk_file
from swe_agent.schemas.code import ChunkKind

SAMPLE = '''"""Module doc."""
import os


@decorator
def alpha(x):
    return x


class Box:
    value = 1

    def get(self):
        return self.value
'''


def assert_lines_exact(chunks, source):
    """The most important property: citations must point at the real text."""
    lines = source.split("\n")
    for c in chunks:
        assert "\n".join(lines[c.start_line - 1 : c.end_line]) == c.text


def test_python_functions_and_classes_become_chunks():
    chunks = chunk_file("pkg/mod.py", SAMPLE, "python")
    assert_lines_exact(chunks, SAMPLE)
    by_name = {c.name: c for c in chunks if c.name}
    assert by_name["alpha"].kind == ChunkKind.FUNCTION
    assert by_name["alpha"].start_line == 5  # the decorator line is included
    assert by_name["alpha"].end_line == 7
    assert by_name["Box"].kind == ChunkKind.CLASS
    module_chunks = [c for c in chunks if c.kind == ChunkKind.MODULE]
    assert module_chunks and module_chunks[0].start_line == 1


def make_big_class():
    parts = ["class Big:", "    x = 1", ""]
    for i in range(5):
        parts += [
            f"    def m{i}(self):", "        a = 1", "        b = 2", "        c = 3",
            "        return a", "",
        ]
    return "\n".join(parts) + "\n"


def test_big_class_is_split_into_methods():
    source = make_big_class()
    chunks = chunk_file("big.py", source, "python", max_lines=12, overlap=2)
    assert_lines_exact(chunks, source)
    names = {c.name for c in chunks}
    assert {"Big.m0", "Big.m4"} <= names
    assert any(c.kind == ChunkKind.CLASS and c.name == "Big" for c in chunks)
    assert all(c.kind == ChunkKind.METHOD for c in chunks if "." in c.name)


def test_plain_text_uses_overlapping_windows():
    source = "\n".join(f"line {i}" for i in range(1, 101))
    chunks = chunk_file("notes.txt", source, "text", max_lines=40, overlap=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 40), (31, 70), (61, 100)]
    assert all(c.kind == ChunkKind.BLOCK for c in chunks)
    assert_lines_exact(chunks, source)


def test_broken_python_falls_back_to_windows():
    source = "def broken(:\n    pass\n"
    chunks = chunk_file("bad.py", source, "python")
    assert len(chunks) == 1
    assert chunks[0].kind == ChunkKind.BLOCK


def test_windows_line_endings_are_handled():
    chunks = chunk_file("a.py", "def a():\r\n    return 1\r\n", "python")
    assert chunks[0].name == "a"
    assert "\r" not in chunks[0].text


def test_empty_and_blank_files_give_no_chunks():
    assert chunk_file("a.py", "", "python") == []
    assert chunk_file("a.py", "\n\n   \n", "python") == []


def test_chunk_id_is_stable_and_embed_text_has_header():
    first = chunk_file("a.py", "def a():\n    return 1\n", "python")[0]
    second = chunk_file("a.py", "def a():\n    return 1\n", "python")[0]
    assert first.chunk_id == second.chunk_id
    assert first.embed_text.startswith("# File: a.py | function: a")
    assert first.citation.label == "a.py:1-2"