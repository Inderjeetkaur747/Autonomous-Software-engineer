"""Cut source files into chunks that have exact line numbers.

Python: split on functions/classes using `ast`.
Everything else (and any oversized piece): overlapping line windows.
"""

from __future__ import annotations

import ast

from swe_agent.schemas.code import ChunkKind, CodeChunk, make_chunk_id

Segment = tuple[int, int, ChunkKind, str]  # (start_line, end_line, kind, name), 1-based


def split_lines(text: str) -> list[str]:
    """Split the same way Python's `ast` counts lines (\\n, \\r\\n and \\r only)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a final newline does not create an extra line
    return lines


def _windows(start: int, end: int, max_lines: int, overlap: int) -> list[tuple[int, int]]:
    if end - start + 1 <= max_lines:
        return [(start, end)]
    step = max_lines - overlap
    result = []
    s = start
    while True:
        e = min(s + max_lines - 1, end)
        result.append((s, e))
        if e >= end:
            break
        s += step
    return result


def _node_range(node: ast.AST) -> tuple[int, int]:
    """First and last line of a def/class, including its decorators."""
    first = node.lineno  # type: ignore[attr-defined]
    for deco in getattr(node, "decorator_list", []):
        first = min(first, deco.lineno)
    return first, getattr(node, "end_lineno", None) or node.lineno  # type: ignore[attr-defined]


def _split_class(node: ast.ClassDef, start: int, end: int) -> list[Segment]:
    """A big class becomes: header/attribute pieces + one chunk per method."""
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if not methods:
        return [(start, end, ChunkKind.CLASS, node.name)]
    segments: list[Segment] = []
    cursor = start
    for method in methods:
        m_start, m_end = _node_range(method)
        if m_start > cursor:
            segments.append((cursor, m_start - 1, ChunkKind.CLASS, node.name))
        segments.append((m_start, m_end, ChunkKind.METHOD, f"{node.name}.{method.name}"))
        cursor = m_end + 1
    if cursor <= end:
        segments.append((cursor, end, ChunkKind.CLASS, node.name))
    return segments


def _python_segments(text: str, total_lines: int, max_lines: int) -> list[Segment] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None  # broken file: caller falls back to line windows

    defs: list[Segment] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start, end = _node_range(node)
            defs.append((start, end, ChunkKind.FUNCTION, node.name))
        elif isinstance(node, ast.ClassDef):
            start, end = _node_range(node)
            if end - start + 1 <= max_lines:
                defs.append((start, end, ChunkKind.CLASS, node.name))
            else:
                defs.extend(_split_class(node, start, end))

    # Fill the gaps (imports, constants, docstring) with MODULE segments.
    segments: list[Segment] = []
    cursor = 1
    for start, end, kind, name in sorted(defs):
        if start > cursor:
            segments.append((cursor, start - 1, ChunkKind.MODULE, ""))
        segments.append((start, min(end, total_lines), kind, name))
        cursor = end + 1
    if cursor <= total_lines:
        segments.append((cursor, total_lines, ChunkKind.MODULE, ""))
    return segments


def chunk_file(
    file_path: str,
    text: str,
    language: str,
    *,
    max_lines: int = 60,
    overlap: int = 10,
) -> list[CodeChunk]:
    lines = split_lines(text)
    if not lines or not text.strip():
        return []
    overlap = min(overlap, max_lines // 2)  # overlap must stay smaller than the window

    segments: list[Segment] | None = None
    if language == "python":
        segments = _python_segments(text, len(lines), max_lines)
    if segments is None:
        segments = [(1, len(lines), ChunkKind.BLOCK, "")]

    chunks: list[CodeChunk] = []
    for seg_start, seg_end, kind, name in segments:
        for start, end in _windows(seg_start, seg_end, max_lines, overlap):
            body = "\n".join(lines[start - 1 : end])
            if not body.strip():
                continue  # skip chunks that are only blank lines
            chunks.append(
                CodeChunk(
                    chunk_id=make_chunk_id(file_path, start, end, body),
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    language=language,
                    kind=kind,
                    name=name,
                    text=body,
                )
            )
    return chunks