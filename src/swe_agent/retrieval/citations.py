"""Everything about trusting citations.

1. format_context()           -> numbered, cited, secret-free code for the LLM prompt
2. check_answer_citations()   -> did the LLM cite only lines that we really showed it?
3. verify_chunk()             -> does the chunk text still match the real file on disk?
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from swe_agent.indexing.chunker import split_lines
from swe_agent.indexing.scanner import language_for
from swe_agent.logging_setup import redact_secrets
from swe_agent.schemas.code import CodeChunk
from swe_agent.schemas.retrieval import SearchHit

# Matches: mathkit/stats.py:4-6   or   mathkit/stats.py:5
_CITATION_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+):(?P<start>\d+)(?:-(?P<end>\d+))?"
)
MAX_CITED_SPAN = 2000  # a citation longer than this is not a real citation


# ----------------------------------------------------------------- 1. context
def format_context(hits: list[SearchHit], max_chars: int = 7500) -> str:
    """Build the code context for a prompt. Secrets are redacted. Output is never longer
    than max_chars (the prompt wrapper cuts at 8000, so we stay below it).
    """
    blocks: list[str] = []
    used = 0
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        title = f"[{number}] {chunk.citation.label}"
        if chunk.name:
            title += f" ({chunk.kind.value}: {chunk.name})"
        block = f"{title}\n```{chunk.language}\n{redact_secrets(chunk.text)}\n```"
        separator = 2 if blocks else 0
        if used + separator + len(block) > max_chars:
            if not blocks:
                blocks.append(block[:max_chars])  # even the first block is too big
            break
        blocks.append(block)
        used += separator + len(block)
    return "\n\n".join(blocks)


# ------------------------------------------------------------ 2. answer check
@dataclass
class CitationCheck:
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return not self.invalid


def _iter_citations(text: str) -> Iterator[tuple[str, int, int]]:
    for match in _CITATION_RE.finditer(text):
        path = match.group("path").replace("\\", "/").removeprefix("./")
        if language_for(path.rsplit("/", 1)[-1]) is None:
            continue  # "localhost.com:8080" is not a code citation
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        yield path, start, end


def shown_lines(hits: list[SearchHit]) -> dict[str, set[int]]:
    """Which lines of which files did we show the model?"""
    shown: dict[str, set[int]] = {}
    for hit in hits:
        c = hit.chunk
        shown.setdefault(c.file_path, set()).update(range(c.start_line, c.end_line + 1))
    return shown


def check_citations_in_lines(text: str, shown: dict[str, set[int]]) -> CitationCheck:
    """A citation is valid only if EVERY cited line is in `shown`."""
    result = CitationCheck()
    for path, start, end in _iter_citations(text):
        label = f"{path}:{start}-{end}"
        well_formed = 1 <= start <= end and end - start <= MAX_CITED_SPAN
        ok = well_formed and set(range(start, end + 1)) <= shown.get(path, set())
        bucket = result.valid if ok else result.invalid
        if label not in bucket:
            bucket.append(label)
    return result


def check_answer_citations(text: str, hits: list[SearchHit]) -> CitationCheck:
    """Same check, when the shown lines come from retrieved hits."""
    return check_citations_in_lines(text, shown_lines(hits))


# -------------------------------------------------------------- 3. disk check
def verify_chunk(repo_dir: Path, chunk: CodeChunk) -> bool:
    """True if the lines the chunk claims really contain the chunk's text right now."""
    root = Path(repo_dir).resolve()
    path = (root / chunk.file_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lines = split_lines(text)
    if chunk.end_line > len(lines):
        return False
    return "\n".join(lines[chunk.start_line - 1 : chunk.end_line]) == chunk.text


def citation_correctness(repo_dir: Path, hits: list[SearchHit]) -> float:
    """Share of hits whose citation matches the real file (0.0 when there are no hits)."""
    if not hits:
        return 0.0
    good = sum(1 for hit in hits if verify_chunk(repo_dir, hit.chunk))
    return good / len(hits)