"""Evidence: the code the LLM has really been shown. Citations are checked against it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from swe_agent.logging_setup import redact_secrets
from swe_agent.schemas.retrieval import SearchHit
from swe_agent.schemas.tools import ToolResult

_NUMBERED_RE = re.compile(r"^\s*(\d+) \| ")  # read_file lines look like "  12 | code"
_SEARCH_SPLIT_RE = re.compile(r"\n\n(?=\[\d+\] )")
_SEARCH_HEAD_RE = re.compile(r"^\[\d+\] (?P<path>\S+):(?P<start>\d+)-(?P<end>\d+)")


@dataclass
class _Block:
    title: str
    text: str
    lines: dict[str, set[int]] = field(default_factory=dict)  # file -> lines fully shown


def _cut_lines(lines: list[str], limit: int) -> list[str]:
    """Keep whole lines until the character budget is used up."""
    kept: list[str] = []
    used = 0
    for line in lines:
        if kept and used + len(line) + 1 > limit:
            break
        kept.append(line)
        used += len(line) + 1
    return kept


class Evidence:
    def __init__(self, max_chars: int = 7000, block_chars: int = 2200) -> None:
        self._max_chars = max_chars  # wrap_untrusted cuts at 8000, so we stay below
        self._block_chars = block_chars
        self._blocks: dict[str, _Block] = {}
        self._shown: dict[str, set[int]] = {}

    @property
    def shown(self) -> dict[str, set[int]]:
        """Lines that were in the LAST rendered text. Call render() first."""
        return self._shown

    # ------------------------------------------------------------------ adding
    def add_hits(self, hits: list[SearchHit]) -> None:
        for hit in hits:
            chunk = hit.chunk
            kept = _cut_lines(redact_secrets(chunk.text).split("\n"), self._block_chars)
            end = chunk.start_line + len(kept) - 1
            label = f"{chunk.file_path}:{chunk.start_line}-{end}"
            title = label + (f" ({chunk.kind.value}: {chunk.name})" if chunk.name else "")
            lines = {chunk.file_path: set(range(chunk.start_line, end + 1))}
            self._blocks.setdefault(f"code:{label}", _Block(title, "\n".join(kept), lines))

    def add_result(self, name: str, result: ToolResult) -> None:
        """Record what a successful tool call showed the model."""
        if not result.ok:
            return
        if name == "read_file":
            self._add_read(result)
        elif name == "search_code":
            self._add_search(result.output)
        elif name in ("list_files", "grep"):
            kept = _cut_lines(result.output.split("\n"), self._block_chars)
            title = "File list" if name == "list_files" else "Grep results"
            self._blocks[f"{name}:{len(self._blocks)}"] = _Block(title, "\n".join(kept))

    def _add_read(self, result: ToolResult) -> None:
        path = str(result.meta.get("path", ""))
        body = result.output.split("\n")[1:]  # the first line is a header
        kept = _cut_lines(body, self._block_chars)
        # The numbers printed in front of each line are the truth about what was shown.
        numbers = [int(m.group(1)) for line in kept if (m := _NUMBERED_RE.match(line))]
        if not path or not numbers:
            return
        first, last = numbers[0], numbers[-1]
        self._blocks.setdefault(
            f"read:{path}:{first}-{last}",
            _Block(f"{path}:{first}-{last} (read_file)", "\n".join(kept), {path: set(numbers)}),
        )

    def _add_search(self, output: str) -> None:
        used = 0
        for piece in _SEARCH_SPLIT_RE.split(output):
            head = _SEARCH_HEAD_RE.match(piece)
            if head is None:
                continue
            if used and used + len(piece) > self._block_chars * 2:
                break  # keep only whole results
            used += len(piece)
            first, _, body = piece.partition("\n")
            path, start, end = head["path"], int(head["start"]), int(head["end"])
            self._blocks.setdefault(
                f"code:{path}:{start}-{end}",
                _Block(first.split("] ", 1)[1], body, {path: set(range(start, end + 1))}),
            )

    # --------------------------------------------------------------- rendering
    def render(self) -> str:
        """The text for the prompt. Also fixes which lines count as 'shown'."""
        parts: list[str] = []
        used = 0
        shown: dict[str, set[int]] = {}
        for block in self._blocks.values():
            part = f"### {block.title}\n{block.text}"
            if used + len(part) + 2 > self._max_chars:
                continue  # does not fit: its lines do NOT count as shown
            parts.append(part)
            used += len(part) + 2
            for path, numbers in block.lines.items():
                shown.setdefault(path, set()).update(numbers)
        self._shown = shown
        return "\n\n".join(parts)