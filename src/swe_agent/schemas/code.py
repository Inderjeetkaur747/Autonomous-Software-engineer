"""Typed data for scanned files and code chunks."""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field

from swe_agent.schemas.common import CodeCitation


class ChunkKind(StrEnum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"  # top-level code that is not a function or class
    BLOCK = "block"  # a plain line window (non-Python files)


def make_chunk_id(file_path: str, start_line: int, end_line: int, text: str) -> str:
    raw = f"{file_path}:{start_line}:{end_line}:{text}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]  # noqa: S324 (an ID, not security)


class CodeChunk(BaseModel):
    chunk_id: str
    file_path: str  # relative POSIX path
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    language: str
    kind: ChunkKind
    name: str = ""
    text: str  # the exact original lines

    @property
    def citation(self) -> CodeCitation:
        return CodeCitation(
            file_path=self.file_path, start_line=self.start_line, end_line=self.end_line
        )

    @property
    def embed_text(self) -> str:
        """What we turn into a vector: the code plus a short header for context."""
        header = f"# File: {self.file_path}"
        if self.name:
            header += f" | {self.kind.value}: {self.name}"
        return f"{header}\n{self.text}"


class ScannedFile(BaseModel):
    path: str  # relative POSIX path
    language: str
    size_bytes: int


class ScanResult(BaseModel):
    files: list[ScannedFile] = Field(default_factory=list)
    skipped: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False  # True if we stopped at MAX_SCAN_FILES

    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[f.language] = counts.get(f.language, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))