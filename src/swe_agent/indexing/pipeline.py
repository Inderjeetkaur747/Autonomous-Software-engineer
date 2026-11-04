"""Scan + read + chunk, and save the result."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from swe_agent.config import Settings, get_settings
from swe_agent.indexing.chunker import chunk_file, split_lines
from swe_agent.indexing.scanner import scan_repo
from swe_agent.schemas.code import CodeChunk, ScanResult

logger = logging.getLogger(__name__)

MAX_LINE_CHARS = 1500  # a longer line usually means minified or generated code


class ChunkingResult(BaseModel):
    scan: ScanResult
    chunks: list[CodeChunk] = Field(default_factory=list)
    skipped: dict[str, int] = Field(default_factory=dict)  # scan + read skips together


def read_text(repo_dir: Path, rel_path: str, max_bytes: int) -> tuple[str | None, str]:
    """Return (text, reason). Text is None when we refuse to read the file."""
    root = repo_dir.resolve()
    path = (root / rel_path).resolve()
    if not path.is_relative_to(root):
        return None, "outside_repo"  # a full path guard comes in Phase 7
    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None, "unreadable"
    if len(data) > max_bytes:
        return None, "too_large"
    if b"\x00" in data[:8000]:
        return None, "binary"
    text = data.decode("utf-8", errors="replace")
    if any(len(line) > MAX_LINE_CHARS for line in split_lines(text)):
        return None, "minified_or_generated"
    return text, "ok"


def build_chunks(repo_dir: Path, settings: Settings | None = None) -> ChunkingResult:
    settings = settings or get_settings()
    repo_dir = Path(repo_dir)
    scan = scan_repo(repo_dir, settings)
    skipped: Counter[str] = Counter(scan.skipped)
    chunks: list[CodeChunk] = []

    for file in scan.files:
        text, reason = read_text(repo_dir, file.path, settings.max_file_kb * 1024)
        if text is None:
            skipped[reason] += 1
            continue
        chunks.extend(
            chunk_file(
                file.path,
                text,
                file.language,
                max_lines=settings.chunk_max_lines,
                overlap=settings.chunk_overlap_lines,
            )
        )
    logger.info("chunking done files=%d chunks=%d", len(scan.files), len(chunks))
    return ChunkingResult(scan=scan, chunks=chunks, skipped=dict(skipped))


def save_chunks(chunks: list[CodeChunk], path: Path) -> None:
    """One JSON object per line (JSONL). Easy to stream and to inspect."""
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[CodeChunk]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) and CodeChunk.model_validate_json(line) for line in handle]