"""Careful file editing: keep each file's own line endings, write atomically."""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


class EditError(Exception):
    """An edit cannot be applied safely. The message is safe to show to the LLM."""


@dataclass(frozen=True)
class LoadedText:
    text: str  # always uses "\n" inside
    newline: str  # the file's real line ending: "\n" or "\r\n"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_text_strict(data: bytes) -> LoadedText:
    """Decode a file for editing. Refuse anything we could corrupt."""
    if b"\x00" in data[:8000]:
        raise EditError("This is a binary file.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EditError("File is not valid UTF-8, so it is not edited.") from exc
    crlf = data.count(b"\r\n")
    if data.count(b"\r") - crlf:
        raise EditError("File uses old-style '\\r' line endings, so it is not edited.")
    if crlf and crlf != data.count(b"\n"):
        raise EditError("File mixes LF and CRLF line endings, so it is not edited.")
    return LoadedText(text.replace("\r\n", "\n"), "\r\n" if crlf else "\n")


def apply_replace(text: str, old: str, new: str) -> str:
    """Replace ONE exact piece of text. It must occur exactly once."""
    if not old:
        raise EditError("old_str must not be empty.")
    old, new = normalize_newlines(old), normalize_newlines(new)
    if old == new:
        raise EditError("old_str and new_str are identical.")
    count = text.count(old)
    if count == 0:
        raise EditError(
            "old_str was not found. Copy it exactly from read_file "
            "(without the 'NN | ' line-number prefix), including spaces and indentation."
        )
    if count > 1:
        raise EditError(
            f"old_str was found {count} times. Add surrounding lines so it matches once."
        )
    return text.replace(old, new, 1)


def to_bytes(text: str, newline: str) -> bytes:
    """Convert back to the file's own line ending style."""
    return (text if newline == "\n" else text.replace("\n", newline)).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    """Write to a temp file in the same folder, then swap it in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".swe_tmp_")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        if path.exists():
            shutil.copymode(path, tmp_name)  # keep file permissions
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def short_diff(rel: str, old: str, new: str, max_lines: int = 60) -> str:
    lines = list(
        difflib.unified_diff(
            old.split("\n"), new.split("\n"), f"a/{rel}", f"b/{rel}", n=2, lineterm=""
        )
    )
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = [*lines[:max_lines], f"... ({extra} more diff lines)"]
    return "\n".join(lines)