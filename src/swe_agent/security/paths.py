"""Path guard: the only door the agent may use to reach files."""

from __future__ import annotations

import re
from pathlib import Path

from swe_agent.security.errors import PathSecurityError

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
# Names Windows treats as devices, even with an extension (con.txt).
_RESERVED = {
    "con", "prn", "aux", "nul",
    *[f"com{i}" for i in range(1, 10)],
    *[f"lpt{i}" for i in range(1, 10)],
}
MAX_PATH_CHARS = 300
PROTECTED_WRITE_PREFIXES = (".github/",)  # the agent must not change CI files


def normalize_relative(raw: str) -> str:
    """Turn user or LLM text into a clean relative POSIX path, or raise."""
    if not isinstance(raw, str) or not raw.strip():
        raise PathSecurityError("Path is empty.")
    if len(raw) > MAX_PATH_CHARS:
        raise PathSecurityError("Path is too long.")

    path = raw.strip().replace("\\", "/")
    if path.startswith("/") or _DRIVE_RE.match(path):
        raise PathSecurityError("Absolute paths are not allowed.")

    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts:
        raise PathSecurityError("Path is empty.")
    for part in parts:
        if part == "..":
            raise PathSecurityError("'..' is not allowed in paths.")
        if any(ord(ch) < 32 for ch in part):
            raise PathSecurityError("Control characters are not allowed in paths.")
        if ":" in part:
            raise PathSecurityError("':' is not allowed (drive letters, data streams).")
        if part.endswith((".", " ")):
            raise PathSecurityError("Names ending in '.' or space are not allowed.")
        if part.split(".")[0].lower() in _RESERVED:
            raise PathSecurityError(f"Reserved Windows name: {part!r}")
    return "/".join(parts)


class PathGuard:
    """All file access for one repository goes through this object."""

    def __init__(self, root: Path, max_read_bytes: int = 256 * 1024) -> None:
        self.root = Path(root).resolve()
        self.max_read_bytes = max_read_bytes

    def check(self, raw: str, *, write: bool = False) -> tuple[str, Path]:
        """Return (clean relative path, absolute path), or raise PathSecurityError."""
        # Imported here: the indexing package is heavy and we want a fast import.
        from swe_agent.indexing.scanner import is_sensitive_name, language_for

        rel = normalize_relative(raw)
        parts = rel.split("/")

        if any(p.lower() == ".git" for p in parts):
            raise PathSecurityError("Access to .git is not allowed.")
        if any(is_sensitive_name(p) for p in parts):
            raise PathSecurityError("This looks like a secret file and is off limits.")
        if write:
            if rel.lower().startswith(PROTECTED_WRITE_PREFIXES):
                raise PathSecurityError("CI configuration files are protected.")
            if language_for(parts[-1]) is None:
                raise PathSecurityError("Only known text and code file types may be written.")

        # resolve() follows symlinks, so a link that points outside is caught here.
        full = (self.root / rel).resolve()
        if not full.is_relative_to(self.root):
            raise PathSecurityError("Path escapes the repository.")
        return rel, full

    def read_text(self, raw: str) -> str:
        rel, full = self.check(raw)
        if (self.root / rel).is_symlink():
            raise PathSecurityError("Symlinks are not read.")
        if not full.is_file():
            raise PathSecurityError(f"Not a file: {rel}")
        if full.stat().st_size > self.max_read_bytes:
            raise PathSecurityError(f"File is too large to read: {rel}")
        data = full.read_bytes()
        if b"\x00" in data[:8000]:
            raise PathSecurityError(f"Binary file: {rel}")
        return data.decode("utf-8", errors="replace")