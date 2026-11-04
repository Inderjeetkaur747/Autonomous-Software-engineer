"""Find the files worth reading. Allow-list of types, deny-list of dangerous things."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

from swe_agent.config import Settings, get_settings
from swe_agent.schemas.code import ScannedFile, ScanResult

logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "dist", "build", "site-packages", "target", ".next", "coverage", ".eggs",
}

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".java": "java", ".go": "go", ".rs": "rust", ".c": "c",
    ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
    ".php": "php", ".sh": "shell", ".sql": "sql", ".html": "html", ".css": "css",
    ".md": "markdown", ".rst": "text", ".txt": "text", ".toml": "toml",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".cfg": "ini", ".ini": "ini",
    ".example": "text",
}
NAME_LANG = {"dockerfile": "dockerfile", "makefile": "makefile"}

LOCK_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock",
    "poetry.lock", "pipfile.lock", "cargo.lock", "composer.lock",
}

SENSITIVE_NAMES = {
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials.json",
    "secrets.json", "secrets.yaml", "secrets.yml", ".npmrc", ".pypirc", ".netrc",
}
SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".kdbx")
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}


def is_sensitive_name(name: str) -> bool:
    """True for files that probably contain secrets. We never read these."""
    lowered = name.lower()
    if lowered in SAFE_ENV_TEMPLATES:
        return False
    if lowered == ".env" or lowered.startswith(".env."):
        return True
    return lowered in SENSITIVE_NAMES or lowered.endswith(SENSITIVE_SUFFIXES)


def language_for(name: str) -> str | None:
    lowered = name.lower()
    if lowered in NAME_LANG:
        return NAME_LANG[lowered]
    if lowered in SAFE_ENV_TEMPLATES:
        return "text"
    return EXT_LANG.get(os.path.splitext(lowered)[1])



def scan_repo(repo_dir: Path, settings: Settings | None = None) -> ScanResult:
    settings = settings or get_settings()
    root = Path(repo_dir).resolve()
    max_bytes = settings.max_file_kb * 1024
    skipped: Counter[str] = Counter()
    files: list[ScannedFile] = []
    truncated = False

    for current, dirs, names in os.walk(root, followlinks=False):
        kept = []
        for d in sorted(dirs):
            full = os.path.join(current, d)
            if d in SKIP_DIRS or d.endswith(".egg-info"):
                skipped["ignored_dir"] += 1
            elif os.path.islink(full):
                skipped["symlink"] += 1
            else:
                kept.append(d)
        dirs[:] = kept  # editing this list in place tells os.walk where NOT to go

        for name in sorted(names):
            full_path = Path(current) / name
            if full_path.is_symlink():
                skipped["symlink"] += 1
                continue
            if is_sensitive_name(name):
                skipped["sensitive_name"] += 1
                continue
            if name.lower() in LOCK_FILES:
                skipped["lock_file"] += 1
                continue
            language = language_for(name)
            if language is None:
                skipped["unsupported_type"] += 1
                continue
            try:
                size = full_path.stat().st_size
            except OSError:
                skipped["unreadable"] += 1
                continue
            if size == 0:
                skipped["empty"] += 1
                continue
            if size > max_bytes:
                skipped["too_large"] += 1
                continue
            if len(files) >= settings.max_scan_files:
                truncated = True
                skipped["over_file_limit"] += 1
                continue
            files.append(
                ScannedFile(
                    path=full_path.relative_to(root).as_posix(),
                    language=language,
                    size_bytes=size,
                )
            )

    logger.info("scan done files=%d skipped=%s truncated=%s", len(files), dict(skipped), truncated)
    return ScanResult(files=files, skipped=dict(skipped), truncated=truncated)