"""Command guard: an allow-list. Anything not listed here is refused.

IMPORTANT: this controls WHICH command runs. It does not make running pytest safe,
because pytest executes the repository's own code. Containment comes from the
sandbox (Phase 12).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from swe_agent.security.errors import CommandNotAllowed, PathSecurityError
from swe_agent.security.paths import normalize_relative

_META_RE = re.compile(r"[;&|`$<>\n\r\x00]")
MAX_TOKENS = 30
MAX_TOKEN_CHARS = 300

# --- pytest: only these options ---
_PYTEST_FLAGS = frozenset(
    {"-x", "-q", "-v", "-vv", "-s", "-ra", "--lf", "--ff", "--no-header", "--disable-warnings"}
)
_PYTEST_PATTERNS = (
    re.compile(r"--tb=(?:auto|long|short|line|native|no)"),
    re.compile(r"--maxfail=\d{1,3}"),
)
_PYTEST_VALUE_FLAGS = frozenset({"-k", "-m"})  # these take one value
_TARGET_SUFFIX_RE = re.compile(r"[A-Za-z0-9_.:\[\]-]+")

# --- git: read-only subcommands and a few safe options ---
_GIT_SUBCOMMANDS = frozenset({"status", "diff", "log", "show", "rev-parse", "ls-files"})
_GIT_FLAGS = frozenset(
    {
        "--stat", "--shortstat", "--numstat", "--summary", "--name-only", "--name-status",
        "--no-color", "--cached", "--staged", "--porcelain", "--short", "-s", "--oneline",
        "--no-ext-diff", "--no-textconv", "--show-toplevel", "--abbrev-ref",
        "--is-inside-work-tree",
    }
)
_GIT_NUMBER_FLAG = re.compile(r"-\d{1,2}")
_GIT_ARG_RE = re.compile(r"[A-Za-z0-9_./~^@{}-]+")
_GIT_DOTDOT_RE = re.compile(r"(^|/)\.\.(/|$)")


@dataclass
class AllowedCommand:
    argv: list[str]  # canonical form, ready for subprocess (never a shell)
    kind: str  # "test" or "git"


def _tokenize(command: str | list[str]) -> list[str]:
    if isinstance(command, str):
        if _META_RE.search(command):
            raise CommandNotAllowed("Shell operators and special characters are not allowed.")
        try:
            tokens = shlex.split(command.replace("\\", "/"))
        except ValueError as exc:
            raise CommandNotAllowed(f"Cannot parse command: {exc}") from exc
    else:
        tokens = [str(item) for item in command]
        if any(_META_RE.search(token) for token in tokens):
            raise CommandNotAllowed("Shell operators and special characters are not allowed.")
    if not tokens:
        raise CommandNotAllowed("Command is empty.")
    if len(tokens) > MAX_TOKENS or any(len(t) > MAX_TOKEN_CHARS for t in tokens):
        raise CommandNotAllowed("Command is too long.")
    return tokens


def _check_test_target(token: str) -> str:
    path, sep, suffix = token.partition("::")
    try:
        rel = normalize_relative(path)
    except PathSecurityError as exc:
        raise CommandNotAllowed(f"Bad test path {token!r}: {exc}") from exc
    if sep and not _TARGET_SUFFIX_RE.fullmatch(suffix):
        raise CommandNotAllowed(f"Bad test name in {token!r}")
    return rel + sep + suffix


def _validate_pytest(rest: list[str]) -> AllowedCommand:
    out = ["python", "-m", "pytest"]
    i = 0
    while i < len(rest):
        token = rest[i]
        if token in _PYTEST_FLAGS or any(p.fullmatch(token) for p in _PYTEST_PATTERNS):
            out.append(token)
        elif token in _PYTEST_VALUE_FLAGS:
            if i + 1 >= len(rest) or rest[i + 1].startswith("-"):
                raise CommandNotAllowed(f"Option {token} needs a value.")
            out.extend([token, rest[i + 1]])
            i += 1
        elif token.startswith("-"):
            raise CommandNotAllowed(f"pytest option not allowed: {token}")
        else:
            out.append(_check_test_target(token))
        i += 1
    return AllowedCommand(out, "test")


def _check_git_arg(token: str) -> str:
    if not _GIT_ARG_RE.fullmatch(token) or token.startswith(("/", "-")):
        raise CommandNotAllowed(f"git argument not allowed: {token!r}")
    if _GIT_DOTDOT_RE.search(token):
        raise CommandNotAllowed("'..' path parts are not allowed in git arguments.")
    return token


def _validate_git(rest: list[str]) -> AllowedCommand:
    if not rest or rest[0] not in _GIT_SUBCOMMANDS:
        allowed = ", ".join(sorted(_GIT_SUBCOMMANDS))
        raise CommandNotAllowed(f"Only these git commands are allowed: {allowed}")
    out = ["git", rest[0]]
    for token in rest[1:]:
        if token == "--" or token in _GIT_FLAGS or _GIT_NUMBER_FLAG.fullmatch(token):
            out.append(token)
        elif token.startswith("-"):
            raise CommandNotAllowed(f"git option not allowed: {token}")
        else:
            out.append(_check_git_arg(token))
    return AllowedCommand(out, "git")


def validate_command(command: str | list[str]) -> AllowedCommand:
    """Return the canonical command, or raise CommandNotAllowed."""
    tokens = _tokenize(command)
    program = tokens[0]
    if program == "pytest":
        return _validate_pytest(tokens[1:])
    if program in ("python", "python3"):
        if tokens[1:3] != ["-m", "pytest"]:
            raise CommandNotAllowed("python may only be used as: python -m pytest")
        return _validate_pytest(tokens[3:])
    if program == "git":
        return _validate_git(tokens[1:])
    raise CommandNotAllowed(f"Program not allowed: {program!r}. Allowed: pytest, git (read-only).")