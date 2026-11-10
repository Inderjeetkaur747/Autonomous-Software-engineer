"""How commands are executed. The default runner REFUSES to run anything.

Phase 11 adds a Docker runner. We never run a repository's code directly on your PC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class RunnerUnavailable(Exception):
    """No safe place to run commands is configured."""


class CommandResult(BaseModel):
    argv: list[str]
    exit_code: int | None = None  # None when the command was killed
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class CommandRunner(Protocol):
    def run(self, argv: list[str], *, cwd: Path, timeout: int) -> CommandResult: ...


class DisabledRunner:
    """Safe default: nothing runs."""

    def run(self, argv: list[str], *, cwd: Path, timeout: int) -> CommandResult:
        raise RunnerUnavailable(
            "Running tests is disabled: no sandbox is configured yet. "
            "The Docker sandbox arrives in Phase 11."
        )