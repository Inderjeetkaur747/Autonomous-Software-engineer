"""Run pytest inside a locked-down Docker container.

Docker REDUCES risk. It is not complete isolation: containers share the host kernel.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from swe_agent.config import Settings, get_settings
from swe_agent.security import validate_command
from swe_agent.security.errors import CommandNotAllowed
from swe_agent.tools.runner import CommandResult, DisabledRunner, RunnerUnavailable

logger = logging.getLogger(__name__)

SANDBOX_UID = 10001
NAME_PREFIX = "swe-sbx-"
_BAD_PATH_CHARS = (",", '"', "\n", "\r", "\x00")  # would break the --mount option
_SMALL_OUTPUT = 10_000


@dataclass
class ExecResult:
    """What happened when a command was started."""

    exit_code: int | None  # None = we killed it (timeout)
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    out_truncated: bool = False
    err_truncated: bool = False


Executor = Callable[[list[str], int, int], ExecResult]  # (command, timeout, output limit)


class _Capture(threading.Thread):
    """Reads a pipe in the background and keeps only the LAST `limit` bytes."""

    def __init__(self, stream, limit: int) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self.data = bytearray()
        self.truncated = False

    def run(self) -> None:
        try:
            while chunk := self._stream.read(4096):
                self.data += chunk
                if len(self.data) > self._limit:
                    del self.data[: len(self.data) - self._limit]
                    self.truncated = True
        except (OSError, ValueError):
            pass


def _execute(cmd: list[str], timeout: int, limit: int) -> ExecResult:
    """Start a process (no shell), wait with a timeout, and capture limited output."""
    try:
        proc = subprocess.Popen(  # list arguments and shell=False: no command injection
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise RunnerUnavailable(
            "Docker is not installed or not on PATH. Install Docker Desktop "
            "and reopen the terminal."
        ) from exc
    out, err = _Capture(proc.stdout, limit), _Capture(proc.stderr, limit)
    out.start()
    err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    out.join(5)
    err.join(5)
    return ExecResult(
        exit_code=None if timed_out else proc.returncode,
        stdout=bytes(out.data),
        stderr=bytes(err.data),
        timed_out=timed_out,
        out_truncated=out.truncated,
        err_truncated=err.truncated,
    )


def _decode(data: bytes, truncated: bool) -> str:
    text = data.decode("utf-8", errors="replace")
    if truncated:
        text = "[... output truncated: only the last part is shown ...]\n" + text
    return text


class DockerRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        execute: Executor | None = None,
        name_factory: Callable[[], str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._exec: Executor = execute or _execute  # tests pass a fake
        self._name_factory = name_factory or (lambda: f"{NAME_PREFIX}{uuid.uuid4().hex[:12]}")
        self._ready = False
        self._server_version = ""

    # ------------------------------------------------------------- preflight
    def ensure_ready(self) -> str:
        """Check that Docker runs and the image exists. Cached after the first success."""
        if self._ready:
            return self._server_version
        s = self._settings
        version = self._exec(
            [s.docker_binary, "version", "--format", "{{.Server.Version}}"], 30, _SMALL_OUTPUT
        )
        if version.timed_out or version.exit_code != 0:
            raise RunnerUnavailable(
                "Docker is not running. Start Docker Desktop, wait until it says "
                "'Engine running', then try again."
            )
        image = self._exec(
            [s.docker_binary, "image", "inspect", s.sandbox_image, "--format", "{{.Id}}"],
            30,
            _SMALL_OUTPUT,
        )
        if image.timed_out or image.exit_code != 0:
            raise RunnerUnavailable(
                f"The sandbox image {s.sandbox_image!r} does not exist. "
                "Build it once with: swe-agent sandbox build"
            )
        self._server_version = version.stdout.decode("utf-8", errors="replace").strip()
        self._ready = True
        return self._server_version

    # --------------------------------------------------------------- command
    def build_command(self, argv: list[str], repo_dir: Path, name: str) -> list[str]:
        s = self._settings
        source = str(Path(repo_dir).resolve())
        if any(ch in source for ch in _BAD_PATH_CHARS):
            raise RunnerUnavailable("The workspace path has characters the sandbox cannot mount.")
        scratch = "rw,noexec,nosuid,nodev,mode=1777"
        return [
            s.docker_binary, "run", "--rm",
            "--name", name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(s.sandbox_pids),
            "--memory", s.sandbox_memory,
            "--memory-swap", s.sandbox_memory,  # same as memory = no swap
            "--cpus", str(s.sandbox_cpus),
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--tmpfs", f"/work:{scratch},size={s.sandbox_tmpfs_mb}m",
            "--tmpfs", f"/tmp:{scratch},size=64m",
            "--mount", f"type=bind,source={source},target=/src,readonly",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTEST_ADDOPTS=-p no:cacheprovider",
            "--env", "HOME=/tmp",
            "--workdir", "/work",
            s.sandbox_image,
            *argv,
        ]

    # ------------------------------------------------------------------- run
    def run(self, argv: list[str], *, cwd: Path, timeout: int) -> CommandResult:
        # Second check. Even if a caller forgot to validate, only canonical pytest gets in.
        allowed = validate_command(list(argv))
        if allowed.kind != "test" or allowed.argv != list(argv):
            raise CommandNotAllowed("The sandbox only runs validated pytest commands.")
        self.ensure_ready()

        name = self._name_factory()
        command = self.build_command(list(argv), cwd, name)
        limit = self._settings.sandbox_output_bytes
        started = time.perf_counter()
        cleanup = True  # stays True if anything raises
        try:
            res = self._exec(command, timeout, limit)
            cleanup = res.timed_out or res.exit_code is None
        finally:
            if cleanup:
                self._remove(name)  # killing the docker client does NOT stop the container

        if res.exit_code == 125:  # docker itself failed (not the tests)
            detail = res.stderr.decode("utf-8", errors="replace").strip()[-500:]
            raise RunnerUnavailable(f"Docker failed to start the sandbox: {detail}")

        stderr = _decode(res.stderr, res.err_truncated)
        if res.exit_code == 137 and not res.timed_out:
            stderr += (
                f"\n[sandbox] the process was killed (exit 137). Most likely it used more "
                f"than {self._settings.sandbox_memory} of memory."
            )
        return CommandResult(
            argv=list(argv),
            exit_code=res.exit_code,
            stdout=_decode(res.stdout, res.out_truncated),
            stderr=stderr,
            timed_out=res.timed_out,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def _remove(self, name: str) -> None:
        try:
            self._exec([self._settings.docker_binary, "rm", "-f", name], 30, _SMALL_OUTPUT)
        except RunnerUnavailable:
            logger.warning("could not remove container %s", name)

    def leftover_containers(self) -> list[str]:
        """Names of sandbox containers that still exist (there should be none)."""
        res = self._exec(
            [
                self._settings.docker_binary, "ps", "-a",
                "--filter", f"name={NAME_PREFIX}", "--format", "{{.Names}}",
            ],
            30,
            _SMALL_OUTPUT,
        )
        return [line for line in res.stdout.decode("utf-8", errors="replace").split() if line]


def get_runner(settings: Settings | None = None) -> DockerRunner | DisabledRunner:
    settings = settings or get_settings()
    if settings.runner_backend == "docker":
        return DockerRunner(settings)
    return DisabledRunner()