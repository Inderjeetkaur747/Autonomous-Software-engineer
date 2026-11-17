"""Attack the sandbox on purpose, to prove it holds."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from swe_agent.tools.runner import CommandResult

SECRET_NAME = "SWE_DEMO_SECRET"

CONTAINMENT = '''\
import os
import socket
import threading
import time

import pytest


def test_network_is_blocked():
    with pytest.raises(OSError):
        socket.create_connection(("1.1.1.1", 53), timeout=3)


def test_dns_is_blocked():
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 80)


def test_source_mount_is_read_only():
    with pytest.raises(OSError):
        open("/src/evil.txt", "w")


def test_root_filesystem_is_read_only():
    with pytest.raises(OSError):
        open("/etc/evil.txt", "w")


def test_runs_as_an_unprivileged_user():
    assert os.getuid() == 10001


def test_cannot_become_root():
    with pytest.raises(OSError):
        os.setuid(0)


def test_host_environment_is_not_visible():
    assert "SWE_DEMO_SECRET" not in os.environ
    assert "GROQ_API_KEY" not in os.environ


def test_git_folder_is_not_copied():
    assert os.path.exists("/work/tests")
    assert not os.path.exists("/work/.git")


def test_thread_bomb_is_limited():
    started = 0
    try:
        for _ in range(600):
            threading.Thread(target=time.sleep, args=(3,), daemon=True).start()
            started += 1
    except RuntimeError:
        pass
    assert started < 600


def test_scratch_space_is_writable_but_temporary():
    with open("/work/scratch.txt", "w") as handle:
        handle.write("gone when the container stops")
'''

MEMORY_BOMB = '''\
def test_memory_bomb():
    hog = []
    while True:
        hog.append(bytearray(b"x") * 50_000_000)
'''

INFINITE_LOOP = '''\
def test_infinite_loop():
    while True:
        pass
'''

_FILES = {
    "test_containment.py": CONTAINMENT,
    "test_memory_bomb.py": MEMORY_BOMB,
    "test_infinite_loop.py": INFINITE_LOOP,
}
_PYTEST = ["python", "-m", "pytest", "-q"]


class DemoRunner(Protocol):
    def run(self, argv: list[str], *, cwd: Path, timeout: int) -> CommandResult: ...

    def leftover_containers(self) -> list[str]: ...


@dataclass
class DemoRow:
    name: str
    expected: str
    observed: str
    ok: bool


def snapshot(root: Path) -> dict[str, str]:
    """A fingerprint of every file under root."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "(no output)"


def run_attack_demo(runner: DemoRunner, base_dir: Path) -> list[DemoRow]:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    repo = base_dir / f"_sandbox_demo_{uuid.uuid4().hex[:8]}"
    (repo / "tests").mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_bytes(b"[core]\n")
    (repo / "conftest.py").write_bytes(b"")
    for name, text in _FILES.items():
        (repo / "tests" / name).write_bytes(text.encode("utf-8"))
    before = snapshot(repo)

    rows: list[DemoRow] = []
    os.environ[SECRET_NAME] = "top-secret-value"  # the container must NOT see this
    try:
        res = runner.run([*_PYTEST, "tests/test_containment.py"], cwd=repo, timeout=90)
        observed = (
            _last_line(res.stdout)
            if res.passed
            else f"exit {res.exit_code}: {(res.stdout + res.stderr)[-250:].strip()}"
        )
        rows.append(
            DemoRow(
                "Containment checks (network, read-only disk, non-root, no host env, threads)",
                "all attacks fail, tests pass",
                observed,
                res.passed,
            )
        )

        res = runner.run([*_PYTEST, "tests/test_memory_bomb.py"], cwd=repo, timeout=90)
        rows.append(
            DemoRow(
                "Memory bomb",
                "killed by the memory limit (exit 137)",
                f"exit {res.exit_code}",
                res.exit_code == 137 and not res.timed_out,
            )
        )

        res = runner.run([*_PYTEST, "tests/test_infinite_loop.py"], cwd=repo, timeout=5)
        rows.append(
            DemoRow(
                "Infinite loop",
                "stopped by the 5 second timeout",
                "timed out" if res.timed_out else f"exit {res.exit_code}",
                res.timed_out,
            )
        )

        unchanged = snapshot(repo) == before
        rows.append(
            DemoRow(
                "Host workspace unchanged",
                "no file added, changed or removed",
                "unchanged" if unchanged else "CHANGED",
                unchanged,
            )
        )

        left = runner.leftover_containers()
        rows.append(
            DemoRow(
                "No leftover containers",
                "none",
                ", ".join(left) if left else "none",
                not left,
            )
        )
    finally:
        os.environ.pop(SECRET_NAME, None)
        shutil.rmtree(repo, ignore_errors=True)
    return rows