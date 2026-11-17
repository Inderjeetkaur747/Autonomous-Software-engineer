import os
import sys

import pytest
from pydantic import ValidationError

from swe_agent.config import Settings
from swe_agent.sandbox import DockerRunner, ExecResult, get_runner
from swe_agent.sandbox.docker_runner import _execute
from swe_agent.security.errors import CommandNotAllowed
from swe_agent.tools import DisabledRunner, RunnerUnavailable

OK = ExecResult(0)
ARGV = ["python", "-m", "pytest", "-q"]


class FakeExec:
    """Returns scripted results and remembers every command it was given."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cmd, timeout, limit):
        self.calls.append((list(cmd), timeout, limit))
        return self.results.pop(0)


def make(fake, **settings_kw):
    settings = Settings(_env_file=None, **settings_kw)
    return DockerRunner(settings, execute=fake, name_factory=lambda: "swe-sbx-test")


def kinds(fake):
    return [call[0][1] for call in fake.calls]


# ------------------------------------------------------------------ command
def test_command_has_every_hardening_flag(tmp_path):
    cmd = make(FakeExec()).build_command(ARGV, tmp_path, "swe-sbx-test")

    def value(flag):
        return cmd[cmd.index(flag) + 1]

    assert "--rm" in cmd and "--read-only" in cmd
    assert value("--network") == "none"
    assert value("--cap-drop") == "ALL"
    assert value("--security-opt") == "no-new-privileges"
    assert value("--user") == "10001:10001"
    assert value("--memory") == value("--memory-swap") == "512m"
    assert value("--pids-limit") == "128"
    assert value("--name") == "swe-sbx-test"
    assert value("--mount") == f"type=bind,source={tmp_path.resolve()},target=/src,readonly"
    tmpfs = [cmd[i + 1] for i, part in enumerate(cmd) if part == "--tmpfs"]
    assert len(tmpfs) == 2 and all("noexec" in t and "nosuid" in t for t in tmpfs)
    assert "--privileged" not in cmd
    assert not any("docker.sock" in part for part in cmd)
    assert cmd[-len(ARGV) - 1] == "swe-agent-sandbox:latest"  # the image, then the command
    assert cmd[-len(ARGV) :] == ARGV


def test_workspace_path_that_could_break_the_mount_option_is_refused(tmp_path):
    with pytest.raises(RunnerUnavailable, match="cannot mount"):
        make(FakeExec()).build_command(ARGV, tmp_path / "a,b", "n")


# ---------------------------------------------------------------------- run
def test_successful_run(tmp_path):
    fake = FakeExec(OK, OK, ExecResult(0, b"3 passed in 0.1s\n"))
    result = make(fake).run(ARGV, cwd=tmp_path, timeout=30)
    assert result.passed and "3 passed" in result.stdout
    assert kinds(fake) == ["version", "image", "run"]
    assert fake.calls[2][1:] == (30, 200_000)  # timeout and output limit are passed on


def test_failing_tests_are_a_result_not_an_error(tmp_path):
    fake = FakeExec(OK, OK, ExecResult(1, b"1 failed\n"))
    result = make(fake).run(ARGV, cwd=tmp_path, timeout=30)
    assert not result.passed and result.exit_code == 1
    assert len(fake.calls) == 3  # nothing to clean up


def test_timeout_removes_the_container(tmp_path):
    fake = FakeExec(OK, OK, ExecResult(None, b"partial", timed_out=True), OK)
    result = make(fake).run(ARGV, cwd=tmp_path, timeout=5)
    assert result.timed_out and not result.passed
    assert fake.calls[-1][0] == ["docker", "rm", "-f", "swe-sbx-test"]


def test_container_is_removed_even_if_running_raises(tmp_path):
    class Boom(FakeExec):
        def __call__(self, cmd, timeout, limit):
            if cmd[1] == "run":
                self.calls.append((list(cmd), timeout, limit))
                raise KeyboardInterrupt
            return super().__call__(cmd, timeout, limit)

    fake = Boom(OK, OK, OK)
    with pytest.raises(KeyboardInterrupt):
        make(fake).run(ARGV, cwd=tmp_path, timeout=5)
    assert fake.calls[-1][0][:3] == ["docker", "rm", "-f"]


def test_docker_start_failure_is_a_runner_error(tmp_path):
    fake = FakeExec(OK, OK, ExecResult(125, b"", b"docker: Error response from daemon: boom"))
    with pytest.raises(RunnerUnavailable, match="failed to start"):
        make(fake).run(ARGV, cwd=tmp_path, timeout=5)


def test_out_of_memory_hint_and_truncation_marker(tmp_path):
    killed = make(FakeExec(OK, OK, ExecResult(137))).run(ARGV, cwd=tmp_path, timeout=5)
    assert "memory" in killed.stderr
    cut = make(FakeExec(OK, OK, ExecResult(0, b"tail", out_truncated=True))).run(
        ARGV, cwd=tmp_path, timeout=5
    )
    assert cut.stdout.startswith("[... output truncated")


def test_only_validated_canonical_pytest_commands_run(tmp_path):
    fake = FakeExec()
    runner = make(fake)
    for bad in (["rm", "-rf", "/"], ["python", "-c", "print(1)"], ["git", "status"],
                ["pytest", "-q"], ["python", "-m", "pytest", "-p", "evil"]):
        with pytest.raises(CommandNotAllowed):
            runner.run(bad, cwd=tmp_path, timeout=5)
    assert fake.calls == []  # docker was never even asked


# ---------------------------------------------------------------- preflight
def test_preflight_messages(tmp_path):
    with pytest.raises(RunnerUnavailable, match="not running"):
        make(FakeExec(ExecResult(1))).run(ARGV, cwd=tmp_path, timeout=5)
    with pytest.raises(RunnerUnavailable, match="swe-agent sandbox build"):
        make(FakeExec(OK, ExecResult(1))).run(ARGV, cwd=tmp_path, timeout=5)


def test_preflight_is_checked_once(tmp_path):
    fake = FakeExec(OK, OK, OK, OK)
    runner = make(fake)
    runner.run(ARGV, cwd=tmp_path, timeout=5)
    runner.run(ARGV, cwd=tmp_path, timeout=5)
    assert kinds(fake) == ["version", "image", "run", "run"]


def test_leftover_containers():
    assert make(FakeExec(ExecResult(0, b"swe-sbx-a\nswe-sbx-b\n"))).leftover_containers() == [
        "swe-sbx-a",
        "swe-sbx-b",
    ]
    assert make(FakeExec(OK)).leftover_containers() == []


# --------------------------------------------------- the real process helper
def test_real_executor_caps_output_and_enforces_the_timeout():
    big = _execute([sys.executable, "-c", "print('x' * 1_000_000)"], 30, 1000)
    assert big.exit_code == 0 and big.out_truncated and len(big.stdout) <= 1000
    slow = _execute([sys.executable, "-c", "import time; time.sleep(30)"], 1, 1000)
    assert slow.timed_out and slow.exit_code is None
    with pytest.raises(RunnerUnavailable):
        _execute(["definitely-not-a-real-program-xyz"], 5, 100)


# ----------------------------------------------------------------- settings
def test_factory_follows_the_setting():
    assert isinstance(get_runner(Settings(_env_file=None, runner_backend="docker")), DockerRunner)
    assert isinstance(
        get_runner(Settings(_env_file=None, runner_backend="disabled")), DisabledRunner
    )


def test_settings_reject_dangerous_values():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sandbox_image="--privileged")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, sandbox_memory="lots")


# --------------------------------------------- real Docker (opt-in, slow)
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_DOCKER") != "1", reason="set RUN_DOCKER=1 to use Docker")
def test_real_sandbox_runs_pytest(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_ok.py").write_bytes(b"def test_ok():\n    assert 1 + 1 == 2\n")
    result = DockerRunner(Settings(_env_file=None)).run(ARGV, cwd=repo, timeout=90)
    assert result.passed, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert [p.name for p in repo.iterdir()] == ["tests"]  # the host folder was not touched