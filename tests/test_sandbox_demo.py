import os
from pathlib import Path

from swe_agent.sandbox import run_attack_demo
from swe_agent.tools import CommandResult


class FakeRunner:
    def __init__(self, results, mutate=False):
        self.results = list(results)
        self.mutate = mutate
        self.calls = []
        self.secret_seen = []

    def run(self, argv, *, cwd, timeout):
        self.calls.append((argv, timeout))
        self.secret_seen.append(os.environ.get("SWE_DEMO_SECRET"))
        if self.mutate:
            (Path(cwd) / "evil.txt").write_text("written by a hostile test")
        return self.results.pop(0)

    def leftover_containers(self):
        return []


def good_results():
    return [
        CommandResult(argv=[], exit_code=0, stdout="9 passed in 1.0s\n"),
        CommandResult(argv=[], exit_code=137),
        CommandResult(argv=[], exit_code=None, timed_out=True),
    ]


def test_demo_reports_every_check_and_cleans_up(tmp_path):
    runner = FakeRunner(good_results())
    rows = run_attack_demo(runner, tmp_path)
    assert len(rows) == 5 and all(row.ok for row in rows)
    assert rows[0].observed == "9 passed in 1.0s"
    assert runner.calls[2][1] == 5  # the loop test gets a short timeout
    assert runner.secret_seen == ["top-secret-value"] * 3  # the secret was set on the host
    assert "SWE_DEMO_SECRET" not in os.environ  # and removed afterwards
    assert list(tmp_path.iterdir()) == []  # the temporary repo was deleted


def test_demo_notices_when_the_host_folder_changes(tmp_path):
    rows = run_attack_demo(FakeRunner(good_results(), mutate=True), tmp_path)
    by_name = {row.name: row for row in rows}
    assert not by_name["Host workspace unchanged"].ok
    assert by_name["Host workspace unchanged"].observed == "CHANGED"


def test_demo_notices_when_an_attack_succeeds(tmp_path):
    results = good_results()
    results[1] = CommandResult(argv=[], exit_code=0)  # the memory bomb was not stopped
    rows = run_attack_demo(FakeRunner(results), tmp_path)
    assert [row.ok for row in rows] == [True, False, True, True, True]