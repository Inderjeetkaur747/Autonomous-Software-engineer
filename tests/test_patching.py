import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from swe_agent.agent import PatchError, approve_patch, build_patch, discard_changes
from swe_agent.config import Settings
from swe_agent.github.cloner import run_git, run_git_bytes

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

SETTINGS = Settings(_env_file=None)
STATS = "mathkit/stats.py"
NEW_TEST = "tests/test_new.py"
SRC = b"def average(numbers):\n    return sum(numbers) / len(numbers)\n"
TEST = b"def test_average():\n    assert average([1]) == 1\n    assert average([2]) == 2\n"
GITHUB = "ghp_" + "b" * 30


def git(root, *args):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t.dev",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "mathkit").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / STATS).write_bytes(SRC)
    (root / "tests" / "test_stats.py").write_bytes(TEST)
    git(root, "init")
    git(root, "add", "-A")
    git(root, "commit", "-m", "init")
    return root


def change(repo, rel, old, new):
    path = repo / rel
    path.write_bytes(path.read_bytes().replace(old, new))


def fix_average(repo):
    change(
        repo,
        STATS,
        b"    return sum",
        b"    if not numbers:\n        return 0.0\n    return sum",
    )


# ------------------------------------------------------------------ happy path
def test_patch_has_edits_and_new_files_and_leaves_git_untouched(repo, tmp_path):
    fix_average(repo)
    (repo / NEW_TEST).write_bytes(b"def test_new():\n    assert True\n")
    out = tmp_path / "patch.diff"
    result, data = build_patch(
        repo, [STATS, NEW_TEST], SETTINGS, write_to=out, base_commit="abc", branch="agent/issue-1",
        execution={"status": "tests_passed", "fixed_tests": ["t::a"]},
    )
    assert result.status == "ready"
    assert {(f.path, f.status) for f in result.files} == {(STATS, "modified"), (NEW_TEST, "added")}
    assert (result.additions, result.deletions) == (4, 0)
    assert b"+++ b/tests/test_new.py" in data and data.endswith(b"\n")
    assert out.read_bytes() == data
    assert result.patch_hash == hashlib.sha256(data).hexdigest()[:16]
    assert result.applies_to_base is True
    assert result.tests_passed and not result.requires_acknowledgement
    assert (result.base_commit, result.branch) == ("abc", "agent/issue-1")
    # git is left exactly as it was: nothing staged, the new file still untracked
    assert run_git(["diff", "--cached", "--name-only"], cwd=repo) == ""
    assert NEW_TEST in run_git(["ls-files", "--others", "--exclude-standard"], cwd=repo)
    again, _ = build_patch(repo, [STATS, NEW_TEST], SETTINGS)
    assert again.patch_hash == result.patch_hash  # deterministic


def test_no_changes(repo):
    result, data = build_patch(repo, [STATS], SETTINGS)
    assert result.status == "no_changes" and data == b""


def test_preview_is_capped_but_the_hash_covers_everything(repo):
    fix_average(repo)
    small = Settings(_env_file=None, patch_preview_chars=2000)
    long_line = b"x = '" + b"a" * 100 + b"'\n"
    (repo / NEW_TEST).write_bytes(long_line * 60)
    result, data = build_patch(repo, [STATS, NEW_TEST], small)
    assert result.truncated and len(result.diff_preview) == 2000 and len(data) > 2000


# ------------------------------------------------------------------ blocked
def blocked(result, data, text):
    assert result.status == "blocked" and data == b""
    assert text in " ".join(result.violations)


def test_files_outside_the_plan_are_blocked(repo):
    fix_average(repo)
    blocked(*build_patch(repo, [], SETTINGS), "approved plan")


def test_deleted_files_are_blocked(repo):
    (repo / STATS).unlink()
    blocked(*build_patch(repo, [STATS], SETTINGS), "never deletes")


def test_new_non_test_files_are_blocked(repo):
    (repo / "mathkit" / "extra.py").write_bytes(b"x = 1\n")
    blocked(*build_patch(repo, ["mathkit/extra.py"], SETTINGS), "only allowed for tests")


def test_secrets_are_blocked_and_never_repeated(repo):
    change(repo, STATS, b"def average", f"T = '{GITHUB}'\ndef average".encode())
    result, data = build_patch(repo, [STATS], SETTINGS)
    blocked(result, data, "looks like")
    assert GITHUB not in " ".join(result.violations)


def test_too_many_files_are_blocked(repo):
    for name in "abc":
        (repo / "tests" / f"test_{name}.py").write_bytes(b"x = 1\n")
    small = Settings(_env_file=None, max_files_changed=2)
    result, _ = build_patch(repo, [f"tests/test_{n}.py" for n in "abc"], small)
    assert result.status == "blocked" and "limit" in result.violations[0]


# -------------------------------------------------------------- risk reasons
def test_failing_tests_need_acknowledgement(repo):
    fix_average(repo)
    result, _ = build_patch(repo, [STATS], SETTINGS, execution={"status": "tests_failed"})
    assert result.requires_acknowledgement
    assert "did not pass" in result.ack_reasons[0]
    no_info, _ = build_patch(repo, [STATS], SETTINGS)  # no execution result at all
    assert no_info.requires_acknowledgement


def test_weakened_tests_need_acknowledgement(repo):
    change(repo, "tests/test_stats.py", b"    assert average([2]) == 2\n", b"")
    result, _ = build_patch(
        repo, ["tests/test_stats.py"], SETTINGS, execution={"status": "tests_passed"}
    )
    assert result.weakened_tests and result.requires_acknowledgement
    assert any("weakened" in r for r in result.ack_reasons)


# ----------------------------------------------------------- approve / discard
def test_approval_saves_the_patch_only_if_it_is_unchanged(repo, tmp_path):
    fix_average(repo)
    result, data = build_patch(repo, [STATS], SETTINGS)
    out = tmp_path / "out"
    out.mkdir()
    info = approve_patch(repo, [STATS], result.patch_hash, out, SETTINGS)
    assert (out / "approved.patch").read_bytes() == data
    assert (out / "approval.json").exists() and info["patch_hash"] == result.patch_hash

    change(repo, STATS, b"return 0.0", b"return 9.9")  # edited after the review
    with pytest.raises(PatchError, match="changed after the review"):
        approve_patch(repo, [STATS], result.patch_hash, out, SETTINGS)
    with pytest.raises(PatchError):
        approve_patch(repo, [STATS], "", out, SETTINGS)


def test_discard_removes_edits_and_new_files(repo):
    fix_average(repo)
    (repo / NEW_TEST).write_bytes(b"def test_new():\n    assert True\n")
    discard_changes(repo)
    restored = (repo / STATS).read_bytes().replace(b"\r\n", b"\n")
    assert restored == SRC  # git may write CRLF on Windows (autocrlf)
    assert not (repo / NEW_TEST).exists()
    assert build_patch(repo, [STATS], SETTINGS)[0].status == "no_changes"


def test_git_commands_refuse_a_folder_that_is_not_its_own_repo(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init")
    child = parent / "child"
    child.mkdir()
    with pytest.raises(PatchError, match="own git repository"):
        discard_changes(child)
    with pytest.raises(PatchError):
        build_patch(child, [], SETTINGS)


# ------------------------------------------------------------------- bytes
def test_run_git_bytes_keeps_carriage_returns(tmp_path):
    root = tmp_path / "crlf"
    root.mkdir()
    (root / "a.txt").write_bytes(b"a\r\nb\r\n")
    git(root, "init")
    git(root, "-c", "core.autocrlf=false", "add", "-A")
    git(root, "-c", "core.autocrlf=false", "commit", "-m", "x")
    assert run_git_bytes(["cat-file", "blob", "HEAD:a.txt"], cwd=root) == b"a\r\nb\r\n"


# ------------------------------------------------------------ never pushes
def test_source_code_never_runs_push_merge_or_commit():
    src = Path(__file__).resolve().parents[1] / "src"
    pattern = re.compile(r'"(?:push|merge|commit|pull|fetch|rebase|remote)"')
    hits = [
        f"{path.name}:{number}"
        for path in src.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert hits == []