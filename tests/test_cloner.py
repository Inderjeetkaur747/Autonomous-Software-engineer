import shutil
import subprocess

import pytest

from swe_agent.github.cloner import GitError, _clone, _git_env, create_branch, get_head_sha, run_git

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def make_local_repo(path):
    path.mkdir()

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t.dev",
             "-c", "commit.gpgsign=false", *args],
            cwd=path, check=True, capture_output=True,
        )

    git("init")
    (path / "a.txt").write_text("hi")
    git("add", ".")
    git("commit", "-m", "init")
    return path


def test_token_is_passed_by_environment_and_scoped_to_github():
    env = _git_env("ghp_" + "a" * 30)
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_CONFIG_KEY_0" not in _git_env("")


@needs_git
def test_head_sha_and_branch(tmp_path):
    repo = make_local_repo(tmp_path / "src")
    assert len(get_head_sha(repo)) == 40
    create_branch(repo, "agent/issue-1")
    assert run_git(["branch", "--show-current"], cwd=repo) == "agent/issue-1"


@needs_git
@pytest.mark.parametrize("bad", ["-x", "a b", "a;b", "", "--force"])
def test_unsafe_branch_names_are_rejected(tmp_path, bad):
    repo = make_local_repo(tmp_path / "src")
    with pytest.raises(GitError, match="Unsafe branch"):
        create_branch(repo, bad)


@needs_git
def test_file_protocol_is_blocked(tmp_path):
    """A malicious 'repo URL' must not be able to read local folders."""
    repo = make_local_repo(tmp_path / "src")
    with pytest.raises(GitError) as info:
        _clone(repo.as_uri(), tmp_path / "copy")
    assert "not allowed" in str(info.value).lower()
    assert not (tmp_path / "copy" / "a.txt").exists()