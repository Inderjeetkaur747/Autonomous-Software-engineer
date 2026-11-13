import json

import pytest

from swe_agent.config import Settings
from swe_agent.github.cloner import GitError
from swe_agent.github.service import RepoTooLargeError, prepare_run
from swe_agent.github.workspace import WorkspaceManager
from swe_agent.schemas.github import IssueData, IssueRef, RepoInfo, RepoRef

REF = IssueRef(repo=RepoRef(owner="o", name="r"), number=3)


class FakeGitHub:
    def __init__(self, size_kb=100):
        self.size_kb = size_kb

    def get_repo_info(self, repo):
        return RepoInfo(full_name=repo.full_name, size_kb=self.size_kb, language="Python")

    def get_issue(self, ref):
        return IssueData(number=ref.number, title="Bug", body="Body", labels=["bug"])


def make_env(tmp_path):
    settings = Settings(
        _env_file=None,
        workspace_dir=tmp_path / "ws",
        max_repo_size_mb=1,
        github_token="ghp_" + "a" * 30,
    )
    return settings, WorkspaceManager(settings)


def test_success_writes_metadata_and_passes_token(tmp_path):
    settings, manager = make_env(tmp_path)
    calls = []

    def fake_cloner(repo, dest, *, work_branch, token, timeout):
        calls.append((work_branch, token, timeout))
        dest.mkdir(parents=True)
        (dest / "a.py").write_text("x = 1")
        return "abc123"

    run = prepare_run(
        REF, settings=settings, github=FakeGitHub(), workspaces=manager, cloner=fake_cloner
    )
    assert run.branch == "agent/issue-3"
    assert run.head_sha == "abc123"
    assert calls[0][0] == "agent/issue-3"
    assert calls[0][1].startswith("ghp_")
    meta = json.loads((run.workspace.root / "run.json").read_text(encoding="utf-8"))
    assert meta["issue"]["title"] == "Bug"
    assert meta["base_commit"] == "abc123"


def test_huge_repo_is_refused_before_any_folder_is_made(tmp_path):
    settings, manager = make_env(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("cloner should not be called")

    with pytest.raises(RepoTooLargeError):
        prepare_run(
            REF,
            settings=settings,
            github=FakeGitHub(size_kb=5000),
            workspaces=manager,
            cloner=must_not_run,
        )
    assert list(manager.base.iterdir()) == []


def test_failed_clone_removes_the_workspace(tmp_path):
    settings, manager = make_env(tmp_path)

    def broken_cloner(repo, dest, *, work_branch, token, timeout):
        dest.mkdir(parents=True)
        raise GitError("network down")

    with pytest.raises(GitError):
        prepare_run(
            REF, settings=settings, github=FakeGitHub(), workspaces=manager, cloner=broken_cloner
        )
    assert list(manager.base.iterdir()) == []

def test_caller_can_choose_the_run_id(tmp_path):
    settings, manager = make_env(tmp_path)

    def fake_cloner(repo, dest, *, work_branch, token, timeout):
        dest.mkdir(parents=True)
        return "abc123"

    run = prepare_run(
        REF, settings=settings, github=FakeGitHub(), workspaces=manager,
        cloner=fake_cloner, run_id="abcdef012345",
    )
    assert run.workspace.run_id == "abcdef012345"