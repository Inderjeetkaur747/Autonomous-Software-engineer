"""prepare_run(): from an issue link to a ready-to-use workspace."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_agent.config import Settings, get_settings
from swe_agent.github.client import GitHubClient
from swe_agent.github.cloner import clone_repo
from swe_agent.github.workspace import Workspace, WorkspaceManager
from swe_agent.schemas.github import IssueData, IssueRef, RepoInfo

logger = logging.getLogger(__name__)


class RepoTooLargeError(Exception):
    """The repository is bigger than MAX_REPO_SIZE_MB."""


@dataclass
class PreparedRun:
    workspace: Workspace
    repo_info: RepoInfo
    issue: IssueData
    head_sha: str
    branch: str


def prepare_run(
    issue_ref: IssueRef,
    *,
    run_id: str | None = None,
    settings: Settings | None = None,
    github: Any | None = None,
    workspaces: WorkspaceManager | None = None,
    cloner: Callable[..., str] = clone_repo,
) -> PreparedRun:
    settings = settings or get_settings()
    workspaces = workspaces or WorkspaceManager(settings)

    # 1) Ask GitHub first. These calls are cheap and can fail early.
    owns_client = github is None
    github = github or GitHubClient(settings)
    try:
        repo_info = github.get_repo_info(issue_ref.repo)
        issue = github.get_issue(issue_ref)
    finally:
        if owns_client:
            github.close()

    # 2) Refuse huge repos BEFORE downloading anything.
    max_kb = settings.max_repo_size_mb * 1024
    if repo_info.size_kb > max_kb:
        raise RepoTooLargeError(
            f"Repository is about {repo_info.size_kb // 1024} MB; "
            f"the limit is {settings.max_repo_size_mb} MB (MAX_REPO_SIZE_MB)."
        )

    # 3) Create the workspace and clone. If anything fails, clean up.
    workspace = workspaces.create(run_id)
    branch = f"agent/issue-{issue_ref.number}"
    try:
        head_sha = cloner(
            issue_ref.repo,
            workspace.repo_dir,
            work_branch=branch,
            token=settings.github_token.get_secret_value(),
            timeout=settings.clone_timeout_seconds,
        )
        _write_metadata(workspace.root, issue_ref, repo_info, issue, head_sha, branch)
    except Exception:
        logger.exception("prepare_run failed, deleting workspace %s", workspace.run_id)
        workspaces.delete(workspace.run_id)
        raise

    return PreparedRun(workspace, repo_info, issue, head_sha, branch)


def _write_metadata(
    root: Path,
    issue_ref: IssueRef,
    repo_info: RepoInfo,
    issue: IssueData,
    head_sha: str,
    branch: str,
) -> None:
    meta = {
        "run_id": root.name,
        "repo": issue_ref.repo.full_name,
        "branch": branch,
        "base_commit": head_sha,
        "repo_info": repo_info.model_dump(mode="json"),
        "issue": issue.model_dump(mode="json"),
    }
    (root / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")