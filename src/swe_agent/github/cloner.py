"""Safe git helpers.

Every command: is a list (no shell), has a timeout, and may only use https.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
from pathlib import Path

from swe_agent.logging_setup import redact_secrets
from swe_agent.schemas.github import RepoRef

logger = logging.getLogger(__name__)

_BRANCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,99}")


class GitError(Exception):
    """A git command failed."""


def _git_env(token: str) -> dict[str, str]:
    """Environment for git. The token lives only here, never in a URL or a file."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"  # never ask for a password
    env["GIT_ALLOW_PROTOCOL"] = "https"  # blocks file://, ext::, ssh, git://
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        # Scoped to github.com so a redirect to another site cannot receive it.
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {basic}"
    return env


def run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    token: str = "",
    timeout: int = 60,
    action: str = "command",
) -> str:
    try:
        result = subprocess.run(  # noqa: S603 (list args, shell is off)
            ["git", *args],
            cwd=cwd,
            env=_git_env(token),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {action} timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise GitError("git is not installed or not on PATH") from exc

    if result.returncode != 0:
        detail = redact_secrets((result.stderr or result.stdout).strip())[:500]
        raise GitError(f"git {action} failed: {detail}")
    return result.stdout.strip()


def _clone(url: str, dest: Path, *, token: str = "", timeout: int = 180) -> None:
    run_git(
        [
            "-c", "core.symlinks=false",  # symlinks become plain files (no escape tricks)
            "-c", "core.longpaths=true",  # helps with long Windows paths
            "clone",
            "--depth", "1",
            "--single-branch",
            "--no-tags",
            "--",  # everything after this is data, never an option
            url,
            str(dest),
        ],
        token=token,
        timeout=timeout,
        action="clone",
    )


def create_branch(repo_dir: Path, name: str) -> None:
    if not _BRANCH_RE.fullmatch(name):
        raise GitError(f"Unsafe branch name: {name!r}")
    run_git(["checkout", "-b", name], cwd=repo_dir, action="checkout")


def get_head_sha(repo_dir: Path) -> str:
    return run_git(["rev-parse", "HEAD"], cwd=repo_dir, action="rev-parse")


def clone_repo(
    repo: RepoRef,
    dest: Path,
    *,
    work_branch: str,
    token: str = "",
    timeout: int = 180,
) -> str:
    """Clone, switch to a work branch, and return the commit SHA we started from."""
    logger.info("cloning %s", repo.full_name)
    _clone(repo.clone_url, dest, token=token, timeout=timeout)
    create_branch(dest, work_branch)
    sha = get_head_sha(dest)
    logger.info("clone done sha=%s branch=%s", sha[:10], work_branch)
    return sha