"""A small GitHub REST client with clear error messages."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from swe_agent.config import Settings, get_settings
from swe_agent.schemas.github import IssueData, IssueRef, RepoInfo, RepoRef

logger = logging.getLogger(__name__)


class GitHubError(Exception):
    """A GitHub API call failed."""


class GitHubClient:
    def __init__(
        self, settings: Settings | None = None, http_client: httpx.Client | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=self._settings.github_api_url,
            timeout=self._settings.github_timeout_seconds,
            follow_redirects=True,  # GitHub redirects renamed repos
        )

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "swe-agent-learning-project",
        }
        token = self._settings.github_token.get_secret_value()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _explain_error(response: httpx.Response) -> str:
        status = response.status_code
        if status in (403, 429) and response.headers.get("x-ratelimit-remaining") == "0":
            reset = response.headers.get("x-ratelimit-reset", "unknown")
            return (
                "GitHub rate limit reached (resets at unix time "
                f"{reset}). Add GITHUB_TOKEN to .env to get a higher limit."
            )
        if status == 401:
            return "GitHub rejected the token (401). Check GITHUB_TOKEN."
        if status == 404:
            return (
                "Not found (404). The repo or issue does not exist, "
                "or it is private and needs GITHUB_TOKEN."
            )
        return f"GitHub API error {status}."

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = self._http.get(path, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise GitHubError("GitHub request timed out.") from exc
        except httpx.HTTPError as exc:
            raise GitHubError(f"Network error while calling GitHub: {exc}") from exc

        if response.status_code != 200:
            raise GitHubError(self._explain_error(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise GitHubError("GitHub returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise GitHubError("GitHub returned an unexpected response.")
        return data

    # --------------------------------------------------------------- public API
    def get_repo_info(self, repo: RepoRef) -> RepoInfo:
        data = self._get_json(f"/repos/{repo.owner}/{repo.name}")
        return RepoInfo(
            full_name=data.get("full_name", repo.full_name),
            description=data.get("description"),
            default_branch=data.get("default_branch") or "main",
            language=data.get("language"),
            size_kb=int(data.get("size") or 0),
            private=bool(data.get("private")),
            stars=int(data.get("stargazers_count") or 0),
            html_url=data.get("html_url", f"https://github.com/{repo.full_name}"),
        )

    def get_issue(self, ref: IssueRef) -> IssueData:
        data = self._get_json(f"/repos/{ref.repo.owner}/{ref.repo.name}/issues/{ref.number}")
        if "pull_request" in data:
            raise GitHubError("This is a pull request, not an issue.")
        labels = [
            label["name"]
            for label in data.get("labels", [])
            if isinstance(label, dict) and "name" in label
        ]
        return IssueData(
            number=int(data["number"]),
            title=data.get("title") or "",
            body=data.get("body") or "",  # GitHub sends null for an empty body
            state=data.get("state", "open"),
            labels=labels,
            author=(data.get("user") or {}).get("login", ""),
            html_url=data.get("html_url", ""),
        )