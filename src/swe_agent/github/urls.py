"""Strict parsing of GitHub links. Only https://github.com is accepted."""

import re
from urllib.parse import urlparse

from swe_agent.schemas.github import IssueRef, RepoRef

ALLOWED_HOSTS = {"github.com", "www.github.com"}

# GitHub usernames: letters, digits, single hyphens, max 39 chars.
_OWNER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
# Repo names: letters, digits, dot, underscore, hyphen. Must not start with '-'.
_REPO_RE = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]{0,99}")
_NUMBER_RE = re.compile(r"#?([0-9]{1,9})")


class InvalidGitHubUrl(ValueError):
    """The text is not a safe GitHub repository or issue link."""


def _make_repo(owner: str, name: str) -> RepoRef:
    name = name.removesuffix(".git")
    if not _OWNER_RE.fullmatch(owner):
        raise InvalidGitHubUrl(f"Invalid GitHub owner: {owner!r}")
    if name in {".", ".."} or not _REPO_RE.fullmatch(name):
        raise InvalidGitHubUrl(f"Invalid repository name: {name!r}")
    return RepoRef(owner=owner, name=name)


def _github_path_parts(text: str) -> list[str]:
    """Check scheme and host, then return the non-empty path parts."""
    try:
        parsed = urlparse(text.strip())
        host = parsed.hostname
        has_credentials = bool(parsed.username or parsed.password)
        has_port = parsed.port is not None
    except ValueError as exc:  # urllib raises ValueError for broken URLs
        raise InvalidGitHubUrl(f"Malformed URL: {exc}") from exc

    if parsed.scheme != "https":
        raise InvalidGitHubUrl("Only https://github.com links are allowed.")
    if host not in ALLOWED_HOSTS:
        raise InvalidGitHubUrl("Only github.com is allowed.")
    if has_credentials or has_port:
        raise InvalidGitHubUrl("Links with credentials or ports are not allowed.")
    return [part for part in parsed.path.split("/") if part]


def parse_repo_url(text: str) -> RepoRef:
    """Accepts 'https://github.com/owner/repo' or the short form 'owner/repo'."""
    text = text.strip()
    if "://" in text:
        parts = _github_path_parts(text)
    else:
        parts = text.strip("/").split("/")
    if len(parts) != 2:
        raise InvalidGitHubUrl("Expected https://github.com/<owner>/<repo> or <owner>/<repo>")
    return _make_repo(parts[0], parts[1])


def parse_issue_url(text: str) -> IssueRef:
    """Accepts 'https://github.com/owner/repo/issues/123'."""
    parts = _github_path_parts(text)
    if len(parts) != 4 or parts[2] != "issues" or not _NUMBER_RE.fullmatch(parts[3]):
        raise InvalidGitHubUrl("Expected https://github.com/<owner>/<repo>/issues/<number>")
    number = int(parts[3])
    if number < 1:
        raise InvalidGitHubUrl("Issue number must be 1 or more.")
    return IssueRef(repo=_make_repo(parts[0], parts[1]), number=number)


def resolve_issue_ref(issue_text: str, repo_text: str | None = None) -> IssueRef:
    """Either a full issue URL, or an issue number plus a repo."""
    issue_text = issue_text.strip()
    match = _NUMBER_RE.fullmatch(issue_text)
    if match:
        if not repo_text:
            raise InvalidGitHubUrl("An issue number needs a repository too.")
        number = int(match.group(1))
        if number < 1:
            raise InvalidGitHubUrl("Issue number must be 1 or more.")
        return IssueRef(repo=parse_repo_url(repo_text), number=number)
    return parse_issue_url(issue_text)