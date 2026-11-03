import httpx
import pytest

from swe_agent.config import Settings
from swe_agent.github.client import GitHubClient, GitHubError
from swe_agent.schemas.github import IssueRef, RepoRef

REPO = RepoRef(owner="o", name="r")
ISSUE = IssueRef(repo=REPO, number=7)


def make_client(handler, token=""):
    settings = Settings(_env_file=None, github_token=token)
    http = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://api.github.com"
    )
    return GitHubClient(settings, http_client=http)


def test_get_issue_success_and_null_body():
    def handler(request):
        assert request.url.path == "/repos/o/r/issues/7"
        return httpx.Response(
            200,
            json={
                "number": 7,
                "title": "Bug",
                "body": None,
                "state": "open",
                "labels": [{"name": "bug"}],
                "user": {"login": "alice"},
                "html_url": "https://github.com/o/r/issues/7",
            },
        )

    issue = make_client(handler).get_issue(ISSUE)
    assert issue.title == "Bug"
    assert issue.body == ""
    assert issue.labels == ["bug"]
    assert issue.author == "alice"


def test_pull_request_is_rejected():
    client = make_client(lambda r: httpx.Response(200, json={"number": 7, "pull_request": {}}))
    with pytest.raises(GitHubError, match="pull request"):
        client.get_issue(ISSUE)


def test_404_mentions_token():
    client = make_client(lambda r: httpx.Response(404, json={}))
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        client.get_issue(ISSUE)


def test_rate_limit_message():
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "123"}
    client = make_client(lambda r: httpx.Response(403, headers=headers, json={}))
    with pytest.raises(GitHubError, match="rate limit"):
        client.get_repo_info(REPO)


def test_timeout_becomes_github_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(GitHubError, match="timed out"):
        make_client(handler).get_repo_info(REPO)


def test_token_is_sent_only_when_set():
    seen = []

    def handler(request):
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"full_name": "o/r", "size": 10})

    make_client(handler, token="").get_repo_info(REPO)
    token = "ghp_" + "a" * 30
    make_client(handler, token=token).get_repo_info(REPO)
    assert seen == [None, f"Bearer {token}"]


def test_repo_info_parsing():
    client = make_client(
        lambda r: httpx.Response(
            200,
            json={
                "full_name": "o/r",
                "default_branch": "main",
                "language": "Python",
                "size": 321,
                "stargazers_count": 5,
                "private": False,
                "html_url": "https://github.com/o/r",
            },
        )
    )
    info = client.get_repo_info(REPO)
    assert info.size_kb == 321
    assert info.language == "Python"