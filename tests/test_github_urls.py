import pytest

from swe_agent.github.urls import (
    InvalidGitHubUrl,
    parse_issue_url,
    parse_repo_url,
    resolve_issue_ref,
)


@pytest.mark.parametrize(
    "text",
    [
        "https://github.com/octocat/Hello-World",
        "https://github.com/octocat/Hello-World.git",
        "https://github.com/octocat/Hello-World/",
        "https://www.github.com/octocat/Hello-World",
        "octocat/Hello-World",
    ],
)
def test_valid_repo_inputs(text):
    repo = parse_repo_url(text)
    assert repo.full_name == "octocat/Hello-World"
    assert repo.clone_url == "https://github.com/octocat/Hello-World.git"


@pytest.mark.parametrize(
    "text",
    [
        "http://github.com/a/b",
        "https://evil.com/a/b",
        "https://github.com.evil.com/a/b",
        "https://user:pw@github.com/a/b",
        "https://github.com\\@evil.com/a/b",
        "file:///etc/passwd",
        "git@github.com:a/b.git",
        "ext::sh -c evil",
        "https://github.com/a",
        "https://github.com/a/b/c",
        "https://github.com/-a/b",
        "https://github.com/a/..",
        "a/b/c",
        "",
    ],
)
def test_dangerous_or_wrong_repo_inputs_are_rejected(text):
    with pytest.raises(InvalidGitHubUrl):
        parse_repo_url(text)


def test_issue_url_ignores_fragment():
    ref = parse_issue_url("https://github.com/octocat/Hello-World/issues/42#issuecomment-1")
    assert ref.number == 42
    assert ref.repo.full_name == "octocat/Hello-World"


@pytest.mark.parametrize(
    "text",
    [
        "https://github.com/a/b/pull/3",
        "https://github.com/a/b/issues/abc",
        "https://github.com/a/b/issues/0",
        "https://github.com/a/b/issues/-1",
        "https://github.com/a/b/issues/12/extra",
        "https://github.com/a/b/issues/\u00b2",
    ],
)
def test_bad_issue_urls_are_rejected(text):
    with pytest.raises(InvalidGitHubUrl):
        parse_issue_url(text)


def test_resolve_with_number_and_repo():
    assert resolve_issue_ref("#12", "me/demo").number == 12
    assert resolve_issue_ref("12", "me/demo").repo.full_name == "me/demo"


def test_resolve_with_full_url():
    assert resolve_issue_ref("https://github.com/me/demo/issues/5").number == 5


def test_number_without_repo_fails():
    with pytest.raises(InvalidGitHubUrl):
        resolve_issue_ref("12")