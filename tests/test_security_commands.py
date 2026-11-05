import pytest

from swe_agent.security import CommandNotAllowed, validate_command


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("pytest", ["python", "-m", "pytest"]),
        ("pytest -q", ["python", "-m", "pytest", "-q"]),
        ("python3 -m pytest tests -x", ["python", "-m", "pytest", "tests", "-x"]),
        ("pytest tests/test_a.py::test_x", ["python", "-m", "pytest", "tests/test_a.py::test_x"]),
        (
            "pytest tests -k 'median and even'",
            ["python", "-m", "pytest", "tests", "-k", "median and even"],
        ),
        ("pytest --tb=short --maxfail=2", ["python", "-m", "pytest", "--tb=short", "--maxfail=2"]),
        (["pytest", "-q"], ["python", "-m", "pytest", "-q"]),
        ("git diff --stat", ["git", "diff", "--stat"]),
        ("git status --porcelain", ["git", "status", "--porcelain"]),
        ("git rev-parse HEAD", ["git", "rev-parse", "HEAD"]),
        ("git diff main..HEAD -- src/a.py", ["git", "diff", "main..HEAD", "--", "src/a.py"]),
    ],
)
def test_allowed_commands_become_canonical(command, expected):
    assert validate_command(command).argv == expected


@pytest.mark.parametrize(
    "command",
    [
        "pytest; del /s *",
        "pytest && echo hi",
        "pytest | more",
        "pytest > out.txt",
        "pytest $(whoami)",
        "pytest `id`",
        "rm -rf /",
        "curl http://evil.example",
        "python script.py",
        "python -c 'print(1)'",
        "python -m pip install x",
        "pytest -p evil_plugin",
        "pytest -c evil.ini",
        "pytest --rootdir=/",
        "pytest --basetemp=/tmp/x",
        "pytest ../other",
        "pytest /etc",
        "pytest -k",
        "./pytest",
        "git push origin main",
        "git commit -m x",
        "git -c core.pager=sh log",
        "git diff --output=x",
        "git diff ../secret",
        "git config --global user.name x",
        "",
        "pytest 'unclosed",
    ],
)
def test_everything_else_is_denied(command):
    with pytest.raises(CommandNotAllowed):
        validate_command(command)


def test_list_form_is_checked_too():
    with pytest.raises(CommandNotAllowed):
        validate_command(["pytest", "a;b"])


def test_too_many_tokens_is_denied():
    with pytest.raises(CommandNotAllowed, match="too long"):
        validate_command(" ".join(["pytest"] + ["tests"] * 30))