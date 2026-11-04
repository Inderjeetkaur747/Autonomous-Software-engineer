import pytest

from swe_agent.config import Settings
from swe_agent.indexing.scanner import is_sensitive_name, language_for, scan_repo


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    for rel, content in {
        "app/main.py": "print('hi')\n",
        "README.md": "# Title\n",
        "Dockerfile": "FROM python:3.11\n",
        ".env": "SECRET=1\n",
        ".env.example": "SECRET=\n",
        "server.pem": "-----BEGIN-----\n",
        "node_modules/lib/index.js": "x = 1\n",
        ".git/config": "[core]\n",
        "uv.lock": "lock\n",
        "image.png": "binary-ish",
        "empty.py": "",
    }.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return repo


def test_scan_keeps_code_and_skips_junk(tmp_path):
    repo = make_repo(tmp_path)
    result = scan_repo(repo, Settings(_env_file=None))
    paths = {f.path for f in result.files}
    assert paths == {"app/main.py", "README.md", "Dockerfile", ".env.example"}
    assert result.skipped["sensitive_name"] == 2  # .env and server.pem
    assert result.skipped["ignored_dir"] == 2  # node_modules and .git
    assert result.skipped["lock_file"] == 1
    assert result.skipped["empty"] == 1


def test_large_files_and_file_limit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.py").write_text("x" * 2000)  # bigger than 1 KB
    for i in range(5):
        (repo / f"f{i}.py").write_text("x = 1\n")
    result = scan_repo(repo, Settings(_env_file=None, max_file_kb=1, max_scan_files=3))
    assert result.skipped["too_large"] == 1
    assert len(result.files) == 3
    assert result.truncated is True


def test_symlinks_are_skipped(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "real.py").write_text("x = 1\n")
    try:
        (repo / "link.py").symlink_to(repo / "real.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not allowed on this machine")
    result = scan_repo(repo, Settings(_env_file=None))
    assert [f.path for f in result.files] == ["real.py"]
    assert result.skipped["symlink"] == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [(".env", True), (".env.production", True), (".env.example", False),
     ("id_rsa", True), ("cert.PEM", True), ("main.py", False)],
)
def test_sensitive_names(name, expected):
    assert is_sensitive_name(name) is expected


def test_language_detection():
    assert language_for("a.PY") == "python"
    assert language_for("Dockerfile") == "dockerfile"
    assert language_for("photo.png") is None