import pytest

from swe_agent.security import PathGuard, PathSecurityError


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_bytes(b"x = 1\n")
    (root / ".env").write_bytes(b"SECRET=1\n")
    (root / ".env.example").write_bytes(b"SECRET=\n")
    return root


def test_good_path_is_normalized_and_inside_root(repo):
    rel, full = PathGuard(repo).check("src\\a.py")
    assert rel == "src/a.py"
    assert full == (repo / "src" / "a.py").resolve()


@pytest.mark.parametrize(
    "bad",
    [
        "../x.py", "/etc/passwd", "C:\\Windows\\x.py", "a/../../b.py", "", "   ",
        "a\x00.py", "src/aux.py", "src/a.py.", "src/a.py:stream",
        "\\\\server\\share\\x.py", "x" * 301,
    ],
)
def test_dangerous_paths_are_rejected(repo, bad):
    with pytest.raises(PathSecurityError):
        PathGuard(repo).check(bad)


@pytest.mark.parametrize("bad", [".git/config", ".GIT/config", "src/.git/hooks/pre-commit"])
def test_git_folder_is_off_limits(repo, bad):
    with pytest.raises(PathSecurityError, match="git"):
        PathGuard(repo).check(bad)


@pytest.mark.parametrize("bad", [".env", "src/.env.local", "keys/server.pem"])
def test_secret_files_are_off_limits(repo, bad):
    with pytest.raises(PathSecurityError):
        PathGuard(repo).check(bad)
    with pytest.raises(PathSecurityError):
        PathGuard(repo).read_text(bad)


def test_env_example_is_allowed(repo):
    assert PathGuard(repo).read_text(".env.example") == "SECRET=\n"


def test_write_rules(repo):
    guard = PathGuard(repo)
    assert guard.check("src/new_mod.py", write=True)[0] == "src/new_mod.py"
    with pytest.raises(PathSecurityError):
        guard.check(".github/workflows/ci.yml", write=True)
    with pytest.raises(PathSecurityError):
        guard.check("logo.png", write=True)


def test_read_text_limits(repo):
    guard = PathGuard(repo, max_read_bytes=10)
    assert guard.read_text("src/a.py") == "x = 1\n"
    (repo / "big.py").write_text("x" * 50)
    (repo / "bin.py").write_bytes(b"\x00\x01")
    with pytest.raises(PathSecurityError, match="too large"):
        guard.read_text("big.py")
    with pytest.raises(PathSecurityError, match="Binary"):
        guard.read_text("bin.py")
    with pytest.raises(PathSecurityError, match="Not a file"):
        guard.read_text("src/missing.py")


def test_symlink_pointing_outside_is_blocked(repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    try:
        (repo / "link.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not allowed on this machine")
    with pytest.raises(PathSecurityError, match="escapes"):
        PathGuard(repo).read_text("link.py")