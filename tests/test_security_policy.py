import pytest

from swe_agent.config import Settings
from swe_agent.schemas.plan import ImplementationPlan, PlanAction, PlanStep
from swe_agent.security import PathGuard, SecurityPolicy

M, C, D = PlanAction.MODIFY, PlanAction.CREATE, PlanAction.DELETE
GITHUB = "ghp_" + "b" * 30
CLEAN_DIFF = (
    "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
    "@@ -1 +1 @@\n-x = 1\n+x = 2\n"
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    return root


def policy(repo, **overrides):
    settings = Settings(_env_file=None, **overrides)
    return SecurityPolicy(PathGuard(repo), settings)


def plan(*steps):
    return ImplementationPlan(
        goal="g",
        test_strategy="t",
        steps=[
            PlanStep(step_number=n, file_path=path, action=action, description="d", rationale="r")
            for n, (path, action) in enumerate(steps, start=1)
        ],
    )


def test_a_normal_plan_passes(repo):
    report = policy(repo).check_plan(plan(("src/a.py", M), ("tests/test_a.py", M)))
    assert report.violations == []
    assert not report.blocked


def test_missing_and_existing_files(repo):
    assert "missing_file" in policy(repo).check_plan(plan(("src/nope.py", M))).codes
    assert "already_exists" in policy(repo).check_plan(plan(("src/a.py", C))).codes


def test_new_files_must_be_tests(repo):
    assert "create_not_test" in policy(repo).check_plan(plan(("src/new.py", C))).codes
    assert not policy(repo).check_plan(plan(("tests/test_new.py", C))).blocked


@pytest.mark.parametrize("path", [".env", ".github/workflows/ci.yml", ".git/config"])
def test_protected_paths_are_blocked(repo, path):
    report = policy(repo).check_plan(plan((path, C)))
    assert report.blocked
    assert "path_denied" in report.codes


def test_delete_is_only_a_warning(repo):
    report = policy(repo).check_plan(plan(("src/a.py", D)))
    assert report.codes == ["delete_requested"]
    assert not report.blocked


def test_too_many_files(repo):
    report = policy(repo, max_files_changed=1).check_plan(
        plan(("src/a.py", M), ("tests/test_a.py", M))
    )
    assert "too_many_files" in report.codes


def test_new_content_checks(repo):
    assert not policy(repo).check_new_content("src/a.py", "x = 2\n").blocked
    big = policy(repo, max_edit_kb=1).check_new_content("src/a.py", "x" * 2000)
    assert "file_too_large" in big.codes
    assert "binary_content" in policy(repo).check_new_content("src/a.py", "a\x00b").codes
    leaked = policy(repo).check_new_content("src/a.py", f"token = '{GITHUB}'")
    assert "secret_in_content" in leaked.codes
    assert GITHUB not in leaked.violations[0].message  # the report never repeats the secret


def test_patch_checks(repo):
    assert not policy(repo).check_patch(CLEAN_DIFF, ["src/a.py"]).blocked
    with_secret = CLEAN_DIFF + f"+key = '{GITHUB}'\n"
    assert "secret_in_patch" in policy(repo).check_patch(with_secret, ["src/a.py"]).codes
    assert "path_denied" in policy(repo).check_patch(CLEAN_DIFF, [".git/config"]).codes
    huge = "+" + "x" * 2000
    assert "patch_too_large" in policy(repo, max_patch_kb=1).check_patch(huge, []).codes