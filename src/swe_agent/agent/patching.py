"""Turn the workspace changes into a reviewed patch.

This module NEVER commits, pushes or merges. It reads the workspace, builds a diff, and
(only after an explicit rejection) throws the workspace changes away.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swe_agent.agent.testparse import find_weakened_tests
from swe_agent.config import Settings, get_settings
from swe_agent.github.cloner import GitError, run_git, run_git_bytes
from swe_agent.schemas.patch import PatchFile, PatchResult
from swe_agent.security import PathGuard, PathSecurityError, SecurityPolicy
from swe_agent.security.policy import is_test_path

logger = logging.getLogger(__name__)

# Options that keep git from running external programs while it makes a diff.
_DIFF = [
    "-c", "core.quotepath=false",
    "diff", "--no-renames", "--no-ext-diff", "--no-textconv", "--no-color",
]


class PatchError(Exception):
    """The patch cannot be built, verified or accepted."""


def ensure_own_repo(repo_dir: Path) -> None:
    """Refuse to run git in a folder that is not its own repository.

    Without this check, git would climb up to a parent repository (your project!).
    """
    try:
        top = run_git(
            ["rev-parse", "--show-toplevel"], cwd=repo_dir, timeout=30, action="rev-parse"
        )
        same = Path(top).samefile(repo_dir)
    except (GitError, OSError) as exc:
        raise PatchError(f"Not a usable git repository: {exc}") from exc
    if not same:
        raise PatchError("This folder is not its own git repository, so git commands are refused.")


def _list_changes(repo: Path) -> tuple[dict[str, str], list[str]]:
    """What changed, according to git: ({path: kind}, untracked paths)."""
    kinds: dict[str, str] = {}
    names = run_git([*_DIFF, "--name-status", "HEAD"], cwd=repo, timeout=60, action="diff")
    for line in names.splitlines():
        status, _, path = line.partition("\t")
        if path:
            kinds[path] = {"A": "added", "D": "deleted"}.get(status[:1], "modified")
    others = run_git(
        ["-c", "core.quotepath=false", "ls-files", "--others", "--exclude-standard"],
        cwd=repo, timeout=60, action="ls-files",
    )
    untracked = sorted(line for line in others.splitlines() if line)
    for path in untracked:
        kinds[path] = "added"
    return kinds, untracked


def _numstat(text: str) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            result[parts[2]] = (parts[0], parts[1])
    return result


def _mode_findings(diff_text: str) -> tuple[list[str], list[str]]:
    """(violations, warnings) from the git header lines of the diff."""
    violations: list[str] = []
    warnings: list[str] = []
    for line in diff_text.splitlines():
        if "120000" in line and line.startswith(("new file mode", "old mode", "new mode")):
            violations.append("The patch creates or changes a symbolic link.")
        elif line.startswith("new file mode 100755") or line.startswith("new mode"):
            warnings.append(f"File permission change: {line}")
    return sorted(set(violations)), warnings


def _acknowledgement(
    execution: dict[str, Any] | None, weakened: list[str]
) -> tuple[bool, str, list[str], list[str], list[str]]:
    """(tests_passed, status, fixed, new_failures, reasons that need --accept-risk)."""
    execution = execution or {}
    status = str(execution.get("status", ""))
    passed = status == "tests_passed"
    fixed = list(execution.get("fixed_tests", []))
    new_failures = list(execution.get("new_failures", []))
    reasons: list[str] = []
    if not passed:
        reasons.append(f"Tests did not pass (execution status: {status or 'unknown'}).")
    if weakened:
        reasons.append("Existing tests may have been weakened.")
    if new_failures:
        reasons.append("Tests that passed before now fail: " + ", ".join(new_failures[:5]))
    return passed, status, fixed, new_failures, reasons


def build_patch(
    repo_dir: Path,
    approved_files: list[str],
    settings: Settings | None = None,
    *,
    execution: dict[str, Any] | None = None,
    base_commit: str = "",
    branch: str = "",
    write_to: Path | None = None,
) -> tuple[PatchResult, bytes]:
    """Build the patch from what git sees. Returns (result, exact patch bytes)."""
    settings = settings or get_settings()
    repo = Path(repo_dir).resolve()
    ensure_own_repo(repo)
    guard = PathGuard(repo)
    approved = set(approved_files)
    base = {"base_commit": base_commit, "branch": branch}

    try:
        kinds, untracked = _list_changes(repo)
    except GitError as exc:
        raise PatchError(f"Could not list the changes: {exc}") from exc
    if not kinds:
        return PatchResult(status="no_changes", **base), b""

    # ---- checks on the FILE LIST, before any diff is made -----------------
    violations: list[str] = []
    if len(kinds) > settings.max_files_changed:
        violations.append(f"{len(kinds)} files changed; the limit is {settings.max_files_changed}.")
    for path, kind in sorted(kinds.items()):
        try:
            rel, _full = guard.check(path, write=True)
        except PathSecurityError as exc:
            violations.append(f"{path}: {exc}")
            continue
        if rel not in approved:
            violations.append(f"{rel}: not part of the approved plan.")
        if kind == "deleted":
            violations.append(f"{rel}: the agent never deletes files.")
        if kind == "added" and not is_test_path(rel):
            violations.append(f"{rel}: new files are only allowed for tests.")
    stubs = [PatchFile(path=p, status=k) for p, k in sorted(kinds.items())]  # type: ignore[arg-type]
    if violations:
        return PatchResult(status="blocked", files=stubs, violations=violations, **base), b""

    # ---- make the diff (new files are marked "intent to add", then unmarked) ----
    marked = False
    try:
        if untracked:
            run_git(
                ["add", "--intent-to-add", "--", *untracked],
                cwd=repo,
                timeout=60,
                action="add",
            )
            marked = True
        data = run_git_bytes([*_DIFF, "HEAD"], cwd=repo, timeout=60, action="diff")
        counts = _numstat(
            run_git([*_DIFF, "--numstat", "HEAD"], cwd=repo, timeout=60, action="numstat")
        )
    except GitError as exc:
        raise PatchError(f"Could not create the diff: {exc}") from exc
    finally:
        if marked:
            try:
                run_git(
                    ["reset", "--quiet", "--", *untracked],
                    cwd=repo,
                    timeout=60,
                    action="reset",
                )
            except GitError as exc:  # the index would keep "intent to add" entries
                logger.warning("could not undo intent-to-add: %s", exc)

    if not data.strip():
        note = ["Git sees no content change (whitespace or line endings only)."]
        return PatchResult(status="no_changes", warnings=note, **base), b""
    if not data.endswith(b"\n"):
        data += b"\n"  # git apply may call a patch without a final newline "corrupt"
    diff_text = data.decode("utf-8", errors="replace")

    # ---- checks on the DIFF ------------------------------------------------
    files: list[PatchFile] = []
    for path, kind in sorted(kinds.items()):
        added, removed = counts.get(path, ("0", "0"))
        if added == "-":
            violations.append(f"{path}: binary changes are not allowed.")
            added = removed = "0"
        files.append(
            PatchFile(path=path, status=kind, additions=int(added), deletions=int(removed))  # type: ignore[arg-type]
        )
    report = SecurityPolicy(guard, settings).check_patch(diff_text, sorted(kinds))
    violations += [v.message for v in report.violations if v.severity == "block"]
    warnings = [v.message for v in report.violations if v.severity == "warn"]
    mode_violations, mode_warnings = _mode_findings(diff_text)
    violations += mode_violations
    warnings += mode_warnings
    if violations:
        return PatchResult(status="blocked", files=files, violations=violations, **base), b""

    # ---- facts for the human ---------------------------------------------
    weakened = find_weakened_tests(diff_text)
    passed, exec_status, fixed, new_failures, reasons = _acknowledgement(execution, weakened)
    applies: bool | None = None
    patch_path = ""
    if write_to is not None:
        target = Path(write_to)
        target.write_bytes(data)
        patch_path = str(target)
        try:  # does it apply to the BASE commit? (checks the index, touches nothing)
            run_git(
                [
                    "apply",
                    "--cached",
                    "--check",
                    "--whitespace=nowarn",
                    "--",
                    str(target.resolve()),
                ],
                cwd=repo,
                timeout=60,
                action="apply --check",
            )
            applies = True
        except GitError as exc:
            applies = False
            warnings.append(f"The patch does not apply cleanly to the base commit: {exc}")

    preview = diff_text[: settings.patch_preview_chars]
    result = PatchResult(
        status="ready",
        patch_hash=hashlib.sha256(data).hexdigest()[:16],
        files=files,
        additions=sum(f.additions for f in files),
        deletions=sum(f.deletions for f in files),
        size_bytes=len(data),
        diff_preview=preview,
        truncated=len(diff_text) > len(preview),
        patch_path=patch_path,
        applies_to_base=applies,
        warnings=warnings,
        weakened_tests=weakened,
        tests_passed=passed,
        execution_status=exec_status,
        fixed_tests=fixed,
        new_failures=new_failures,
        requires_acknowledgement=bool(reasons),
        ack_reasons=reasons,
        **base,
    )
    return result, data


def approve_patch(
    repo_dir: Path,
    approved_files: list[str],
    expected_hash: str,
    out_dir: Path,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Save the patch as 'approved.patch', but ONLY if it is still exactly what was reviewed."""
    result, data = build_patch(repo_dir, approved_files, settings)
    if result.status != "ready":
        raise PatchError(f"The workspace no longer has a valid patch ({result.status}).")
    if not expected_hash or result.patch_hash != expected_hash:
        raise PatchError("The workspace changed after the review, so the approval was refused.")
    out = Path(out_dir)
    path = out / "approved.patch"
    path.write_bytes(data)
    record = {
        "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "patch_hash": result.patch_hash,
        "files": [f.path for f in result.files],
        "size_bytes": result.size_bytes,
    }
    (out / "approval.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"path": str(path), **record}


def discard_changes(repo_dir: Path) -> None:
    """Throw away ALL changes in a workspace repository (after an explicit rejection)."""
    repo = Path(repo_dir)
    ensure_own_repo(repo)
    try:
        run_git(["reset", "--hard", "HEAD"], cwd=repo, timeout=60, action="reset")
        run_git(["clean", "-fd"], cwd=repo, timeout=60, action="clean")
    except GitError as exc:
        raise PatchError(f"Could not discard the changes: {exc}") from exc