"""Hard rules for what the agent may change. Checked in code, never left to the LLM."""

from __future__ import annotations

from dataclasses import dataclass, field

from swe_agent.config import Settings, get_settings
from swe_agent.schemas.plan import ImplementationPlan, PlanAction
from swe_agent.security.errors import PathSecurityError
from swe_agent.security.paths import PathGuard
from swe_agent.security.secrets import scan_added_lines, scan_text


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    severity: str = "block"  # "block" stops the run, "warn" is shown to the human


@dataclass
class PolicyReport:
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(v.severity == "block" for v in self.violations)

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.violations]

    def add(self, code: str, message: str, severity: str = "block") -> None:
        self.violations.append(Violation(code, message, severity))


def _is_test_path(rel: str) -> bool:
    parts = rel.lower().split("/")
    name = parts[-1]
    return (
        "tests" in parts[:-1]
        or "test" in parts[:-1]
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".spec.js"))
    )


class SecurityPolicy:
    def __init__(self, guard: PathGuard, settings: Settings | None = None) -> None:
        self._guard = guard
        self._settings = settings or get_settings()

    def check_plan(self, plan: ImplementationPlan) -> PolicyReport:
        """Run BEFORE asking the human to approve a plan."""
        report = PolicyReport()
        files: set[str] = set()
        for step in plan.steps:
            label = f"step {step.step_number}"
            try:
                rel, full = self._guard.check(step.file_path, write=True)
            except PathSecurityError as exc:
                report.add("path_denied", f"{label}: {exc}")
                continue
            files.add(rel)
            exists = full.is_file()
            if step.action == PlanAction.MODIFY and not exists:
                report.add("missing_file", f"{label}: {rel} does not exist.")
            elif step.action == PlanAction.CREATE:
                if exists:
                    report.add("already_exists", f"{label}: {rel} already exists.")
                if not _is_test_path(rel):
                    report.add("create_not_test", f"{label}: new files are only allowed for tests.")
            elif step.action == PlanAction.DELETE:
                if not exists:
                    report.add("missing_file", f"{label}: {rel} does not exist.")
                else:
                    report.add("delete_requested", f"{label}: deletes {rel}.", "warn")
        if len(files) > self._settings.max_files_changed:
                report.add(
                "too_many_files",
                f"Plan touches {len(files)} files; "
                f"the limit is {self._settings.max_files_changed}.",
            )
        return report

    def check_new_content(self, rel_path: str, content: str) -> PolicyReport:
        """Run on every file the LLM wants to write."""
        report = PolicyReport()
        try:
            self._guard.check(rel_path, write=True)
        except PathSecurityError as exc:
            report.add("path_denied", str(exc))
        if len(content.encode("utf-8")) > self._settings.max_edit_kb * 1024:
            report.add("file_too_large", f"Content is over {self._settings.max_edit_kb} KB.")
        if "\x00" in content:
            report.add("binary_content", "Content contains NUL bytes.")
        kinds = sorted({f.kind for f in scan_text(content)})
        if kinds:
            report.add("secret_in_content", f"Content looks like it contains: {', '.join(kinds)}")
        return report

    def check_patch(self, diff_text: str, changed_files: list[str]) -> PolicyReport:
        """Run on the final diff, before showing it for approval."""
        report = PolicyReport()
        if len(diff_text.encode("utf-8")) > self._settings.max_patch_kb * 1024:
            report.add("patch_too_large", f"Patch is over {self._settings.max_patch_kb} KB.")
        if len(changed_files) > self._settings.max_files_changed:
            report.add("too_many_files", f"{len(changed_files)} files changed.")
        for path in changed_files:
            try:
                self._guard.check(path, write=True)
            except PathSecurityError as exc:
                report.add("path_denied", f"{path}: {exc}")
        kinds = sorted({f.kind for f in scan_added_lines(diff_text)})
        if kinds:
            report.add(
                "secret_in_patch",
                f"Patch adds something that looks like: {', '.join(kinds)}",
            )
        return report