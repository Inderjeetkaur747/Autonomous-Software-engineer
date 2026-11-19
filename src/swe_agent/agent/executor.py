"""execute_plan(): implement, test in the sandbox, diagnose, repair. Everything is bounded."""

from __future__ import annotations

import logging
import time
from typing import Any

from swe_agent.agent.editor import (
    LLM_EDIT_TOOLS,
    EditOutcome,
    preload_files,
    run_edit_loop,
)
from swe_agent.agent.testparse import describe_outcome, find_weakened_tests, parse_test_run
from swe_agent.config import Settings, get_settings
from swe_agent.llm.client import LLMError
from swe_agent.llm.prompts import (
    build_diagnosis_prompts,
    build_edit_prompts,
    build_repair_prompts,
)
from swe_agent.logging_setup import redact_secrets
from swe_agent.schemas.execution import (
    AttemptRecord,
    ExecutionResult,
    ExecutionStatus,
    FailureCategory,
    FailureDiagnosis,
    PytestOutcome,
)
from swe_agent.schemas.github import IssueData
from swe_agent.schemas.plan import ImplementationPlan
from swe_agent.tools import RepoTools

logger = logging.getLogger(__name__)

PRELOAD_CHARS = 6500


class _SandboxProblem(Exception):
    """The tests could not be run (Docker off, runner disabled, call limit...)."""


def _current_diff(tools: RepoTools) -> tuple[str, bool | None]:
    """(diff text, has_changes). has_changes is None when git could not tell us."""
    result = tools.call("git_diff", {})
    if not result.ok:
        return "", None
    return result.output, bool(result.meta.get("has_changes"))


class _Run:
    def __init__(
        self,
        *,
        issue: IssueData,
        plan: ImplementationPlan,
        approved_files: list[str],
        tools: RepoTools,
        llm: Any,
        settings: Settings,
    ) -> None:
        self.issue = issue
        self.plan_json = plan.model_dump_json()
        self.approved = sorted(approved_files)
        self.tools = tools
        self.llm = llm
        self.s = settings
        self.notes: list[str] = []
        self.attempts: list[AttemptRecord] = []
        self.baseline: PytestOutcome | None = None
        self.final: PytestOutcome | None = None
        self._llm_before = len(llm.calls)
        self._tokens_before = llm.total_tokens
        self._tools_before = len(tools.calls)

    # ------------------------------------------------------------------ steps
    def _run_tests(self) -> PytestOutcome:
        result = self.tools.call("run_tests", {"command": self.s.test_command})
        if not result.ok:
            raise _SandboxProblem(result.error)
        outcome = parse_test_run(result, self.s.diagnosis_output_chars)
        if outcome.timed_out:
            self.notes.append("A test run hit the time limit.")
        return outcome

    def _edit(self, system: str, user: str) -> EditOutcome:
        return run_edit_loop(
            llm=self.llm,
            tools=self.tools,
            system=system,
            user=user,
            allowed=LLM_EDIT_TOOLS,
            max_steps=self.s.max_edit_steps,
            result_chars=self.s.edit_result_chars,
        )

    def _only_old_failures(self, outcome: PytestOutcome) -> bool:
        before = set(self.baseline.failed_tests) if self.baseline else set()
        return bool(outcome.failed_tests) and set(outcome.failed_tests) <= before

    def _test_text(self, outcome: PytestOutcome) -> str:
        text = describe_outcome(outcome)
        if self.baseline and self.baseline.failed_tests:
            text += "\n\nThese tests already failed BEFORE any change: " + ", ".join(
                self.baseline.failed_tests[:10]
            )
        return text

    def _diagnose(self, outcome: PytestOutcome) -> FailureDiagnosis:
        diff_text, _ = _current_diff(self.tools)
        system, user = build_diagnosis_prompts(
            self.issue.title, self.issue.body, self.plan_json, self._test_text(outcome), diff_text
        )
        return self.llm.generate_structured(
            FailureDiagnosis, system=system, user=user, purpose="diagnosis"
        )

    # --------------------------------------------------------------- workflow
    def _workflow(self) -> ExecutionStatus:
        _, dirty = _current_diff(self.tools)
        if dirty:
            self.notes.append(
                "The workspace already has uncommitted changes. Run: swe-agent reset <run_id>"
            )
            return ExecutionStatus.NEEDS_HUMAN

        # 1) Baseline: what happens BEFORE we change anything?
        self.baseline = self._run_tests()
        if self.baseline.failed_tests:
            self.notes.append(
                "Tests that already failed before any change: "
                + ", ".join(self.baseline.failed_tests[:5])
            )

        # 2) Implement the approved plan.
        context = preload_files(self.tools, self.approved, PRELOAD_CHARS)
        system, user = build_edit_prompts(
            self.issue.title, self.issue.body, self.plan_json, self.approved, context
        )
        edit = self._edit(system, user)
        if edit.edits == 0:
            self.notes.extend(edit.notes)
            if edit.stopped == "llm_error":
                return ExecutionStatus.LLM_ERROR
            self.notes.append("The model made no changes. See the tool log for refused edits.")
            return ExecutionStatus.NO_CHANGES

        # 3) Test.
        outcome = self._run_tests()
        self.final = outcome
        self.attempts.append(
            AttemptRecord(
                number=0,
                kind="implement",
                edit_steps=edit.steps,
                edits_made=edit.edits,
                files_changed=self.tools.changed_files,
                outcome=outcome,
                notes=edit.notes,
            )
        )

        # 4) Diagnose and repair, with hard limits.
        previous = outcome.signature()
        for number in range(1, self.s.max_repair_attempts + 1):
            if outcome.passed:
                break
            if self._only_old_failures(outcome):
                self.notes.append(
                    "The remaining failures also failed before any change. "
                    "They are not repaired automatically."
                )
                return ExecutionStatus.TESTS_FAILED

            diagnosis = self._diagnose(outcome)
            self.attempts[-1].diagnosis = diagnosis
            outside = [f for f in diagnosis.files_to_change if f not in self.approved]
            if outside:
                self.notes.append(
                    "The diagnosis suggests files outside the plan (edits will be refused): "
                    + ", ".join(outside)
                )
            if diagnosis.category == FailureCategory.ENVIRONMENT:
                self.notes.append("The failure comes from the environment. A human must look.")
                return ExecutionStatus.NEEDS_HUMAN
            if diagnosis.category == FailureCategory.UNRELATED:
                self.notes.append("The failure looks unrelated to this change.")
                return ExecutionStatus.TESTS_FAILED

            diff_text, _ = _current_diff(self.tools)
            system, user = build_repair_prompts(
                self.issue.title,
                self.issue.body,
                self.plan_json,
                self.approved,
                diagnosis.model_dump_json(),
                self._test_text(outcome),
                diff_text,
            )
            repair = self._edit(system, user)
            if repair.edits == 0:
                self.notes.extend(repair.notes)
                self.notes.append("The repair attempt changed nothing, so the loop stopped.")
                return (
                    ExecutionStatus.LLM_ERROR
                    if repair.stopped == "llm_error"
                    else ExecutionStatus.STUCK
                )

            outcome = self._run_tests()
            self.final = outcome
            self.attempts.append(
                AttemptRecord(
                    number=number,
                    kind="repair",
                    edit_steps=repair.steps,
                    edits_made=repair.edits,
                    files_changed=self.tools.changed_files,
                    outcome=outcome,
                    notes=repair.notes,
                )
            )
            if not outcome.passed and outcome.signature() == previous:
                self.notes.append("The same failure happened again, so the loop stopped.")
                return ExecutionStatus.STUCK
            previous = outcome.signature()

        if outcome.passed:
            return ExecutionStatus.TESTS_PASSED
        self.notes.append(f"The repair limit was reached ({self.s.max_repair_attempts}).")
        return ExecutionStatus.TESTS_FAILED

    # ----------------------------------------------------------------- result
    def go(self) -> ExecutionResult:
        started = time.perf_counter()
        try:
            status = self._workflow()
        except _SandboxProblem as exc:
            self.notes.append(f"The tests could not run: {exc}")
            status = ExecutionStatus.SANDBOX_UNAVAILABLE
        except LLMError as exc:
            self.notes.append(f"LLM error: {exc}")
            status = ExecutionStatus.LLM_ERROR
        return self._result(status, started)

    def _result(self, status: ExecutionStatus, started: float) -> ExecutionResult:
        diff_text, _ = _current_diff(self.tools)
        before = set(self.baseline.failed_tests) if self.baseline else set()
        after = set(self.final.failed_tests) if self.final else set()
        outside = [f for f in self.tools.changed_files if f not in self.approved]
        if outside:  # cannot happen (write_paths), but we check anyway
            self.notes.append("Files outside the approved plan changed: " + ", ".join(outside))
        return ExecutionResult(
            status=status,
            attempts=self.attempts,
            baseline=self.baseline,
            final=self.final,
            changed_files=self.tools.changed_files,
            created_files=sorted(self.tools.created_files),
            fixed_tests=sorted(before - after) if self.final else [],
            new_failures=sorted(after - before),
            weakened_tests=find_weakened_tests(diff_text),
            diff_preview=diff_text[:6000],
            notes=[redact_secrets(n)[:400] for n in self.notes],
            tool_log=self.tools.calls[self._tools_before :],
            repairs_used=max(0, len(self.attempts) - 1),
            llm_calls=len(self.llm.calls) - self._llm_before,
            total_tokens=self.llm.total_tokens - self._tokens_before,
            seconds=round(time.perf_counter() - started, 2),
        )


def execute_plan(
    *,
    issue: IssueData,
    plan: ImplementationPlan,
    approved_files: list[str],
    tools: RepoTools,
    llm: Any,
    settings: Settings | None = None,
) -> ExecutionResult:
    """Run the approved plan. The tools must have write_paths = approved_files."""
    run = _Run(
        issue=issue,
        plan=plan,
        approved_files=approved_files,
        tools=tools,
        llm=llm,
        settings=settings or get_settings(),
    )
    return run.go()