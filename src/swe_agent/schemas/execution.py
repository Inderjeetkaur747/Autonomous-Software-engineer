"""Typed data for editing, testing, diagnosis and repair."""

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from swe_agent.schemas.tools import ToolCallRecord


class FailureCategory(StrEnum):
    CODE_BUG = "code_bug"
    TEST_BUG = "test_bug"
    ENVIRONMENT = "environment"
    UNRELATED = "unrelated"
    UNCLEAR = "unclear"


class FailureDiagnosis(BaseModel):
    """The LLM's explanation of a failing test run."""

    category: FailureCategory
    summary: str = Field(description="One sentence: what fails")
    root_cause: str = Field(description="Why it fails, based only on the output and the diff")
    failing_tests: list[str] = Field(default_factory=list)
    suggested_fix: str = Field(description="A small, concrete change")
    files_to_change: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class PytestOutcome(BaseModel):
    """One test run, read by plain code."""

    exit_code: int | None = None
    passed: bool = False
    timed_out: bool = False
    counts: dict[str, int] = Field(default_factory=dict)  # {"passed": 3, "failed": 1}
    failed_tests: list[str] = Field(default_factory=list)
    failure_messages: dict[str, str] = Field(default_factory=dict)
    summary_line: str = ""
    output_tail: str = ""

    def signature(self) -> str:
        """Same failure gives the same text. Timings are left out on purpose."""
        parts = [str(self.exit_code), str(self.timed_out), repr(sorted(self.counts.items()))]
        parts += sorted(
            f"{test}|{self.failure_messages.get(test, '')[:100]}" for test in self.failed_tests
        )
        return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]  # noqa: S324


class AttemptRecord(BaseModel):
    number: int  # 0 = first implementation, 1 and more = repairs
    kind: Literal["implement", "repair"]
    edit_steps: int = 0
    edits_made: int = 0
    files_changed: list[str] = Field(default_factory=list)  # all changed files so far
    outcome: PytestOutcome | None = None
    diagnosis: FailureDiagnosis | None = None
    notes: list[str] = Field(default_factory=list)


class ExecutionStatus(StrEnum):
    TESTS_PASSED = "tests_passed"
    TESTS_FAILED = "tests_failed"
    NO_CHANGES = "no_changes"
    STUCK = "stuck"
    NEEDS_HUMAN = "needs_human"
    SANDBOX_UNAVAILABLE = "sandbox_unavailable"
    LLM_ERROR = "llm_error"


class ExecutionResult(BaseModel):
    status: ExecutionStatus
    attempts: list[AttemptRecord] = Field(default_factory=list)
    baseline: PytestOutcome | None = None
    final: PytestOutcome | None = None
    changed_files: list[str] = Field(default_factory=list)
    created_files: list[str] = Field(default_factory=list)
    fixed_tests: list[str] = Field(default_factory=list)  # failed before, pass now
    new_failures: list[str] = Field(default_factory=list)  # passed before, fail now
    weakened_tests: list[str] = Field(default_factory=list)
    diff_preview: str = ""
    notes: list[str] = Field(default_factory=list)
    tool_log: list[ToolCallRecord] = Field(default_factory=list)
    repairs_used: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    seconds: float = 0.0