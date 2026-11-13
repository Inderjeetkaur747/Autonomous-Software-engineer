"""Structured output of the issue-analysis step."""

from enum import StrEnum

from pydantic import BaseModel, Field


class IssueType(StrEnum):
    BUG = "bug"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCS = "docs"
    QUESTION = "question"
    OTHER = "other"


class IssueAnalysis(BaseModel):
    """What the agent understood about the issue."""

    summary: str = Field(description="One or two sentences describing the problem")
    issue_type: IssueType
    root_cause_hypothesis: str = Field(
        description="Best guess of why this happens, based only on the given code"
    )
    affected_areas: list[str] = Field(
        description="Relative file paths or module names likely involved"
    )
    acceptance_criteria: list[str] = Field(
        min_length=1, description="Checkable conditions that mean the issue is fixed"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0 = guessing, 1 = certain")
    needs_clarification: bool = False
    clarification_questions: list[str] = Field(default_factory=list)

    evidence: list[str] = Field(
        default_factory=list,
        description="Citations like path:start-end for the code that supports the hypothesis",
    )