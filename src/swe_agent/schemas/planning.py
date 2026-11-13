"""The result of the analysis and planning stage."""

from enum import StrEnum

from pydantic import BaseModel, Field

from swe_agent.schemas.analysis import IssueAnalysis
from swe_agent.schemas.plan import ImplementationPlan
from swe_agent.schemas.tools import ToolCallRecord


class PlanningStatus(StrEnum):
    READY = "ready"  # a plan passed all checks, waiting for human approval
    NEEDS_CLARIFICATION = "needs_clarification"  # the issue is too unclear to plan
    BLOCKED = "blocked"  # the plan broke a security rule, even after a retry


class PlanningResult(BaseModel):
    status: PlanningStatus
    analysis: IssueAnalysis | None = None
    plan: ImplementationPlan | None = None
    citations_valid: list[str] = Field(default_factory=list)
    citations_invalid: list[str] = Field(default_factory=list)
    retrieved_citations: list[str] = Field(default_factory=list)
    inspected_files: list[str] = Field(default_factory=list)
    injection_risk: str = "none"
    injection_flags: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    policy_warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    explore_stopped: str = ""
    tool_log: list[ToolCallRecord] = Field(default_factory=list)
    llm_calls: int = 0
    total_tokens: int = 0
    seconds: float = 0.0