"""Typed data for the workflow: the human's decision and the run status."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PlanDecision(BaseModel):
    """The human's answer to 'do you approve this plan?'"""

    action: Literal["approve", "reject", "revise"]
    feedback: str = Field(default="", max_length=2000)
    plan_hash: str = Field(default="", max_length=64)  # hash of the plan that was shown

    @model_validator(mode="after")
    def _revise_needs_feedback(self) -> "PlanDecision":
        if self.action == "revise" and not self.feedback.strip():
            raise ValueError("Asking for a revision needs feedback text.")
        return self


class RunStatus(BaseModel):
    """Everything a UI needs to know about one run."""

    run_id: str
    status: str
    error: str = ""
    paused: bool = False  # True = waiting for a human
    waiting_for: str = ""  # name of the node that is waiting
    approval: dict[str, Any] = Field(default_factory=dict)  # what the human must review
    approved_files: list[str] = Field(default_factory=list)
    plan_revisions: int = 0
    planning: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)