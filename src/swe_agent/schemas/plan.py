"""Structured output of the planning step."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class PlanAction(StrEnum):
    MODIFY = "modify"
    CREATE = "create"
    DELETE = "delete"


class PlanStep(BaseModel):
    step_number: int = Field(ge=1)
    file_path: str = Field(description="Relative POSIX path, e.g. src/auth/login.py")
    action: PlanAction
    description: str = Field(description="What exactly will change in this file")
    rationale: str = Field(description="Why this change is needed")

    @field_validator("file_path")
    @classmethod
    def _must_be_safe_relative_path(cls, value: str) -> str:
        """Early safety net. The full path guard comes in Phase 7."""
        path = value.strip().replace("\\", "/")
        parts = path.split("/")
        if not path:
            raise ValueError("file_path is empty")
        if path.startswith("/") or _WINDOWS_DRIVE.match(path):
            raise ValueError("file_path must be relative, not absolute")
        if ".." in parts:
            raise ValueError("file_path must not contain '..'")
        return path


class ImplementationPlan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(min_length=1, max_length=15)
    test_strategy: str = Field(description="How we will prove the fix works")
    risks: list[str] = Field(default_factory=list)