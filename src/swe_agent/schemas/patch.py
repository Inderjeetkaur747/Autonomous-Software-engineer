"""Typed data for the patch that a human reviews."""

from typing import Literal

from pydantic import BaseModel, Field


class PatchFile(BaseModel):
    path: str
    status: Literal["added", "modified", "deleted"]
    additions: int = 0
    deletions: int = 0


class PatchResult(BaseModel):
    status: Literal["ready", "no_changes", "blocked"]
    patch_hash: str = ""
    base_commit: str = ""
    branch: str = ""
    files: list[PatchFile] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    size_bytes: int = 0
    diff_preview: str = ""  # the full patch is in the file; this is capped for the UI
    truncated: bool = False
    patch_path: str = ""
    applies_to_base: bool | None = None  # git apply --cached --check against the base commit
    violations: list[str] = Field(default_factory=list)  # block the patch
    warnings: list[str] = Field(default_factory=list)  # shown to the human
    weakened_tests: list[str] = Field(default_factory=list)
    tests_passed: bool = False
    execution_status: str = ""
    fixed_tests: list[str] = Field(default_factory=list)
    new_failures: list[str] = Field(default_factory=list)
    requires_acknowledgement: bool = False  # approving needs --accept-risk
    ack_reasons: list[str] = Field(default_factory=list)
    approved_path: str = ""  # set after approval