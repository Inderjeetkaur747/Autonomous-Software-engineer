"""Small building blocks shared by many schemas."""

from pydantic import BaseModel, Field, model_validator


class CodeCitation(BaseModel):
    """Points to an exact place in the repository. Used in retrieval (Phase 6)."""

    file_path: str = Field(description="Relative path inside the repository")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_line_order(self) -> "CodeCitation":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self

    @property
    def label(self) -> str:
        """Human readable form, e.g. src/auth/login.py:10-22"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"