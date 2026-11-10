"""Typed inputs and outputs for the agent's tools."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolResult(BaseModel):
    """What every tool returns. ok=False means the tool refused or failed."""

    ok: bool
    output: str = ""
    error: str = ""
    warnings: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """One line of the tool log (shown in the dashboard, used for metrics)."""

    seq: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error: str = ""
    warnings: int = 0
    duration_ms: float = 0.0
    output_chars: int = 0
    timestamp: str = ""


class ToolSpec(BaseModel):
    """A tool description for the LLM and for MCP."""

    name: str
    description: str
    input_schema: dict[str, Any]
    explore: bool = False  # True = does not change files and does not run code

    def to_openai(self) -> dict[str, Any]:
        """The format Groq (OpenAI-compatible) uses for function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


# ---------------------------------------------------------------- arguments
class _Args(BaseModel):
    """extra='forbid': an unknown argument is an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class ListFilesArgs(_Args):
    prefix: str = Field(default="", description="Only list files under this folder, e.g. 'src'")
    max_results: int = Field(default=100, ge=1, le=500)


class ReadFileArgs(_Args):
    path: str = Field(description="Relative path, e.g. 'mathkit/stats.py'")
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=0, ge=0, description="0 means: read up to the page limit")


class SearchCodeArgs(_Args):
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=15)
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"


class GrepArgs(_Args):
    pattern: str = Field(min_length=1, max_length=200, description="Plain text, not a regex")
    path_prefix: str = ""
    ignore_case: bool = True
    max_results: int = Field(default=30, ge=1, le=200)


class GitDiffArgs(_Args):
    stat_only: bool = False
    path: str = Field(default="", description="Only this file or folder (optional)")


class EditFileArgs(_Args):
    path: str
    old_str: str = Field(min_length=1, max_length=20000)
    new_str: str = Field(max_length=20000)


class CreateFileArgs(_Args):
    path: str
    content: str = Field(min_length=1, max_length=100000)


class RunTestsArgs(_Args):
    command: str = Field(default="pytest -q", max_length=300)
    timeout_seconds: int = Field(default=0, ge=0, le=900, description="0 means: use the default")