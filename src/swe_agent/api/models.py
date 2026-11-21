"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from swe_agent.schemas.workflow import RunStatus


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown field is an error

    repo: str = Field(min_length=3, max_length=140, description="owner/repo or a github.com URL")
    issue_number: int = Field(ge=1, le=999_999_999)
    run_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")


class JobView(BaseModel):
    """What the background worker is doing for one run."""

    state: Literal["idle", "queued", "running", "done", "error"] = "idle"
    action: str = ""
    error: str = ""
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""


class RunView(RunStatus):
    """The workflow status plus the job state. This is what the dashboard shows."""

    job: JobView = Field(default_factory=JobView)
    busy: bool = False  # True = a job is queued or running: do not send decisions
    interrupted: bool = False  # True = the run stopped midway (for example a server restart)


class RunSummary(BaseModel):
    run_id: str
    repo: str = ""
    issue_number: int | None = None
    issue_title: str = ""
    status: str = "not_started"
    paused: bool = False
    busy: bool = False
    updated_at: str = ""


class RunMetrics(BaseModel):
    """A small summary for the dashboard. Phase 16 adds proper tracing."""

    run_id: str
    status: str
    llm_calls: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    tool_errors: int = 0
    repairs_used: int = 0
    tests_passed: bool | None = None
    plan_revisions: int = 0
    events: int = 0