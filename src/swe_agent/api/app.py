"""FastAPI backend: a thin HTTP layer over AgentRunner. Long work runs as background jobs."""

import json
import logging
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from swe_agent import __version__
from swe_agent.agent import (
    AgentRunner,
    WorkflowError,
    close_checkpointer,
    make_default_deps,
    open_checkpointer,
)
from swe_agent.api.jobs import JobBusy, JobManager
from swe_agent.api.models import (
    RunMetrics,
    RunSummary,
    RunView,
    StartRequest,
)
from swe_agent.config import Settings, get_settings
from swe_agent.github import InvalidGitHubUrl, WorkspaceError, WorkspaceManager, resolve_issue_ref
from swe_agent.schemas.workflow import PatchDecision, PlanDecision

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MIN_EXPOSED_KEY_CHARS = 16
_RUN_ID_RE = re.compile(r"[a-f0-9]{12}")
# A run in one of these states, with no job and no pause, was cut off midway.
_IN_PROGRESS = {"running", "prepared", "indexed", "revising", "plan_approved"}

RunId = Annotated[str, PathParam(pattern=r"^[a-f0-9]{12}$", description="12 lowercase hex chars")]


def ensure_safe_bind(host: str, settings: Settings) -> None:
    """Never listen on the network without a real API key."""
    if host in LOOPBACK_HOSTS:
        return
    if len(settings.api_key.get_secret_value()) < MIN_EXPOSED_KEY_CHARS:
        raise ValueError(
            f"Refusing to listen on {host}: set API_KEY (at least {MIN_EXPOSED_KEY_CHARS} "
            "characters) in .env first. Anyone who can reach this API can approve patches."
        )


# ---------------------------------------------------------------- dependencies
def require_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.settings.api_key.get_secret_value()
    if not expected:
        return
    given = (x_api_key or "").encode("utf-8")
    if not secrets.compare_digest(given, expected.encode("utf-8")):  # constant-time compare
        raise HTTPException(
            401, "Missing or wrong API key.", headers={"WWW-Authenticate": "ApiKey"}
        )


def _get_runner(request: Request) -> AgentRunner:
    return request.app.state.runner


def _get_jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


Runner = Annotated[AgentRunner, Depends(_get_runner)]
Jobs = Annotated[JobManager, Depends(_get_jobs)]
Cfg = Annotated[Settings, Depends(_get_settings)]

router = APIRouter(dependencies=[Depends(require_key)])


# --------------------------------------------------------------------- helpers
def _load_view(runner: AgentRunner, jobs: JobManager, run_id: str) -> RunView:
    """Combine the durable workflow state with the in-memory job state."""
    job = jobs.view(run_id)
    busy = job.state in ("queued", "running")
    try:
        status = runner.status(run_id)
    except WorkflowError as exc:
        if busy:  # the run has no checkpoint yet
            return RunView(run_id=run_id, status=job.state, job=job, busy=True)
        if job.state == "error":
            return RunView(run_id=run_id, status="failed", error=job.error, job=job)
        raise HTTPException(404, "No run with that id.") from exc

    data = status.model_dump()
    if busy:  # the agent is working: nobody may answer a question right now
        data.update(paused=False, waiting_for="", approval={})
    interrupted = (not status.paused) and not busy and status.status in _IN_PROGRESS
    return RunView(**data, job=job, busy=busy, interrupted=interrupted)


def _run_exists(runner: AgentRunner, run_id: str) -> bool:
    try:
        runner.status(run_id)
    except WorkflowError:
        return False
    return True


def _require_idle(view: RunView) -> None:
    if view.busy:
        raise HTTPException(409, "This run is busy. Poll GET /runs/{id} until busy is false.")


def _submit(jobs: JobManager, run_id: str, action: str, work: Any) -> None:
    try:
        jobs.submit(run_id, action, work)
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc


def _metrics(view: RunView) -> RunMetrics:
    planning, execution = view.planning or {}, view.execution or {}
    calls = [*(planning.get("tool_log") or []), *(execution.get("tool_log") or [])]
    final = execution.get("final") or {}
    return RunMetrics(
        run_id=view.run_id,
        status=view.status,
        llm_calls=int(planning.get("llm_calls", 0)) + int(execution.get("llm_calls", 0)),
        total_tokens=int(planning.get("total_tokens", 0)) + int(execution.get("total_tokens", 0)),
        seconds=round(float(planning.get("seconds", 0)) + float(execution.get("seconds", 0)), 2),
        tool_calls=len(calls),
        tool_errors=sum(1 for call in calls if not call.get("ok", True)),
        repairs_used=int(execution.get("repairs_used", 0)),
        tests_passed=final.get("passed") if final else None,
        plan_revisions=view.plan_revisions,
        events=len(view.events),
    )


# ------------------------------------------------------------------- endpoints
@router.post("/runs", status_code=202, response_model=RunView)
def start_run(body: StartRequest, runner: Runner, jobs: Jobs) -> RunView:
    """Start a run (clone, index, plan). Returns at once. Poll GET /runs/{id}."""
    try:
        ref = resolve_issue_ref(str(body.issue_number), body.repo)
    except InvalidGitHubUrl as exc:
        raise HTTPException(422, str(exc)) from exc
    run_id = body.run_id or uuid.uuid4().hex[:12]
    if jobs.busy(run_id) or _run_exists(runner, run_id):
        raise HTTPException(409, f"Run {run_id} already exists.")
    _submit(jobs, run_id, "start", lambda: runner.start(ref.repo.full_name, ref.number, run_id))
    return _load_view(runner, jobs, run_id)


@router.get("/runs", response_model=list[RunSummary])
def list_runs(
    runner: Runner, jobs: Jobs, settings: Cfg, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[RunSummary]:
    """The newest runs that have a workspace."""
    base = WorkspaceManager(settings).base
    found: list[tuple[float, str]] = []
    for child in base.iterdir():
        meta = child / "run.json"
        if _RUN_ID_RE.fullmatch(child.name) and child.is_dir() and meta.is_file():
            found.append((meta.stat().st_mtime, child.name))
    found.sort(reverse=True)

    rows: list[RunSummary] = []
    for mtime, run_id in found[:limit]:
        try:
            data = json.loads((base / run_id / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        issue = data.get("issue") if isinstance(data.get("issue"), dict) else {}
        summary = RunSummary(
            run_id=run_id,
            repo=str(data.get("repo", "")),
            issue_number=issue.get("number"),
            issue_title=str(issue.get("title", ""))[:200],
            busy=jobs.busy(run_id),
            updated_at=datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds"),
        )
        try:
            status = runner.status(run_id)
            summary.status, summary.paused = status.status, status.paused
        except WorkflowError:
            pass  # a workspace made by the CLI 'prepare' command only
        rows.append(summary)
    return rows


@router.get("/runs/{run_id}", response_model=RunView)
def get_run(run_id: RunId, runner: Runner, jobs: Jobs) -> RunView:
    return _load_view(runner, jobs, run_id)


@router.post("/runs/{run_id}/plan-decision", status_code=202, response_model=RunView)
def plan_decision(run_id: RunId, decision: PlanDecision, runner: Runner, jobs: Jobs) -> RunView:
    """Answer 'do you approve this plan?'. approve | reject | revise (needs feedback)."""
    view = _load_view(runner, jobs, run_id)
    _require_idle(view)
    if view.waiting_for != "approve_plan":
        raise HTTPException(409, "This run is not waiting for a plan decision.")
    if decision.plan_hash and decision.plan_hash != view.approval.get("plan_hash"):
        raise HTTPException(409, "That decision is for a different plan. Reload the run.")
    _submit(jobs, run_id, "plan_decision", lambda: runner.decide(run_id, decision))
    return _load_view(runner, jobs, run_id)


@router.post("/runs/{run_id}/patch-decision", status_code=202, response_model=RunView)
def patch_decision(run_id: RunId, decision: PatchDecision, runner: Runner, jobs: Jobs) -> RunView:
    """Answer 'do you accept this patch?'. Risky patches need accept_risk = true."""
    view = _load_view(runner, jobs, run_id)
    _require_idle(view)
    if view.waiting_for != "review_patch":
        raise HTTPException(409, "This run is not waiting for a patch decision.")
    if decision.patch_hash and decision.patch_hash != view.approval.get("patch_hash"):
        raise HTTPException(409, "That decision is for a different patch. Reload the run.")
    risky = view.approval.get("requires_acknowledgement")
    if decision.action == "approve" and risky and not decision.accept_risk:
        reasons = " ".join(view.approval.get("ack_reasons", []))
        raise HTTPException(422, f"This patch needs accept_risk = true. Reasons: {reasons}")
    _submit(jobs, run_id, "patch_decision", lambda: runner.decide_patch(run_id, decision))
    return _load_view(runner, jobs, run_id)


@router.get("/runs/{run_id}/events")
def get_events(
    run_id: RunId, runner: Runner, jobs: Jobs, after: Annotated[int, Query(ge=0)] = 0
) -> dict[str, Any]:
    """Workflow events. Use ?after=N to get only the new ones."""
    view = _load_view(runner, jobs, run_id)
    return {"total": len(view.events), "events": view.events[after:], "busy": view.busy}


@router.get("/runs/{run_id}/tools")
def get_tool_log(run_id: RunId, runner: Runner, jobs: Jobs) -> dict[str, Any]:
    """Every tool call, from the planning and the execution stage."""
    view = _load_view(runner, jobs, run_id)
    stages = (("planning", view.planning), ("execution", view.execution))
    rows = [
        {"stage": stage, **record}
        for stage, data in stages
        for record in (data.get("tool_log") or [])
    ]
    return {"count": len(rows), "calls": rows}


@router.get("/runs/{run_id}/metrics", response_model=RunMetrics)
def get_metrics(run_id: RunId, runner: Runner, jobs: Jobs) -> RunMetrics:
    return _metrics(_load_view(runner, jobs, run_id))


@router.get("/runs/{run_id}/patch", response_class=PlainTextResponse)
def get_patch(run_id: RunId, settings: Cfg) -> PlainTextResponse:
    """The exact patch text: approved.patch, else the candidate, else rejected.patch."""
    try:
        root = WorkspaceManager(settings).get(run_id).root
    except WorkspaceError as exc:
        raise HTTPException(404, "No run with that id.") from exc
    for kind, name in (
        ("approved", "approved.patch"),
        ("candidate", "patch.diff"),
        ("rejected", "rejected.patch"),
    ):
        path = root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            return PlainTextResponse(text, headers={"X-Patch-Kind": kind})
    raise HTTPException(404, "There is no patch yet.")


# ------------------------------------------------------------------------ app
def create_app(settings: Settings | None = None, runner: AgentRunner | None = None) -> FastAPI:
    """Build the app. Tests pass a runner made from fake deps."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        saver = None
        active = runner
        if active is None:
            saver = open_checkpointer(settings)  # ONE shared connection for the whole server
            active = AgentRunner(make_default_deps(settings), saver)
        app.state.settings = settings
        app.state.runner = active
        app.state.jobs = JobManager(settings.api_workers)
        logger.info(
            "api ready workers=%d auth=%s",
            settings.api_workers,
            bool(settings.api_key.get_secret_value()),
        )
        try:
            yield
        finally:
            app.state.jobs.shutdown()
            if saver is not None:
                close_checkpointer(saver)

    app = FastAPI(title="Autonomous AI Software Engineer", version=__version__, lifespan=lifespan)

    origins = [
        o.strip()
        for o in settings.api_cors_origins.split(",")
        if o.strip() not in ("", "*")
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # never "*"
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "auth_required": bool(settings.api_key.get_secret_value()),
        }

    app.include_router(router)
    return app