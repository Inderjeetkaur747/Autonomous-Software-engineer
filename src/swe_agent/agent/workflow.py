"""The LangGraph workflow.

prepare -> index -> plan -> approve_plan (HUMAN) -> execute (placeholder until Phase 12)
"""

import hashlib
import json
import logging
import operator
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from swe_agent.config import Settings
from swe_agent.github.urls import resolve_issue_ref
from swe_agent.github.workspace import WorkspaceError, WorkspaceManager
from swe_agent.logging_setup import redact_secrets
from swe_agent.schemas.planning import PlanningResult
from swe_agent.schemas.workflow import PlanDecision, RunStatus

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"[a-f0-9]{12}")
MAX_FEEDBACK_CHARS = 4000


class WorkflowError(Exception):
    """A workflow request cannot be done (unknown run, wrong moment, stale plan...)."""


class Status:
    RUNNING = "running"
    PREPARED = "prepared"
    INDEXED = "indexed"
    PLAN_READY = "plan_ready"  # waiting for the human
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    REVISING = "revising"
    REVISION_LIMIT = "revision_limit"
    REJECTED = "rejected"
    PLAN_APPROVED = "plan_approved"
    FAILED = "failed"


class AgentState(TypedDict, total=False):
    """Plain JSON values only. `events` is appended to, not replaced (reducer)."""

    run_id: str
    repo: str
    issue_number: int
    branch: str
    head_sha: str
    status: str
    error: str
    planning: dict[str, Any]
    plan_revisions: int
    feedback: str
    decision: str  # approve | revise | reject | stop
    approved_files: list[str]
    events: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentDeps:
    """The real work, injected. Tests pass fakes."""

    settings: Settings
    prepare: Callable[[str, int, str], dict[str, Any]]  # (repo, issue number, run_id)
    index: Callable[[str], dict[str, Any]]  # (run_id)
    plan: Callable[[str, str], PlanningResult]  # (run_id, reviewer feedback)


# ------------------------------------------------------------------ helpers
def _event(node: str, detail: str) -> dict[str, Any]:
    return {
        "node": node,
        "detail": detail[:300],
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _failed(node: str, exc: Exception) -> dict[str, Any]:
    message = redact_secrets(f"{type(exc).__name__}: {exc}")[:500]
    logger.warning("node %s failed: %s", node, message)
    return {
        "status": Status.FAILED,
        "error": f"{node}: {message}",
        "events": [_event(node, "FAILED")],
    }


def plan_hash(plan: dict[str, Any] | None) -> str:
    """A short fingerprint of a plan. Same plan, same hash."""
    if not plan:
        return ""
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def plan_files(plan: dict[str, Any] | None) -> list[str]:
    """The files an approval would allow. Computed from the plan, never from human input."""
    if not plan:
        return []
    return sorted({str(step["file_path"]) for step in plan.get("steps", [])})


_PLANNING_TO_STATUS = {
    "ready": Status.PLAN_READY,
    "needs_clarification": Status.NEEDS_CLARIFICATION,
    "blocked": Status.BLOCKED,
}


# ------------------------------------------------------------------- routing
def route_after_step(state: AgentState) -> str:
    return "end" if state.get("status") == Status.FAILED else "next"


def route_after_plan(state: AgentState) -> str:
    return "approve" if state.get("status") == Status.PLAN_READY else "end"


def route_after_decision(state: AgentState) -> str:
    return {"approve": "execute", "revise": "plan"}.get(state.get("decision", ""), "end")


# --------------------------------------------------------------------- graph
def build_graph(deps: AgentDeps, checkpointer: Any):
    settings = deps.settings

    def prepare_node(state: AgentState) -> dict[str, Any]:
        try:
            info = deps.prepare(state["repo"], state["issue_number"], state["run_id"])
        except Exception as exc:  # noqa: BLE001 (any failure becomes a clean 'failed' status)
            return _failed("prepare", exc)
        note = "reused existing workspace" if info.get("reused") else "cloned repository"
        return {
            "status": Status.PREPARED,
            "branch": info.get("branch", ""),
            "head_sha": info.get("head_sha", ""),
            "events": [_event("prepare", note)],
        }

    def index_node(state: AgentState) -> dict[str, Any]:
        try:
            info = deps.index(state["run_id"])
        except Exception as exc:  # noqa: BLE001
            return _failed("index", exc)
        note = "already indexed" if info.get("skipped") else "indexed"
        return {
            "status": Status.INDEXED,
            "events": [_event("index", f"{note}, {info.get('chunks', '?')} chunks")],
        }

    def plan_node(state: AgentState) -> dict[str, Any]:
        try:
            result = deps.plan(state["run_id"], state.get("feedback", ""))
        except Exception as exc:  # noqa: BLE001
            return _failed("plan", exc)
        steps = len(result.plan.steps) if result.plan else 0
        return {
            "status": _PLANNING_TO_STATUS[result.status.value],
            "planning": result.model_dump(mode="json"),
            "decision": "",
            "events": [_event("plan", f"{result.status.value}, {steps} step(s)")],
        }

    def approve_plan_node(state: AgentState) -> dict[str, Any]:
        # RULE: no side effects before interrupt(). On resume this node runs again from here.
        planning = state.get("planning") or {}
        plan = planning.get("plan")
        if not plan:
            return {"decision": "stop", **_failed("approve_plan", ValueError("no plan"))}
        revisions = state.get("plan_revisions", 0)
        request = {
            "type": "plan_approval",
            "run_id": state["run_id"],
            "plan_hash": plan_hash(plan),
            "analysis": planning.get("analysis"),
            "plan": plan,
            "files_in_scope": plan_files(plan),
            "injection_risk": planning.get("injection_risk", "none"),
            "injection_flags": planning.get("injection_flags", []),
            "citations_invalid": planning.get("citations_invalid", []),
            "policy_warnings": planning.get("policy_warnings", []),
            "notes": planning.get("notes", []),
            "revision": revisions,
            "revisions_left": max(0, settings.max_plan_revisions - revisions),
        }

        answer = interrupt(request)  # the graph stops here until a human answers

        try:
            decision = PlanDecision.model_validate(answer)
        except ValidationError:
            return {
                "decision": "reject",
                "status": Status.REJECTED,
                "events": [_event("approve_plan", "invalid answer, treated as reject")],
            }
        if decision.plan_hash and decision.plan_hash != request["plan_hash"]:
            return {
                "decision": "reject",
                "status": Status.REJECTED,
                "events": [_event("approve_plan", "answer was for a different plan: rejected")],
            }
        if decision.action == "approve":
            return {
                "decision": "approve",
                "status": Status.PLAN_APPROVED,
                "approved_files": request["files_in_scope"],
                "events": [_event("approve_plan", "plan approved by the human")],
            }
        if decision.action == "revise":
            if revisions >= settings.max_plan_revisions:
                return {
                    "decision": "stop",
                    "status": Status.REVISION_LIMIT,
                    "events": [_event("approve_plan", "revision limit reached")],
                }
            combined = f"{state.get('feedback', '')}\n{decision.feedback}".strip()
            return {
                "decision": "revise",
                "status": Status.REVISING,
                "feedback": combined[-MAX_FEEDBACK_CHARS:],
                "plan_revisions": revisions + 1,
                "events": [_event("approve_plan", f"revision {revisions + 1} requested")],
            }
        return {
            "decision": "reject",
            "status": Status.REJECTED,
            "events": [_event("approve_plan", "plan rejected by the human")],
        }

    def execute_node(state: AgentState) -> dict[str, Any]:
        # PLACEHOLDER. Phases 11 to 13 add: Docker sandbox, editing, tests, repair loop, diff.
        files = ", ".join(state.get("approved_files", []))
        return {"events": [_event("execute", f"not implemented yet; will be allowed to edit: {files}")]}  # noqa: E501

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("index", index_node)
    graph.add_node("plan", plan_node)
    graph.add_node("approve_plan", approve_plan_node)
    graph.add_node("execute", execute_node)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", route_after_step, {"next": "index", "end": END})
    graph.add_conditional_edges("index", route_after_step, {"next": "plan", "end": END})
    graph.add_conditional_edges("plan", route_after_plan, {"approve": "approve_plan", "end": END})
    graph.add_conditional_edges(
        "approve_plan",
        route_after_decision,
        {"execute": "execute", "plan": "plan", "end": END},
    )
    graph.add_edge("execute", END)
    return graph.compile(checkpointer=checkpointer)


# ------------------------------------------------------------- checkpointing
def open_checkpointer(settings: Settings) -> SqliteSaver:
    """A SQLite file that stores the state after every node. Survives restarts."""
    path = Path(settings.checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def close_checkpointer(saver: SqliteSaver) -> None:
    saver.conn.close()


# -------------------------------------------------------------------- runner
class AgentRunner:
    """start / status / decide. The CLI, the API and the dashboard all use this."""

    def __init__(self, deps: AgentDeps, checkpointer: Any) -> None:
        self._deps = deps
        self._graph = build_graph(deps, checkpointer)

    def _config(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise WorkflowError(f"Invalid run id: {run_id!r}")
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 40}

    def mermaid(self) -> str:
        """The graph as Mermaid text (paste it into docs)."""
        return self._graph.get_graph().draw_mermaid()

    def start(self, repo: str, issue_number: int, run_id: str | None = None) -> RunStatus:
        """Run until the graph ends or pauses for the human."""
        run_id = run_id or uuid.uuid4().hex[:12]
        config = self._config(run_id)
        if self._graph.get_state(config).values:
            raise WorkflowError(f"Run {run_id} already exists. Use: status or decide.")
        ref = resolve_issue_ref(str(issue_number), repo)  # rejects unsafe or invalid input
        self._graph.invoke(
            {
                "run_id": run_id,
                "repo": ref.repo.full_name,
                "issue_number": ref.number,
                "status": Status.RUNNING,
                "plan_revisions": 0,
                "feedback": "",
                "events": [],
            },
            config,
        )
        return self.status(run_id)

    def status(self, run_id: str) -> RunStatus:
        snapshot = self._graph.get_state(self._config(run_id))
        values = dict(snapshot.values or {})
        if not values:
            raise WorkflowError(f"No run with id {run_id}.")
        waiting, approval = "", {}
        for task in snapshot.tasks:
            if task.interrupts:
                waiting = task.name
                payload = task.interrupts[0].value
                approval = payload if isinstance(payload, dict) else {}
                break
        return RunStatus(
            run_id=run_id,
            status=values.get("status", Status.RUNNING),
            error=values.get("error", ""),
            paused=bool(waiting),
            waiting_for=waiting,
            approval=approval,
            approved_files=values.get("approved_files", []),
            plan_revisions=values.get("plan_revisions", 0),
            planning=values.get("planning", {}),
            events=values.get("events", []),
        )

    def decide(self, run_id: str, decision: PlanDecision) -> RunStatus:
        """Give the human's answer to the waiting graph, then continue."""
        current = self.status(run_id)
        if current.waiting_for != "approve_plan":
            raise WorkflowError(f"Run {run_id} is not waiting for a plan decision.")
        shown = str(current.approval.get("plan_hash", ""))
        if decision.plan_hash and decision.plan_hash != shown:
            raise WorkflowError("That decision is for a different plan. Look at the current one.")
        decision = decision.model_copy(update={"plan_hash": shown})
        self._audit(
            run_id,
            {
                "event": "plan_decision",
                "action": decision.action,
                "plan_hash": shown,
                "feedback_chars": len(decision.feedback),  # the text itself is not copied
            },
        )
        self._graph.invoke(Command(resume=decision.model_dump()), self._config(run_id))
        return self.status(run_id)

    def _audit(self, run_id: str, record: dict[str, Any]) -> None:
        try:
            workspace = WorkspaceManager(self._deps.settings).get(run_id)
        except WorkspaceError:
            return
        record = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **record}
        with open(workspace.root / "audit.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")