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
from swe_agent.schemas.execution import ExecutionResult
from swe_agent.schemas.patch import PatchResult
from swe_agent.schemas.planning import PlanningResult
from swe_agent.schemas.workflow import PatchDecision, PlanDecision, RunStatus

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
    PATCH_READY = "patch_ready"  # waiting for the human
    PATCH_APPROVED = "patch_approved"
    PATCH_REJECTED = "patch_rejected"
    PATCH_BLOCKED = "patch_blocked"
    NO_PATCH = "no_patch"


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
    execution: dict[str, Any]
    patch: dict[str, Any]


@dataclass
class AgentDeps:
    """The real work, injected. Tests pass fakes."""

    settings: Settings
    prepare: Callable[[str, int, str], dict[str, Any]]  # (repo, issue number, run_id)
    index: Callable[[str], dict[str, Any]]  # (run_id)
    plan: Callable[[str, str], PlanningResult]  # (run_id, reviewer feedback)
    execute: Callable[[str, list[str], dict[str, Any]], ExecutionResult] | None = None
    make_patch: Callable[[str, list[str], dict[str, Any]], PatchResult] | None = None
    accept_patch: Callable[[str, str, list[str]], dict[str, Any]] | None = None
    discard_patch: Callable[[str], None] | None = None


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

def route_after_patch(state: AgentState) -> str:
    return "review" if state.get("status") == Status.PATCH_READY else "end"


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
        files = state.get("approved_files", [])
        if deps.execute is None:  # no executor configured (tests of the flow only)
            detail = f"not configured; would be allowed to edit: {', '.join(files)}"
            return {"events": [_event("execute", detail)]}
        plan = (state.get("planning") or {}).get("plan") or {}
        try:
            result = deps.execute(state["run_id"], files, plan)
        except Exception as exc:  # noqa: BLE001
            return _failed("execute", exc)
        detail = (
            f"{result.status.value}: {len(result.changed_files)} file(s) changed, "
            f"{len(result.attempts)} test run(s), {result.repairs_used} repair(s)"
        )
        return {
            "status": result.status.value,
            "execution": result.model_dump(mode="json"),
            "events": [_event("execute", detail)],
        }

    def route_after_execute(state: AgentState) -> str:
        if deps.make_patch is None or state.get("status") == Status.FAILED:
            return "end"
        return "patch"

    def make_patch_node(state: AgentState) -> dict[str, Any]:
        make = deps.make_patch
        if make is None:
            return _failed("make_patch", RuntimeError("no patch builder configured"))
        try:
            result = make(
                state["run_id"], state.get("approved_files", []), state.get("execution") or {}
            )
        except Exception as exc:  # noqa: BLE001
            return _failed("make_patch", exc)
        status = {
            "ready": Status.PATCH_READY,
            "no_changes": Status.NO_PATCH,
            "blocked": Status.PATCH_BLOCKED,
        }[result.status]
        detail = (
            f"{result.status}: {len(result.files)} file(s), "
            f"+{result.additions} -{result.deletions}"
        )
        return {
            "status": status,
            "patch": result.model_dump(mode="json"),
            "events": [_event("make_patch", detail)],
        }

    def review_patch_node(state: AgentState) -> dict[str, Any]:
        # RULE: no side effects before interrupt(). On resume this node runs again from here.
        patch = state.get("patch") or {}
        request = {
            "type": "patch_approval",
            "run_id": state["run_id"],
            "patch_hash": patch.get("patch_hash", ""),
            "files": patch.get("files", []),
            "additions": patch.get("additions", 0),
            "deletions": patch.get("deletions", 0),
            "size_bytes": patch.get("size_bytes", 0),
            "diff_preview": patch.get("diff_preview", ""),
            "truncated": patch.get("truncated", False),
            "patch_path": patch.get("patch_path", ""),
            "applies_to_base": patch.get("applies_to_base"),
            "warnings": patch.get("warnings", []),
            "weakened_tests": patch.get("weakened_tests", []),
            "tests_passed": patch.get("tests_passed", False),
            "execution_status": patch.get("execution_status", ""),
            "fixed_tests": patch.get("fixed_tests", []),
            "new_failures": patch.get("new_failures", []),
            "requires_acknowledgement": patch.get("requires_acknowledgement", False),
            "ack_reasons": patch.get("ack_reasons", []),
            "base_commit": patch.get("base_commit", ""),
            "branch": patch.get("branch", ""),
        }

        answer = interrupt(request)  # the graph stops here until a human answers

        kept = {"status": Status.PATCH_REJECTED}  # NOT approved, and nothing is deleted
        try:
            decision = PatchDecision.model_validate(answer)
        except ValidationError:
            return {
                **kept,
                "events": [
                    _event("review_patch", "invalid answer: not approved, changes kept")
                ],
            }
        if decision.patch_hash and decision.patch_hash != request["patch_hash"]:
            return {
                **kept,
                "events": [
                    _event(
                        "review_patch",
                        "answer was for another patch: not approved, changes kept",
                    )
                ],
            }

        if decision.action == "reject":
            note = "patch rejected; workspace changes discarded"
            if deps.discard_patch is not None:
                try:
                    deps.discard_patch(state["run_id"])
                except Exception as exc:  # noqa: BLE001
                    note = "patch rejected; discarding FAILED: " + redact_secrets(str(exc))[:200]
            return {**kept, "events": [_event("review_patch", note)]}

        if request["requires_acknowledgement"] and not decision.accept_risk:
            return {
                **kept,
                "events": [
                    _event("review_patch", "risk not acknowledged: not approved, changes kept")
                ],
            }
        if deps.accept_patch is None:
            return _failed("review_patch", RuntimeError("no patch saver configured"))
        try:
            info = deps.accept_patch(
                state["run_id"], request["patch_hash"], state.get("approved_files", [])
            )
        except Exception as exc:  # noqa: BLE001
            return _failed("review_patch", exc)
        return {
            "status": Status.PATCH_APPROVED,
            "patch": {**patch, "approved_path": info.get("path", "")},
            "events": [
                _event(
                    "review_patch",
                    f"patch approved and saved ({len(request['files'])} file(s))",
                )
            ],
        }

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_node)
    graph.add_node("index", index_node)
    graph.add_node("plan", plan_node)
    graph.add_node("approve_plan", approve_plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("make_patch", make_patch_node)
    graph.add_node("review_patch", review_patch_node)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", route_after_step, {"next": "index", "end": END})
    graph.add_conditional_edges("index", route_after_step, {"next": "plan", "end": END})
    graph.add_conditional_edges("plan", route_after_plan, {"approve": "approve_plan", "end": END})
    graph.add_conditional_edges(
        "approve_plan",
        route_after_decision,
        {"execute": "execute", "plan": "plan", "end": END},
    )
    graph.add_conditional_edges(
        "execute", route_after_execute, {"patch": "make_patch", "end": END}
    )
    graph.add_conditional_edges(
        "make_patch", route_after_patch, {"review": "review_patch", "end": END}
    )
    graph.add_edge("review_patch", END)
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
            execution=values.get("execution", {}),
            patch=values.get("patch", {}),
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

    def decide_patch(self, run_id: str, decision: PatchDecision) -> RunStatus:
        """Give the human's answer about the final patch, then continue."""
        current = self.status(run_id)
        if current.waiting_for != "review_patch":
            raise WorkflowError(f"Run {run_id} is not waiting for a patch decision.")
        shown = str(current.approval.get("patch_hash", ""))
        if decision.patch_hash and decision.patch_hash != shown:
            raise WorkflowError("That decision is for a different patch. Look at the current one.")
        needs_ack = current.approval.get("requires_acknowledgement")
        if decision.action == "approve" and needs_ack and not decision.accept_risk:
            reasons = " ".join(current.approval.get("ack_reasons", []))
            raise WorkflowError(f"This patch needs --accept-risk. Reasons: {reasons}")
        decision = decision.model_copy(update={"patch_hash": shown})
        self._audit(
            run_id,
            {
                "event": "patch_decision",
                "action": decision.action,
                "patch_hash": shown,
                "accept_risk": decision.accept_risk,
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