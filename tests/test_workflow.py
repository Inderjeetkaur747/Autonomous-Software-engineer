import json

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from swe_agent.agent import (
    AgentDeps,
    AgentRunner,
    WorkflowError,
    close_checkpointer,
    open_checkpointer,
    plan_hash,
)
from swe_agent.config import Settings
from swe_agent.schemas.analysis import IssueAnalysis
from swe_agent.schemas.plan import ImplementationPlan, PlanStep
from swe_agent.schemas.planning import PlanningResult
from swe_agent.schemas.workflow import PlanDecision

RUN = "a" * 12
FILES = ("mathkit/stats.py", "tests/test_stats.py")


def planning(status="ready", paths=FILES):
    analysis = IssueAnalysis(
        summary="s", issue_type="bug", root_cause_hypothesis="r",
        affected_areas=list(paths), acceptance_criteria=["c"], confidence=0.9,
    )
    plan = None
    if status != "needs_clarification":
        steps = [
            PlanStep(step_number=i, file_path=p, action="modify", description="d", rationale="r")
            for i, p in enumerate(paths, start=1)
        ]
        plan = ImplementationPlan(goal="g", steps=steps, test_strategy="t")
    return PlanningResult(
        status=status, analysis=analysis, plan=plan,
        policy_violations=["a blocked step"] if status == "blocked" else [],
    )


class Env:
    """A runner built from fake deps and a real SQLite checkpoint file."""

    def __init__(self, tmp_path, statuses=("ready",), fail_in="", **settings_kw):
        self.settings = Settings(
            _env_file=None,
            checkpoint_path=tmp_path / "cp.sqlite",
            workspace_dir=tmp_path / "ws",
            **settings_kw,
        )
        self.prepare_calls = self.index_calls = 0
        self.plan_feedback: list[str] = []
        self._statuses, self._fail_in = list(statuses), fail_in
        self.saver = open_checkpointer(self.settings)
        deps = AgentDeps(
            settings=self.settings, prepare=self._prepare, index=self._index, plan=self._plan
        )
        self.runner = AgentRunner(deps, self.saver)

    def _prepare(self, repo, number, run_id):
        self.prepare_calls += 1
        if self._fail_in == "prepare":
            raise RuntimeError("clone failed, token gsk_" + "a" * 30)
        return {"branch": f"agent/issue-{number}", "head_sha": "abc123", "reused": False}

    def _index(self, run_id):
        self.index_calls += 1
        return {"skipped": False, "chunks": 11}

    def _plan(self, run_id, feedback):
        self.plan_feedback.append(feedback)
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return planning(status)

    def close(self):
        close_checkpointer(self.saver)


@pytest.fixture
def make_env(tmp_path):
    envs = []

    def factory(**kwargs):
        env = Env(tmp_path, **kwargs)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()


def nodes(status):
    return [e["node"] for e in status.events]


# ---------------------------------------------------------------- happy path
def test_graph_pauses_for_the_human_then_continues_after_approval(make_env):
    env = make_env()
    paused = env.runner.start("me/demo", 1, RUN)
    assert paused.paused and paused.waiting_for == "approve_plan"
    assert paused.status == "plan_ready"
    assert nodes(paused) == ["prepare", "index", "plan"]
    assert paused.approval["files_in_scope"] == list(FILES)
    assert paused.approval["plan_hash"] == plan_hash(paused.planning["plan"])
    assert paused.approved_files == []  # nothing is approved yet

    done = env.runner.decide(RUN, PlanDecision(action="approve"))
    assert done.status == "plan_approved" and not done.paused
    assert done.approved_files == list(FILES)
    assert nodes(done) == ["prepare", "index", "plan", "approve_plan", "execute"]
    assert (env.prepare_calls, env.index_calls, len(env.plan_feedback)) == (1, 1, 1)


def test_pause_survives_a_restart(make_env):
    first = make_env()
    first.runner.start("me/demo", 1, RUN)
    first.close()  # like closing the terminal

    second = make_env()  # new objects, same SQLite file
    status = second.runner.status(RUN)
    assert status.paused and status.waiting_for == "approve_plan"
    done = second.runner.decide(RUN, PlanDecision(action="approve"))
    assert done.status == "plan_approved"
    assert second.plan_feedback == [] and second.prepare_calls == 0  # nothing was repeated


def test_start_generates_a_run_id(make_env):
    status = make_env().runner.start("me/demo", 1)
    assert len(status.run_id) == 12 and status.paused


# ------------------------------------------------------------------ decisions
def test_reject_stops_without_executing(make_env):
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    done = env.runner.decide(RUN, PlanDecision(action="reject"))
    assert done.status == "rejected" and not done.paused
    assert done.approved_files == []
    assert "execute" not in nodes(done)


def test_revise_replans_with_feedback_and_pauses_again(make_env):
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    status = env.runner.decide(RUN, PlanDecision(action="revise", feedback="cover negatives"))
    assert env.plan_feedback == ["", "cover negatives"]
    assert status.paused and status.plan_revisions == 1
    assert status.approval["revisions_left"] == 1
    assert env.runner.decide(RUN, PlanDecision(action="approve")).status == "plan_approved"


def test_feedback_accumulates_across_revisions(make_env):
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    env.runner.decide(RUN, PlanDecision(action="revise", feedback="first"))
    env.runner.decide(RUN, PlanDecision(action="revise", feedback="second"))
    assert env.plan_feedback[-1] == "first\nsecond"


def test_revisions_are_bounded(make_env):
    env = make_env(max_plan_revisions=1)
    env.runner.start("me/demo", 1, RUN)
    env.runner.decide(RUN, PlanDecision(action="revise", feedback="one"))
    stopped = env.runner.decide(RUN, PlanDecision(action="revise", feedback="two"))
    assert stopped.status == "revision_limit" and not stopped.paused
    assert len(env.plan_feedback) == 2  # no third plan


# ------------------------------------------------------------- early endings
@pytest.mark.parametrize("status", ["needs_clarification", "blocked"])
def test_unclear_or_blocked_plans_never_reach_the_human(make_env, status):
    result = make_env(statuses=(status,)).runner.start("me/demo", 1, RUN)
    assert result.status == status
    assert not result.paused
    assert "approve_plan" not in nodes(result)


def test_failures_end_the_run_and_hide_secrets(make_env):
    env = make_env(fail_in="prepare")
    result = env.runner.start("me/demo", 1, RUN)
    assert result.status == "failed"
    assert result.error.startswith("prepare:")
    assert "gsk_" not in result.error
    assert env.index_calls == 0


# ------------------------------------------------------------- fail closed
def test_a_garbage_answer_counts_as_reject(make_env):
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    env.runner._graph.invoke(Command(resume={"action": "yolo"}), env.runner._config(RUN))
    result = env.runner.status(RUN)
    assert result.status == "rejected" and result.approved_files == []


def test_an_answer_for_another_plan_is_refused(make_env):
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    with pytest.raises(WorkflowError, match="different plan"):
        env.runner.decide(RUN, PlanDecision(action="approve", plan_hash="0" * 16))
    assert env.runner.status(RUN).paused  # still waiting
    # Even if it slipped past the runner, the node itself rejects it:
    answer = {"action": "approve", "feedback": "", "plan_hash": "0" * 16}
    env.runner._graph.invoke(Command(resume=answer), env.runner._config(RUN))
    assert env.runner.status(RUN).status == "rejected"


def test_request_errors(make_env):
    env = make_env()
    with pytest.raises(WorkflowError, match="Invalid run id"):
        env.runner.status("../x")
    with pytest.raises(WorkflowError, match="No run"):
        env.runner.status("b" * 12)
    env.runner.start("me/demo", 1, RUN)
    with pytest.raises(WorkflowError, match="already exists"):
        env.runner.start("me/demo", 1, RUN)
    env.runner.decide(RUN, PlanDecision(action="reject"))
    with pytest.raises(WorkflowError, match="not waiting"):
        env.runner.decide(RUN, PlanDecision(action="approve"))
    with pytest.raises(ValueError):
        env.runner.start("https://evil.com/a/b", 1)


# --------------------------------------------------------------------- audit
def test_decisions_are_logged_without_the_feedback_text(make_env, tmp_path):
    (tmp_path / "ws" / RUN).mkdir(parents=True)
    env = make_env()
    env.runner.start("me/demo", 1, RUN)
    env.runner.decide(RUN, PlanDecision(action="revise", feedback="private words"))
    env.runner.decide(RUN, PlanDecision(action="approve"))
    text = (tmp_path / "ws" / RUN / "audit.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(line) for line in text.splitlines()]
    assert [line["action"] for line in lines] == ["revise", "approve"]
    assert lines[0]["feedback_chars"] == len("private words")
    assert "private words" not in text
    assert all(line["plan_hash"] for line in lines)


# ----------------------------------------------------------------- small units
def test_decision_validation():
    assert PlanDecision(action="approve").feedback == ""
    with pytest.raises(ValidationError):
        PlanDecision(action="revise", feedback="   ")
    with pytest.raises(ValidationError):
        PlanDecision(action="merge")  # type: ignore[arg-type]


def test_plan_hash_is_stable_and_sensitive():
    plan = planning().plan.model_dump(mode="json")
    assert plan_hash(plan) == plan_hash(dict(reversed(list(plan.items()))))
    assert plan_hash(plan) != plan_hash({**plan, "goal": "something else"})
    assert plan_hash(None) == ""


def test_graph_diagram_shows_the_approval_node(make_env):
    text = make_env().runner.mermaid()
    assert "approve_plan" in text and "execute" in text