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
from swe_agent.schemas.execution import ExecutionResult
from swe_agent.schemas.patch import PatchFile, PatchResult
from swe_agent.schemas.plan import ImplementationPlan, PlanStep
from swe_agent.schemas.planning import PlanningResult
from swe_agent.schemas.workflow import PatchDecision, PlanDecision

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

    def __init__(
        self, tmp_path, statuses=("ready",), fail_in="", execute=None,
        make_patch=None, accept_patch=None, discard_patch=None, **settings_kw,
    ):
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
            settings=self.settings,
            prepare=self._prepare,
            index=self._index,
            plan=self._plan,
            execute=execute,
            make_patch=make_patch,
            accept_patch=accept_patch,
            discard_patch=discard_patch,
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


def test_approval_runs_the_executor_with_exactly_the_approved_files(make_env):
    seen = []

    def fake_execute(run_id, files, plan):
        seen.append((run_id, files, plan["goal"]))
        return ExecutionResult(status="tests_passed", changed_files=["mathkit/stats.py"])

    env = make_env(execute=fake_execute)
    env.runner.start("me/demo", 1, RUN)
    done = env.runner.decide(RUN, PlanDecision(action="approve"))
    assert seen == [(RUN, list(FILES), "g")]
    assert done.status == "tests_passed"
    assert done.execution["changed_files"] == ["mathkit/stats.py"]
    assert done.events[-1]["node"] == "execute"


def test_the_executor_never_runs_without_approval(make_env):
    calls = []
    env = make_env(execute=lambda *args: calls.append(args))
    env.runner.start("me/demo", 1, RUN)
    env.runner.decide(RUN, PlanDecision(action="reject"))
    assert calls == []


def test_a_crashing_executor_becomes_a_failed_run(make_env):
    def boom(*_args):
        raise RuntimeError("sandbox exploded gsk_" + "a" * 30)

    env = make_env(execute=boom)
    env.runner.start("me/demo", 1, RUN)
    done = env.runner.decide(RUN, PlanDecision(action="approve"))
    assert done.status == "failed"
    assert done.error.startswith("execute:") and "gsk_" not in done.error


# ------------------------------------------------------------ patch review
PHASH = "a1b2c3d4e5f60718"


def good_execution(run_id, files, plan):
    return ExecutionResult(status="tests_passed", changed_files=list(files))


def patch_result(status="ready", **kw):
    return PatchResult(
        status=status,
        patch_hash=PHASH if status == "ready" else "",
        files=[PatchFile(path="mathkit/stats.py", status="modified", additions=2)],
        additions=2,
        diff_preview="diff --git a/x b/x",
        tests_passed=True,
        **kw,
    )


class PatchSpy:
    def __init__(self, result=None, accept_error=None):
        self.result = result or patch_result()
        self.accept_error = accept_error
        self.made, self.accepted, self.discarded = [], [], []

    def make(self, run_id, files, execution):
        self.made.append((run_id, list(files), execution.get("status")))
        return self.result

    def accept(self, run_id, patch_hash, files):
        if self.accept_error:
            raise self.accept_error
        self.accepted.append((run_id, patch_hash, list(files)))
        return {"path": "ws/approved.patch"}

    def discard(self, run_id):
        self.discarded.append(run_id)


def at_review(make_env, spy):
    env = make_env(
        execute=good_execution, make_patch=spy.make,
        accept_patch=spy.accept, discard_patch=spy.discard,
    )
    env.runner.start("me/demo", 1, RUN)
    return env, env.runner.decide(RUN, PlanDecision(action="approve"))


def test_a_ready_patch_pauses_for_the_final_review_then_saves_on_approval(make_env):
    spy = PatchSpy()
    env, paused = at_review(make_env, spy)
    assert paused.paused and paused.waiting_for == "review_patch"
    assert paused.status == "patch_ready"
    assert paused.approval["type"] == "patch_approval"
    assert paused.approval["patch_hash"] == PHASH
    assert spy.made == [(RUN, list(FILES), "tests_passed")]
    assert spy.accepted == []  # nothing is saved before the human decides

    done = env.runner.decide_patch(RUN, PatchDecision(action="approve"))
    assert done.status == "patch_approved" and not done.paused
    assert spy.accepted == [(RUN, PHASH, list(FILES))]
    assert done.patch["approved_path"] == "ws/approved.patch"
    assert nodes(done) == [
        "prepare", "index", "plan", "approve_plan", "execute", "make_patch", "review_patch",
    ]


def test_rejecting_discards_the_changes(make_env):
    spy = PatchSpy()
    env, _ = at_review(make_env, spy)
    done = env.runner.decide_patch(RUN, PatchDecision(action="reject"))
    assert done.status == "patch_rejected"
    assert spy.discarded == [RUN] and spy.accepted == []


def test_no_changes_and_blocked_patches_never_reach_the_human(make_env):
    for kind, expected in (("no_changes", "no_patch"), ("blocked", "patch_blocked")):
        result = patch_result(kind, violations=["x"]) if kind == "blocked" else patch_result(kind)
        spy = PatchSpy(result)
        env, status = at_review(make_env, spy)
        assert status.status == expected and not status.paused
        assert "review_patch" not in nodes(status)
        env.close()
        (env.settings.checkpoint_path).unlink()


def test_risky_patches_need_an_explicit_acknowledgement(make_env):
    spy = PatchSpy(patch_result(requires_acknowledgement=True, ack_reasons=["Tests did not pass."]))
    env, paused = at_review(make_env, spy)
    assert paused.approval["requires_acknowledgement"] is True
    with pytest.raises(WorkflowError, match="accept-risk"):
        env.runner.decide_patch(RUN, PatchDecision(action="approve"))
    assert spy.accepted == [] and env.runner.status(RUN).paused
    done = env.runner.decide_patch(RUN, PatchDecision(action="approve", accept_risk=True))
    assert done.status == "patch_approved"


def test_a_decision_for_another_patch_is_refused(make_env):
    spy = PatchSpy()
    env, _ = at_review(make_env, spy)
    with pytest.raises(WorkflowError, match="different patch"):
        env.runner.decide_patch(RUN, PatchDecision(action="approve", patch_hash="0" * 16))
    assert env.runner.status(RUN).paused


def test_garbage_answers_are_not_approvals_and_do_not_delete_anything(make_env):
    spy = PatchSpy(patch_result(requires_acknowledgement=True, ack_reasons=["r"]))
    env, _ = at_review(make_env, spy)
    config = env.runner._config(RUN)
    env.runner._graph.invoke(Command(resume={"action": "merge"}), config)
    status = env.runner.status(RUN)
    assert status.status == "patch_rejected"
    assert spy.accepted == [] and spy.discarded == []  # nothing saved, nothing destroyed


def test_a_failing_save_becomes_a_failed_run_without_secrets(make_env):
    spy = PatchSpy(accept_error=RuntimeError("disk full gsk_" + "a" * 30))
    env, _ = at_review(make_env, spy)
    done = env.runner.decide_patch(RUN, PatchDecision(action="approve"))
    assert done.status == "failed"
    assert done.error.startswith("review_patch:") and "gsk_" not in done.error


def test_plan_and_patch_decisions_cannot_be_mixed_up(make_env):
    spy = PatchSpy()
    env = make_env(execute=good_execution, make_patch=spy.make)
    env.runner.start("me/demo", 1, RUN)  # waiting for the PLAN decision
    with pytest.raises(WorkflowError, match="not waiting for a patch decision"):
        env.runner.decide_patch(RUN, PatchDecision(action="approve"))
    env.runner.decide(RUN, PlanDecision(action="approve"))  # now waiting for the PATCH
    with pytest.raises(WorkflowError, match="not waiting for a plan decision"):
        env.runner.decide(RUN, PlanDecision(action="approve"))


def test_the_patch_pause_survives_a_restart(make_env):
    spy = PatchSpy()
    first, _ = at_review(make_env, spy)
    first.close()
    second = make_env(
        execute=good_execution,
        make_patch=spy.make,
        accept_patch=spy.accept,
        discard_patch=spy.discard,
    )
    assert second.runner.status(RUN).waiting_for == "review_patch"
    result = second.runner.decide_patch(RUN, PatchDecision(action="approve"))
    assert result.status == "patch_approved"
    assert len(spy.made) == 1  # the patch was not rebuilt