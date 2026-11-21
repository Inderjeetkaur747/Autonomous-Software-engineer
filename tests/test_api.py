import threading
import time

import pytest
from fastapi.testclient import TestClient

from swe_agent.agent import AgentDeps, AgentRunner, close_checkpointer, open_checkpointer
from swe_agent.api import JobBusy, JobManager, create_app, ensure_safe_bind
from swe_agent.config import Settings
from swe_agent.schemas.analysis import IssueAnalysis
from swe_agent.schemas.execution import ExecutionResult
from swe_agent.schemas.patch import PatchFile, PatchResult
from swe_agent.schemas.plan import ImplementationPlan, PlanStep
from swe_agent.schemas.planning import PlanningResult

RUN, OTHER = "a" * 12, "b" * 12
FILES = ("mathkit/stats.py", "tests/test_stats.py")
PHASH = "a1b2c3d4e5f60718"
KEY = "k" * 20
HEADERS = {"X-API-Key": KEY}


def planning():
    analysis = IssueAnalysis(
        summary="s", issue_type="bug", root_cause_hypothesis="r",
        affected_areas=list(FILES), acceptance_criteria=["c"], confidence=0.9,
    )
    steps = [
        PlanStep(step_number=i, file_path=p, action="modify", description="d", rationale="r")
        for i, p in enumerate(FILES, start=1)
    ]
    plan = ImplementationPlan(goal="g", steps=steps, test_strategy="t")
    return PlanningResult(status="ready", analysis=analysis, plan=plan)


@pytest.fixture
def make_app(tmp_path):
    opened = []

    def factory(api_key="", gate=None, ack=False):
        settings = Settings(
            _env_file=None,
            checkpoint_path=tmp_path / "cp.sqlite",
            workspace_dir=tmp_path / "ws",
            api_key=api_key,
        )
        saver = open_checkpointer(settings)

        def prepare(repo, number, run_id):
            if gate is not None:
                gate.wait(10)
            return {"branch": f"agent/issue-{number}", "head_sha": "abc", "reused": False}

        def make_patch(run_id, files, execution):
            return PatchResult(
                status="ready", patch_hash=PHASH, tests_passed=True, additions=2,
                files=[PatchFile(path=FILES[0], status="modified", additions=2)],
                diff_preview="diff --git a/x b/x",
                requires_acknowledgement=ack,
                ack_reasons=["Tests did not pass."] if ack else [],
            )

        deps = AgentDeps(
            settings=settings,
            prepare=prepare,
            index=lambda run_id: {"skipped": False, "chunks": 5},
            plan=lambda run_id, feedback: planning(),
            execute=lambda run_id, files, plan: ExecutionResult(
                status="tests_passed", changed_files=list(files)
            ),
            make_patch=make_patch,
            accept_patch=lambda run_id, patch_hash, files: {"path": "x/approved.patch"},
            discard_patch=lambda run_id: None,
        )
        client = TestClient(create_app(settings, AgentRunner(deps, saver)))
        client.__enter__()  # runs the lifespan: creates the JobManager
        opened.append((client, saver))
        return client

    yield factory
    for client, saver in opened:
        client.__exit__(None, None, None)
        close_checkpointer(saver)


def wait_idle(client, run_id, headers=None, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/runs/{run_id}", headers=headers).json()
        if not view["busy"]:
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not finish")


def start(client, run_id=RUN, headers=None):
    body = {"repo": "me/demo", "issue_number": 1, "run_id": run_id}
    return client.post("/runs", json=body, headers=headers)


def at_plan_approval(client):
    assert start(client).status_code == 202
    view = wait_idle(client, RUN)
    assert view["paused"] and view["waiting_for"] == "approve_plan"
    return view


# ----------------------------------------------------------------- the flow
def test_full_flow_through_the_api(make_app):
    client = make_app()
    started = start(client)
    assert started.status_code == 202 and started.json()["busy"] is True

    view = wait_idle(client, RUN)
    assert view["status"] == "plan_ready" and view["paused"]
    assert view["approval"]["files_in_scope"] == list(FILES)

    plan_hash = view["approval"]["plan_hash"]
    body = {"action": "approve", "plan_hash": plan_hash}
    assert client.post(f"/runs/{RUN}/plan-decision", json=body).status_code == 202
    view = wait_idle(client, RUN)
    assert view["status"] == "patch_ready" and view["waiting_for"] == "review_patch"
    assert view["approval"]["patch_hash"] == PHASH

    assert client.post(f"/runs/{RUN}/patch-decision", json={"action": "approve"}).status_code == 202
    view = wait_idle(client, RUN)
    assert view["status"] == "patch_approved" and not view["paused"]
    assert view["patch"]["approved_path"] == "x/approved.patch"
    assert view["interrupted"] is False


def test_risky_patches_need_accept_risk(make_app):
    client = make_app(ack=True)
    at_plan_approval(client)
    client.post(f"/runs/{RUN}/plan-decision", json={"action": "approve"})
    wait_idle(client, RUN)

    refused = client.post(f"/runs/{RUN}/patch-decision", json={"action": "approve"})
    assert refused.status_code == 422 and "accept_risk" in refused.json()["detail"]
    assert client.get(f"/runs/{RUN}").json()["paused"] is True  # still waiting

    ok = client.post(f"/runs/{RUN}/patch-decision", json={"action": "approve", "accept_risk": True})
    assert ok.status_code == 202
    assert wait_idle(client, RUN)["status"] == "patch_approved"


def test_reject_finishes_the_run(make_app):
    client = make_app()
    at_plan_approval(client)
    client.post(f"/runs/{RUN}/plan-decision", json={"action": "reject"})
    assert wait_idle(client, RUN)["status"] == "rejected"


# ------------------------------------------------------------- bad requests
def test_start_validation(make_app):
    client = make_app()
    bad = [
        {"repo": "https://evil.com/a/b", "issue_number": 1},
        {"repo": "me/demo", "issue_number": 0},
        {"repo": "me/demo", "issue_number": 1, "unknown": True},
        {"repo": "me/demo", "issue_number": 1, "run_id": "../../x"},
    ]
    for body in bad:
        assert client.post("/runs", json=body).status_code == 422
    assert start(client).status_code == 202
    assert start(client).status_code == 409  # the same run id again


def test_decisions_in_the_wrong_state_are_refused(make_app):
    client = make_app()
    view = at_plan_approval(client)
    patch_body = {"action": "approve"}
    assert client.post(f"/runs/{RUN}/patch-decision", json=patch_body).status_code == 409
    stale = {"action": "approve", "plan_hash": "0" * 16}
    assert client.post(f"/runs/{RUN}/plan-decision", json=stale).status_code == 409
    assert client.post(f"/runs/{RUN}/plan-decision", json={"action": "revise"}).status_code == 422
    assert client.post(f"/runs/{RUN}/plan-decision", json={"action": "merge"}).status_code == 422
    assert (
        client.get(f"/runs/{RUN}").json()["approval"]["plan_hash"]
        == view["approval"]["plan_hash"]
    )


def test_unknown_and_malformed_run_ids(make_app):
    client = make_app()
    assert client.get(f"/runs/{'c' * 12}").status_code == 404
    assert client.get("/runs/not-a-run-id").status_code == 422
    assert (
        client.post(
            f"/runs/{'c' * 12}/plan-decision", json={"action": "approve"}
        ).status_code
        == 404
    )


# ----------------------------------------------------------- jobs and queue
def test_a_busy_run_refuses_decisions_and_other_runs_wait_in_the_queue(make_app):
    gate = threading.Event()
    client = make_app(gate=gate)
    assert start(client, RUN).status_code == 202
    assert start(client, OTHER).status_code == 202

    first = client.get(f"/runs/{RUN}").json()
    second = client.get(f"/runs/{OTHER}").json()
    assert first["busy"] and second["busy"]
    assert second["job"]["state"] == "queued"  # one worker: it waits for the first
    busy = client.post(f"/runs/{RUN}/plan-decision", json={"action": "approve"})
    assert busy.status_code == 409

    gate.set()
    assert wait_idle(client, RUN)["paused"]
    assert wait_idle(client, OTHER)["paused"]


def test_job_manager_rejects_double_submit_and_records_redacted_errors():
    jobs = JobManager(1)
    release = threading.Event()
    jobs.submit(RUN, "first", lambda: release.wait(5))
    with pytest.raises(JobBusy):
        jobs.submit(RUN, "second", lambda: None)
    release.set()
    deadline = time.time() + 5
    while jobs.busy(RUN) and time.time() < deadline:
        time.sleep(0.02)

    def boom():
        raise RuntimeError("failed with gsk_" + "a" * 30)

    jobs.submit(RUN, "third", boom)
    while jobs.busy(RUN) and time.time() < deadline:
        time.sleep(0.02)
    view = jobs.view(RUN)
    assert view.state == "error" and "gsk_" not in view.error and "RuntimeError" in view.error
    assert jobs.view(OTHER).state == "idle"
    jobs.shutdown()


# ------------------------------------------------------------ read endpoints
def test_events_tools_metrics_and_list(make_app):
    client = make_app()
    at_plan_approval(client)
    root = client.app.state.settings.workspace_dir / RUN
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(
        '{"repo": "me/demo", "issue": {"number": 1, "title": "T"}}', encoding="utf-8"
    )

    events = client.get(f"/runs/{RUN}/events?after=2").json()
    assert events["total"] == 3 and [e["node"] for e in events["events"]] == ["plan"]
    assert client.get(f"/runs/{RUN}/tools").json() == {"count": 0, "calls": []}
    metrics = client.get(f"/runs/{RUN}/metrics").json()
    assert metrics["events"] == 3 and metrics["llm_calls"] == 0 and metrics["tests_passed"] is None

    rows = client.get("/runs").json()
    assert [(r["run_id"], r["repo"], r["issue_title"], r["status"], r["paused"]) for r in rows] == [
        (RUN, "me/demo", "T", "plan_ready", True)
    ]


def test_patch_file_endpoint(make_app):
    client = make_app()
    root = client.app.state.settings.workspace_dir / RUN
    root.mkdir(parents=True, exist_ok=True)
    assert client.get(f"/runs/{RUN}/patch").status_code == 404
    (root / "patch.diff").write_bytes(b"candidate\n")
    reply = client.get(f"/runs/{RUN}/patch")
    assert reply.text == "candidate\n" and reply.headers["x-patch-kind"] == "candidate"
    (root / "approved.patch").write_bytes(b"approved\n")
    assert client.get(f"/runs/{RUN}/patch").headers["x-patch-kind"] == "approved"
    assert client.get(f"/runs/{OTHER}/patch").status_code == 404


# ---------------------------------------------------------------- security
def test_api_key_is_required_when_set(make_app):
    client = make_app(api_key=KEY)
    assert client.get("/health").status_code == 200  # health is open
    assert client.get("/health").json()["auth_required"] is True
    assert client.get("/runs").status_code == 401
    assert client.get("/runs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/runs", headers=HEADERS).status_code == 200
    assert start(client).status_code == 401
    assert start(client, headers=HEADERS).status_code == 202
    assert wait_idle(client, RUN, headers=HEADERS)["paused"]


def test_security_headers_and_cors(make_app):
    client = make_app()
    reply = client.get("/health")
    assert reply.headers["x-content-type-options"] == "nosniff"
    assert reply.headers["cache-control"] == "no-store"
    allowed = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    other = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_refuses_to_listen_on_the_network_without_a_strong_key():
    ensure_safe_bind("127.0.0.1", Settings(_env_file=None))
    with pytest.raises(ValueError, match="API_KEY"):
        ensure_safe_bind("0.0.0.0", Settings(_env_file=None))
    with pytest.raises(ValueError):
        ensure_safe_bind("0.0.0.0", Settings(_env_file=None, api_key="short"))
    ensure_safe_bind("0.0.0.0", Settings(_env_file=None, api_key=KEY))