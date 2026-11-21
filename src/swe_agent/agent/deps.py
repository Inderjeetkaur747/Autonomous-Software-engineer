"""The real implementations of prepare, index and plan (Phases 3 to 9)."""

from __future__ import annotations

import json
from typing import Any

from swe_agent.agent.editor import EXECUTION_TOOLS
from swe_agent.agent.executor import execute_plan
from swe_agent.agent.patching import approve_patch, build_patch, discard_changes
from swe_agent.agent.planning import load_issue, plan_issue
from swe_agent.agent.workflow import AgentDeps
from swe_agent.config import Settings, get_settings
from swe_agent.github import WorkspaceError, WorkspaceManager, prepare_run, resolve_issue_ref
from swe_agent.indexing import (
    QdrantStore,
    build_chunks,
    collection_for,
    get_embedder,
    index_chunks,
    save_chunks,
)
from swe_agent.llm import GroqLLM
from swe_agent.sandbox import get_runner
from swe_agent.schemas.execution import ExecutionResult
from swe_agent.schemas.patch import PatchResult
from swe_agent.schemas.plan import ImplementationPlan
from swe_agent.schemas.planning import PlanningResult
from swe_agent.tools import EXPLORE_TOOLS, CodeSearcher, RepoTools


def make_default_deps(settings: Settings | None = None) -> AgentDeps:
    settings = settings or get_settings()

    def prepare(repo: str, number: int, run_id: str) -> dict[str, Any]:
        workspaces = WorkspaceManager(settings)
        try:
            existing = workspaces.get(run_id)
        except WorkspaceError:
            existing = None

        if existing is not None:  # reuse a workspace made earlier by `swe-agent prepare`
            meta = json.loads((existing.root / "run.json").read_text(encoding="utf-8"))
            same_repo = str(meta.get("repo", "")).lower() == repo.lower()
            same_issue = (meta.get("issue") or {}).get("number") == number
            if not (same_repo and same_issue):
                raise ValueError("This run id belongs to a different repository or issue.")
            return {
                "branch": meta.get("branch", ""),
                "head_sha": meta.get("base_commit", ""),
                "reused": True,
            }

        run = prepare_run(resolve_issue_ref(str(number), repo), settings=settings, run_id=run_id)
        return {"branch": run.branch, "head_sha": run.head_sha, "reused": False}

    def index(run_id: str) -> dict[str, Any]:
        workspace = WorkspaceManager(settings).get(run_id)
        chunks_file = workspace.root / "chunks.jsonl"
        collection = collection_for(run_id)
        store = QdrantStore(settings)
        try:
            if chunks_file.exists() and store.collection_exists(collection):
                return {"skipped": True, "chunks": store.count(collection)}
            chunks = build_chunks(workspace.repo_dir, settings).chunks
            save_chunks(chunks, chunks_file)
            stats = index_chunks(
                chunks,
                collection=collection,
                embedder=get_embedder(settings),
                store=store,
                batch_size=settings.embedding_batch_size,
            )
            return {"skipped": False, "chunks": stats.chunk_count, "seconds": stats.seconds}
        finally:
            store.close()  # local Qdrant: the planner opens it next

    def plan(run_id: str, feedback: str) -> PlanningResult:
        workspace = WorkspaceManager(settings).get(run_id)
        issue = load_issue(workspace.root)
        if feedback.strip():
            # The issue text is wrapped as untrusted data by the planner, feedback included.
            issue = issue.model_copy(
                update={"body": f"{issue.body}\n\nReviewer feedback on the previous plan:\n{feedback}"}  # noqa: E501
            )
        searcher = CodeSearcher(workspace.run_id, workspace.root, settings)
        tools = RepoTools(
            workspace.repo_dir,
            settings=settings,
            searcher=searcher,
            allowed_tools=EXPLORE_TOOLS,  # exploring only
            write_paths=[],  # nothing may change before approval
        )
        result = plan_issue(
            issue=issue, tools=tools, llm=GroqLLM(settings), searcher=searcher, settings=settings
        )
        (workspace.root / "plan.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def execute(run_id: str, approved_files: list[str], plan: dict[str, Any]) -> ExecutionResult:
        workspace = WorkspaceManager(settings).get(run_id)
        issue = load_issue(workspace.root)
        tools = RepoTools(
            workspace.repo_dir,
            settings=settings,
            searcher=CodeSearcher(workspace.run_id, workspace.root, settings),
            runner=get_runner(settings),  # the Docker sandbox from Phase 11
            allowed_tools=EXECUTION_TOOLS,
            write_paths=approved_files,  # ONLY the files of the approved plan
        )
        result = execute_plan(
            issue=issue,
            plan=ImplementationPlan.model_validate(plan),
            approved_files=approved_files,
            tools=tools,
            llm=GroqLLM(settings),
            settings=settings,
        )
        (workspace.root / "execution.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    def make_patch(
        run_id: str, approved_files: list[str], execution: dict[str, Any]
    ) -> PatchResult:
        workspace = WorkspaceManager(settings).get(run_id)
        meta = json.loads((workspace.root / "run.json").read_text(encoding="utf-8"))
        result, _data = build_patch(
            workspace.repo_dir,
            approved_files,
            settings,
            execution=execution,
            base_commit=meta.get("base_commit", ""),
            branch=meta.get("branch", ""),
            write_to=workspace.root / "patch.diff",  # outside the repo, so it is not in the diff
        )
        (workspace.root / "patch.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    def accept_patch(run_id: str, patch_hash: str, approved_files: list[str]) -> dict[str, Any]:
        workspace = WorkspaceManager(settings).get(run_id)
        return approve_patch(
            workspace.repo_dir,
            approved_files,
            patch_hash,
            workspace.root,
            settings,
        )

    def discard_patch(run_id: str) -> None:
        workspace = WorkspaceManager(settings).get(run_id)
        discard_changes(workspace.repo_dir)
        candidate = workspace.root / "patch.diff"
        if candidate.exists():
            candidate.replace(workspace.root / "rejected.patch")

    return AgentDeps(
        settings=settings,
        prepare=prepare,
        index=index,
        plan=plan,
        execute=execute,
        make_patch=make_patch,
        accept_patch=accept_patch,
        discard_patch=discard_patch,
    )