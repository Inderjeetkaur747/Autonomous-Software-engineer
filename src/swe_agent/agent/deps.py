"""The real implementations of prepare, index and plan (Phases 3 to 9)."""

from __future__ import annotations

import json
from typing import Any

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

    return AgentDeps(settings=settings, prepare=prepare, index=index, plan=plan)