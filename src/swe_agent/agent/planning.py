"""plan_issue(): retrieve, explore, analyse, plan, check. Nothing here changes the repo."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from swe_agent.agent.evidence import Evidence
from swe_agent.agent.explorer import explore
from swe_agent.config import Settings, get_settings
from swe_agent.llm.prompts import build_analysis_prompts, build_plan_prompts
from swe_agent.retrieval.citations import check_citations_in_lines
from swe_agent.schemas.analysis import IssueAnalysis
from swe_agent.schemas.github import IssueData
from swe_agent.schemas.plan import ImplementationPlan, PlanAction
from swe_agent.schemas.planning import PlanningResult, PlanningStatus
from swe_agent.security import PathGuard, SecurityPolicy, scan_for_injection
from swe_agent.tools import CodeSearch, RepoTools

logger = logging.getLogger(__name__)

RETRIEVAL_K = 6
CLARIFY_BELOW = 0.5


class PlanningError(Exception):
    """The planning stage could not start."""


def load_issue(run_root: Path) -> IssueData:
    """Read the issue that `swe-agent prepare` saved in run.json."""
    meta_file = Path(run_root) / "run.json"
    if not meta_file.exists():
        raise PlanningError("run.json not found. Run: swe-agent prepare ...")
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        return IssueData.model_validate(data["issue"])
    except (KeyError, ValueError, ValidationError) as exc:
        raise PlanningError(f"run.json is not valid: {exc}") from exc


def _clean(listed: list[str], shown: dict[str, set[int]]) -> list[str]:
    """Keep only the citations whose lines the model really saw."""
    return check_citations_in_lines(" ".join(listed), shown).valid


def plan_issue(
    *,
    issue: IssueData,
    tools: RepoTools,
    llm: Any,
    searcher: CodeSearch | None = None,
    settings: Settings | None = None,
) -> PlanningResult:
    settings = settings or get_settings()
    started = time.perf_counter()
    llm_before, tokens_before, tools_before = len(llm.calls), llm.total_tokens, len(tools.calls)
    notes: list[str] = []

    def result_of(status: PlanningStatus, **extra: Any) -> PlanningResult:
        return PlanningResult(
            status=status,
            injection_risk=injection.risk,
            injection_flags=[f"{f.name} ({f.severity}) line {f.line}" for f in injection.findings],
            retrieved_citations=retrieved,
            inspected_files=sorted(evidence.shown),
            notes=notes,
            explore_stopped=outcome.stopped,
            tool_log=tools.calls[tools_before:],
            llm_calls=len(llm.calls) - llm_before,
            total_tokens=llm.total_tokens - tokens_before,
            seconds=round(time.perf_counter() - started, 2),
            **extra,
        )

    # 1) Warn about suspicious issue text. We never silently edit it.
    injection = scan_for_injection(f"{issue.title}\n{issue.body}")

    # 2) Automatic retrieval gives the model a map before it uses any tool.
    evidence = Evidence()
    retrieved: list[str] = []
    if searcher is not None:
        try:
            found = searcher.search(f"{issue.title}\n{issue.body}"[:600], RETRIEVAL_K, "hybrid")
            evidence.add_hits(found.hits)
            retrieved = [h.chunk.citation.label for h in found.hits]
        except Exception as exc:  # noqa: BLE001
            logger.warning("initial retrieval failed: %s", exc)
            notes.append(f"Initial retrieval failed: {exc}")

    # 3) Bounded exploration with read-only tools.
    outcome = explore(
        llm=llm,
        tools=tools,
        issue=issue,
        evidence=evidence,
        initial_context=evidence.render(),
        max_steps=settings.max_explore_steps,
        result_chars=settings.explore_result_chars,
    )
    notes.extend(outcome.notes)
    context = evidence.render()  # exactly what the next prompts will contain
    shown = evidence.shown

    # 4) Analysis.
    system, user = build_analysis_prompts(issue.title, issue.body, context)
    analysis = llm.generate_structured(IssueAnalysis, system=system, user=user, purpose="analysis")
    if analysis.needs_clarification and analysis.confidence < CLARIFY_BELOW:
        cleaned = analysis.model_copy(update={"evidence": _clean(analysis.evidence, shown)})
        full = check_citations_in_lines(analysis.model_dump_json(), shown)
        return result_of(
            PlanningStatus.NEEDS_CLARIFICATION,
            analysis=cleaned,
            citations_valid=full.valid,
            citations_invalid=full.invalid,
        )

    # 5) Planning, with a bounded repair loop driven by the security policy.
    policy = SecurityPolicy(PathGuard(tools.repo_dir), settings)
    feedback = ""
    plan: ImplementationPlan | None = None
    blocked_messages: list[str] = []
    for attempt in range(1, settings.max_plan_attempts + 1):
        system, user = build_plan_prompts(
            issue.title, issue.body, analysis.model_dump_json(), context
        )
        plan = llm.generate_structured(
            ImplementationPlan, system=system, user=user + feedback, purpose="plan"
        )
        report = policy.check_plan(plan)
        blocked_messages = [v.message for v in report.violations if v.severity == "block"]
        if not blocked_messages:
            break
        logger.warning("plan attempt %d blocked: %s", attempt, blocked_messages)
        feedback = (
            "\n\nYour previous plan was rejected by the security policy:\n"
            + "\n".join(f"- {m}" for m in blocked_messages)
            + "\nReturn a corrected plan."
        )
    assert plan is not None  # max_plan_attempts >= 1

    warnings = [v.message for v in report.violations if v.severity == "warn"]
    for step in plan.steps:
        if step.action == PlanAction.MODIFY and step.file_path not in shown:
            notes.append(f"step {step.step_number}: {step.file_path} was never read. Review it.")

    # 6) Citations: remove the ones the model could not have seen, and report them.
    full = check_citations_in_lines(
        analysis.model_dump_json() + " " + plan.model_dump_json(), shown)
    analysis = analysis.model_copy(update={"evidence": _clean(analysis.evidence, shown)})
    plan = plan.model_copy(update={"citations": _clean(plan.citations, shown)})

    return result_of(
        PlanningStatus.BLOCKED if blocked_messages else PlanningStatus.READY,
        analysis=analysis,
        plan=plan,
        citations_valid=full.valid,
        citations_invalid=full.invalid,
        policy_violations=blocked_messages,
        policy_warnings=warnings,
    )