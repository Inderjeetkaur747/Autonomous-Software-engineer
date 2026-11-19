"""Edit loop: the LLM requests read and edit tools. OUR code validates and runs them."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from swe_agent.agent.explorer import MAX_PARALLEL_CALLS, _tool_message
from swe_agent.llm.client import LLMError
from swe_agent.schemas.tools import ToolResult
from swe_agent.tools import EXPLORE_TOOLS, RepoTools

logger = logging.getLogger(__name__)

CHANGE_TOOLS = frozenset({"edit_file", "create_file"})
LLM_EDIT_TOOLS = EXPLORE_TOOLS | CHANGE_TOOLS  # what the LLM may ask for
EXECUTION_TOOLS = LLM_EDIT_TOOLS | {"run_tests"}  # run_tests is for OUR code only


@dataclass
class EditOutcome:
    steps: int = 0  # model calls
    tool_calls: int = 0  # tools really executed
    edits: int = 0  # successful edit_file / create_file calls
    stopped: str = "done"  # done | max_steps | llm_error
    notes: list[str] = field(default_factory=list)


def preload_files(tools: RepoTools, files: list[str], budget: int) -> str:
    """Read the approved files first, so the model starts with their exact text."""
    parts: list[str] = []
    used = 0
    for path in files:
        if used >= budget:
            break
        result = tools.call("read_file", {"path": path})
        if not result.ok:
            continue  # a file that will be created does not exist yet
        piece = result.output[: budget - used]
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def _run_allowed(
    tools: RepoTools, allowed: frozenset[str], name: str, raw_arguments: str
) -> tuple[ToolResult, bool]:
    """Run one requested tool, but only if it is on the allow-list. Returns (result, executed)."""
    if name not in allowed:
        return ToolResult(ok=False, error=f"Tool not available at this stage: {name!r}."), False
    try:
        arguments: Any = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return ToolResult(ok=False, error="The arguments are not valid JSON."), False
    if not isinstance(arguments, dict):
        return ToolResult(ok=False, error="The arguments must be a JSON object."), False
    return tools.call(name, arguments), True


def run_edit_loop(
    *,
    llm: Any,
    tools: RepoTools,
    system: str,
    user: str,
    allowed: frozenset[str],
    max_steps: int,
    result_chars: int,
) -> EditOutcome:
    outcome = EditOutcome()
    specs = [spec.to_openai() for spec in tools.specs() if spec.name in allowed]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    for _ in range(max_steps):
        try:
            turn = llm.chat_tools(messages, specs, purpose="edit")
        except LLMError as exc:
            outcome.stopped = "llm_error"
            outcome.notes.append(f"Editing stopped early: {exc}")
            logger.warning("edit loop stopped: %s", exc)
            return outcome
        outcome.steps += 1
        if not turn.tool_calls:
            return outcome  # the model said DONE

        messages.append(turn.to_message())
        for index, call in enumerate(turn.tool_calls):
            if index >= MAX_PARALLEL_CALLS:
                content = f"Skipped: call at most {MAX_PARALLEL_CALLS} tools at a time."
            else:
                result, executed = _run_allowed(tools, allowed, call.name, call.arguments)
                outcome.tool_calls += int(executed)
                if executed and result.ok and call.name in CHANGE_TOOLS:
                    outcome.edits += 1
                content = _tool_message(result, result_chars)
            # Every tool call MUST get an answer, even a refusal.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    outcome.stopped = "max_steps"
    outcome.notes.append(f"The edit step limit was reached ({max_steps}).")
    return outcome