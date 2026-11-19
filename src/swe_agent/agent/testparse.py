"""Read pytest output with plain code, and describe a run for the LLM."""

from __future__ import annotations

import re
from collections import Counter

from swe_agent.schemas.execution import PytestOutcome
from swe_agent.schemas.tools import ToolResult
from swe_agent.security.policy import is_test_path

_RESULT_RE = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")
_COUNT_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b"
)
_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.*))?$")
_NAMES = {"error": "errors", "warning": "warnings"}


def parse_test_run(result: ToolResult, tail_chars: int = 2800) -> PytestOutcome:
    """Turn the output of the run_tests tool into numbers and test names."""
    meta = result.meta
    lines = result.output.splitlines()
    body = lines[1:] if lines and lines[0].startswith("exit_code=") else lines

    counts: dict[str, int] = {}
    summary = ""
    failed: list[str] = []
    messages: dict[str, str] = {}
    for raw in body:
        text = raw.strip()
        match = _FAIL_RE.match(text)
        if match:
            test_id = match.group(1)
            if test_id not in messages:
                failed.append(test_id)
            messages[test_id] = (match.group(2) or "")[:200]
            continue
        if _RESULT_RE.search(text):
            found = _COUNT_RE.findall(text)
            if found or "no tests ran" in text:
                summary = text.strip("= ")
                counts = {}
                for number, name in found:
                    key = _NAMES.get(name, name)
                    counts[key] = counts.get(key, 0) + int(number)

    tail = "\n".join(body).strip()
    if len(tail) > tail_chars:
        tail = "[... cut ...]\n" + tail[-tail_chars:]
    return PytestOutcome(
        exit_code=meta.get("exit_code"),
        passed=bool(meta.get("passed")),
        timed_out=bool(meta.get("timed_out")),
        counts=counts,
        failed_tests=failed,
        failure_messages=messages,
        summary_line=summary,
        output_tail=tail,
    )


def describe_outcome(outcome: PytestOutcome) -> str:
    """Text for the diagnosis prompt. It is wrapped as untrusted data there."""
    failing = "\n".join(
        f"- {test}: {outcome.failure_messages.get(test, '')}" for test in outcome.failed_tests[:20]
    )
    return (
        f"exit_code={outcome.exit_code} timed_out={outcome.timed_out} counts={outcome.counts}\n"
        f"Failing tests:\n{failing or '(none parsed)'}\n\nEnd of the output:\n{outcome.output_tail}"
    )


# ------------------------------------------------------------ weakened tests
_ASSERT_RE = re.compile(r"\bassert\b|pytest\.raises|\.assert[A-Z]\w*\(")
_SKIP_RE = re.compile(r"pytest\.mark\.(?:skip|xfail)|unittest\.skip|pytest\.skip\(")


def find_weakened_tests(diff_text: str) -> list[str]:
    """Warn if a diff removes more assertions than it adds, or adds a skip marker."""
    removed: Counter[str] = Counter()
    added: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    path: str | None = None
    in_hunk = False
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            in_hunk, path = False, None
            continue
        if not in_hunk:
            if line.startswith("--- "):
                old = line[4:].strip()
                path = old[2:] if old.startswith("a/") else None
            elif line.startswith("+++ "):
                new = line[4:].strip()
                if new.startswith("b/"):  # "+++ /dev/null" (deleted file) keeps the old path
                    path = new[2:]
            elif line.startswith("@@"):
                in_hunk = True
            continue
        if line.startswith("@@") or path is None or not is_test_path(path):
            continue
        if line.startswith("-") and _ASSERT_RE.search(line):
            removed[path] += 1
        elif line.startswith("+"):
            if _ASSERT_RE.search(line):
                added[path] += 1
            if _SKIP_RE.search(line):
                skipped[path] += 1

    warnings = [
        f"{name}: {count} assertion line(s) removed, {added[name]} added"
        for name, count in removed.items()
        if count > added[name]
    ]
    warnings += [f"{name}: a skip or xfail marker was added" for name in skipped]
    return warnings