"""RepoTools: every tool the agent may use. Each one goes through the security guards."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from swe_agent.config import Settings, get_settings
from swe_agent.github.cloner import GitError, run_git
from swe_agent.indexing.chunker import split_lines
from swe_agent.indexing.scanner import scan_repo
from swe_agent.indexing.service import IndexingError
from swe_agent.logging_setup import redact_secrets
from swe_agent.retrieval.citations import format_context, verify_chunk
from swe_agent.schemas.code import ScanResult
from swe_agent.schemas.tools import (
    CreateFileArgs,
    EditFileArgs,
    GitDiffArgs,
    GrepArgs,
    ListFilesArgs,
    ReadFileArgs,
    RunTestsArgs,
    SearchCodeArgs,
    ToolCallRecord,
    ToolResult,
    ToolSpec,
)
from swe_agent.security import (
    PathGuard,
    SecurityError,
    SecurityPolicy,
    normalize_relative,
    scan_for_injection,
    validate_command,
)
from swe_agent.security.policy import is_test_path
from swe_agent.tools.fileio import (
    EditError,
    apply_replace,
    atomic_write,
    load_text_strict,
    short_diff,
    to_bytes,
)
from swe_agent.tools.runner import CommandRunner, DisabledRunner, RunnerUnavailable
from swe_agent.tools.search import CodeSearch

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool refused or failed. The message is safe to show to the LLM."""


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class _ToolDef:
    name: str
    description: str
    args_model: type[BaseModel]
    explore: bool  # True = read-only and runs no code
    method: str


_DEFS: dict[str, _ToolDef] = {
    d.name: d
    for d in (
        _ToolDef(
            "list_files",
            "List the source files in the repository. Use 'prefix' to look inside one folder.",
            ListFilesArgs, True, "list_files",
        ),
        _ToolDef(
            "read_file",
            "Read a file with line numbers. Lines look like '  12 | code': the 'NN | ' part "
            "is NOT in the file. Long files come in pages.",
            ReadFileArgs, True, "read_file",
        ),
        _ToolDef(
            "search_code",
            "Search the indexed repository by meaning and keywords. Returns cited code "
            "chunks (file:start-end). Use it first when you do not know where something is.",
            SearchCodeArgs, True, "search_code",
        ),
        _ToolDef(
            "grep",
            "Find exact text (not a regex) in the repository. Returns path:line: text.",
            GrepArgs, True, "grep",
        ),
        _ToolDef(
            "git_diff",
            "Show the current uncommitted changes: changed files, new files and the diff.",
            GitDiffArgs, True, "git_diff",
        ),
        _ToolDef(
            "edit_file",
            "Replace ONE exact piece of text in an existing file. old_str must match the file "
            "exactly (copy it from read_file, without line-number prefixes) and occur once. "
            "Include enough surrounding lines to make it unique.",
            EditFileArgs, False, "edit_file",
        ),
        _ToolDef(
            "create_file",
            "Create a NEW test file, for example tests/test_x.py. Existing files cannot be "
            "overwritten; use edit_file for them.",
            CreateFileArgs, False, "create_file",
        ),
        _ToolDef(
            "run_tests",
            "Run pytest in the sandbox. Example: 'pytest -q' or 'pytest tests/test_x.py -x'. "
            "Returns the exit code and the end of the output.",
            RunTestsArgs, False, "run_tests",
        ),
    )
}

EXPLORE_TOOLS = frozenset(name for name, d in _DEFS.items() if d.explore)


def all_tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            name=d.name,
            description=d.description,
            input_schema=d.args_model.model_json_schema(),
            explore=d.explore,
        )
        for d in _DEFS.values()
    ]


# ------------------------------------------------------------------- helpers
def summarize_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """A log-safe copy of the arguments: secrets hidden, long text replaced by its size."""
    summary: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str):
            text = redact_secrets(value)
            summary[str(key)] = text if len(text) <= 120 else f"<{len(text)} chars>"
        elif isinstance(value, bool | int | float) or value is None:
            summary[str(key)] = value
        else:
            summary[str(key)] = redact_secrets(str(value))[:120]
    return summary


def _short_validation(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors()[:5]:
        where = ".".join(str(p) for p in error["loc"]) or "arguments"
        parts.append(f"{where}: {error['msg']}")
    return "; ".join(parts)


def _tail(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"... [first {len(text) - limit} characters cut]\n{text[-limit:]}"


def _under(path: str, prefix: str) -> bool:
    return not prefix or path == prefix or path.startswith(prefix.rstrip("/") + "/")


# ------------------------------------------------------------------ the tools
class RepoTools:
    def __init__(
        self,
        repo_dir: Path,
        *,
        settings: Settings | None = None,
        searcher: CodeSearch | None = None,
        runner: CommandRunner | None = None,
        allowed_tools: Iterable[str] | None = None,
        write_paths: Iterable[str] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.repo_dir = Path(repo_dir).resolve()
        self._guard = PathGuard(self.repo_dir, max_read_bytes=self._settings.max_file_kb * 1024)
        self._policy = SecurityPolicy(self._guard, self._settings)
        self._searcher = searcher
        self._runner: CommandRunner = runner or DisabledRunner()

        if allowed_tools is None:
            self._allowed = set(_DEFS)
        else:
            self._allowed = set(allowed_tools)
            unknown = self._allowed - set(_DEFS)
            if unknown:
                raise ValueError(f"Unknown tools: {sorted(unknown)}")
        # None = any file the guards allow. A list (even empty) = ONLY those files.
        self._write_paths = (
            None if write_paths is None else {normalize_relative(p) for p in write_paths}
        )

        self._lock = threading.Lock()
        self._scan_cache: ScanResult | None = None
        self.calls: list[ToolCallRecord] = []
        self.modified_files: set[str] = set()  # edited or created
        self.created_files: set[str] = set()

    # ---------------------------------------------------------- public API
    @property
    def changed_files(self) -> list[str]:
        return sorted(self.modified_files)

    def specs(self) -> list[ToolSpec]:
        return [spec for spec in all_tool_specs() if spec.name in self._allowed]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Run one tool. It never raises: failures come back as ToolResult(ok=False)."""
        with self._lock:
            started = time.perf_counter()
            result = self._finish(self._dispatch(name, arguments))
            self._record(name, arguments, result, started)
            return result

    # ------------------------------------------------------------ internals
    def _dispatch(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return ToolResult(ok=False, error="Tool arguments must be a JSON object.")
        definition = _DEFS.get(name)
        if definition is None or name not in self._allowed:
            return ToolResult(ok=False, error=f"Unknown or disabled tool: {name!r}.")
        limit = self._settings.max_tool_calls
        if len(self.calls) >= limit:
            return ToolResult(ok=False, error=f"Tool call limit reached ({limit}). Stop now.")
        try:
            args = definition.args_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult(ok=False, error="Invalid arguments: " + _short_validation(exc))
        try:
            return getattr(self, definition.method)(args)
        except (ToolError, EditError, SecurityError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except OSError as exc:
            logger.warning("tool %s file system error: %s", name, exc)
            return ToolResult(ok=False, error=f"File system error: {type(exc).__name__}")
        except Exception:
            logger.exception("tool %s crashed", name)
            return ToolResult(ok=False, error="Internal tool error. See the logs.")

    def _finish(self, result: ToolResult) -> ToolResult:
        """Last step for every result: hide secrets and cut very long output."""
        limit = self._settings.tool_output_max_chars
        output = redact_secrets(result.output)
        if len(output) > limit:
            output = output[:limit] + f"\n... [truncated {len(output) - limit} characters]"
        return result.model_copy(update={"output": output, "error": redact_secrets(result.error)})

    def _record(
        self, name: str, arguments: dict[str, Any] | None, result: ToolResult, started: float
    ) -> None:
        record = ToolCallRecord(
            seq=len(self.calls) + 1,
            tool=name,
            args=summarize_args(arguments if isinstance(arguments, dict) else {}),
            ok=result.ok,
            error=result.error[:300],
            warnings=len(result.warnings),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            output_chars=len(result.output),
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self.calls.append(record)
        logger.info(
            "tool_call seq=%d tool=%s ok=%s ms=%s chars=%d",
            record.seq, record.tool, record.ok, record.duration_ms, record.output_chars,
        )

    def _scan(self) -> ScanResult:
        if self._scan_cache is None:
            self._scan_cache = scan_repo(self.repo_dir, self._settings)
        return self._scan_cache

    @staticmethod
    def _injection_warnings(rel: str, text: str, first_line: int = 1) -> list[str]:
        report = scan_for_injection(text)
        return [
            f"{rel}:{first_line + f.line - 1}: possible prompt injection ({f.name}, {f.severity})"
            for f in report.findings
            if f.severity in ("high", "medium")
        ][:5]

    def _check_writable(self, raw: str) -> tuple[str, Path]:
        rel, full = self._guard.check(raw, write=True)
        if self._write_paths is not None and rel not in self._write_paths:
            allowed = ", ".join(sorted(self._write_paths)) or "none"
            raise ToolError(f"{rel} is not part of the approved plan. Allowed files: {allowed}")
        return rel, full

    def _check_content(self, rel: str, added_text: str, final_bytes: bytes) -> None:
        if len(final_bytes) > self._settings.max_edit_kb * 1024:
            raise ToolError(f"The file would be larger than {self._settings.max_edit_kb} KB.")
        report = self._policy.check_new_content(rel, added_text)
        if report.blocked:
            reasons = "; ".join(v.message for v in report.violations if v.severity == "block")
            raise ToolError(f"Blocked by security policy: {reasons}")

    def _after_write(self, rel: str, *, created: bool) -> None:
        self.modified_files.add(rel)
        if created:
            self.created_files.add(rel)
        self._scan_cache = None  # the file list may have changed

    # ------------------------------------------------------------ explore tools
    def list_files(self, args: ListFilesArgs) -> ToolResult:
        prefix = normalize_relative(args.prefix) if args.prefix.strip() else ""
        scan = self._scan()
        files = [f for f in scan.files if _under(f.path, prefix)]
        shown = files[: args.max_results]
        header = f"{len(shown)} of {len(files)} files"
        if scan.truncated:
            header += " (the scan hit the file limit)"
        lines = [f"{f.path}  ({f.language}, {f.size_bytes} bytes)" for f in shown]
        return ToolResult(ok=True, output=header + "\n" + "\n".join(lines),
                          meta={"total": len(files)})

    def read_file(self, args: ReadFileArgs) -> ToolResult:
        rel, _full = self._guard.check(args.path)
        lines = split_lines(self._guard.read_text(rel))
        total = len(lines)
        if total == 0:
            return ToolResult(ok=True, output=f"{rel} is empty.", meta={"total_lines": 0})
        start = args.start_line
        if start > total:
            raise ToolError(f"start_line {start} is beyond the end of the file ({total} lines).")
        page = self._settings.read_max_lines
        end = min(args.end_line or start + page - 1, total, start + page - 1)
        if end < start:
            raise ToolError("end_line must be greater than or equal to start_line.")
        width = len(str(end))
        body = "\n".join(f"{n:>{width}} | {lines[n - 1]}" for n in range(start, end + 1))
        header = f"{rel} (lines {start}-{end} of {total})"
        if end < total:
            header += f". More: call read_file with start_line={end + 1}."
        warnings = self._injection_warnings(rel, "\n".join(lines[start - 1 : end]), start)
        return ToolResult(
            ok=True,
            output=f"{header}\n{body}",
            warnings=warnings,
            meta={"path": rel, "start_line": start, "end_line": end, "total_lines": total},
        )

    def search_code(self, args: SearchCodeArgs) -> ToolResult:
        if self._searcher is None:
            raise ToolError("Search is not configured: the repository is not indexed.")
        try:
            result = self._searcher.search(args.query, args.k, args.mode)
        except IndexingError as exc:
            raise ToolError(str(exc)) from exc
        if not result.hits:
            return ToolResult(ok=True, output="No results.", meta={"citations": []})
        labels = [h.chunk.citation.label for h in result.hits]
        stale = [
            h.chunk.citation.label
            for h in result.hits
            if not verify_chunk(self.repo_dir, h.chunk)
        ]
        budget = int(self._settings.tool_output_max_chars * 0.9)
        output = format_context(result.hits, max_chars=budget)
        if stale:
            output += "\n\nSTALE (file changed since indexing, use read_file): " + ", ".join(stale)
        return ToolResult(ok=True, output=output, meta={"citations": labels, "stale": stale})

    def grep(self, args: GrepArgs) -> ToolResult:
        prefix = normalize_relative(args.path_prefix) if args.path_prefix.strip() else ""
        needle = args.pattern.lower() if args.ignore_case else args.pattern
        matches: list[str] = []
        limit_hit = False
        for file in self._scan().files:
            if not _under(file.path, prefix):
                continue
            try:
                text = self._guard.read_text(file.path)
            except SecurityError:
                continue
            for number, line in enumerate(split_lines(text), start=1):
                haystack = line.lower() if args.ignore_case else line
                if needle in haystack:
                    if len(matches) >= args.max_results:
                        limit_hit = True
                        break
                    matches.append(f"{file.path}:{number}: {line.strip()[:160]}")
            if limit_hit:
                break
        header = f"{len(matches)} match(es)."
        if limit_hit:
            header += " More exist; narrow the search."
        return ToolResult(ok=True, output=header + "\n" + "\n".join(matches),
                          meta={"count": len(matches)})

    def git_diff(self, args: GitDiffArgs) -> ToolResult:
        rel = normalize_relative(args.path) if args.path.strip() else ""
        pathspec = ["--", rel] if rel else []
        base = ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-color"]
        diff_argv = validate_command([*base, *(["--stat"] if args.stat_only else []), *pathspec])
        names_argv = validate_command(["git", "diff", "--name-status", *pathspec])
        status_argv = validate_command(["git", "status", "--porcelain"])
        try:
            diff = run_git(diff_argv.argv[1:], cwd=self.repo_dir, timeout=30, action="diff")
            names = run_git(names_argv.argv[1:], cwd=self.repo_dir, timeout=30, action="diff")
            status = run_git(status_argv.argv[1:], cwd=self.repo_dir, timeout=30, action="status")
        except GitError as exc:
            raise ToolError(str(exc)) from exc

        untracked = [
            line[3:] for line in status.splitlines()
            if line.startswith("?? ") and _under(line[3:], rel)
        ]
        parts = [
            "Changed tracked files:\n" + (names or "(none)"),
            "New untracked files:\n" + ("\n".join(untracked) or "(none)"),
        ]
        if diff:
            parts.append(("Diff stat:\n" if args.stat_only else "Diff:\n") + diff)
        return ToolResult(
            ok=True,
            output="\n\n".join(parts),
            meta={"has_changes": bool(names or untracked), "untracked": untracked},
        )

    # ------------------------------------------------------------ change tools
    def edit_file(self, args: EditFileArgs) -> ToolResult:
        rel, full = self._check_writable(args.path)
        if (self.repo_dir / rel).is_symlink():
            raise ToolError("Symlinks are not edited.")
        if not full.is_file():
            raise ToolError(f"{rel} does not exist. Use create_file for new test files.")
        data = full.read_bytes()
        if len(data) > self._settings.max_edit_kb * 1024:
            raise ToolError(f"{rel} is too large to edit.")
        loaded = load_text_strict(data)
        new_text = apply_replace(loaded.text, args.old_str, args.new_str)
        new_bytes = to_bytes(new_text, loaded.newline)
        self._check_content(rel, args.new_str, new_bytes)
        atomic_write(full, new_bytes)
        self._after_write(rel, created=False)
        diff = short_diff(rel, loaded.text, new_text)
        return ToolResult(
            ok=True,
            output=f"Edited {rel} (1 replacement).\n{diff}",
            meta={"path": rel, "newline": "crlf" if loaded.newline == "\r\n" else "lf"},
        )

    def create_file(self, args: CreateFileArgs) -> ToolResult:
        rel, full = self._check_writable(args.path)
        if not is_test_path(rel):
            raise ToolError("New files are only allowed for tests, e.g. tests/test_x.py.")
        if full.exists() or (self.repo_dir / rel).is_symlink():
            raise ToolError(f"{rel} already exists. Use edit_file to change it.")
        content = args.content.replace("\r\n", "\n").replace("\r", "\n")
        if not content.endswith("\n"):
            content += "\n"
        data = content.encode("utf-8")
        self._check_content(rel, content, data)
        atomic_write(full, data)
        self._after_write(rel, created=True)
        return ToolResult(
            ok=True,
            output=f"Created {rel} ({len(content.splitlines())} lines).",
            meta={"path": rel},
        )

    def run_tests(self, args: RunTestsArgs) -> ToolResult:
        allowed = validate_command(args.command)
        if allowed.kind != "test":
            raise ToolError("run_tests only runs pytest. Use git_diff to look at changes.")
        cap = self._settings.command_timeout_seconds
        timeout = min(args.timeout_seconds or cap, cap)  # the LLM can never raise the cap
        try:
            result = self._runner.run(allowed.argv, cwd=self.repo_dir, timeout=timeout)
        except RunnerUnavailable as exc:
            raise ToolError(str(exc)) from exc

        budget = max(200, self._settings.tool_output_max_chars // 2 - 150)
        parts = [
            f"exit_code={result.exit_code} passed={result.passed} "
            f"timed_out={result.timed_out} duration_ms={result.duration_ms}"
        ]
        if result.stdout.strip():
            parts.append("--- stdout (end) ---\n" + _tail(result.stdout, budget))
        if result.stderr.strip():
            parts.append("--- stderr (end) ---\n" + _tail(result.stderr, budget))
        # ok=True means "the tool ran". Failing tests are information, not a tool error.
        return ToolResult(
            ok=True,
            output="\n".join(parts),
            meta={
                "exit_code": result.exit_code,
                "passed": result.passed,
                "timed_out": result.timed_out,
                "argv": result.argv,
            },
        )