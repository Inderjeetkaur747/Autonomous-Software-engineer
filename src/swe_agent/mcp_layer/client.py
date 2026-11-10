"""MCP client: starts our MCP server as a child process and calls its tools."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from swe_agent.schemas.tools import ToolCallRecord, ToolResult, ToolSpec
from swe_agent.tools import EXPLORE_TOOLS, summarize_args

logger = logging.getLogger(__name__)

_RUN_ID_RE = re.compile(r"[a-f0-9]{12}")


class McpClientError(Exception):
    """The MCP connection is not usable."""


class McpToolClient:
    """Use it as:  async with McpToolClient(run_id) as client: ..."""

    def __init__(
        self,
        run_id: str,
        *,
        explore_only: bool = False,
        write_paths: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        call_timeout_seconds: float = 180.0,
    ) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):  # also stops option injection like "--x"
            raise ValueError(f"Invalid run id: {run_id!r}")
        args = ["-m", "swe_agent.mcp_layer.server", run_id]
        if explore_only:
            args.append("--explore-only")
        for path in write_paths or []:
            args.append(f"--write-path={path}")  # '=' form: a value can never become an option
        self._params = StdioServerParameters(command=sys.executable, args=args, env=extra_env)
        self._timeout = call_timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self.calls: list[ToolCallRecord] = []

    async def __aenter__(self) -> McpToolClient:
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(self._params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=180)
        except BaseException:
            await self._stack.aclose()
            raise
        self._session = session
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise McpClientError("Use the client inside 'async with'.")
        return self._session

    async def list_tools(self) -> list[ToolSpec]:
        response = await self._require_session().list_tools()
        return [
            ToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                explore=tool.name in EXPLORE_TOOLS,
            )
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        session = self._require_session()
        started = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                session.call_tool(name, arguments or {}), timeout=self._timeout
            )
            text = "\n".join(
                block.text for block in raw.content if getattr(block, "type", "") == "text"
            )
            if raw.isError:
                result = ToolResult(ok=False, error=text or "The tool failed.")
            else:
                result = ToolResult(ok=True, output=text)
        except TimeoutError:
            result = ToolResult(ok=False, error=f"MCP call timed out after {self._timeout}s.")
        except Exception as exc:
            logger.warning("mcp call failed: %s", exc)
            result = ToolResult(ok=False, error=f"MCP error: {exc}")

        self.calls.append(
            ToolCallRecord(
                seq=len(self.calls) + 1,
                tool=name,
                args=summarize_args(arguments),
                ok=result.ok,
                error=result.error[:300],
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                output_chars=len(result.output),
                timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        return result