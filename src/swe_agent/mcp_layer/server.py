"""MCP server: exposes RepoTools over stdio.

Run by hand:  python -m swe_agent.mcp_layer.server <run_id> [--explore-only]
              [--write-path=src/a.py ...]

IMPORTANT: stdout belongs to the MCP protocol. Never print() in this file.
Logs go to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from swe_agent.config import get_settings
from swe_agent.github.workspace import WorkspaceManager
from swe_agent.logging_setup import setup_logging
from swe_agent.tools import EXPLORE_TOOLS, CodeSearcher, RepoTools

logger = logging.getLogger(__name__)


class ToolFailed(Exception):
    """Raised so the SDK sends the result back with isError=True."""


def build_server(tools: RepoTools) -> Server:
    server: Server = Server("swe-agent-repo-tools")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=s.name, description=s.description, inputSchema=s.input_schema)
            for s in tools.specs()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        # Our tools are plain blocking code, so we run them in a worker thread.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(tools.call, name, arguments or {}), timeout=150
            )
        except TimeoutError:
            raise ToolFailed("The tool timed out on the server.") from None
        if not result.ok:
            raise ToolFailed(result.error)
        text = result.output
        if result.warnings:
            text += "\n\nSECURITY WARNINGS:\n" + "\n".join(f"- {w}" for w in result.warnings)
        return [types.TextContent(type="text", text=text)]
    return server


async def serve(tools: RepoTools) -> None:
    server = build_server(tools)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="swe-agent-mcp")
    parser.add_argument("run_id")
    parser.add_argument("--explore-only", action="store_true", help="expose only read tools")
    parser.add_argument(
        "--write-path",
        action="append",
        default=[],
        help="a file the agent may change (repeat). With none, nothing can be changed.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.models_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must be set before the models load
    setup_logging(settings.log_level, stream=sys.stderr)
    workspace = WorkspaceManager(settings).get(args.run_id)
    searcher = CodeSearcher(workspace.run_id, workspace.root, settings)
    try:
        searcher.warm_up()
    except Exception as exc:  # noqa: BLE001
        logger.warning("search warm-up failed (search_code may not work): %s", exc)
    tools = RepoTools(
        workspace.repo_dir,
        settings=settings,
        searcher=searcher,
        allowed_tools=EXPLORE_TOOLS if args.explore_only else None,
        write_paths=args.write_path,  # fail closed: empty list = no edits
    )
    logger.info("mcp server ready run_id=%s tools=%d", args.run_id, len(tools.specs()))
    asyncio.run(serve(tools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())