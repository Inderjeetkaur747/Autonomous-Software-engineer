import asyncio

import pytest

pytest.importorskip("mcp")

from swe_agent.mcp_layer import McpToolClient  # noqa: E402

RUN_ID = "a" * 12


def make_workspace(tmp_path):
    ws = tmp_path / "ws"
    repo = ws / RUN_ID / "repo"
    (repo / "mathkit").mkdir(parents=True)
    (repo / "mathkit" / "stats.py").write_bytes(b"def average(n):\n    return sum(n) / len(n)\n")
    return ws


def test_real_server_round_trip_over_stdio(tmp_path):
    env = {"WORKSPACE_DIR": str(make_workspace(tmp_path))}

    async def scenario():
        async with McpToolClient(RUN_ID, explore_only=True, extra_env=env) as client:
            specs = await client.list_tools()
            read = await client.call_tool("read_file", {"path": "mathkit/stats.py"})
            traversal = await client.call_tool("read_file", {"path": "../x.py"})
            edit = await client.call_tool(
                "edit_file", {"path": "mathkit/stats.py", "old_str": "sum", "new_str": "min"}
            )
            return specs, read, traversal, edit, client.calls

    specs, read, traversal, edit, calls = asyncio.run(asyncio.wait_for(scenario(), 90))

    names = {s.name for s in specs}
    assert names == {"list_files", "read_file", "search_code", "grep", "git_diff"}
    assert read.ok
    assert "def average" in read.output
    assert not traversal.ok
    assert "not allowed" in traversal.error
    assert not edit.ok  # edit_file is not even offered in explore-only mode
    assert "disabled" in edit.error.lower()
    assert [c.tool for c in calls] == ["read_file", "read_file", "edit_file"]


def test_client_rejects_unsafe_run_ids():
    for bad in ["--help", "../x", "ABC", "a" * 11]:
        with pytest.raises(ValueError):
            McpToolClient(bad)