#!/usr/bin/env python3
"""Exercise the installed wheel's public entry points without the source tree."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from agentdescent.integrations import hooks_text, install, skill_text


MCP_SESSION_TIMEOUT_SECONDS = 45


async def run_mcp_check(check_server, *, timeout=MCP_SESSION_TIMEOUT_SECONDS) -> None:
    """Bound the complete MCP initialize/list/call sequence for CI."""
    try:
        await asyncio.wait_for(check_server(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"MCP stdio smoke session exceeded {timeout:g} seconds"
        ) from exc


def main() -> None:
    import agentdescent

    installed_root = Path(sys.prefix).resolve()
    package_path = Path(agentdescent.__file__).resolve()
    assert package_path.is_relative_to(installed_root), (
        f"agentdescent imported from outside the clean environment: {package_path}"
    )
    version = importlib.metadata.version("agentdescent")
    assert agentdescent.__version__ == version, (
        f"imported version {agentdescent.__version__!r} != metadata {version!r}"
    )

    with tempfile.TemporaryDirectory(prefix="agentdescent-wheel-smoke-") as temp:
        workdir = Path(temp)
        host_home = workdir / "host-home"
        install("claude-code", home=str(host_home))
        plugin = host_home / ".agentdescent" / "plugins" / "claude-code"
        installed_skill = plugin / "skills" / "agentdescent" / "SKILL.md"
        installed_hooks = plugin / "hooks" / "hooks.json"
        assert installed_skill.read_text(encoding="utf-8") == skill_text(), (
            "installed wheel did not provide the shared SKILL.md resource"
        )
        assert installed_hooks.read_text(encoding="utf-8") == hooks_text(), (
            "installed wheel did not provide the hooks.json resource"
        )
        assert "SessionStart" in json.loads(hooks_text())["hooks"], (
            "installed hooks.json resource is not valid host configuration"
        )

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["AGENTDESCENT_HOME"] = str(workdir / "home")

        demo = subprocess.run(
            ["agentdescent", "demo"], cwd=workdir, env=env,
            text=True, capture_output=True, timeout=120,
        )
        assert demo.returncode == 0, (
            f"offline CLI demo failed ({demo.returncode})\n{demo.stdout}\n{demo.stderr}"
        )
        assert "held-out reward" in demo.stdout and "what it learned" in demo.stdout, (
            f"offline CLI demo did not complete its learning example:\n{demo.stdout}"
        )

        if sys.version_info < (3, 10):
            unavailable = subprocess.run(
                ["agentdescent", "mcp"], cwd=workdir, env=env,
                text=True, capture_output=True, timeout=30,
            )
            assert unavailable.returncode == 3, (
                f"MCP should be unavailable on Python <3.10, got {unavailable.returncode}"
            )
            assert "Python >= 3.10" in unavailable.stderr, unavailable.stderr
        else:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def check_server() -> None:
                command = shutil.which("agentdescent")
                assert command, "installed agentdescent console script is missing"
                params = StdioServerParameters(
                    command=command,
                    args=["mcp"],
                    env=env,
                )
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        tools = {tool.name for tool in (await session.list_tools()).tools}
                        assert {"doctor", "plan", "start", "status", "show", "apply"} <= tools, (
                            f"MCP server returned an incomplete tool set: {sorted(tools)}"
                        )
                        result = await session.call_tool("doctor", {})
                        assert not result.is_error, f"MCP doctor failed: {result}"

            asyncio.run(run_mcp_check(check_server))


if __name__ == "__main__":
    main()
