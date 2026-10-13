#!/usr/bin/env python3
"""Exercise the installed wheel's public entry points without the source tree."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

from agentdescent.integrations import (
    codex_config_block, hooks_text, install, opencode_mcp_entry, skill_text,
)


MCP_SESSION_TIMEOUT_SECONDS = 45


def mcp_tool_result_is_error(result) -> bool:
    """Read the error field exposed by MCP Python SDK v1 and v2."""
    for name in ("is_error", "isError"):
        value = getattr(result, name, None)
        if value is not None:
            return bool(value)
    raise AssertionError(
        "MCP CallToolResult exposes neither `is_error` nor `isError`"
    )


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
        launcher = [os.path.abspath(sys.executable), "-m", "agentdescent.cli"]
        assert installed_hooks.read_text(encoding="utf-8") == hooks_text(launcher), (
            "installed wheel did not provide the hooks.json resource"
        )
        assert "SessionStart" in json.loads(hooks_text())["hooks"], (
            "installed hooks.json resource is not valid host configuration"
        )

        env = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        env["AGENTDESCENT_HOME"] = str(workdir / "home")
        env["HOME"] = str(host_home)
        env["USERPROFILE"] = str(host_home)
        env["CODEX_HOME"] = str(host_home / ".codex")
        env["DSH_HOME"] = str(host_home / ".dsh")
        env["XDG_CONFIG_HOME"] = str(host_home / ".config")
        # Model a host started after the installer exits: no pip scripts on
        # PATH, no source checkout, and no ambient Python import overrides.
        host_env = dict(env, PATH=str(workdir / "empty-path"))
        (workdir / "empty-path").mkdir()
        # Reinstall must repair an untouched launcher pinned by an earlier
        # install, even when that Python environment no longer exists. Launch
        # the repaired manifests below to test the real installed CLI, not just
        # string replacement in the source checkout.
        obsolete = [str(workdir / "removed-env" / "python"), "-m", "agentdescent.cli"]
        codex_config = host_home / ".codex" / "config.toml"
        codex_config.parent.mkdir(parents=True, exist_ok=True)
        codex_config.write_text(codex_config_block(obsolete), encoding="utf-8")
        opencode_config = host_home / ".config" / "opencode" / "opencode.jsonc"
        opencode_config.parent.mkdir(parents=True, exist_ok=True)
        opencode_config.write_text(json.dumps({
            "mcp": {"agentdescent": opencode_mcp_entry(obsolete)},
        }), encoding="utf-8")
        manifests = {}
        for host in ("claude-code", "codex", "dsh", "opencode"):
            configured = subprocess.run(
                [*launcher, "install", host, "--home", str(host_home)],
                cwd=workdir, env=host_env,
                text=True, capture_output=True, timeout=30,
            )
            assert configured.returncode == 0, configured.stdout + configured.stderr
        manifests["claude-code"] = json.loads(
            (plugin / ".mcp.json").read_text())["mcpServers"]["agentdescent"]
        oc = json.loads((host_home / ".config/opencode/opencode.jsonc").read_text())
        argv = oc["mcp"]["agentdescent"]["command"]
        manifests["opencode"] = {"command": argv[0], "args": argv[1:]}
        # These fields are JSON-quoted strings/arrays in our generated TOML
        # and YAML. Parse without adding dependencies to the wheel smoke gate.
        for host, path, separator in (
            ("codex", host_home / ".codex/config.toml", "="),
            ("dsh", host_home / ".dsh/cordis.patch.yml", ":"),
        ):
            text = path.read_text()
            manifests[host] = {
                field: json.loads(re.search(
                    rf"^\s*{field}\s*{separator}\s*(.+)$", text, re.M).group(1))
                for field in ("command", "args")
            }
        for host, manifest in manifests.items():
            assert [manifest["command"], *manifest["args"]] == [*launcher, "mcp"]
            launched = subprocess.run(
                [manifest["command"], *manifest["args"][:-1], "--help"],
                cwd=workdir, env=host_env, text=True, capture_output=True, timeout=30,
            )
            assert launched.returncode == 0, (host, launched.stdout, launched.stderr)
        if os.name != "nt":
            for path in (installed_hooks, host_home / ".dsh/skills/agentdescent/hooks.json"):
                hook = json.loads(path.read_text())["hooks"]["SessionStart"][0]["hooks"][0]
                # Drop the intentional best-effort shell suffix so a missing
                # executable cannot be hidden by `|| true` in this regression.
                command = hook["command"].removesuffix(" 2>/dev/null || true")
                status = subprocess.run(command, shell=True, cwd=workdir, env=host_env,
                                        text=True, capture_output=True, timeout=30)
                assert status.returncode == 0, status.stdout + status.stderr

        policy_init = subprocess.run(
            ["agentdescent", "init", "selection", "--kind", "policy_slot"],
            cwd=workdir, env=env, text=True, capture_output=True, timeout=30,
        )
        assert policy_init.returncode == 0, (
            f"installed policy-slot init failed ({policy_init.returncode})\n"
            f"{policy_init.stdout}\n{policy_init.stderr}"
        )
        policy_spec = json.loads(
            (workdir / ".agentdescent" / "selection.evolve.json").read_text(encoding="utf-8")
        )
        assert policy_spec["target"] == "selection", policy_spec
        assert policy_spec["data"] == {"problems": "mypkg.problems:build", "seeds": [0]}, (
            policy_spec
        )
        assert "mypkg.problems:build" in policy_init.stdout, policy_init.stdout
        assert "allow list" in policy_init.stdout, policy_init.stdout
        assert "JSON object per line" not in policy_init.stdout, policy_init.stdout

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

            async def check_server(manifest) -> None:
                params = StdioServerParameters(
                    command=manifest["command"],
                    args=manifest["args"],
                    env=host_env,
                )
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        tools = {tool.name for tool in (await session.list_tools()).tools}
                        assert {"doctor", "plan", "start", "status", "show", "apply"} <= tools, (
                            f"MCP server returned an incomplete tool set: {sorted(tools)}"
                        )
                        result = await session.call_tool("doctor", {})
                        assert not mcp_tool_result_is_error(result), (
                            f"MCP doctor failed: {result}"
                        )

            for manifest in manifests.values():
                asyncio.run(run_mcp_check(lambda: check_server(manifest)))


if __name__ == "__main__":
    main()
