"""``install dsh`` and the native dsh plugin forward Codex's list (#415).

dsh removes every ambient variable matching ``KEY|PASSWORD|SECRET|TOKEN`` before
starting an MCP server, so a key that is not *declared* never arrives. The dsh
paths used to declare four names and left ``ARK_API_KEY`` -- the key the docs'
own Ark examples use -- scrubbed, while Codex had been fixed for it (#404/#407).
There is now one tuple, ``FORWARDED_ENV``.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from agentdescent import cli
from agentdescent.integrations import (
    DSH_BLOCK_END, DSH_BLOCK_START, FORWARDED_ENV, _dsh_env_expression, codex_config_block,
    dsh_patch_block, dsh_plugin_patch, host_status, install, mcp_manifest, render_dsh_plugin,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: What dsh strips from an MCP server's environment unless it is declared.
_DSH_SCRUBBED = re.compile(r"KEY|PASSWORD|SECRET|TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)


def _names(text: str):
    """The variable names an expression / env block forwards, in order."""
    return re.findall(r"(\w+): process\.env\.\1", text)


#: What `install dsh` wrote before #415: the same block with the old four names.
_OLD_FOUR = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_API_KEY")


def _pre_fix_block() -> str:
    old_expr = _dsh_env_expression().replace(
        ", ".join(f"{k}: process.env.{k}" for k in FORWARDED_ENV),
        ", ".join(f"{k}: process.env.{k}" for k in _OLD_FOUR))
    assert old_expr != _dsh_env_expression()
    return dsh_patch_block().replace(_dsh_env_expression(), old_expr)


def _old_patch_file(extra_before: str = "", extra_after: str = "") -> str:
    return (f"{extra_before}{DSH_BLOCK_START}\n{_pre_fix_block().rstrip()}\n"
            f"{DSH_BLOCK_END}\n{extra_after}")


def test_every_name_dsh_would_scrub_is_forwarded_and_ark_is_one_of_them():
    assert "ARK_API_KEY" in FORWARDED_ENV
    expr_names = _names(_dsh_env_expression())
    assert expr_names == list(FORWARDED_ENV)
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ARK_API_KEY"):
        assert _DSH_SCRUBBED.search(name) and name in expr_names


def test_install_dsh_patch_forwards_what_codex_forwards(tmp_path):
    install("dsh", home=str(tmp_path))
    patch = (tmp_path / ".dsh" / "cordis.patch.yml").read_text()
    assert "ARK_API_KEY: process.env.ARK_API_KEY" in patch
    dsh_names = _names(patch)
    # Codex: the `env_vars` line of its block and the plugin's `.mcp.json`
    codex = json.loads(re.search(r"^env_vars = (\[.*\])$", codex_config_block(), re.M).group(1))
    assert dsh_names == codex == list(FORWARDED_ENV)
    assert mcp_manifest()["mcpServers"]["agentdescent"].get("env_vars", codex) == codex


def test_native_dsh_plugin_and_checked_in_bundle_forward_the_same_list(tmp_path):
    assert _names(dsh_plugin_patch()) == list(FORWARDED_ENV)
    render_dsh_plugin(str(tmp_path))
    rendered = (tmp_path / "cordis.patch.yml").read_text()
    checked_in = open(os.path.join(ROOT, "integrations", "dsh-agentdescent",
                                   "cordis.patch.yml"), encoding="utf-8").read()
    assert rendered == checked_in
    assert "ARK_API_KEY: process.env.ARK_API_KEY" in checked_in
    assert _names(checked_in) == list(FORWARDED_ENV)


def test_install_dsh_dry_run_prints_the_full_list(tmp_path):
    lines = install("dsh", home=str(tmp_path), dry_run=True)
    note = next(line for line in lines if "forwarded into the MCP server env" in line)
    for name in FORWARDED_ENV:
        assert name in note
    assert not (tmp_path / ".dsh").exists()


def test_install_over_a_pre_fix_block_rewrites_it_and_keeps_the_rest(tmp_path):
    path = tmp_path / ".dsh" / "cordis.patch.yml"
    path.parent.mkdir(parents=True)
    path.write_text(_old_patch_file("# mine\n- insert:\n    - id: other\n      name: x\n",
                                    "# also mine\n"))
    assert "ARK_API_KEY" not in path.read_text()
    lines = install("dsh", home=str(tmp_path))
    text = path.read_text()
    assert any("rewrote" in line for line in lines), lines
    assert "ARK_API_KEY: process.env.ARK_API_KEY" in text
    assert text.startswith("# mine\n") and text.rstrip().endswith("# also mine")
    assert text.count(DSH_BLOCK_START) == 1
    again = install("dsh", home=str(tmp_path))
    assert path.read_text() == text and any(line.startswith("kept ") for line in again)


def test_host_status_flags_a_pre_fix_block_as_stale_until_reinstalled(tmp_path):
    install("dsh", home=str(tmp_path))
    assert host_status("dsh", home=str(tmp_path))["state"] == "ok"
    path = tmp_path / ".dsh" / "cordis.patch.yml"
    path.write_text(_old_patch_file())
    status = host_status("dsh", home=str(tmp_path))
    assert status["state"] == "stale" and status["mcp"] == "ok" and status["skill"] == "ok"
    install("dsh", home=str(tmp_path))
    assert host_status("dsh", home=str(tmp_path))["state"] == "ok"


def test_a_pre_fix_block_without_an_end_marker_is_still_flagged_and_repaired(tmp_path):
    install("dsh", home=str(tmp_path))
    path = tmp_path / ".dsh" / "cordis.patch.yml"
    path.write_text(f"{DSH_BLOCK_START}\n{_pre_fix_block()}")
    assert host_status("dsh", home=str(tmp_path))["state"] in ("stale", "partial")
    install("dsh", home=str(tmp_path))
    assert host_status("dsh", home=str(tmp_path))["state"] == "ok"


def test_doctor_reports_the_stale_dsh_block(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install("dsh", home=str(tmp_path))
    (tmp_path / ".dsh" / "cordis.patch.yml").write_text(_old_patch_file())
    rep = cli.doctor_report()
    assert rep["hosts"]["dsh"]["state"] == "stale"
    assert any(p.startswith("dsh: the installed integration is stale") for p in rep["problems"])


def test_a_hand_added_custom_key_is_not_stale(tmp_path):
    """The documented fix for a custom `api_key_env` must not trip `doctor`."""
    install("dsh", home=str(tmp_path))
    path = tmp_path / ".dsh" / "cordis.patch.yml"
    text = path.read_text()
    assert "ARK_API_KEY: process.env.ARK_API_KEY, " in text
    path.write_text(text.replace("ARK_API_KEY: process.env.ARK_API_KEY, ",
                                 "ARK_API_KEY: process.env.ARK_API_KEY, "
                                 "MY_PROVIDER_KEY: process.env.MY_PROVIDER_KEY, "))
    assert "MY_PROVIDER_KEY" in path.read_text()
    assert host_status("dsh", home=str(tmp_path))["state"] == "ok"
