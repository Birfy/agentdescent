"""The `plugin` kind, offline: a stub host CLI stands in for dsh / claude / codex.

The stub is a Python program that behaves like a host loading a plugin from a
path: it reads the plugin directory it was pointed at (via an argument, like
Claude Code's --plugin-dir, or via $HOME, like dsh) and answers according to
what the plugin says. So the test exercises exactly what `plugin_runner` adds
over `code_runner`: the layout, the isolated host home, the gate, the nested
marker, and the frozen list.
"""

import json
import os
import sys

import pytest

from agentdescent import EvolveSpec, SpecError, Task, compose, plugin_runner
from agentdescent.agents import NESTED_MARKER, cli_agent, worker_env
from agentdescent.runners import (
    PLUGIN_CONTEXT, PLUGIN_FROZEN, PLUGIN_HOSTS, TEST_FAILURE_MARKER, PluginHost,
)
from agentdescent.treestrategy import FileTree

# A host that takes the plugin path on its command line (Claude Code shape) and
# proves it ran with HOME inside the workspace and the nested marker set.
_HOST_SRC = r"""
import json, os, sys
args = sys.argv[1:]
plugin = args[args.index("--plugin-dir") + 1]
task = args[-1]
manifest = json.load(open(os.path.join(plugin, "plugin.json")))
assert os.environ.get("AGENTDESCENT_NESTED") == "1", "nested marker missing"
assert os.environ["HOME"] == os.getcwd(), "HOME is not the workspace"
assert "CLAUDECODE" not in os.environ
print(task[::-1] if manifest.get("mode") == "reverse" else task)
"""

_VALIDATE_SRC = r"""
import json, sys
m = json.load(open(sys.argv[1] + "/plugin.json"))
if "name" not in m:
    print("plugin.json has no name", file=sys.stderr); sys.exit(1)
"""


def _stub_host(tmp_path):
    host = tmp_path / "host.py"
    host.write_text(_HOST_SRC)
    val = tmp_path / "validate.py"
    val.write_text(_VALIDATE_SRC)
    return PluginHost(
        "stub",
        entrypoint=[sys.executable, str(host), "--plugin-dir", "{plugin_dir}"],
        validate=[sys.executable, str(val), "{plugin_dir}"],
    )


def _plugin_dir(tmp_path, mode="forward"):
    d = tmp_path / "my-plugin"
    d.mkdir()
    (d / "plugin.json").write_text(json.dumps({"name": "my-plugin", "mode": mode}))
    (d / "hooks").mkdir()
    (d / "hooks" / "hooks.json").write_text('{"PreToolUse": []}')
    return str(d)


def _tree(path):
    from agentdescent.filetree import load_tree
    return load_tree(path)


# ---------------------------------------------------------------------------
# worker_env
# ---------------------------------------------------------------------------


def test_worker_env_drops_session_markers_and_points_homes_inside(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("DSH_HOME", "/real/home/.dsh")
    monkeypatch.setenv("CODEX_SANDBOX", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    env = worker_env(str(tmp_path))
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CODEX_SANDBOX"):
        assert k not in env
    assert env["OPENAI_API_KEY"] == "k"                # keys stay: the worker needs them
    assert env[NESTED_MARKER] == "1"
    assert env["DSH_HOME"].startswith(str(tmp_path))  # not the real one
    assert env["CLAUDE_CONFIG_DIR"].startswith(str(tmp_path))
    assert env["CODEX_HOME"].startswith(str(tmp_path))
    assert worker_env(str(tmp_path), {"DSH_HOME": "/x"})["DSH_HOME"] == "/x"   # extra wins
    loose = worker_env(str(tmp_path), isolate=False)
    assert loose["DSH_HOME"] == "/real/home/.dsh" and loose[NESTED_MARKER] == "1"


def test_cli_agent_runs_with_the_worker_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDECODE", "1")
    agent = cli_agent([sys.executable, "-c",
                       "import os,sys;print(os.environ.get('CLAUDECODE','none'),"
                       "os.environ['AGENTDESCENT_NESTED'],os.environ['DSH_HOME'])"])
    out = agent.in_workspace(str(tmp_path))("q")
    none, nested, dsh_home = out.split()
    assert none == "none" and nested == "1" and dsh_home.startswith(str(tmp_path))
    assert os.path.isdir(dsh_home)


# ---------------------------------------------------------------------------
# plugin_runner
# ---------------------------------------------------------------------------


def test_plugin_runner_loads_the_candidate_into_an_isolated_host(tmp_path):
    host = _stub_host(tmp_path)
    tree = _tree(_plugin_dir(tmp_path))
    strategy = FileTree(tree, frozen=["hooks/**"])
    run = plugin_runner(host, name="my-plugin", overlay=strategy.frozen_files(tree))
    task = Task(id="t", prompt="alpha", meta={"gold": "ahpla"})
    assert run(strategy.render(tree), task) == "alpha"
    evolved = dict(tree, **{"plugin.json": json.dumps({"name": "my-plugin", "mode": "reverse"})})
    assert run(strategy.render(evolved), task) == "ahpla"


def test_a_plugin_that_fails_validation_scores_zero_in_band(tmp_path):
    host = _stub_host(tmp_path)
    tree = _tree(_plugin_dir(tmp_path))
    strategy = FileTree(tree)
    run = plugin_runner(host, name="my-plugin")
    broken = dict(tree, **{"plugin.json": json.dumps({"mode": "reverse"})})   # no name
    out = run(strategy.render(broken), Task(id="t", prompt="alpha"))
    assert out.startswith(TEST_FAILURE_MARKER) and "no name" in out


def test_frozen_hooks_survive_a_candidate_that_rewrites_them(tmp_path):
    host = PluginHost("stub", entrypoint=[
        sys.executable, "-c", "import sys;print(open(sys.argv[1]+'/hooks/hooks.json').read())",
        "{plugin_dir}"])
    tree = _tree(_plugin_dir(tmp_path))
    strategy = FileTree(tree, frozen=list(PLUGIN_FROZEN["claude_code"]))
    run = plugin_runner(host, name="my-plugin", overlay=strategy.frozen_files(tree))
    tampered = dict(tree, **{"hooks/hooks.json": '{"PreToolUse": "REMOVED"}'})
    out = run(strategy.render(tampered), Task(id="t", prompt="x"))
    assert out == '{"PreToolUse": []}'                 # the overlay put the pristine hook back
    # and the strategy refuses the proposal in the first place
    assert not strategy.writable("hooks/hooks.json")


def test_env_passthrough_forwards_named_variables_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "secret")
    monkeypatch.setenv("OTHER_SECRET", "no")
    host = PluginHost("stub", entrypoint=[
        sys.executable, "-c",
        "import os;print(os.environ.get('MY_PROVIDER_KEY'), os.environ.get('OTHER_SECRET'))"])
    tree = _tree(_plugin_dir(tmp_path))
    run = plugin_runner(host, name="p", env_passthrough=["MY_PROVIDER_KEY"])
    assert run(FileTree(tree).render(tree), Task(id="t", prompt="x")) == "secret None"


def test_the_host_table_is_complete():
    assert set(PLUGIN_HOSTS) == set(PLUGIN_FROZEN) == set(PLUGIN_CONTEXT) == {
        "dsh", "claude_code", "codex", "opencode"}
    for name, host in PLUGIN_HOSTS.items():
        assert host.entrypoint[0] in ("dsh", "claude", "codex", "opencode")
        rendered = host.render("plugin/x", "x")
        assert not any("{plugin_dir}" in c for c in rendered.entrypoint)
    assert "hooks/**" in PLUGIN_FROZEN["claude_code"]
    assert "hooks.json" in PLUGIN_FROZEN["dsh"] and "pnpm-lock.yaml" in PLUGIN_FROZEN["dsh"]
    assert "--plugin-dir" in PLUGIN_HOSTS["claude_code"].entrypoint
    assert "--strict-mcp-config" in PLUGIN_HOSTS["claude_code"].entrypoint
    assert PLUGIN_HOSTS["dsh"].entrypoint == ["dsh", "--profile", "headless"]
    assert PLUGIN_HOSTS["opencode"].entrypoint == ["opencode", "run"]
    # `--full-auto` does not exist in codex-cli 0.153; these are the real flags
    codex = PLUGIN_HOSTS["codex"].entrypoint
    assert "--full-auto" not in codex
    assert codex[:2] == ["codex", "exec"] and "--skip-git-repo-check" in codex
    assert "workspace-write" in codex


# ---------------------------------------------------------------------------
# through the spec
# ---------------------------------------------------------------------------


def test_plugin_spec_composes_at_the_harness_layer_with_frozen_hooks(tmp_path, monkeypatch):
    from agentdescent import runners
    from agentdescent.governance import HARNESS_BLAST_RADIUS

    monkeypatch.setitem(runners.PLUGIN_HOSTS, "stub", _stub_host(tmp_path))
    monkeypatch.setitem(runners.PLUGIN_FROZEN, "stub", ("hooks/**",))
    monkeypatch.setitem(runners.PLUGIN_CONTEXT, "stub", ("plugin.json",))
    rows = [{"prompt": w, "gold": w[::-1]} for w in ["alpha", "bravo", "charlie", "delta",
                                                     "echo", "foxtrot", "golf", "hotel"]]
    spec = EvolveSpec(kind="plugin", host="stub", target=_plugin_dir(tmp_path),
                      data={"inline": rows}, score="exact", agent="echo",
                      frozen=["README.md"], evolve={"rounds": 2, "n_workers": 2})
    comp = compose(spec)
    k = comp.kwargs
    assert k["blast_radius"] == HARNESS_BLAST_RADIUS
    assert list(k["strategy"].frozen) == ["hooks/**", "README.md"]
    assert k["self_verify"] is False and k["cheap_eval_tasks"] == 4
    assert any("container" in n for n in comp.notes)
    t = Task(id="t", prompt="alpha", meta={"gold": "ahpla"})
    assert comp.reward(t, TEST_FAILURE_MARKER + " x") == 0.0 and comp.reward(t, "ahpla") == 1.0
    assert k["run"](k["strategy"].render(comp.tree), t) == "alpha"   # the stub host really ran


def test_plugin_spec_needs_a_known_host(tmp_path):
    spec = EvolveSpec(kind="plugin", target=_plugin_dir(tmp_path), agent="echo",
                      data={"inline": [{"prompt": "q", "gold": "q"}] * 4})
    with pytest.raises(SpecError, match="host"):
        compose(spec)
