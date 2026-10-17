"""`doctor` says whether each host is wired up and whether it is still current (#382).

Before this, `doctor` reported CLIs, keys and optional pieces and nothing about
the integration: a host CLI that `install` had never touched, and a `SKILL.md`
copied by an older package version, were both invisible.
"""

import io
import json
import os
from contextlib import redirect_stdout

import pytest

from agentdescent import cli
from agentdescent.integrations import HOSTS, host_status, install, skill_text

HOST_NAMES = sorted(HOSTS)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home, with every variable `install` honours pointed back at it."""
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)
    return tmp_path


def _report(monkeypatch, *, present=()):
    """`doctor_report()`, with only the named host binaries on PATH."""
    import shutil

    real = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: (
        f"/fake/bin/{name}" if name in present else real(name, *a, **k)
        if name in ("git", "node", "pnpm") else None))
    return cli.doctor_report()


def _skill_path(home, host):
    return host_status(host, home=str(home))["path"]


def _root(home, host):
    """`<root>/<skills|skill>/agentdescent/SKILL.md` -> `<root>`."""
    path = _skill_path(home, host)
    for _ in range(3):
        path = os.path.dirname(path)
    return path


@pytest.mark.parametrize("host", HOST_NAMES)
def test_a_host_nobody_installed_into_is_missing(home, host):
    info = host_status(host, home=str(home))
    assert (info["state"], info["skill"], info["mcp"]) == ("missing", "missing", "missing")


@pytest.mark.parametrize("host", HOST_NAMES)
def test_install_makes_the_host_ok(home, host):
    install(host, home=str(home))
    info = host_status(host, home=str(home))
    assert (info["state"], info["skill"], info["mcp"]) == ("ok", "ok", "ok"), info


@pytest.mark.parametrize("host", HOST_NAMES)
def test_a_skill_from_an_older_package_is_stale(home, host):
    install(host, home=str(home))
    path = _skill_path(home, host)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(skill_text().replace("doctor", "doctr", 1))
    assert host_status(host, home=str(home))["state"] == "stale"
    install(host, home=str(home))                 # what the problem tells you to run
    assert host_status(host, home=str(home))["state"] == "ok"


@pytest.mark.parametrize("host", HOST_NAMES)
def test_a_skill_without_its_mcp_entry_is_partial(home, host):
    install(host, home=str(home))
    if host == "opencode":
        with open(os.path.join(_root(home, host), "opencode.jsonc"), "w",
                  encoding="utf-8") as fh:
            json.dump({"mcp": {}}, fh)
    elif host == "codex":
        os.remove(os.path.join(_root(home, host), "config.toml"))
    elif host == "dsh":
        os.remove(os.path.join(_root(home, host), "cordis.patch.yml"))
    else:                                          # claude-code: .mcp.json in the plugin
        os.remove(os.path.join(_root(home, host), ".mcp.json"))
    assert host_status(host, home=str(home))["state"] == "partial"


def test_an_opencode_config_with_comments_is_not_guessed_at(home):
    install("opencode", home=str(home))
    cfg = os.path.join(_root(home, "opencode"), "opencode.jsonc")
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write('{\n  // mine\n  "mcp": {"agentdescent": {}}\n}\n')
    info = host_status("opencode", home=str(home))
    assert info["mcp"] == "unknown" and info["state"] == "ok"


def test_status_follows_the_directories_install_honours(home, monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("CODEX_HOME", str(elsewhere / "codex"))
    monkeypatch.setenv("DSH_HOME", str(elsewhere / "dsh"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(elsewhere / "xdg"))
    for host in ("codex", "dsh", "opencode"):
        install(host, home=str(home))
        info = host_status(host, home=str(home))
        assert info["state"] == "ok" and str(elsewhere) in info["path"], info


def test_host_status_rejects_an_unknown_host(home):
    with pytest.raises(ValueError, match="unknown host"):
        host_status("emacs", home=str(home))


# -- doctor ------------------------------------------------------------------


def test_doctor_lists_every_host(home, monkeypatch):
    rep = _report(monkeypatch)
    assert sorted(rep["hosts"]) == HOST_NAMES
    assert all(info["state"] == "missing" for info in rep["hosts"].values())
    assert rep["hosts"]["opencode"]["cli"] is None


def test_doctor_hints_at_a_host_that_is_on_path_but_not_wired(home, monkeypatch):
    rep = _report(monkeypatch, present=("opencode", "codex"))
    assert rep["hosts"]["opencode"]["cli"] == "/fake/bin/opencode"
    hints = " ".join(rep["hints"])
    assert "agentdescent install opencode" in hints
    assert "agentdescent install codex" in hints and "codex plugin add" in hints
    assert "install dsh" not in hints, "dsh is not on PATH; nothing to say"
    assert not any("integration" in p for p in rep["problems"]), \
        "a host nobody wired is information, not a failure"


def test_doctor_stops_hinting_once_the_host_is_wired(home, monkeypatch):
    install("opencode", home=str(home))
    rep = _report(monkeypatch, present=("opencode",))
    assert rep["hosts"]["opencode"]["state"] == "ok"
    assert not any("install opencode" in h for h in rep["hints"])


def test_doctor_calls_a_stale_integration_a_problem(home, monkeypatch):
    install("opencode", home=str(home))
    with open(_skill_path(home, "opencode"), "w", encoding="utf-8") as fh:
        fh.write("an older procedure\n")
    rep = _report(monkeypatch)
    assert any("opencode" in p and "stale" in p and "agentdescent install opencode" in p
               for p in rep["problems"]), rep["problems"]


def test_doctor_text_output_names_the_hosts(home, monkeypatch, capsys):
    install("opencode", home=str(home))
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: None)
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "host opencode: ok" in out and "host dsh: missing" in out
    assert code == 1                                    # no agent CLI on PATH: unchanged


# -- review of #383 -----------------------------------------------------------


def test_an_unreadable_opencode_config_on_a_fresh_home_is_missing_not_partial(
        home, monkeypatch):
    """A user who never installed anything but keeps comments in their own
    opencode.jsonc must not get exit code 1 for it."""
    cfg = home / ".config" / "opencode"
    cfg.mkdir(parents=True)
    (cfg / "opencode.jsonc").write_text('{\n  // my providers\n  "provider": {}\n}\n')
    info = host_status("opencode", home=str(home))
    assert (info["state"], info["skill"], info["mcp"]) == ("missing", "missing", "unknown")
    rep = _report(monkeypatch, present=("opencode",))
    assert not any("integration" in p for p in rep["problems"]), rep["problems"]
    assert any("install opencode" in h for h in rep["hints"])


def test_an_mcp_entry_kept_in_opencode_json_counts(home):
    install("opencode", home=str(home))
    root = _root(home, "opencode")
    os.replace(os.path.join(root, "opencode.jsonc"), os.path.join(root, "opencode.json"))
    assert host_status("opencode", home=str(home))["mcp"] == "ok"


def test_the_marketplace_route_is_named_for_claude_code_too(home, monkeypatch):
    rep = _report(monkeypatch, present=("claude",))
    assert any("/plugin install agentdescent@agentdescent" in h for h in rep["hints"])


def test_a_surviving_config_without_our_live_entry_is_not_ok(home):
    """Review of #383: the file or a sentinel being present is not the entry."""
    install("claude-code", home=str(home))
    (open(os.path.join(_root(home, "claude-code"), ".mcp.json"), "w")).write("{}")
    assert host_status("claude-code", home=str(home))["mcp"] == "missing"

    install("dsh", home=str(home))
    patch = os.path.join(_root(home, "dsh"), "cordis.patch.yml")
    text = open(patch).read()
    open(patch, "w").write(text[:text.index("# --- end agentdescent ---")])
    assert host_status("dsh", home=str(home))["mcp"] == "missing"

    install("codex", home=str(home))
    cfg = os.path.join(_root(home, "codex"), "config.toml")
    open(cfg, "w").write("# [mcp_servers.agentdescent]\n# command = \"agentdescent\"\n")
    assert host_status("codex", home=str(home))["mcp"] == "missing"
    assert host_status("codex", home=str(home))["state"] == "partial"
