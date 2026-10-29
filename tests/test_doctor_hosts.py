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
    # Grok Bot has no host-native MCP config; scaffold reports mcp n/a.
    expect_mcp = "n/a" if host == "grok-bot" else "ok"
    assert (info["state"], info["skill"], info["mcp"]) == ("ok", "ok", expect_mcp), info


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
    elif host == "grok-bot":
        # HOST.md is the scaffold marker that makes mcp n/a; without it, skill-only
        # is partial (honest: incomplete scaffold).
        os.remove(os.path.join(_root(home, host), "HOST.md"))
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
    monkeypatch.setenv("HOME", str(home))           # no explicit home: the variables apply
    for host in ("codex", "dsh", "opencode"):
        install(host)
        info = host_status(host)
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
    assert "host opencode: ok (skill ok, mcp ok; cli not on PATH)" in out
    assert "host dsh: no cli on PATH" in out
    assert code == 1                                    # no agent CLI on PATH: unchanged


def test_doctor_does_not_call_an_unwired_cli_missing(home, monkeypatch, capsys):
    """#535: a CLI on PATH with no integration is "not installed", not "missing",
    and the summary must not claim nothing is missing."""
    rep = _report(monkeypatch, present=("opencode",))
    monkeypatch.setattr(cli, "doctor_report", lambda **_kw: rep)
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "host opencode: not installed (cli /fake/bin/opencode)" in out
    assert "agentdescent install opencode" in out
    assert "host opencode: missing" not in out
    assert "host codex: no cli on PATH" in out
    assert "nothing missing" not in out
    if not rep["problems"]:
        assert "ok: no blocking problems (1 host can be wired: opencode)" in out
        assert code == 0


def test_doctor_host_line_names_skill_and_mcp_for_partial(home, monkeypatch, capsys):
    """#165: stale vs MCP-missing must be readable on the host line itself."""
    install("opencode", home=str(home))
    with open(os.path.join(_root(home, "opencode"), "opencode.jsonc"), "w",
              encoding="utf-8") as fh:
        json.dump({"mcp": {}}, fh)
    rep = _report(monkeypatch, present=("opencode",))
    monkeypatch.setattr(cli, "doctor_report", lambda **_kw: rep)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "host opencode: partial (skill ok, mcp missing; cli /fake/bin/opencode)" in out


def test_doctor_host_line_names_skill_and_mcp_for_stale(home, monkeypatch, capsys):
    install("opencode", home=str(home))
    with open(_skill_path(home, "opencode"), "w", encoding="utf-8") as fh:
        fh.write("an older procedure\n")
    rep = _report(monkeypatch, present=("opencode",))
    monkeypatch.setattr(cli, "doctor_report", lambda **_kw: rep)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "host opencode: stale (skill stale, mcp ok; cli /fake/bin/opencode)" in out


# -- #605: doctor --home -----------------------------------------------------


def test_doctor_help_lists_home():
    ns = cli.build_parser().parse_args(["doctor", "--home", "/tmp/probe"])
    assert ns.home == "/tmp/probe"
    for action in cli.build_parser()._actions:
        choices = getattr(action, "choices", None) or {}
        if "doctor" in choices:
            assert "--home" in choices["doctor"].format_help()
            break
    else:
        raise AssertionError("doctor subparser not found")


def test_doctor_home_sees_an_install_into_a_probe(tmp_path, monkeypatch, capsys):
    """install --home then doctor --home: host is not missing (the #605 bug)."""
    real = tmp_path / "real"
    probe = tmp_path / "probe"
    real.mkdir(); probe.mkdir()
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name, *a, **k: (
        "/usr/bin/git" if name == "git" else None))
    install("claude-code", home=str(probe))
    install("grok-bot", home=str(probe))
    # Without --home, doctor still reads the real home → missing.
    code = cli.main(["doctor", "--json"])
    without = json.loads(capsys.readouterr().out)
    assert without["hosts"]["claude-code"]["state"] == "missing"
    assert without["hosts"]["grok-bot"]["state"] == "missing"
    # With --home, same probe install reads ok.
    code = cli.main(["doctor", "--json", "--home", str(probe)])
    with_home = json.loads(capsys.readouterr().out)
    assert with_home["hosts"]["claude-code"]["state"] == "ok", with_home["hosts"]["claude-code"]
    assert with_home["hosts"]["grok-bot"]["state"] == "ok", with_home["hosts"]["grok-bot"]
    assert str(probe) in with_home["hosts"]["claude-code"]["path"]
    assert code in (0, 1)  # may still flag missing agent CLIs


def test_doctor_home_wins_over_host_env_variables(tmp_path, monkeypatch):
    """Same #410 rule as install: explicit --home ignores CODEX_HOME etc."""
    real = tmp_path / "real"
    probe = tmp_path / "probe"
    elsewhere = tmp_path / "elsewhere"
    for d in (real, probe, elsewhere):
        d.mkdir()
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("CODEX_HOME", str(elsewhere / "codex"))
    monkeypatch.setenv("DSH_HOME", str(elsewhere / "dsh"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(elsewhere / "xdg"))
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)
    install("codex", home=str(probe))
    install("dsh", home=str(probe))
    install("opencode", home=str(probe))
    rep = cli.doctor_report(home=str(probe))
    for host in ("codex", "dsh", "opencode"):
        info = rep["hosts"][host]
        assert info["state"] == "ok", info
        assert str(probe) in info["path"], info
        assert str(elsewhere) not in info["path"], info
    # Without home, the exported directories apply — and they are empty.
    bare = cli.doctor_report()
    for host in ("codex", "dsh", "opencode"):
        assert bare["hosts"][host]["state"] == "missing"


def test_doctor_empty_home_is_refused(capsys):
    code = cli.main(["doctor", "--home", ""])
    err = capsys.readouterr().err
    assert code == 2
    assert "home=" in err and "empty" in err


def test_mcp_doctor_forwards_home(tmp_path, monkeypatch):
    from agentdescent.mcp import Tools
    real = tmp_path / "real"
    probe = tmp_path / "probe"
    real.mkdir(); probe.mkdir()
    for var in ("DSH_HOME", "CODEX_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(real))
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "ad-home"))
    install("claude-code", home=str(probe))
    t = Tools(str(tmp_path / "runs"))
    assert t.doctor()["hosts"]["claude-code"]["state"] == "missing"
    assert t.doctor(home=str(probe))["hosts"]["claude-code"]["state"] == "ok"


# -- review of #383 -----------------------------------------------------------


def test_an_unreadable_opencode_config_on_a_fresh_home_is_missing_not_partial(
        home, monkeypatch):
    """A user who never installed anything but keeps comments in their own
    opencode.jsonc must not get exit code 1 for it."""
    cfg = home / ".config" / "opencode"
    cfg.mkdir(parents=True)
    (cfg / "opencode.jsonc").write_text('{\n  // my providers\n  "provider": {}\n}\n')
    info = host_status("opencode", home=str(home))
    # It never names agentdescent, so it holds no entry of ours (#516).
    assert (info["state"], info["skill"], info["mcp"]) == ("missing", "missing", "missing")
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


# -- #516 ---------------------------------------------------------------------

_COMMENTED = ('{\n  // my settings\n  "model": "ark/deepseek-v4-flash",\n'
              '  "provider": {"ark": {"models": {"deepseek-v4-flash": {}},},},\n}\n')


def test_install_beside_a_commented_opencode_jsonc_is_ok(home):
    root = home / ".config" / "opencode"
    root.mkdir(parents=True)
    (root / "opencode.jsonc").write_text(_COMMENTED)
    install("opencode", home=str(home))
    assert (root / "opencode.jsonc").read_text() == _COMMENTED
    info = host_status("opencode", home=str(home))
    assert (info["state"], info["mcp"]) == ("ok", "ok")


def test_a_skill_beside_a_commented_config_without_our_entry_is_partial(home):
    """Before #516 this read `ok: nothing missing` while OpenCode had no tools."""
    install("opencode", home=str(home))
    root = home / ".config" / "opencode"
    os.remove(root / "opencode.jsonc")
    (root / "opencode.jsonc").write_text(_COMMENTED)
    info = host_status("opencode", home=str(home))
    assert (info["state"], info["mcp"]) == ("partial", "missing")
