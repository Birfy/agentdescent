"""doctor sees the Grok Bot user skill, read-only (#675)."""

from __future__ import annotations

import json
import os

import pytest

from agentdescent import cli
from agentdescent.integrations import (
    GROK_BOT_SKILLS_ENV,
    _skill_body,
    host_status,
    install,
    skill_text,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(GROK_BOT_SKILLS_ENV, raising=False)
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)
    return tmp_path


def _save_user_skill(home, body, name="agentdescent", slug="skill-123"):
    d = home / "agent-data" / "workflows" / slug
    d.mkdir(parents=True)
    # Grok Bot keeps its own frontmatter keys; only `name:` identifies us.
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: >-\n  saved by the Bot\nmetadata: x\n---\n\n{body}",
        encoding="utf-8")
    return d / "SKILL.md"


def _doctor(home, capsys, *extra):
    rc = cli.main(["doctor", "--home", str(home), *extra])
    return rc, capsys.readouterr().out


def _grok_line(out):
    return next(l for l in out.splitlines() if l.startswith("host grok-bot"))


def test_current_user_skill_is_ok(home, capsys):
    install("grok-bot", home=str(home))
    path = _save_user_skill(home, _skill_body())
    info = host_status("grok-bot", home=str(home))
    assert (info["state"], info["user_skill"], info["user_skill_path"]) == (
        "ok", "ok", str(path))
    _, out = _doctor(home, capsys)
    line = _grok_line(out)
    assert "user skill ok" in line and "cli" not in line, line
    _, out = _doctor(home, capsys, "--json")
    rep = json.loads(out)["hosts"]["grok-bot"]
    assert rep["user_skill"] == "ok" and rep["user_skill_path"] == str(path)


def test_old_user_skill_is_stale_and_a_problem(home, capsys):
    install("grok-bot", home=str(home))
    _save_user_skill(home, _skill_body().replace("doctor", "doctr", 1),
                     name="AgentDescent")
    info = host_status("grok-bot", home=str(home))
    assert (info["state"], info["skill"], info["user_skill"]) == ("stale", "ok", "stale")
    rc, out = _doctor(home, capsys)
    assert rc == 1
    assert "user skill stale" in _grok_line(out)
    problem = next(l for l in out.splitlines() if l.startswith("! grok-bot"))
    assert "rewrite" in problem and info["path"] in problem
    rc, out = _doctor(home, capsys, "--json")
    assert json.loads(out)["hosts"]["grok-bot"]["user_skill"] == "stale"


def test_no_user_skill_is_missing_and_line_has_no_cli(home, capsys):
    install("grok-bot", home=str(home))
    _save_user_skill(home, _skill_body(), name="something-else")
    info = host_status("grok-bot", home=str(home))
    assert (info["state"], info["user_skill"]) == ("ok", "missing")
    _, out = _doctor(home, capsys)
    line = _grok_line(out)
    assert "user skill missing" in line and "cli" not in line, line


def test_saved_skill_without_scaffold_is_ok(home):
    _save_user_skill(home, _skill_body())
    info = host_status("grok-bot", home=str(home))
    assert (info["state"], info["skill"], info["user_skill"]) == ("ok", "missing", "ok")


def test_env_is_ignored_under_explicit_home(home, tmp_path_factory, monkeypatch):
    other = tmp_path_factory.mktemp("elsewhere")
    d = other / "skill-9"
    d.mkdir()
    (d / "SKILL.md").write_text(skill_text(), encoding="utf-8")
    monkeypatch.setenv(GROK_BOT_SKILLS_ENV, str(other))
    assert host_status("grok-bot", home=str(home))["user_skill"] == "missing"
    # The real home honours the variable.
    assert host_status("grok-bot")["user_skill"] == "ok"


def test_doctor_does_not_write_the_skill_store(home, capsys):
    install("grok-bot", home=str(home))
    _save_user_skill(home, "old body\n")
    store = home / "agent-data" / "workflows"
    before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    _doctor(home, capsys)
    install("grok-bot", home=str(home))
    after = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    assert before == after
