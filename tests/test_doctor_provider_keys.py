"""``doctor`` reads provider keys from the shared ``FORWARDED_ENV`` list (#423).

It used to hard-code ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``,
so a machine with only ``ARK_API_KEY`` -- forwarded by every installer, and the key
the docs' Ark example uses -- exited 1 with "no provider key in the environment"."""

from __future__ import annotations

import json

import pytest

from agentdescent import cli
from agentdescent.integrations import FORWARDED_ENV, provider_key_names, skill_text

NO_KEY = "no provider key in the environment"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in FORWARDED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "mcp_unavailable", lambda: None)


def _problems(rep):
    return [p for p in rep["problems"] if p.startswith(NO_KEY)]


def test_ark_api_key_alone_is_a_provider_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    rep = cli.doctor_report()
    assert rep["provider_keys_present"]["ARK_API_KEY"] is True
    assert _problems(rep) == []


def test_nothing_set_still_reports_no_provider_key():
    rep = cli.doctor_report()
    assert len(_problems(rep)) == 1
    assert not any(rep["provider_keys_present"].values())


@pytest.mark.parametrize("name", provider_key_names())
def test_every_forwarded_key_alone_is_enough(monkeypatch, name):
    monkeypatch.setenv(name, "k")
    rep = cli.doctor_report()
    assert rep["provider_keys_present"][name] is True and _problems(rep) == []


def test_every_key_like_name_in_forwarded_env_is_reported():
    """The two lists cannot drift: a name added to FORWARDED_ENV is a key to doctor."""
    reported = cli.doctor_report()["provider_keys_present"]
    keys = [n for n in FORWARDED_ENV if not n.endswith("_BASE_URL") and n != "AGENTDESCENT_HOME"]
    assert keys and set(keys) <= set(reported)
    assert "ARK_API_KEY" in keys and set(keys) == set(provider_key_names())
    # endpoint variables and the run-store path are not keys, and not "a key" for
    # the no-key problem
    assert "AGENTDESCENT_HOME" not in reported and "ANTHROPIC_BASE_URL" not in reported


def test_a_base_url_alone_is_not_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://x")
    monkeypatch.setenv("AGENTDESCENT_HOME", "/tmp/x")
    assert len(_problems(cli.doctor_report())) == 1


def test_the_output_shape_keeps_the_old_names_first(monkeypatch):
    names = list(cli.doctor_report()["provider_keys_present"])
    assert names[:4] == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                         "DEEPSEEK_API_KEY"]


def test_only_a_non_openai_key_comes_with_a_how_to_hint(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "k")
    hints = [h for h in cli.doctor_report()["hints"] if "ARK_API_KEY" in h]
    reflect = [h for h in hints if "openai_compatible" in h]
    [hint] = reflect
    for word in ("openai_compatible", "api_key_env", "default_base_url", "base_url_env",
                 "never guess"):
        assert word in hint, word


@pytest.mark.parametrize("key", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
def test_no_hint_when_a_default_key_is_present(monkeypatch, key):
    monkeypatch.setenv(key, "k")
    monkeypatch.setenv("ARK_API_KEY", "k")
    assert not [h for h in cli.doctor_report()["hints"] if "api_key_env" in h]


def test_the_cli_does_not_print_the_no_key_problem_with_only_ark_api_key(monkeypatch, capsys):
    monkeypatch.setenv("ARK_API_KEY", "k")
    cli.main(["doctor"])        # the exit code also reflects machine-dependent problems
    out = capsys.readouterr().out
    assert "ARK_API_KEY': True" in out and NO_KEY not in out
    cli.main(["--json", "doctor"])
    assert json.loads(capsys.readouterr().out)["provider_keys_present"]["ARK_API_KEY"] is True


def test_the_skill_documents_the_other_provider_knobs():
    text = skill_text()
    for word in ("api_key_env", "base_url_env", "default_base_url", "Other providers",
                 "never guess"):
        assert word in text, word
    for name in provider_key_names():       # the skill names every key doctor checks
        assert name in text, name
