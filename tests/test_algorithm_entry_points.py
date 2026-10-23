"""Third-party algorithms register through an entry point (#384, slice 6).

A package installed beside AgentDescent can advertise an algorithm under the
``agentdescent.algorithms`` group. The shipped registry must keep working if one
plugin is broken, and a duplicate name is an error rather than a silent shadow.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from agentdescent import algorithms
from agentdescent.algorithms import Algorithm, AlgorithmError, Overrides, Param


def _algorithm(name: str) -> Algorithm:
    def build(ctx, params):
        return Overrides(notes=[f"external {name}"])

    return Algorithm(name=name, summary=f"external {name}", kinds=("text",),
                     fidelity="preset", params=(Param("x", "int", 1),), build=build)


class _EntryPoint:
    def __init__(self, name, value=None, error=None):
        self.name = name
        self._value = value
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._value


@pytest.fixture
def clean_registry(monkeypatch):
    algorithms.names()          # ensure the shipped algorithms are loaded first
    before = dict(algorithms._REGISTRY)
    monkeypatch.setattr(algorithms, "_external_loaded", False)
    monkeypatch.setattr(algorithms, "_EXTERNAL_ERRORS", {})
    yield
    algorithms._REGISTRY.clear()
    algorithms._REGISTRY.update(before)


def _points(monkeypatch, *points):
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda **kwargs: list(points))
    monkeypatch.setattr(algorithms, "_external_loaded", False)


def test_an_entry_point_adds_a_listed_and_gettable_algorithm(monkeypatch, clean_registry):
    _points(monkeypatch, _EntryPoint("acme_method", _algorithm("acme_method")))
    assert "acme_method" in algorithms.names()
    listed = [a.name for a in algorithms.listing("text")]
    assert "acme_method" in listed and listed == sorted(listed)
    assert algorithms.get("acme_method").summary == "external acme_method"


def test_a_factory_returning_an_algorithm_is_accepted(monkeypatch, clean_registry):
    _points(monkeypatch, _EntryPoint("made", lambda: _algorithm("made")))
    assert algorithms.get("made").fidelity == "preset"


def test_a_duplicate_name_is_an_error_and_does_not_shadow_the_builtin(monkeypatch, clean_registry):
    _points(monkeypatch, _EntryPoint("rrsi", _algorithm("rrsi")))
    assert "already registered" in algorithms.external_errors()["rrsi"]
    assert algorithms.get("rrsi").fidelity == "mechanism_microport"


def test_a_broken_plugin_does_not_hide_the_shipped_ones(monkeypatch, clean_registry):
    _points(monkeypatch, _EntryPoint("broken", error=RuntimeError("boom")))
    assert "broken" not in algorithms.names()
    assert "boom" in algorithms.external_errors()["broken"]
    with pytest.raises(AlgorithmError, match="failed to load.*boom"):
        algorithms.get("broken")
    assert algorithms.get("rrsi")


def test_a_broken_plugin_does_not_escape_as_a_process_exit(monkeypatch, clean_registry):
    _points(monkeypatch, _EntryPoint("exits", error=SystemExit(3)))
    assert "exits" not in algorithms.names()
    assert "SystemExit" in algorithms.external_errors()["exits"]
    assert algorithms.get("rrsi")


def test_an_installed_plugin_does_not_change_the_generated_catalog(monkeypatch, clean_registry):
    from tools import gen_algorithms_docs as gen

    before = gen.render()
    _points(monkeypatch, _EntryPoint("acme_method", _algorithm("acme_method")))
    assert "acme_method" in algorithms.names()
    assert gen.render() == before
    assert "## `acme_method`" not in gen.render()


def test_cli_warns_about_a_broken_plugin_on_stderr(monkeypatch, clean_registry, capsys):
    from agentdescent import cli

    _points(monkeypatch, _EntryPoint("broken", error=RuntimeError("boom")))
    assert cli.main(["algorithms", "list"]) == 0
    assert "algorithm plugin 'broken' failed to load" in capsys.readouterr().err


def test_mcp_list_algorithms_reports_load_errors(monkeypatch, clean_registry):
    from agentdescent.mcp import Tools

    _points(monkeypatch, _EntryPoint("broken", error=RuntimeError("boom")))
    out = Tools(None).list_algorithms()
    assert "boom" in out["load_errors"]["broken"]
    assert "broken" not in [a["name"] for a in out["algorithms"]]
