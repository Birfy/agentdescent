"""The command-line contract shared by the six faithful algorithm ports."""

from __future__ import annotations

import argparse
import inspect
import pathlib
import socket
import urllib.request

import pytest

from examples import ace_context_evolution as ace
from examples import adas_meta_agent_search as adas
from examples import dgm_self_improve as dgm
from examples import evoskill_skill_discovery as evoskill
from examples import gepa_prompt_evolution as gepa
from examples import skillopt_skill_training as skillopt
from examples import _TEMPLATE as port_template
from examples import _common as common
from examples._common import add_standard_args


PORTS = (
    (ace, "rounds", "claude-haiku-4-5", 30.0, "load_dataset"),
    (gepa, "rounds", "claude-haiku-4-5", 45.0, "load_dataset"),
    (evoskill, "iterations", "claude-haiku-4-5", 40.0, "load_dataset"),
    (skillopt, "steps", "claude-haiku-4-5", 40.0, "load_dataset"),
    (adas, "generations", "claude-haiku-4-5", 60.0, "build_examples"),
    (dgm, "generations", None, 15.0, "load_dataset"),
)


def test_standard_args_have_one_definition():
    parser = add_standard_args(argparse.ArgumentParser())
    args = parser.parse_args([
        "--provider", "openai",
        "--model", "test-model",
        "--seed", "7",
        "--async",
        "--async-ratio", "5",
        "--max-seconds", "9.5",
        "--dry-run",
        "--yes",
    ])
    assert args.provider == "openai"
    assert args.model == "test-model"
    assert args.seed == 7
    assert args.asynchronous is True
    assert args.async_ratio == 5
    assert args.max_seconds == 9.5
    assert args.dry_run is True
    assert args.yes is True


@pytest.mark.parametrize("module,iteration,model,seconds,_loader", PORTS)
def test_every_port_uses_the_standard_contract(module, iteration, model, seconds,
                                                _loader):
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.provider == "claude"
    assert args.model == model
    assert args.seed == 0
    assert args.asynchronous is False
    assert args.async_ratio == 3
    assert args.max_seconds == seconds
    assert args.dry_run is False
    assert args.yes is False
    assert hasattr(args, iteration)

    option_strings = [opt for action in parser._actions for opt in action.option_strings]
    for option in ("--provider", "--model", "--seed", "--async", "--async-ratio",
                   "--max-seconds", "--dry-run", "--yes"):
        assert option_strings.count(option) == 1, f"{module.__name__}: {option}"
    iteration_options = {"--rounds", "--generations", "--iterations", "--steps"}
    assert iteration_options.intersection(option_strings) == {f"--{iteration}"}


@pytest.mark.parametrize("module,_iteration,_model,_seconds,_loader", PORTS)
def test_every_port_calls_the_shared_helper_once(
        module, _iteration, _model, _seconds, _loader, monkeypatch):
    calls = []

    def recording_helper(parser, **kwargs):
        calls.append(kwargs)
        return common.add_standard_args(parser, **kwargs)

    monkeypatch.setattr(module, "add_standard_args", recording_helper)
    module.build_parser()
    assert len(calls) == 1


@pytest.mark.parametrize("module,_iteration,_model,_seconds,loader", PORTS)
def test_dry_run_never_touches_data_network_or_models(
        module, _iteration, _model, _seconds, loader, monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run crossed an external boundary")

    monkeypatch.setattr(module, loader, forbidden)
    monkeypatch.setattr(module, "completion_for", forbidden)
    monkeypatch.setattr(common, "claude", forbidden)
    monkeypatch.setattr(common, "openai_compatible", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    module.main(["--dry-run"])
    assert "dry-run" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# The flags have to be *honoured*, not only declared. A port that grows a
# `--yes` it never reads is exactly the drift #74 was filed about, and a shape
# test over the parser cannot see it -- so these exercise the behaviour and
# then check no port reimplements it locally.
# ---------------------------------------------------------------------------


class _Stdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _never_asks(*_args, **_kwargs):
    raise AssertionError("confirm() prompted when it should not have")


@pytest.mark.parametrize("yes,tty", [(True, True), (True, False), (False, False)])
def test_confirm_does_not_prompt_when_yes_or_non_interactive(yes, tty, monkeypatch):
    monkeypatch.setattr(common.sys, "stdin", _Stdin(tty))
    monkeypatch.setattr("builtins.input", _never_asks)
    assert common.confirm(argparse.Namespace(yes=yes)) is True


@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("YES", True), ("n", False), ("", False),
])
def test_confirm_honours_an_interactive_answer(answer, expected, monkeypatch, capsys):
    monkeypatch.setattr(common.sys, "stdin", _Stdin(True))
    monkeypatch.setattr("builtins.input", lambda *_args: answer)
    assert common.confirm(argparse.Namespace(yes=False)) is expected
    assert ("aborted." in capsys.readouterr().out) is not expected


@pytest.mark.parametrize("provider,factory", [
    ("claude", "claude"), ("openai", "openai_compatible"), ("glm", "openai_compatible"),
])
def test_completion_for_dispatches_on_provider(provider, factory, monkeypatch):
    seen = {}

    def record(name):
        def factory_stub(**kwargs):
            seen[name] = kwargs
            return name
        return factory_stub

    monkeypatch.setattr(common, "claude", record("claude"))
    monkeypatch.setattr(common, "openai_compatible", record("openai_compatible"))
    args = argparse.Namespace(provider=provider, model="test-model")

    assert common.completion_for(args, max_tokens=99) == factory
    assert seen == {factory: {"model": "test-model", "usage": None, "max_tokens": 99}}


@pytest.mark.parametrize("module,_i,_m,_s,_l", PORTS + ((port_template, "", "", 0.0, ""),))
def test_no_port_reimplements_the_shared_behaviour(module, _i, _m, _s, _l):
    source = inspect.getsource(module)
    assert "Proceed with real API calls?" not in source, (
        f"{module.__name__}: use examples._common.confirm(args)")
    assert 'provider in ("openai", "glm")' not in source, (
        f"{module.__name__}: use examples._common.completion_for/is_openai_compatible")
    assert "confirm(args)" in source, f"{module.__name__}: --yes is declared but never read"
    assert "completion_for(" in source, f"{module.__name__}: --provider is never dispatched"


def test_ports_table_covers_every_standardised_entrypoint():
    """A seventh port must join PORTS, or it escapes the whole contract."""
    examples_dir = pathlib.Path(common.__file__).resolve().parent
    on_disk = {
        path.stem for path in examples_dir.glob("*.py")
        if path.stem != "_common" and "add_standard_args" in path.read_text()
    }
    listed = {module.__name__.rsplit(".", 1)[-1] for module, *_ in PORTS}
    assert on_disk == listed | {"_TEMPLATE"}


def test_porting_checklist_stays_short():
    root = pathlib.Path(__file__).resolve().parent.parent
    lines = (root / "docs" / "porting-checklist.md").read_text().splitlines()
    assert len(lines) <= 20


def test_port_template_is_importable_and_offline(capsys):
    args = port_template.build_parser().parse_args([])
    assert args.iterations == 6
    assert args.provider == "claude"
    port_template.main(["--dry-run"])
    assert "no dataset or model api was accessed" in capsys.readouterr().out.lower()
