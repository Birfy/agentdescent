"""The command-line contract shared by the six faithful algorithm ports."""

from __future__ import annotations

import argparse
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
    monkeypatch.setattr(module, "claude", forbidden)
    monkeypatch.setattr(module, "openai_compatible", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    module.main(["--dry-run"])
    assert "dry-run" in capsys.readouterr().out.lower()


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
