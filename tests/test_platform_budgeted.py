"""A platform spec can use the budgeted PRM/verifier search as its step."""

import json

import pytest

from agentdescent import cli
from agentdescent.mcp import Tools
from agentdescent.planning import PlanTransition
from agentdescent.process_reward import ProcessRewardModel, StepExample
from agentdescent.search_fabric import SearchFabric
from agentdescent.surface_platform import platform_plan, platform_run
from agentdescent.surface_common import SurfaceError


class _Fork:
    def actions(self, state):
        return ("good", "bad") if state == "root" else ()

    def step(self, state, action):
        return PlanTransition(action, reward=0.0, terminal=False)


def _features(state):
    return (1.0, 0.0) if state == "good" else (0.0, 1.0)


def build_fabric():
    model = ProcessRewardModel()
    model.fit([
        StepExample((1.0, 0.0), 1.0),
        StepExample((0.9, 0.1), 1.0),
        StepExample((0.0, 1.0), 0.0),
        StepExample((0.1, 0.9), 0.0),
    ], min_n=4)
    return SearchFabric(_Fork(), model, _features, budget=5, seed=0)


def state(context):
    return "root"


def _spec(tmp_path):
    return {
        "runs_db": str(tmp_path / "runs.sqlite3"),
        "platform": {"tenant": "t1", "workspace": "ws", "max_generations": 1,
                     "target": 0.4},
        "campaign": {"max_cycles": 1, "target": 0.4},
        "step": {
            "budgeted": {
                "fabric": "tests.test_platform_budgeted:build_fabric",
                "state": "tests.test_platform_budgeted:state",
                "config": {"budgets": [4, 8],
                           "min_gain_per_expansion": 0.0,
                           "patience": 10},
            }
        },
    }


def test_platform_plan_accepts_a_budgeted_step(tmp_path):
    plan = platform_plan(_spec(tmp_path))
    assert plan["ok"] is True
    assert plan["spec"] == "<inline>"
    assert plan["refs"]["step"]["budgeted"]["config"]["budgets"] == [4, 8]


def test_platform_run_uses_the_budgeted_step(tmp_path):
    payload = platform_run(_spec(tmp_path))
    assert payload["ok"] is True
    assert payload["report"]["complete"] is True
    generation = payload["report"]["cycles"][0]["loop"]["generations"][0]
    assert generation["evidence"]["stopped_reason"]
    assert generation["evidence"]["rounds"]
    assert generation["evidence"]["best_value"] >= 0.4


def test_budgeted_step_config_invalid_value_is_refused(tmp_path):
    spec = _spec(tmp_path)
    spec["step"]["budgeted"]["config"]["budgets"] = 0
    with pytest.raises(SurfaceError, match="budgeted"):
        platform_plan(spec)


def test_budgeted_step_unknown_keys_are_refused(tmp_path):
    spec = _spec(tmp_path)
    spec["step"]["budgeted"]["evidence_off"] = "x:y"
    with pytest.raises(SurfaceError, match="unknown step.budgeted"):
        platform_plan(spec)
    with pytest.raises(SurfaceError, match="unknown step.budgeted"):
        platform_run(spec)


def test_budgeted_config_unknown_key_is_refused(tmp_path):
    spec = _spec(tmp_path)
    spec["step"]["budgeted"]["config"]["budget"] = [4, 8]
    with pytest.raises(SurfaceError, match="budget"):
        platform_plan(spec)


def test_cli_platform_plan_and_run_budgeted(tmp_path, capsys):
    path = tmp_path / "platform.json"
    path.write_text(json.dumps(_spec(tmp_path)), encoding="utf-8")
    assert cli.main(["platform", "plan", "--spec", str(path), "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ok"] is True
    assert cli.main(["platform", "run", "--spec", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["complete"] is True


def test_mcp_platform_budgeted(tmp_path):
    tools = Tools(surface_allow=("tests.", "agentdescent."),
                  surface_write_dirs=(str(tmp_path),))
    spec = _spec(tmp_path)
    assert tools.platform(spec, dry_run=True)["ok"] is True
    run = tools.platform(spec, dry_run=False)
    assert run["report"]["complete"] is True
