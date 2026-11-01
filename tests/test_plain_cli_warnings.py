"""Foreground CLI prints AgentDescent warnings as plain lines (#654).

``campaign`` / ``tournament`` used to leak Python's ``file:line: Category``
format for ``SmallHeldOutWarning``; the early-stop ``RuntimeWarning`` did the
same on every verb and duplicated the status-line error. Library ``evolve()``
callers keep the standard ``warnings`` behaviour.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

from agentdescent.evolution import AppendRules, SmallHeldOutWarning, Task, evolve
from agentdescent.evolvespec import compose

from tests.test_evolvespec import _WORDS, _dir_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _small_dir_spec(tmp, n=8, **extra):
    """Stub skill_dir spec with fewer than 12 cases (tiny held-out set)."""
    rows = [{"prompt": w, "gold": w[::-1]} for w in _WORDS[:n]]
    path = os.path.join(tmp, f"cases{n}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return _dir_spec(tmp, data={"path": path, "prompt": "prompt", "gold": "gold"},
                     **extra)


def _run_cli(args, *, tmp_path, store, timeout=180):
    env = {**os.environ, "AGENTDESCENT_HOME": str(tmp_path / "home"),
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("AGENTDESCENT_DEBUG", None)
    return subprocess.run(
        [sys.executable, "-m", "agentdescent.cli", "--store", store, *args],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=timeout)


def test_campaign_prints_plain_held_out_warning(tmp_path):
    spec = _small_dir_spec(str(tmp_path), n=8,
                           evolve={"rounds": 1, "n_workers": 1, "seed": 0})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    store = str(tmp_path / "runs")
    proc = _run_cli(["campaign", str(path), "--sessions", "1", "--rounds", "1",
                     "--workdir", str(tmp_path / "c")],
                    tmp_path=tmp_path, store=store)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = [ln for ln in proc.stderr.splitlines() if "held-out set has only" in ln]
    assert len(lines) == 1 and lines[0].startswith("warning: the held-out set has only")
    assert "evolvespec.py:" not in proc.stderr
    assert "return evolve(" not in proc.stderr


def test_tournament_prints_plain_held_out_warning(tmp_path):
    spec = _small_dir_spec(str(tmp_path), n=8,
                           evolve={"rounds": 1, "n_workers": 1, "seed": 0})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    store = str(tmp_path / "runs")
    proc = _run_cli(["tournament", str(path), "--algorithm", "default",
                     "--rounds", "1"],
                    tmp_path=tmp_path, store=store)
    assert proc.returncode in (0, 1), proc.stderr + proc.stdout
    lines = [ln for ln in proc.stderr.splitlines() if "held-out set has only" in ln]
    assert len(lines) == 1 and lines[0].startswith("warning: the held-out set has only")
    assert "evolvespec.py:" not in proc.stderr
    assert "return evolve(" not in proc.stderr


def boom_agent(prompt: str) -> str:
    """Always-failing completion used as a text agent."""
    raise RuntimeError("401 unauthorized")


def test_evolve_suppresses_duplicate_early_stop_warning(tmp_path):
    """A dead agent must show the error once, without a raw RuntimeWarning dump."""
    rows = [{"prompt": f"q{i}", "gold": f"a{i}"} for i in range(12)]
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Answer.\n")
    spec = {
        "kind": "text", "target": str(prompt),
        "data": {"inline": rows}, "score": "contains",
        "agent": {"ref": "tests.test_plain_cli_warnings:boom_agent", "call": False},
        "reflect": {"ref": "tests.test_evolvespec:text_reflect", "call": False},
        "allow": ["tests."],
        "evolve": {"rounds": 2, "n_workers": 1, "seed": 0},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    store = str(tmp_path / "runs")
    proc = _run_cli(["evolve", str(path)], tmp_path=tmp_path, store=store)
    combined = proc.stderr + proc.stdout
    assert "RuntimeWarning:" not in combined
    assert "EarlyStopWarning:" not in combined
    assert "return evolve(" not in combined
    assert "401 unauthorized" in combined
    # status line and/or a dedicated error: line may both mention it
    assert combined.count("401 unauthorized") <= 2


def test_library_evolve_still_emits_early_stop_warning():
    """Direct ``evolve()`` callers keep standard warnings (#654)."""
    from agentdescent.evolution import EarlyStopWarning

    tasks = [Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(8)]

    def dead_run(rendered, task):
        raise RuntimeError("401 unauthorized")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = evolve(tasks, lambda t, o: 0.0, run=dead_run, strategy=AppendRules(),
                     propose=lambda rendered, t, o, s: "x",
                     rounds=2, n_workers=1, seed=0, held_out_frac=0.4)
    assert res.error and "401" in res.error
    early = [w for w in caught if issubclass(w.category, EarlyStopWarning)]
    assert early and "stopped early" in str(early[0].message)
    formatted = warnings.formatwarning(
        early[0].message, early[0].category, early[0].filename, early[0].lineno)
    assert "EarlyStopWarning" in formatted


def test_library_small_held_out_warning_unchanged(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compose(_small_dir_spec(str(tmp_path), n=8,
                                evolve={"rounds": 1, "n_workers": 1, "seed": 0})).run()
    held = [w for w in caught if issubclass(w.category, SmallHeldOutWarning)]
    assert held
