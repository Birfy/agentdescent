"""``plan`` reports the held-out set ``evolve()`` will warn about (#414).

``evolve()`` warns below four held-out tasks, but ``plan`` used to key on the
*task count* (``4 <= n < 8``) while the held-out size comes from
``held_out_frac``: 8-11 tasks at the default 0.3 hold out 2-3 and ``plan`` said
nothing, so a detached run only recorded it in ``log.txt``.
"""

import json
import os
import subprocess
import sys
import time
import warnings

import pytest

from agentdescent import runstore
from agentdescent.cli import main as cli_main, plan_payload
from agentdescent.evolution import (MIN_HELD_OUT, SmallHeldOutWarning, Task,
                                    held_out_size, min_tasks_for_held_out)
from agentdescent.evolvespec import EvolveSpec, compose, estimate
from agentdescent.mcp import Tools

from tests.test_evolvespec import _dir_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    return str(tmp_path / "runs")


def _text_spec(tmp_path, n, **evolve):
    rows = [{"prompt": f"q{i}", "gold": f"a{i}"} for i in range(n)]
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Answer.\n")
    return EvolveSpec.from_dict({
        "kind": "text", "target": str(prompt),
        "data": {"inline": rows}, "score": "contains",
        "agent": "echo", "reflect": "echo",
        "evolve": {"rounds": 1, "n_workers": 1, "seed": 0, **evolve},
    })


def _held_out_notes(notes):
    return [n for n in notes if "the held-out set has only" in n]


@pytest.mark.parametrize("n,held", [(8, 2), (10, 3), (11, 3)])
def test_plan_notes_a_tiny_held_out_set_at_the_default_fraction(tmp_path, n, held):
    payload = plan_payload(_text_spec(tmp_path, n))
    assert payload["ok"] is True
    (note,) = _held_out_notes(payload["notes"])
    assert f"only {held} task(s)" in note
    assert f"{n} tasks x held_out_frac=0.3" in note
    # both fixes are named, and the size that would be enough
    assert "more tasks" in note and "raise held_out_frac" in note
    assert "at least 12 tasks" in note


def test_plan_is_quiet_when_the_held_out_set_is_big_enough(tmp_path):
    assert not _held_out_notes(plan_payload(_text_spec(tmp_path, 12))["notes"])
    assert not _held_out_notes(plan_payload(_text_spec(tmp_path, 14))["notes"])
    # a higher fraction fixes 8 tasks: 8 x 0.5 holds out 4
    assert not _held_out_notes(
        plan_payload(_text_spec(tmp_path, 8, held_out_frac=0.5))["notes"])


def test_plan_still_refuses_fewer_than_four_tasks(tmp_path):
    payload = plan_payload(_text_spec(tmp_path, 3))
    assert payload["ok"] is False
    assert "need at least 4 tasks to split train/held-out" in payload["error"]


def test_plan_and_evolve_share_one_held_out_computation():
    """What `plan` reports is what `evolve()` splits and warns about."""
    from agentdescent import evolution as ev
    from tests.test_provider_and_split_contracts import _split

    for n in range(4, 40):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SmallHeldOutWarning)
            _, held = _split([Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(n)])
        # `_split` runs _build_engine at held_out_frac=0.4
        assert len(held) == held_out_size(n, 0.4), n
    for frac in (0.1, 0.25, 0.3, 0.4, 0.5, 0.7):
        need = min_tasks_for_held_out(frac)
        assert held_out_size(need, frac) >= MIN_HELD_OUT > held_out_size(need - 1, frac)
    assert ev.MIN_HELD_OUT == MIN_HELD_OUT == 4


def test_evolve_warning_uses_the_same_message_as_plan(tmp_path):
    spec = _text_spec(tmp_path, 8)
    (note,) = _held_out_notes(compose(spec).notes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compose(spec).run()
    (w,) = [w for w in caught if issubclass(w.category, SmallHeldOutWarning)]
    assert isinstance(w.message, RuntimeWarning) and str(w.message) == note


def test_estimate_counts_the_same_held_out_tasks_as_the_split(tmp_path):
    comp = compose(_text_spec(tmp_path, 10))
    assert estimate(comp)["held_out_tasks"] == held_out_size(10, 0.3) == 3


def _rows_spec(tmp_path, n):
    """The shared stub skill_dir spec, but with ``n`` cases."""
    rows = [{"prompt": w, "gold": w[::-1]}
            for w in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
                      "hotel", "india", "juliet", "kilo", "lima"][:n]]
    path = tmp_path / f"cases{n}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return _dir_spec(str(tmp_path), data={"path": str(path), "prompt": "prompt",
                                          "gold": "gold"})


def _wait(rd, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = rd.status()
        if st.state in ("done", "failed", "cancelled"):
            return st
        time.sleep(0.2)
    raise AssertionError(rd.status())


def test_a_detached_run_carries_the_note_in_status(store, tmp_path, capsys):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_rows_spec(tmp_path, 8).to_dict()))
    assert cli_main(["--store", store, "--json", "evolve", str(path), "--detach"]) == 0
    started = json.loads(capsys.readouterr().out)
    assert _held_out_notes(started["notes"])
    rd = runstore.get(started["run_id"], store=store)
    assert _wait(rd).state == "done"
    # `status` -- JSON and text -- shows it, not just log.txt
    assert cli_main(["--store", store, "--json", "status", started["run_id"]]) == 0
    st = json.loads(capsys.readouterr().out)
    (note,) = _held_out_notes(st["notes"])
    assert "only 2 task(s)" in note
    assert cli_main(["--store", store, "status", started["run_id"]]) == 0
    assert "note: the held-out set has only 2 task(s)" in capsys.readouterr().out


def test_mcp_start_returns_the_note_and_status_keeps_it(store, tmp_path):
    t = Tools(store)
    spec = _rows_spec(tmp_path, 8).to_dict()
    assert _held_out_notes(t.plan(spec)["notes"])
    started = t.start(spec)
    assert started["ok"] and _held_out_notes(started["notes"])
    _wait(runstore.get(started["run_id"], store=store))
    assert _held_out_notes(t.status(started["run_id"])["notes"])


def test_mcp_start_without_a_small_held_out_set_has_no_such_note(store, tmp_path):
    started = Tools(store).start(_rows_spec(tmp_path, 12).to_dict())
    assert started["ok"] and not _held_out_notes(started["notes"])
    _wait(runstore.get(started["run_id"], store=store))


def test_foreground_evolve_prints_one_plain_line(store, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_rows_spec(tmp_path, 8).to_dict()))
    env = {**os.environ, "AGENTDESCENT_HOME": str(tmp_path / "home"),
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run([sys.executable, "-m", "agentdescent.cli", "--store", store,
                           "evolve", str(path)], capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stderr.splitlines() if "held-out set has only" in ln]
    assert len(lines) == 1 and lines[0].startswith("warning: the held-out set has only 2")
    # no `file.py:123: RuntimeWarning:` prefix and no echoed source line
    assert "Warning:" not in proc.stderr and "return evolve(" not in proc.stderr


def test_one_task_is_all_held_out_and_the_train_count_agrees():
    """`_train_count` is `evolve()`'s slice: train + held-out never exceeds n."""
    from agentdescent.evolvespec import _train_count

    assert held_out_size(1, 0.4) == 1 and _train_count(1, 0.4) == 0
    for n in range(1, 40):
        for frac in (0.1, 0.3, 0.4, 0.5, 0.9):
            assert _train_count(n, frac) + held_out_size(n, frac) == n, (n, frac)
