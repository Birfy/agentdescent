"""A reflector that raises must be seen, and a run it cannot serve must fail (#395).

Before: ``propose`` raising was folded into the worker failures *after* that
worker's rollout had already counted as a success, so every round looked healthy
-- ``considered: 0``, ``reasons: {}``, ``state: done``, exit 0 -- and the cause
appeared nowhere.
"""

import json
import warnings

import pytest

from agentdescent import cli, evolution, runstore
from agentdescent.evolution import AppendRules, Task, evolve

from tests.test_evolvespec import _dir_spec
from tests.test_runstore_cli import _cli, store  # noqa: F401  (fixture)


class _Agent:
    """Never solves a task, so every task reaches ``propose``."""

    def __init__(self, propose):
        self._propose = propose

    def solve(self, rendered, task):
        return "no"

    def propose(self, rendered, task, output, reward):
        return self._propose(task)


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt="q", meta={"h": f"H{i % 3}"}) for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def _boom(task):
    raise RuntimeError("no key")


def _run(propose, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = evolve(_tasks(), REWARD, agent=_Agent(propose), strategy=AppendRules(),
                     rounds=kw.pop("rounds", 4), **kw)
    return res, [str(w.message) for w in caught]


def test_a_reflector_that_always_raises_fails_the_run_with_the_cause():
    res, warned = _run(_boom)
    assert res.error and "no key" in res.error
    assert "reflector failed on every call" in res.error
    assert res.stop_reason == "error"
    assert len(res.history) == evolution.PROPOSE_FAIL_ROUNDS < 4      # stopped early
    for h in res.history:
        assert h.propose_errors > 0 and h.propose_error == "RuntimeError: no key"
        assert h.considered == 0
    assert any("no key" in m for m in warned)


def test_a_flaky_reflector_does_not_fail_the_run_but_is_recorded():
    def flaky(task):
        if task.meta["h"] == "H0":                  # a third of the calls
            raise RuntimeError("429 slow down")
        return task.meta["h"]

    res, warned = _run(flaky)
    assert res.error is None and len(res.history) == 4
    assert sum(h.propose_errors for h in res.history) > 0
    assert all(h.propose_error in (None, "RuntimeError: 429 slow down") for h in res.history)
    assert any("propose() raised" in m and "429 slow down" in m for m in warned)


def test_one_bad_round_is_not_a_dead_reflector():
    done = []                                        # rounds completed so far

    def down_for_round_one(task):
        if not done:
            raise RuntimeError("blip")
        return task.meta["h"]

    res, _ = _run(down_for_round_one, rounds=4, on_round=done.append)
    assert res.error is None and len(res.history) == 4
    assert res.history[0].propose_errors > 0
    assert all(h.propose_errors == 0 for h in res.history[1:])


def test_a_one_round_run_cannot_tell_so_it_warns_instead_of_failing():
    res, warned = _run(_boom, rounds=1)
    assert res.error is None
    assert res.history[0].propose_errors > 0
    assert any("propose() raised" in m and "no key" in m for m in warned)


def test_a_healthy_run_records_no_errors_and_no_warning():
    res, warned = _run(lambda task: task.meta["h"])
    assert res.error is None
    assert all(h.propose_errors == 0 and h.propose_error is None for h in res.history)
    assert not [m for m in warned if "propose" in m]


def test_the_new_round_fields_survive_save_and_load(tmp_path):
    res, _ = _run(_boom)
    path = tmp_path / "r.json"
    res.save(str(path))
    saved = json.loads(path.read_text())["history"][0]
    assert saved["propose_errors"] > 0 and saved["propose_error"] == "RuntimeError: no key"
    back = evolution.EvolutionResult.load(str(path))
    assert back.history[0].propose_errors == res.history[0].propose_errors
    # a result.json from before the fields existed still loads
    saved.pop("propose_errors"), saved.pop("propose_error")
    legacy = {**json.loads(path.read_text()), "history": [saved]}
    path.write_text(json.dumps(legacy))
    assert evolution.EvolutionResult.load(str(path)).history[0].propose_errors == 0


def raising_reflect(prompt):
    raise RuntimeError("no key")


def test_cli_evolve_fails_loudly_and_records_the_reflector_error(store, tmp_path, capsys):
    spec = _dir_spec(str(tmp_path), reflect={"ref": "tests.test_propose_errors:raising_reflect",
                                             "call": False})
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        code, out = _cli("--store", store, "--json", "evolve", str(path))
    assert code != 0
    payload = json.loads(out)
    assert payload["state"] == "failed" and "no key" in payload["error"]
    rd = runstore.get(payload["run_id"], store=store)
    st = rd.status()
    assert st.state == "failed" and "reflector failed on every call" in st.error
    assert st.propose_errors > 0 and st.propose_error == "RuntimeError: no key"
    rows = rd.rounds()
    assert rows and all(r["propose_errors"] > 0 and "no key" in r["propose_error"]
                        for r in rows)
    assert all(r["considered"] == 0 for r in rows)
    # status / watch / show surface it
    code, out = _cli("--store", store, "status", payload["run_id"])
    assert "propose_errors=" in out and "no key" in out
    code, out = _cli("--store", store, "show", payload["run_id"])
    assert "propose_errors:" in out
    code, out = _cli("--store", store, "--json", "show", payload["run_id"])
    assert json.loads(out)["propose_errors"] > 0
    code, out = _cli("--store", store, "--json", "status", payload["run_id"])
    assert json.loads(out)["propose_errors"] > 0


def test_a_run_with_no_reflector_errors_adds_nothing_to_the_status_line():
    st = runstore.RunStatus(run_id="r", state="done")
    assert "propose_errors" not in cli._status_line(st)
    st.propose_errors, st.propose_error = 3, "RuntimeError: x"
    assert "propose_errors=3" in cli._status_line(st)


def test_init_warns_when_the_starter_reflector_cannot_be_called(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    skill = tmp_path / "skill"
    skill.mkdir()
    code = cli.main(["init", str(skill), "--out", str(tmp_path / "spec.json")])
    out = capsys.readouterr().out
    assert code == 0 and "OPENAI_API_KEY is not set" in out
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cli.main(["init", str(skill), "--out", str(tmp_path / "spec2.json")])
    assert "OPENAI_API_KEY is not set" not in capsys.readouterr().out


def test_a_result_with_round_fields_from_a_newer_version_still_loads(tmp_path):
    res, _ = _run(lambda task: task.meta["h"], rounds=2)
    path = tmp_path / "r.json"
    res.save(str(path))
    data = json.loads(path.read_text())
    data["history"][0]["a_field_from_the_future"] = 1
    path.write_text(json.dumps(data))
    assert len(evolution.EvolutionResult.load(str(path)).history) == 2
