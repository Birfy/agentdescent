"""bench/metasearch_algotune: the plumbing, offline. The sandbox, the task files
and the model are all replaced; the outer loop and the report are real."""

from types import SimpleNamespace

import pytest

from agentdescent.meta import MetaOutcome, PrioritySelection, PRIORITY_SEED

from bench import metasearch_algotune as bench


def _fake_run(seen):
    def run_agentdescent_era(complete, **kwargs):
        seen.append(kwargs)
        history = [SimpleNamespace(held_out_reward=r) for r in (0.2, 0.5, 0.5)]
        result = SimpleNamespace(history=history, final_reward=0.5, rollouts=3,
                                 outcomes=lambda: {"committed": 1}, stop_reason="rounds",
                                 error=None)
        tree = SimpleNamespace(nodes=[1, 2, 3, 4],
                               summary=lambda: {"selection": type(kwargs["selection"]).__name__})
        return SimpleNamespace(result=result, tree=tree,
                               baseline_test_metrics={"speedup": 1.0},
                               best_test_metrics={"speedup": 1.7})
    return run_agentdescent_era


def test_algotune_problem_threads_the_policy_and_the_seed(monkeypatch):
    seen = []
    monkeypatch.setattr(bench, "prepare_suite", lambda task, **kw: SimpleNamespace(task=task))
    monkeypatch.setattr(bench, "algotune_domain", lambda suite, **kw: {"task": suite.task})
    monkeypatch.setattr(bench, "run_agentdescent_era", _fake_run(seen))
    problem = bench.algotune_problem("svd", lambda p: "", iterations=6)
    policy = PrioritySelection(PRIORITY_SEED)
    outcome = problem(policy, 7)
    assert seen[0]["selection"] is policy and seen[0]["seed"] == 7
    assert seen[0]["domain"] == {"task": "svd"} and seen[0]["iterations"] == 6
    assert outcome.curve == [0.2, 0.5, 0.5] and outcome.final == 0.5
    assert outcome.detail["best_speedup"] == 1.7 and outcome.detail["selection"] == "PrioritySelection"
    with pytest.raises(ValueError, match="not an AlgoTune task"):
        bench.algotune_problem("no_such_task", lambda p: "")


def _scripted_problem(bias):
    def problem(policy, seed):
        # A greedier rule (larger rank weight) does better here by `bias`.
        rank_weight = policy.priority(1.0, 0, 0, 1.0, 0, 1) - policy.priority(0.0, 0, 0, 1.0, 0, 1)
        base = min(1.0, 0.3 + bias * (rank_weight - 1.0))
        return MetaOutcome(curve=[base, base, base], final=base)
    return problem


PROPOSAL = """```python
def priority(rank, visits, total, prior, depth, n_nodes):
    # weight rank more
    return 2.0 * rank + (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)
```"""


def test_run_experiment_reports_train_validate_and_the_transfer_ratio():
    payload = bench.run_experiment(
        lambda prompt: PROPOSAL,
        train={"a": _scripted_problem(0.2), "b": _scripted_problem(0.2)},
        validate={"c": _scripted_problem(0.0), "d": _scripted_problem(0.1)},
        seeds=[0, 1], validate_seeds=[100, 101], rounds=2, workers=2)
    assert payload["outer"]["error"] is None
    assert "2.0 * rank" in payload["evolved_source"]
    assert set(payload["validation"]) == {"a", "b", "c", "d"}
    assert payload["by_set"]["train"]["gain"] == pytest.approx(0.2)
    assert payload["by_set"]["validate"]["gain"] == pytest.approx(0.05)
    assert payload["transfer_ratio"] == pytest.approx(0.25)
    text = bench.format_report(payload)
    assert "transfer ratio" in text and "validate" in text


def test_run_experiment_refuses_overlap():
    with pytest.raises(ValueError, match="share tasks"):
        bench.run_experiment(lambda p: "", train={"a": _scripted_problem(0)},
                             validate={"a": _scripted_problem(0)}, seeds=[0],
                             validate_seeds=[1], rounds=1, workers=1)
    with pytest.raises(ValueError, match="seeds must not overlap"):
        bench.run_experiment(lambda p: "", train={"a": _scripted_problem(0)},
                             validate={"b": _scripted_problem(0)}, seeds=[0],
                             validate_seeds=[0], rounds=1, workers=1)


def test_dry_run_touches_nothing(monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError("dry-run crossed a boundary")

    for name in ("prepare_suite", "run_agentdescent_era", "completion_for"):
        monkeypatch.setattr(bench, name, forbidden)
    assert bench.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "psd_cone_projection" in out and "affine_transform_2d" in out
    with pytest.raises(SystemExit, match="unknown AlgoTune"):
        bench.main(["--dry-run", "--train-tasks", "nope"])
