"""bench/metasearch_gsm: evolving the task_sampler on GSM word problems, offline.

No network and no model: the row loaders are replaced by synthetic arithmetic
and the completion is scripted. The engine, the inner `evolve()`, the slot
installation and the report are real.
"""

import pytest

from agentdescent.meta import MetaOutcome, SLOT_PROTOCOLS, policy_source
from agentdescent.sampling import DifficultyWeighted

from bench import metasearch_gsm as bench


def _rows(n, offset=0):
    return [{"question": f"What is {i} plus 1?", "answer": f"#### {i + 1}"}
            for i in range(offset, offset + n)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setitem(bench.LOADERS, "gsmhard", lambda limit: _rows(limit))
    monkeypatch.setitem(bench.LOADERS, "gsm8k", lambda limit: _rows(limit, offset=10_000))


def test_windows_are_disjoint_and_named(monkeypatch):
    got = bench.windows("gsmhard", 3, 5, seed=0, pool=60)
    assert list(got) == ["gsmhard-0", "gsmhard-1", "gsmhard-2"]
    ids = [t.id for tasks in got.values() for t in tasks]
    assert len(ids) == len(set(ids)) == 15
    assert all(t.meta["gold"].startswith("#### ") for t in got["gsmhard-0"])
    # `pool` is a floor on what is fetched, not a cap: the guard fires when the
    # benchmark itself is smaller than the request, as GSM-Hard's 1319 rows are.
    monkeypatch.setitem(bench.LOADERS, "small", lambda limit: _rows(min(limit, 10)))
    with pytest.raises(ValueError, match="tasks requested"):
        bench.windows("small", 3, 5)


def _scripted(prompt: str) -> str:
    """A solver that only answers correctly once the instruction says to add."""
    if "improve" in prompt.lower() or "propose" in prompt.lower():
        return "Add one to the number in the problem and state the result."
    number = [int(w) for w in prompt.replace("?", " ").split() if w.isdigit()]
    if not number:
        return "0"
    return str(number[0] + 1) if "Add one" in prompt else str(number[0])


def test_gsm_problem_runs_an_inner_evolve_with_the_candidate_sampler():
    tasks = bench.windows("gsmhard", 1, 10, pool=40)["gsmhard-0"]
    problem = bench.gsm_problem(tasks, _scripted, rounds=2, workers=1)
    spec = policy_source("task_sampler")
    seed_sampler = spec.compile(spec.render(spec.initial()))
    assert isinstance(seed_sampler, SLOT_PROTOCOLS["task_sampler"])
    outcome = problem(seed_sampler, 0)
    assert isinstance(outcome, MetaOutcome)
    assert outcome.curve and outcome.detail["error"] is None
    # ...and a different sampler is honoured rather than ignored.
    picked = []

    class Recording(DifficultyWeighted):
        def pick(self, keys, round_index):
            choice = super().pick(keys, round_index)
            picked.append(choice)
            return choice

    problem(Recording(), 0)
    assert picked, "the installed sampler was never asked for a task"


def _flat(value):
    return lambda policy, seed: MetaOutcome(curve=[value] * 3, final=value)


def _sampler_sensitive(bonus):
    """Rewards a sampler whose class name the reflector's proposal introduces."""
    def problem(policy, seed):
        base = 0.4 + (bonus if type(policy).__name__ == "Policy"
                      and getattr(policy, "greedy", False) else 0.0)
        return MetaOutcome(curve=[base] * 3, final=base)
    return problem


PROPOSAL = """```python
class Policy:
    greedy = True

    def pick(self, keys, round_index):
        return min(self.seen, key=self.seen.get) if self.seen else keys[0]

    def record(self, task_id, score):
        self.seen[task_id] = score

    seen = {}
```"""


def test_run_experiment_groups_the_report_and_gives_a_ratio_per_group():
    payload = bench.run_experiment(
        lambda prompt: PROPOSAL,
        train={"s0": _sampler_sensitive(0.2), "s1": _sampler_sensitive(0.2)},
        validate={"u0": _sampler_sensitive(0.1), "o0": _flat(0.4)},
        groups={"train": ["s0", "s1"], "unseen": ["u0"], "other": ["o0"]},
        seeds=[0, 1], validate_seeds=[100, 101], rounds=2, workers=2)
    assert payload["outer"]["error"] is None and "greedy" in payload["evolved_source"]
    assert payload["by_group"]["train"]["gain"] == pytest.approx(0.2)
    assert payload["by_group"]["unseen"]["gain"] == pytest.approx(0.1)
    assert payload["by_group"]["other"]["gain"] == 0.0
    assert payload["transfer_ratio"] == {"unseen": pytest.approx(0.5), "other": 0.0}
    text = bench.format_report(payload)
    assert "transfer ratio (unseen" in text and "transfer ratio (other" in text


def test_run_experiment_refuses_overlapping_problems_or_seeds():
    with pytest.raises(ValueError, match="share problems"):
        bench.run_experiment(lambda p: "", train={"a": _flat(0.5)}, validate={"a": _flat(0.5)},
                             groups={"train": ["a"]}, seeds=[0], validate_seeds=[1],
                             rounds=1, workers=1)
    with pytest.raises(ValueError, match="seeds must not overlap"):
        bench.run_experiment(lambda p: "", train={"a": _flat(0.5)}, validate={"b": _flat(0.5)},
                             groups={"train": ["a"], "unseen": ["b"]}, seeds=[0],
                             validate_seeds=[0], rounds=1, workers=1)


def test_build_problems_keeps_the_three_groups_disjoint():
    train, validate, groups = bench.build_problems(
        _scripted, source="gsmhard", other="gsm8k", train_windows=2, unseen_windows=1,
        other_windows=1, size=6, data_seed=0, inner={"rounds": 1, "workers": 1})
    assert set(train) == {"gsmhard-0", "gsmhard-1"} == set(groups["train"])
    assert set(validate) == {"gsmhard-2", "gsm8k-0"}
    assert groups["unseen"] == ["gsmhard-2"] and groups["other"] == ["gsm8k-0"]
    assert not set(train) & set(validate)


def test_dry_run_touches_nothing(monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError("dry-run crossed a boundary")

    monkeypatch.setattr(bench, "completion_for", forbidden)
    monkeypatch.setattr(bench, "build_problems", forbidden)
    assert bench.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "task_sampler" in out and "gsmhard" in out
