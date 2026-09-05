"""bench/metasearch_slots: evolving a decision slot on a real dataset, offline.

No network and no model: the row loaders are replaced by synthetic arithmetic
and the completion is scripted. The engine, the inner `evolve()`, the slot
installation and the report are real.
"""

import pytest

from agentdescent.meta import MetaOutcome, SLOT_PROTOCOLS, policy_source
from agentdescent.sampling import DifficultyWeighted

from bench import metasearch_slots as bench


def _rows(n, offset=0):
    return [{"question": f"What is {i} plus 1?", "answer": f"#### {i + 1}"}
            for i in range(offset, offset + n)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import dataclasses

    def fake(name, offset=0):
        return dataclasses.replace(bench.BENCHMARKS[name], load=lambda limit: _rows(limit, offset))

    monkeypatch.setitem(bench.BENCHMARKS, "gsmhard", fake("gsmhard"))
    monkeypatch.setitem(bench.BENCHMARKS, "gsm8k", fake("gsm8k", 10_000))


def test_windows_are_disjoint_and_named(monkeypatch):
    got = bench.windows("gsmhard", 3, 5, seed=0, pool=60)
    assert list(got) == ["gsmhard-0", "gsmhard-1", "gsmhard-2"]
    ids = [t.id for tasks in got.values() for t in tasks]
    assert len(ids) == len(set(ids)) == 15
    assert all(t.meta["gold"].startswith("#### ") for t in got["gsmhard-0"])
    # `pool` is a floor on what is fetched, not a cap: the guard fires when the
    # benchmark itself is smaller than the request, as GSM-Hard's 1319 rows are.
    import dataclasses
    monkeypatch.setitem(bench.BENCHMARKS, "small", dataclasses.replace(
        bench.BENCHMARKS["gsmhard"], name="small", load=lambda limit: _rows(min(limit, 10))))
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
    problem = bench.inner_problem(tasks, _scripted, benchmark=bench.BENCHMARKS["gsmhard"], rounds=2, workers=1)
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


#: A correct greedy sampler. The first version of this fixture answered from
#: its own memory (`min(self.seen, ...)`) rather than from `keys`, which is the
#: stale-id bug every live reflector proposal had -- and the stricter smoke test
#: in `agentdescent.meta` caught the fixture too, which is the gate working.
PROPOSAL = """```python
class Policy:
    greedy = True

    def __init__(self):
        self.seen = {}

    def pick(self, keys, round_index):
        unseen = [k for k in keys if k not in self.seen]
        if unseen:
            return unseen[0]
        return min(keys, key=lambda k: self.seen.get(k, 0.0))

    def record(self, task_id, score):
        self.seen[task_id] = score
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


def test_every_shared_flag_is_honoured_or_refused(monkeypatch):
    """A flag the parser declares and nothing reads is the defect this checks."""
    import inspect

    source = inspect.getsource(bench)
    for dest, read_by in (("max_seconds", "max_seconds=args.max_seconds"),
                          ("budget_rollouts", "args.budget_rollouts"),
                          ("eval_concurrency", "eval_concurrency=args.eval_concurrency"),
                          ("seed", "outer_seed=args.seed"),
                          ("serial", "worker_count(")):
        assert read_by in source, f"--{dest.replace('_', '-')} is declared and never read"
    with pytest.raises(SystemExit, match="not supported"):
        bench.main(["--async"])
    with pytest.raises(SystemExit, match="not supported"):
        bench.main(["--pipelined-gate"])


def test_main_writes_a_complete_result_file(monkeypatch, tmp_path):
    """The payload path, end to end -- the one the tests above did not cover.

    `main()` assembled its own usage dict with the Anthropic SDK's field names,
    which `Usage` does not have. Every assertion in this file passed and the
    AttributeError landed on the last line of a live run, after an hour of
    measurement and before any of it was written. So this test drives `main()`
    itself, with only the model and the problems replaced.
    """
    import json

    from agentdescent.agents import Usage
    from agentdescent.meta import MetaOutcome

    calls = {"n": 0}

    def fake_completion(*a, **k):
        usage = k.get("usage")

        def complete(prompt: str) -> str:
            calls["n"] += 1
            if usage is not None:
                usage.record(prompt_tokens=7, completion_tokens=3, seconds=0.01)
            return PROPOSAL
        return complete

    def fake_build(complete, **kwargs):
        train = {"t0": _sampler_sensitive(0.2)}
        validate = {"u0": _sampler_sensitive(0.1), "o0": _flat(0.4)}
        return train, validate, {"train": ["t0"], "unseen": ["u0"], "other": ["o0"]}

    monkeypatch.setattr(bench, "completion_for", fake_completion)
    monkeypatch.setattr(bench, "build_problems", fake_build)
    out = tmp_path / "result.json"
    assert bench.main(["--yes", "--rounds", "2", "--workers", "2", "--seeds", "4",
                       "--validate-seeds", "2", "--output", str(out)]) == 0
    payload = json.loads(out.read_text())
    # `calls` counts the engine's own rollout accounting too, so it is only
    # bounded below; the token counts come from the adapter alone.
    assert payload["usage"]["calls"] >= calls["n"] > 0
    assert payload["usage"]["prompt_tokens"] == 7 * calls["n"]
    assert payload["usage"]["completion_tokens"] == 3 * calls["n"]
    assert payload["usage"]["total_tokens"] == 10 * calls["n"]
    assert payload["usage"]["wall_seconds"] > 0
    assert set(payload["by_group"]) == {"train", "unseen", "other"}
    assert set(payload["transfer_ratio"]) == {"unseen", "other"}
    assert payload["config"]["template"] == bench.BENCHMARKS["gsmhard"].template
    assert payload["outer"]["error"] is None and payload["evolved_source"]
    assert isinstance(Usage().calls, int)      # the field the bug got wrong


def test_meta_reward_choices_and_their_shapes():
    from agentdescent.meta import MetaOutcome

    rising = MetaOutcome(curve=[0.4, 0.5, 0.8, 0.8], final=0.8)
    flat = MetaOutcome(curve=[0.9, 0.9, 0.9, 0.9], final=0.9)
    ttq = bench.meta_reward_for("time-to-quality", 0.75)
    assert ttq(rising) == pytest.approx(1 / 3)      # reached at the third sweep
    assert ttq(MetaOutcome(curve=[0.4, 0.4], final=0.4)) == 0.0
    # The saturation this option exists for: a bar below the seed's own score
    # gives every run 1.0, which is the failure mode `auc` hit on this domain.
    assert bench.meta_reward_for("time-to-quality", 0.3)(rising) == 1.0
    assert bench.meta_reward_for("auc", 0.0)(flat) == pytest.approx(0.9)
    assert bench.meta_reward_for("final", 0.0)(rising) == 0.8
    with pytest.raises(SystemExit, match="unknown --meta-reward"):
        bench.meta_reward_for("nope", 0.5)


def test_the_meta_reward_reaches_both_the_outer_loop_and_the_validation(monkeypatch):
    seen = {}
    real_evolve, real_validate = bench.meta_evolve, bench.meta_validate

    def spy_evolve(problems, **kw):
        seen["evolve"] = kw.get("meta_reward")
        return real_evolve(problems, **kw)

    def spy_validate(spec, before, after, problems, **kw):
        seen["validate"] = kw.get("meta_reward")
        return real_validate(spec, before, after, problems, **kw)

    monkeypatch.setattr(bench, "meta_evolve", spy_evolve)
    monkeypatch.setattr(bench, "meta_validate", spy_validate)
    reward = bench.meta_reward_for("time-to-quality", 0.5)
    bench.run_experiment(lambda p: PROPOSAL,
                         train={"t0": _sampler_sensitive(0.2), "t1": _sampler_sensitive(0.2)},
                         validate={"v": _flat(0.4)},
                         groups={"train": ["t0", "t1"], "unseen": ["v"]}, seeds=[0, 1],
                         validate_seeds=[9], rounds=1, workers=1, meta_reward=reward)
    assert seen["evolve"] is reward is seen["validate"], (
        "the gate and the report must score an inner run the same way")


def test_cached_completion_makes_a_flaky_model_reproducible(tmp_path):
    """The property the paired comparison rests on, pinned.

    Validating the seed sampler against itself reported a gain of -0.0625 on a
    live window because the rollouts were fresh model calls and the endpoint
    was not perfectly deterministic. A prompt cache is what closes that.
    """
    answers = iter(["first", "second", "third"])
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        return next(answers)

    cached = bench.cached_completion(flaky, str(tmp_path / "c"), key_extra="m|0")
    assert [cached("same prompt") for _ in range(3)] == ["first"] * 3
    assert calls["n"] == 1
    assert cached.stats == {"hits": 2, "misses": 1}
    # A different prompt, or a different request shape, is a different entry.
    assert cached("other prompt") == "second"
    other = bench.cached_completion(flaky, str(tmp_path / "c"), key_extra="m|0.7")
    assert other("same prompt") == "third", "temperature must not share an entry"
    # A half-written entry is a miss, not a crash.
    import glob
    path = sorted(glob.glob(str(tmp_path / "c" / "*.json")))[0]
    open(path, "w").write("{ truncated")
    assert bench.cached_completion(lambda p: "recomputed", str(tmp_path / "c"),
                                   key_extra="m|0")("same prompt") in {"recomputed", "first"}


def test_an_inner_run_is_a_function_of_the_sampler_when_completions_are_cached(tmp_path):
    """Two runs of the same sampler must give the same curve, even if the model
    answers differently the second time."""
    tasks = bench.windows("gsmhard", 1, 8, pool=40)["gsmhard-0"]
    seq = {"n": 0}

    def drifting(prompt: str) -> str:
        # Answers correctly only on the first pass, so an uncached second run
        # would produce a different curve.
        seq["n"] += 1
        if "improve" in prompt.lower() or "propose" in prompt.lower():
            return "Add one to the number in the problem and state the result."
        number = [int(w) for w in prompt.replace("?", " ").split() if w.isdigit()]
        if not number:
            return "0"
        base = number[0] + 1 if "Add one" in prompt else number[0]
        return str(base if seq["n"] % 2 else base + 100)

    cached = bench.cached_completion(drifting, str(tmp_path / "c"), key_extra="k")
    problem = bench.inner_problem(tasks, cached, benchmark=bench.BENCHMARKS["gsmhard"], rounds=2, workers=1)
    spec = policy_source("task_sampler")
    sampler = spec.render(spec.initial())
    first = problem(spec.compile(sampler), 0)
    second = problem(spec.compile(sampler), 0)
    assert first.curve == second.curve, "the inner run is not a function of the sampler"
    assert cached.stats["hits"] > 0


def test_a_run_that_commits_nothing_still_says_what_it_tried():
    """The gap the first four live runs had: `{'oracle-rejected': 3}` could not
    distinguish three samplers that lost from three that never compiled."""
    from agentdescent.evolution import Task
    from agentdescent.meta import policy_source

    spec = policy_source("task_sampler")
    good = ("class Policy:\n"
            "    def pick(self, keys, round_index): return keys[0]\n"
            "    def record(self, task_id, score): pass\n")
    task = Task(id="p0:0", prompt="a problem")
    propose, log = bench.recording_reflector(lambda prompt: good, spec)
    assert propose("rendered", task, "{}", 0.0) == good
    assert log == [{"source": good, "accepted_by_gate": True, "reason": "",
                    "on_task": "p0:0"}]

    propose, log = bench.recording_reflector(lambda prompt: "import os", spec)
    propose("rendered", task, "{}", 0.0)
    assert log[0]["accepted_by_gate"] is False and log[0]["reason"]


def test_the_payload_carries_the_proposals(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(bench, "completion_for",
                        lambda *a, **k: (lambda prompt: PROPOSAL))
    monkeypatch.setattr(bench, "build_problems", lambda complete, **kw: (
        {"t0": _sampler_sensitive(0.2), "t1": _sampler_sensitive(0.2)},
        {"u0": _flat(0.4)}, {"train": ["t0", "t1"], "unseen": ["u0"], "other": []}))
    out = tmp_path / "r.json"
    assert bench.main(["--yes", "--rounds", "2", "--workers", "2", "--seeds", "2",
                       "--validate-seeds", "1", "--output", str(out)]) == 0
    outer = json.loads(out.read_text())["outer"]
    assert outer["proposals"] and all(p["accepted_by_gate"] for p in outer["proposals"])
    assert outer["proposals_rejected_by_gate"] == 0
    assert "invalid_proposals" in outer


def test_hard_rows_keeps_only_what_the_seed_gets_wrong():
    """A saturated validation group can only move down -- measured, the seed
    instruction scored 1.000 on both plain GSM8K windows."""
    rows = [{"question": f"What is {i} plus 1?", "answer": f"#### {i + 1}"}
            for i in range(20)]

    def solver(prompt):
        # Right on the even numbers, wrong on the odd ones.
        n = [int(w) for w in prompt.replace("?", " ").split() if w.isdigit()][0]
        return str(n + 1) if n % 2 == 0 else "0"

    kept = bench.hard_rows(rows, solver, keep=5, pool=20)
    assert len(kept) == 5
    assert all(int(r["answer"].split()[-1]) % 2 == 0 for r in kept), \
        "the kept rows must be the ones the seed answered wrongly"
    with pytest.raises(ValueError, match="are hard for the seed"):
        bench.hard_rows(rows, solver, keep=15, pool=20)


def test_build_problems_can_take_the_hard_subset_for_the_other_benchmark(monkeypatch):
    seen = {}

    def fake_hard(rows, complete, *, keep, pool, benchmark=None):
        seen["keep"], seen["pool"] = keep, pool
        seen["benchmark"] = benchmark.name if benchmark else None
        return list(rows)[:keep]

    monkeypatch.setattr(bench, "hard_rows", fake_hard)
    train, validate, groups = bench.build_problems(
        _scripted, source="gsmhard", other="gsm8k", train_windows=1, unseen_windows=1,
        other_windows=1, size=6, data_seed=0, inner={"rounds": 1, "workers": 1},
        hard_other=True, hard_pool=50)
    assert seen == {"keep": 6, "pool": 50, "benchmark": "gsm8k"}
    assert groups["other"] == ["gsm8k-0"] and "gsm8k-0" in validate
    # ...and off by default, so the plain path is unchanged.
    seen.clear()
    bench.build_problems(_scripted, source="gsmhard", other="gsm8k", train_windows=1,
                         unseen_windows=1, other_windows=1, size=6, data_seed=0,
                         inner={"rounds": 1, "workers": 1})
    assert seen == {}
