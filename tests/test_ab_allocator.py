"""Offline main-path accounting tests for the allocator A/B harness."""

import json
from types import SimpleNamespace


def _run_main(monkeypatch, capsys, tmp_path, *, scores, spend, seeds=(7,),
              evolve_failures=None, build_fail_call=None,
              allocator_counts=None):
    import bench.ab_allocator as ab

    class Workload:
        evolve_kwargs = {}
        tasks = []
        reward = None
        agent = strategy = None

        def __init__(self, seed):
            self.seed = seed

        def test_eval(self, result):
            value = scores[(result.seed, result.arm)]
            if isinstance(value, BaseException):
                raise value
            return value

    build_calls = 0

    def build(_pool, _top_k, seed, _completion, **_shared):
        nonlocal build_calls
        build_calls += 1
        if build_calls == build_fail_call:
            raise RuntimeError("mock build failure")
        return Workload(seed)

    class StubAllocator:
        seeds = []

        def __init__(self, seed=0, **_kwargs):
            self.seed = seed
            self.n_skipped, self.n_observed = (allocator_counts or {}).get(
                seed, (0, 0))
            self.n_counterfactual = 0
            type(self).seeds.append(seed)

    evolve_seeds = []

    def fake_evolve(*_args, usage, seed, allocator=None, **_kwargs):
        arm = "on" if allocator is not None else "off"
        evolve_seeds.append((seed, arm))
        usage.record(prompt_tokens=spend[(seed, arm)])
        failure = (evolve_failures or {}).get((seed, arm))
        if failure is not None:
            raise failure
        return SimpleNamespace(
            error=None,
            final_reward=0.5,
            rollouts=1,
            outcomes=lambda: {"committed": 1},
            seed=seed,
            arm=arm,
        )

    monkeypatch.setattr(ab, "confirm", lambda _args: True)
    monkeypatch.setattr(ab, "completion_for", lambda *_a, **_k: None)
    monkeypatch.setattr(ab, "_finer", build)
    monkeypatch.setattr(ab, "ValueBudgetAllocator", StubAllocator)
    monkeypatch.setattr(ab, "evolve", fake_evolve)

    json_path = tmp_path / "ab-rows.json"
    argv = ["--dataset", "finer", "--seeds", ",".join(map(str, seeds)),
            "--max-tokens", "100", "--yes", "--json", str(json_path)]
    rc = ab.main(argv)
    captured = capsys.readouterr()
    payload = json.loads(json_path.read_text())
    return rc, captured.out, captured.err, payload, StubAllocator.seeds, evolve_seeds


def _spend(seeds, off, on):
    return {(seed, arm): (off if arm == "off" else on)
            for seed in seeds for arm in ("off", "on")}


def test_one_arm_scoring_failure_drops_pair_but_keeps_partner_spend(
        monkeypatch, capsys, tmp_path):
    rc, out, err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        scores={(7, "off"): 0.75,
                (7, "on"): RuntimeError("final scorer unavailable")},
        spend=_spend((7,), 11, 23))

    assert rc == 4
    assert "scoring failed: RuntimeError" in err
    assert "1 failed (spent 23 tokens before failing)" in out
    assert "incomplete seed pairs dropped from quality/cost comparison: [7]" in out
    assert "34 tokens (23 in failed runs; 11 in non-failed runs)" in out
    assert "No complete successfully scored seed pairs" in out
    assert "| arm | complete seeds |" not in out
    rows = payload["rows"]
    assert [(r["arm"], r["tokens"]) for r in rows] == [("off", 11), ("on", 23)]
    assert rows[0]["error"] is None
    assert rows[1]["scoring_error"] == (
        "final scoring failed: RuntimeError: final scorer unavailable")
    assert rows[1]["error"] == rows[1]["scoring_error"]


def test_both_arm_scoring_failures_report_empty_comparison_and_all_spend(
        monkeypatch, capsys, tmp_path):
    rc, out, err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(9,),
        scores={(9, "off"): ValueError("off scorer"),
                (9, "on"): RuntimeError("on scorer")},
        spend=_spend((9,), 7, 13))

    assert rc == 4
    assert "2 failed (spent 20 tokens before failing)" in out
    assert "20 tokens (20 in failed runs; 0 in non-failed runs)" in out
    assert "No complete successfully scored seed pairs" in out
    assert len(payload["rows"]) == 2
    assert all(r["scoring_error"] for r in payload["rows"])
    assert sum(r["tokens"] for r in payload["rows"]) == 20


def test_missing_final_score_is_not_a_measurable_pair(
        monkeypatch, capsys, tmp_path):
    rc, out, err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        scores={(7, "off"): 0.75, (7, "on"): None},
        spend=_spend((7,), 11, 23))

    assert rc == 4
    assert "scoring failed: final scoring returned no score" in err
    assert "No complete successfully scored seed pairs" in out
    assert payload["rows"][1]["scoring_error"] == "final scoring returned no score"
    assert payload["rows"][1]["tokens"] == 23


def test_mixed_pairs_use_same_scored_seeds_for_quality_and_cost(
        monkeypatch, capsys, tmp_path):
    rc, out, _err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(3, 5),
        scores={(3, "off"): 0.4, (3, "on"): 0.6,
                (5, "off"): 0.99,
                (5, "on"): RuntimeError("one final score failed")},
        spend={(3, "off"): 10, (3, "on"): 20,
               (5, "off"): 90, (5, "on"): 80})

    assert rc == 0
    assert "complete seeds | test (min/med/max) | tokens (med)" in out
    assert "incomplete seed pairs dropped from quality/cost comparison: [5]" in out
    assert "170 tokens (80 in failed runs; 90 in non-failed runs)" in out
    assert "| off | 1 | 0.400 / 0.400 / 0.400 | 10 |" in out
    assert "| on | 1 | 0.600 / 0.600 / 0.600 | 20 |" in out
    table = out.split("|---|---|---|---|---|---|---|\n", 1)[1]
    assert "0.990" not in table.split("\n\n", 1)[0]
    assert len(payload["rows"]) == 4


def test_treatment_allocator_receives_each_distinct_run_seed(
        monkeypatch, capsys, tmp_path):
    seeds = (17, 42)
    rc, out, _err, payload, alloc_seeds, evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=seeds,
        scores={(seed, arm): 0.5 for seed in seeds for arm in ("off", "on")},
        spend=_spend(seeds, 10, 20))

    assert rc == 0
    assert alloc_seeds == [17, 42]
    assert evolve_seeds == [(17, "off"), (17, "on"),
                            (42, "off"), (42, "on")]
    assert "| off | 2 | 0.500 / 0.500 / 0.500 | 10 |" in out
    assert "| on | 2 | 0.500 / 0.500 / 0.500 | 20 |" in out
    assert len(payload["rows"]) == 4


def test_successful_pair_with_allocator_decisions_keeps_paired_summary(
        monkeypatch, capsys, tmp_path):
    rc, out, _err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(12,),
        scores={(12, "off"): 0.4, (12, "on"): 0.6},
        spend=_spend((12,), 10, 20),
        allocator_counts={12: (2, 5)})

    assert rc == 0
    assert "| off | 1 | 0.400 / 0.400 / 0.400 | 10 |" in out
    assert "| on | 1 | 0.600 / 0.600 / 0.600 | 20 |" in out
    assert "2/5 |" in out
    assert "No complete successfully scored seed pairs" not in out
    assert "but no complete successfully scored seed pairs survived" not in out
    assert len(payload["rows"]) == 2


def test_empty_pair_with_allocator_decisions_reports_no_comparison(
        monkeypatch, capsys, tmp_path):
    rc, out, _err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(13,),
        scores={(13, "off"): 0.4,
                (13, "on"): RuntimeError("final scorer unavailable")},
        spend=_spend((13,), 10, 20),
        allocator_counts={13: (2, 5)})

    assert rc == 4
    assert "No complete successfully scored seed pairs" in out
    assert "allocator observed 5 decision(s), but no complete" in out
    assert "| arm | complete seeds |" not in out
    assert len(payload["rows"]) == 2


def test_evolution_failure_still_retains_its_usage_and_build_failure_row(
        monkeypatch, capsys, tmp_path):
    # Warm build and off-arm build succeed; the on-arm build then fails.
    rc, out, err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(4,),
        scores={(4, "off"): 0.5, (4, "on"): 0.5},
        spend={(4, "off"): 12, (4, "on"): 31},
        build_fail_call=3)
    assert rc == 4
    assert "BUILD FAILED RuntimeError: mock build failure" in err
    assert payload["rows"][1]["error"].startswith("build failed:")
    assert payload["rows"][1]["tokens"] == 0

    # Now exercise the existing evolve-exception path after accrued spend.
    rc, out, _err, payload, _alloc_seeds, _evolve_seeds = _run_main(
        monkeypatch, capsys, tmp_path,
        seeds=(4,),
        scores={(4, "off"): 0.5, (4, "on"): 0.5},
        spend={(4, "off"): 12, (4, "on"): 31},
        evolve_failures={(4, "on"): RuntimeError("mock evolve failure")})
    assert rc == 4
    assert "1 failed (spent 31 tokens before failing)" in out
    assert payload["rows"][1]["tokens"] == 31
    assert payload["rows"][1]["error"].startswith("evolve failed:")
