"""The harness has to be trustworthy before anything it measures is.

Every test here is about the measuring instrument, not the thing measured. The
one that matters most is determinism: on a deterministic domain, the same seed
twice must produce the same numbers. If it does not, every difference the harness
reports between two configurations could be that instead.
"""

import json

import pytest

from bench.harness import Config, Measurement, Summary, summarise, to_markdown
from bench.run import CONFIGS, main, truncate_to_budget, workload


def _measure(config_name="sync-1", seed=0, target=0.75):
    from bench.run import test_quality

    result = workload(CONFIGS[config_name], seed)
    return Measurement.of(config_name, seed, result, target,
                          test_quality(seed)(result))


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


def test_the_same_seed_twice_gives_the_same_numbers():
    """The harness's self-check.

    The domain is deterministic, so any difference between two runs of one seed
    is the harness's own noise -- and every difference it reports between two
    configurations could be that instead."""
    a, b = _measure(seed=0), _measure(seed=0)
    assert a.rollouts == b.rollouts
    assert a.calls == b.calls
    assert a.cost_to_quality == b.cost_to_quality
    assert a.test_quality == b.test_quality
    assert a.stale_rate == b.stale_rate


def test_quality_is_measured_on_a_split_the_gate_never_saw():
    """`evolve()` stops when *its* held-out split says so. Reporting that number
    would be reporting a training score."""
    from bench.run import _dataset

    train, test = _dataset(0)
    assert train and test
    train_ids = {t.id for t in train}
    assert not (train_ids & {t.id for t in test}), "the splits overlap"


def test_the_split_does_not_move_between_configurations():
    """Splitting inside a configuration would let two of them see different data
    and let the difference be called parallelism."""
    from bench.run import _dataset

    a_train, a_test = _dataset(3)
    b_train, b_test = _dataset(3)
    assert [t.id for t in a_train] == [t.id for t in b_train]
    assert [t.id for t in a_test] == [t.id for t in b_test]


def test_learning_generalises_in_this_domain():
    """A domain where a rule names a *task* cannot generalise, so every quality
    column is structurally zero -- a table of honest numbers measuring nothing.
    An earlier version of this harness had exactly that."""
    m = _measure("sync-4", seed=0)
    assert m.test_quality is not None and m.test_quality > 0.0, (
        "nothing learned on train applied to test; the domain cannot show a "
        "quality difference between configurations")


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_the_budget_is_counted_in_calls_not_rounds():
    """Configurations differ in how many calls a round makes, so a budget fixed
    in rounds hands the wider one more model -- and then reports the extra model
    as a win for parallelism."""
    result = workload(CONFIGS["sync-4"], 0)
    assert result.history and result.history[-1].calls > 0

    budget = result.history[0].calls
    cut = truncate_to_budget(result, budget)
    assert all(h.calls <= budget for h in cut.history)
    assert len(cut.history) < len(result.history) or len(result.history) == 1


def test_a_budget_nobody_exceeds_changes_nothing():
    result = workload(CONFIGS["sync-1"], 0)
    assert truncate_to_budget(result, 10 ** 9) is result


def test_per_round_calls_are_cumulative_and_monotonic():
    """The budget is applied against them, so a non-monotonic sequence would
    truncate somewhere arbitrary."""
    result = workload(CONFIGS["sync-4"], 1)
    calls = [h.calls for h in result.history]
    assert calls == sorted(calls)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_a_configuration_that_never_reached_the_bar_says_so():
    """`—` and `0/3`, not a number borrowed from the total wall-clock. A
    configuration that never converged has no time-to-quality, and filling one in
    would make the slowest row look like the fastest."""
    never = [Measurement(config="c", seed=s, time_to_quality=None,
                         cost_to_quality=None, wallclock=5.0, rollouts=10,
                         calls=20, stale_rate=0.0, duplicate_rate=0.0,
                         sandbox_wait_s=0.0, sandbox_setup_s=0.0,
                         sandboxes_created=0, test_quality=0.1) for s in range(3)]
    [summary] = summarise(never)
    assert summary.reached == 0 and summary.time_to_quality is None
    table = to_markdown([summary])
    assert "0/3" in table and "—" in table


def test_the_interval_reports_the_spread_that_was_observed():
    """Three seeds cannot support a confidence interval, and one dressed up as
    arithmetic would be worse than the spread it hides."""
    ms = [Measurement(config="c", seed=s, time_to_quality=t,
                      cost_to_quality=10 * s + 10, wallclock=t, rollouts=5,
                      calls=10, stale_rate=0.1, duplicate_rate=0.2,
                      sandbox_wait_s=0.0, sandbox_setup_s=0.0,
                      sandboxes_created=0, test_quality=0.5)
          for s, t in enumerate([1.0, 4.0, 2.0])]
    [summary] = summarise(ms)
    assert summary.time_to_quality == [1.0, 2.0, 4.0]        # min, median, max


def test_a_heterogeneous_run_is_marked_rather_than_averaged_in():
    """`env_mismatch` non-zero means acceptance compared measurements from
    different environments, so that row's quality is not comparable."""
    ms = [Measurement(config="c", seed=0, time_to_quality=1.0, cost_to_quality=5,
                      wallclock=1.0, rollouts=5, calls=10, stale_rate=0.0,
                      duplicate_rate=0.0, sandbox_wait_s=0.0, sandbox_setup_s=0.0,
                      sandboxes_created=0, test_quality=0.9, env_mismatch=3)]
    [summary] = summarise(ms)
    assert summary.heterogeneous
    assert "⚠︎" in to_markdown([summary])


def test_the_sandbox_share_is_reported():
    """"8 workers only bought 2x" reads as staleness, a slow model, and a queue
    of containers installing dependencies. The share is what tells them apart."""
    ms = [Measurement(config="c", seed=0, time_to_quality=1.0, cost_to_quality=5,
                      wallclock=10.0, rollouts=5, calls=10, stale_rate=0.0,
                      duplicate_rate=0.0, sandbox_wait_s=2.0, sandbox_setup_s=3.0,
                      sandboxes_created=4, test_quality=0.9)]
    [summary] = summarise(ms)
    assert summary.sandbox_share == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_the_cli_runs_offline_and_writes_its_measurements(tmp_path, capsys):
    out = tmp_path / "bench.json"
    assert main(["--config", "sync-1", "--seeds", "0,1,2",
                 "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| config |" in printed and "sync-1" in printed

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["measurements"]) == 3
    assert payload["summaries"][0]["seeds"] == 3


def test_too_few_seeds_is_warned_about(tmp_path, capsys):
    """This repository has published a point estimate that moved 4.8 points
    between two runs of one configuration."""
    main(["--config", "sync-1", "--seeds", "0"])
    assert "not a result" in capsys.readouterr().err


def test_staleness_is_measured_on_both_paths():
    """A counter that is never written reports 0%, which in a table is
    indistinguishable from "there was no staleness".

    The async path filters staleness inline and never reaches `Aggregator`'s
    filter, so its stale column read 0% while 83% of its evidence was being
    discarded -- and that 83% is the explanation for its row."""
    sync = workload(CONFIGS["sync-4"], 0)
    asyn = workload(CONFIGS["async-4"], 0)
    assert sync.stale_considered > 0, "the synchronous path stopped counting"
    assert asyn.stale_considered > 0, (
        "the async path reports a stale rate it never measured")


def test_the_lag_budget_is_what_made_the_async_row_look_broken():
    """The first version of this matrix reported `async-4` as 0/3 and concluded
    the barrier-free path was worse. The path was fine; its default lag budget is
    a budget in *versions*, and a version is worth no wall-clock at all when a
    rollout is a dictionary lookup.

    Pinned because the conclusion in `docs/efficiency.md` rests on it."""
    default = workload(CONFIGS["async-4"], 0)
    tight = workload(CONFIGS["async-4-lag1"], 0)
    assert default.stale_rate() > 0.5, (
        "the default lag budget no longer produces chronic staleness here; the "
        "documented explanation needs re-checking")
    assert tight.stale_rate() < default.stale_rate()
    assert tight.final_reward > default.final_reward


def test_a_run_that_discards_most_of_its_evidence_says_so():
    """Every number such a run reports is a number about the fraction that
    survived, and it used to say nothing."""
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        workload(CONFIGS["async-4"], 0)
    messages = [str(w.message) for w in caught if "stale" in str(w.message)]
    assert messages, "a run discarding most of its evidence reported nothing"
    assert "async_ratio" in messages[0], "the warning must name the knob"
