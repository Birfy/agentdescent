"""Tests for the barrier-free async runtime (agentdescent.async_evolve).

Deterministic no-network stub agents; assertions avoid brittle timing by checking
monotonicity (the aggregator never regresses the head) and that the pipeline runs
and commits, rather than exact wall-clock outcomes.
"""

import warnings

import pytest

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, Task, evolve


class _Composer:
    """Composes complementary hints: correct once the hint is in the artifact."""

    def solve(self, rendered, task):
        return "yes" if task.meta["hint"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def _tasks(n=18):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"target": "yes", "hint": f"H{i % 3}"})
            for i in range(n)]


REWARD = lambda t, o: 1.0 if o.strip().lower() == "yes" else 0.0


def test_async_evolve_runs_barrier_free_and_improves():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=4, async_ratio=3, max_seconds=8.0, target_reward=1.0,
                     held_out_frac=0.5)
    assert len(r.history) >= 1                       # the merger swept at least once
    assert r.final_reward >= r.history[0].held_out_reward   # head never regresses
    assert r.final_reward >= 0.66                     # composed >= 2 of the 3 hints


def test_async_evolve_tolerates_high_staleness():
    # a large lag budget -> workers propose against stale snapshots; the staleness
    # policy must rebase/discard so the run still completes without error.
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=4, async_ratio=12, max_seconds=8.0, target_reward=1.0,
                     held_out_frac=0.5)
    assert r.final_reward >= r.history[0].held_out_reward
    assert 0.0 <= r.final_reward <= 1.0


def test_async_evolve_stops_at_max_iters():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=30.0, max_iters=6,
                     held_out_frac=0.5)
    # capped early -> few rules committed, but the pipeline produced a valid result.
    assert len(r.state) <= 3
    assert r.final_reward >= 0.0


def test_evolve_asynchronous_flag_delegates_to_async():
    r = evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
               asynchronous=True, n_workers=4, rounds=12, async_ratio=3, max_seconds=8.0,
               held_out_frac=0.5)
    assert r.final_reward >= r.history[0].held_out_reward


def test_async_evolve_works_with_custom_pareto_aggregator():
    from examples.gepa_prompt_evolution import InstructionSlot, pareto_aggregator_factory

    class _Gepa:
        def solve(self, rendered, task):
            return "yes" if task.meta["hint"] in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return (rendered + " " + task.meta["hint"]).strip()

    factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=1)
    async_evolve(_tasks(), REWARD, agent=_Gepa(), strategy=InstructionSlot(),
                 initial_state={"instruction": "start"}, artifact_id="gepa_prompt",
                 n_workers=4, async_ratio=5, max_seconds=8.0, target_reward=1.0,
                 held_out_frac=0.5, aggregator_factory=factory)
    agg = factory.holder["agg"]
    seed_avg = sum(agg.scores[0]) / len(agg.scores[0])
    assert agg.best_avg >= seed_avg                  # illumination held up under async


class _FlakyAgent:
    """Fails the first ``n_fail`` rollouts, then behaves like _Composer."""

    def __init__(self, n_fail=2):
        self.n_fail, self.calls = n_fail, 0

    def solve(self, rendered, task):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("transient backend failure")
        return "yes" if task.meta["hint"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


class _DeadAgent:
    """Every rollout fails -- simulates credit exhaustion / a dead endpoint."""

    def solve(self, rendered, task):
        raise RuntimeError("backend is down")

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def test_async_survives_transient_backend_errors():
    """A few transient failures must NOT kill the run (they are retried)."""
    r = async_evolve(_tasks(), REWARD, agent=_FlakyAgent(n_fail=2),
                     strategy=AppendRules(), n_workers=2, async_ratio=3,
                     max_seconds=8.0, target_reward=1.0, held_out_frac=0.5)
    assert r.final_reward > 0.0            # recovered and still made progress
    assert len(r.history) >= 1
    # a survived blip is not a failed run: `error` means "ended because of it"
    assert r.error is None


def test_async_reports_persistent_backend_failure():
    """A dead backend ends the run, but the reason is reported, not swallowed."""
    with pytest.warns(RuntimeWarning, match="backend is down"):
        r = async_evolve(_tasks(), REWARD, agent=_DeadAgent(), strategy=AppendRules(),
                         n_workers=2, async_ratio=3, max_seconds=20.0, held_out_frac=0.5)
    assert r.error is not None                  # the failure is surfaced ...
    assert "backend is down" in r.error         # ... with the real cause
    assert r.state == {}                        # nothing bogus was committed


def test_clean_run_reports_no_error():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=6.0,
                     target_reward=1.0, held_out_frac=0.5)
    assert r.error is None


def test_history_reports_real_reward_not_a_probability():
    """RoundInfo.held_out_reward must be the held-out reward.

    A previous optimisation reported MergeReport.prob_improve (a Beta-posterior
    P(delta>0)) instead, which corrupted `history` and made `target_reward` fire
    spuriously. Pin the units: the last sweep's reward must equal a fresh
    held-out score of the final artifact.
    """
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=3, async_ratio=3, max_seconds=6.0, held_out_frac=0.5)
    assert r.history, "expected at least one sweep"
    assert abs(r.history[-1].held_out_reward - r.final_reward) < 1e-9


def test_target_reward_is_compared_against_the_real_reward():
    """Stopping early must require the true reward, not a probability >= target."""
    # _DeadEnd never solves anything -> true reward stays 0.0, so a target of 1.0
    # must never be reached even though acceptance probabilities can approach 1.
    class _DeadEnd:
        def solve(self, rendered, task):
            return "no"
        def propose(self, rendered, task, output, reward):
            return task.meta["hint"]

    r = async_evolve(_tasks(), REWARD, agent=_DeadEnd(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=3.0,
                     target_reward=1.0, held_out_frac=0.5)
    assert r.final_reward == 0.0
    assert all(info.held_out_reward == 0.0 for info in r.history)


class _SlowAgent:
    """A backend whose rollouts outlast a short budget."""

    def __init__(self, delay=0.6):
        self.delay = delay

    def solve(self, rendered, task):
        import time as _t
        _t.sleep(self.delay)
        return "yes"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def test_shutdown_does_not_scale_with_worker_count():
    """Joining 2s per worker plus 10s for the merger made the overshoot grow with
    n_workers; the joins now share one bounded grace period."""
    import time as _t

    def elapsed(n_workers):
        t0 = _t.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            async_evolve(_tasks(), lambda t, o: 1.0, agent=_SlowAgent(),
                         strategy=AppendRules(), n_workers=n_workers,
                         max_seconds=0.5, held_out_frac=0.5, shutdown_grace=0.5)
        return _t.time() - t0

    few, many = elapsed(2), elapsed(8)
    # 4x the workers must not cost meaningfully more shutdown time
    assert many < few + 3.0, f"shutdown scaled with workers: {few:.1f}s -> {many:.1f}s"


def test_shutdown_grace_is_respected_and_warns():
    import time as _t

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        async_evolve(_tasks(), lambda t, o: 1.0, agent=_SlowAgent(delay=2.0),
                     strategy=AppendRules(), n_workers=2, max_seconds=0.3,
                     held_out_frac=0.5, shutdown_grace=0.2)
    assert any("still" in str(x.message) for x in w), \
        "abandoning in-flight rollouts must be reported, not silent"


def test_async_staleness_alpha_comes_from_agg_config():
    """The merger's staleness gate hardcoded 5/1, ignoring agg_config.

    The aggregator honoured alpha_head/alpha_tail while the async gate in front of
    it did not, so a tightened tolerance was applied in one place only.
    """
    import inspect

    from agentdescent.aggregator import AggregatorConfig
    from agentdescent.async_evolve import async_evolve as fn

    src = inspect.getsource(fn)
    assert "alpha_head" in src and "alpha = 5 if" not in src

    # and it still runs with a strict configuration
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=2, max_seconds=3.0, held_out_frac=0.5,
                     agg_config=AggregatorConfig(alpha_head=0, alpha_tail=0))
    assert r.error is None
