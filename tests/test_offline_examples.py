"""The five examples that run end-to-end with no dependencies, and had no test.

The coverage was inverted: the six algorithm ports, which need credentials to do
anything real, all had offline tests of their pure helpers -- while the five that
run fully offline in seconds had none, and CI runs `pytest` only, so no example
was executed at all.

They produce numbers the docs quote. `run_demo` is the RQ1 result in the README,
`index.md` and `usage.md`; `rq2_staleness` is the RQ2 table. Three of the defects
found in this repo would have been caught here:

* `stable` sat at 0.000 for all 40 rounds of `run_demo` while the docs published a
  table where it caught up at round 8 -- output that code could not produce;
* `rq2_staleness` swept a parameter the run never read, so three of its four
  published rows were the same configuration;
* every sweep reported `dev_acc=1.000` for every setting, so the experiments had
  no dynamic range to report at all.

Each test asserts the *shape of the finding*, not an exact number: these are
seeded but wall-clock-bounded in places, and pinning absolutes would make them
flaky and would encode the machine they were written on.
"""

import tempfile

import pytest

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.ledger import Ledger
from agentdescent.orchestrator import AgentDescent, run_fork_baseline
from agentdescent.scheduler import DurationEstimator, fifo_makespan, lpt_schedule
from agentdescent.staleness import get_policy


# -- run_demo: the RQ1 number three doc pages quote ------------------------------


def test_merging_beats_forking_on_the_same_budget():
    """RQ1. The whole premise: N workers merging beat N workers forking."""
    universe = make_task_universe(seed=7)
    with tempfile.TemporaryDirectory() as repo:
        system = AgentDescent(repo, universe, n_workers=6, noise=0.15,
                              refresh_interval=2, seed=0)
        system.run(rounds=20)
        merged = system.final_accuracy()
    forked = run_fork_baseline(universe, n_workers=6, noise=0.15, rounds=20, seed=0)
    assert merged > forked + 0.2, (
        f"merge={merged:.3f} vs best fork={forked:.3f} -- the advantage RQ1 "
        "reports is not there")


def test_the_demo_reaches_the_stable_branch():
    """`stable` was 0.000 for all 40 rounds while the docs showed it catching up.

    Also covered by `test_dual_branch_promotion.py`; kept here because this is the
    run the docs actually publish.
    """
    universe = make_task_universe(seed=7)
    with tempfile.TemporaryDirectory() as repo:
        system = AgentDescent(repo, universe, n_workers=6, noise=0.15,
                              refresh_interval=2, seed=0)
        history = system.run(rounds=20)
        assert history[-1].stable_accuracy > 0.9, \
            f"stable never caught up: {history[-1].stable_accuracy}"
        assert system.final_accuracy(branch=Ledger.STABLE) > 0.9


# -- rq2_staleness: the sweep must sweep something -------------------------------


def _rq2_row(alpha):
    from agentdescent.aggregator import AggregatorConfig

    universe = make_task_universe(seed=7)
    cfg = AggregatorConfig(alpha_head=alpha, alpha_tail=alpha)
    with tempfile.TemporaryDirectory() as repo:
        system = AgentDescent(repo, universe, n_workers=6, noise=0.1,
                              refresh_interval=3, config=cfg, seed=2)
        history = system.run(rounds=40)
    return {
        "stale": sum(h.discarded_stale for h in history),
        "rounds": next((h.round for h in history if h.dev_accuracy >= 0.999), None),
    }


def test_the_staleness_sweep_actually_varies_with_alpha():
    """It swept `alpha_head`, which `_alpha_for` only reads for an L1 artifact --
    and `RouterSkill` is 0.2. Three of the four published rows were the same run.
    """
    tight, loose = _rq2_row(0), _rq2_row(5)
    assert tight != loose, f"alpha=0 and alpha=5 produced identical results: {tight}"
    assert tight["stale"] > loose["stale"], (
        "a zero staleness budget must discard more than a generous one; "
        f"alpha=0 discarded {tight['stale']}, alpha=5 discarded {loose['stale']}")


def test_tolerating_staleness_is_what_rq2_claims_it_is():
    """The research question: a rebasable diff should cost less than a lost one."""
    tight, loose = _rq2_row(0), _rq2_row(5)
    assert tight["rounds"] is not None and loose["rounds"] is not None
    assert loose["rounds"] <= tight["rounds"], (
        f"tolerating staleness converged slower ({loose['rounds']} rounds) than "
        f"discarding it ({tight['rounds']})")


# -- run_async: the policies must differ in cost, since they cannot in accuracy --


def _async_stats(policy_name, ratio=4, seconds=8.0):
    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=6, async_ratio=ratio, noise=0.15,
                      target_accuracy=0.98, max_seconds=seconds, seed=1)
    with tempfile.TemporaryDirectory() as repo:
        return AsyncAgentDescent(repo, universe, config=cfg,
                                 staleness_policy=get_policy(policy_name)).run()


def test_full_discards_nothing_and_guarded_discards_the_most():
    """Accuracy saturates on this domain, so the policies are only separable by
    what they spend -- which is why the experiment now reports cost."""
    full, guarded = _async_stats("full"), _async_stats("guarded")
    assert full.discarded_stale == 0, "Full accepts stale diffs by definition"
    assert guarded.discarded_stale > 0, "Guarded must reject something past alpha"


def test_the_async_runtime_actually_pipelines():
    """Many rollouts per merge sweep, or nothing is overlapping."""
    s = _async_stats("full")
    assert s.sweeps > 0 and s.rollouts > s.commits * 5


# -- parallelism: the demo now drives evolve(), so it must still converge --------


def test_the_parallelism_demo_delivers_and_converges():
    from examples.parallelism import CATEGORIES, category_of, measure
    from agentdescent.parallel import DataParallel, TensorParallel

    dp = measure(DataParallel())
    assert dp["reward"] == 1.0 and dp["violations"] == 0

    routed = measure(TensorParallel(4, keys=CATEGORIES, route=category_of))
    assert routed["delivered"] == routed["proposals"], \
        "routed TP must not reject any proposal"
    assert routed["violations"] == 0

    unrouted = measure(TensorParallel(4, keys=CATEGORIES))
    assert unrouted["violations"] > 0, \
        "unrouted TP rejects out-of-section proposals, and must say so"


# -- duration_scheduling: the two claims the page makes --------------------------


def test_the_duration_estimator_recovers_a_linear_cost_law():
    est = DurationEstimator()
    for cost in range(10, 400, 7):
        est.observe(cost=cost, seconds=0.05 + 0.0008 * cost)
    intercept, slope = est.params
    assert intercept == pytest.approx(0.05, abs=0.01)
    assert slope == pytest.approx(0.0008, abs=0.0002)


def test_lpt_beats_round_robin_on_a_heavy_tail():
    """Dispatching the tail early is the whole point of estimating duration."""
    # Round-robin assigns item i to worker i % 4, so a tail that happens to fall
    # on one worker is its worst case -- and the position of a heavy rollout in
    # the queue is exactly what a scheduler should not depend on.
    weights = [1.2 if i % 4 == 0 else 0.05 for i in range(40)]
    _, lpt = lpt_schedule(weights, 4)
    fifo = fifo_makespan(weights, 4)
    assert lpt < fifo, f"LPT makespan {lpt:.2f} did not beat round-robin {fifo:.2f}"
    assert lpt >= sum(weights) / 4, "cannot beat the perfect-balance lower bound"
