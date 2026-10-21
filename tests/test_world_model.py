"""World model has to be right where the framework dreams before it acts.

The simulator is a pure-Python nearest-neighbour model fit on observed
rollouts; a problem can run on it instead of the real inner run; and
`imagine_then_evolve` spends its budget dreaming, then verifies on the real
problem and reports how much the dream was worth. The fixtures pin the surfaces
that make this honest: the simulator cannot extrapolate beyond what it has seen,
the simulated problem is cheap and deterministic, the trust score measures
error and rank agreement against the real problem, and the loop verifies its
gain on the real landscape.
"""

import pytest

from agentdescent import (
    ImagineReport,
    LearnedSimulator,
    MetaOutcome,
    ParamSlot,
    imagine_then_evolve,
    simulator_problem,
    simulator_score,
)
from agentdescent.sampling import DifficultyWeighted


def _landscape(value, seed):
    c = getattr(value, "c", 0.2)
    r = min(1.0, c / 2.0 + (seed % 5) / 10.0)
    return MetaOutcome(curve=[r, r], final=r, rollouts=1)


def _landscape_val(value, seed):
    return _landscape(value, seed)


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _features(value, seed):
    return [float(getattr(value, "c", 0.2)), float(seed % 10)]


def _nudge_up():
    import re
    def propose(rendered, task, output, reward):
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose


def test_the_simulator_predicts_nearest_neighbours():
    sim = LearnedSimulator(k=1)
    assert sim.predict([0.2, 0.0]) == 0.0                 # no examples -> 0
    sim.fit([([0.2, 1.0], 0.3), ([0.5, 1.0], 0.6), ([0.8, 1.0], 0.9)])
    assert sim.predict([0.5, 1.0]) == pytest.approx(0.6)  # exact neighbour
    assert 0.3 <= sim.predict([0.4, 1.0]) <= 0.6           # between neighbours


def test_the_simulated_problem_is_cheap_and_deterministic():
    sim = LearnedSimulator(k=1)
    sim.fit([([0.2, 0.0], 0.2), ([0.8, 0.0], 0.8)])
    problem = simulator_problem(sim, features=_features)
    outcome = problem(_sampler_slot().compile("# Policy parameters\nc: 0.8"), 0)
    assert isinstance(outcome, MetaOutcome)
    assert outcome.final == pytest.approx(0.8)             # nearest example
    assert outcome.detail["predicted"] == outcome.final


def test_simulator_score_measures_trust():
    sim = LearnedSimulator(k=1)
    sim.fit([([c, d], min(1.0, c / 2 + d / 10)) for c in (0.2, 0.5, 0.8)
             for d in (0.0, 2.0)])
    value = _sampler_slot().compile("# Policy parameters\nc: 0.6")
    report = simulator_score(sim, _landscape, value=value, seeds=(0, 1, 2, 3),
                             features=_features)
    assert report["mean_abs_error"] < 0.5                  # roughly right
    assert 0.0 <= report["rank_agreement"] <= 1.0
    assert report["n"] == 4


def test_a_constant_predictor_cannot_claim_perfect_trust():
    # an untrained simulator predicts 0.0 everywhere; on any landscape the old
    # tie-blind 'tau' handed it agreement 1.0 (or 0.0 on the reverse). tau-b
    # counts ties half, so a constant predictor gets ~0.5 either way.
    sim = LearnedSimulator()                          # no examples -> 0.0
    value = _sampler_slot().compile("# Policy parameters\nc: 0.6")
    dec = lambda value, seed: MetaOutcome(curve=[1 - seed / 10] * 2,
                                          final=1 - seed / 10, rollouts=1)
    inc = lambda value, seed: MetaOutcome(curve=[seed / 10] * 2,
                                          final=seed / 10, rollouts=1)
    down = simulator_score(sim, dec, value=value, seeds=range(5),
                           features=_features)
    up = simulator_score(sim, inc, value=value, seeds=range(5),
                         features=_features)
    assert down["rank_agreement"] == pytest.approx(0.5)
    assert up["rank_agreement"] == pytest.approx(0.5)


def test_imagine_then_evolve_dreams_then_verifies():
    sim = LearnedSimulator(k=1)
    sim.fit([([c, d], min(1.0, c / 2 + d / 10)) for c in (0.2, 0.4, 0.6, 0.8)
             for d in (0.0, 2.0, 4.0)])
    train = {"t": _landscape}
    val = {"v": _landscape_val}
    report, result = imagine_then_evolve(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        simulator=sim, propose=_nudge_up(),
        seeds=(0, 1, 2, 3), val_seeds=(4, 5), rounds=1, n_workers=1,
        repo_path=None)
    assert isinstance(report, ImagineReport)
    assert report.dreamed_rollouts >= 4
    assert report.verified_rollouts > 2                 # trust + before/after
    assert 0.0 <= report.simulator_error <= 1.0
    assert report.value
    assert result.meta["dreamed_rollouts"] == report.dreamed_rollouts


def test_imagine_then_evolve_counts_real_calls():
    sim = LearnedSimulator(k=1)
    sim.fit([([c, d], min(1.0, c / 2 + d / 10)) for c in (0.2, 0.4, 0.6, 0.8)
             for d in (0.0, 2.0, 4.0)])
    real = {"n": 0}

    def counted_real(value, seed):
        real["n"] += 1
        return _landscape_val(value, seed)

    report, result = imagine_then_evolve(
        {"t": _landscape}, {"v": counted_real}, slot="task_sampler",
        spec=_sampler_slot(), simulator=sim, propose=_nudge_up(),
        seeds=(0, 1, 2, 3), val_seeds=(4, 5), rounds=1, n_workers=1,
        repo_path=None)
    # the report's verified count is the actual number of real problem calls
    assert report.verified_rollouts == real["n"]


def test_imagine_then_evolve_needs_a_proposer():
    with pytest.raises(ValueError):
        imagine_then_evolve({"t": _landscape}, {"v": _landscape_val},
                            slot="task_sampler", spec=_sampler_slot(),
                            simulator=LearnedSimulator())


def test_imagine_then_evolve_runs_with_default_seeds():
    # the documented call passes no seeds; the default must provide >= 4 outer
    # tasks (one problem x seeds), or evolve raises
    sim = LearnedSimulator(k=1)
    sim.fit([([c, d], min(1.0, c / 2 + d / 10))
             for c in (0.2, 0.4) for d in (0.0, 2.0)])
    report, result = imagine_then_evolve(
        {"t": _landscape}, {"v": _landscape_val}, slot="task_sampler",
        spec=_sampler_slot(), simulator=sim, propose=_nudge_up(),
        rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, ImagineReport)
    assert report.dreamed_rollouts >= 4
