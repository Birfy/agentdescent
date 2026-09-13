"""Constraint 7: the acceptance gate eats only the held-out re-measurement.

Not the score that selected a winner, and not the cheap layer's sub-sample. The
repository has shipped the violation once -- the regression guard read
`cheap_eval`, so lowering `cheap_eval_tasks` silently made "quality dropped" a
judgement from four tasks -- and `MergeContext` names the two differently to
make it awkward to write again. Awkward is not impossible, so these lock it.

The residual risk issue #179 §1.2 names is real and different:
`ThreeLayerVerifier._subset` draws its sub-sample **once** and reuses it, and
the full held-out set contains it. Ranking can therefore overfit a fixed sample
the gate then reads. Phase 0 measured that on a real run and found no evidence
of it (`score(inside) - score(outside) = -0.1929`, the wrong sign, with the
tournament off). What is locked here is the part that is structural rather than
measured: with the tournament off, nothing on the commit path reads the cheap
layer at all.
"""

from difflib import SequenceMatcher

import pytest

from agentdescent.defaults import DefaultAcceptance
from agentdescent.evolution import Task, evolve
from agentdescent.policies import MergeContext

from tests.test_audit_sparse import GoodAgent


def _tasks(n=40):
    """A workload the agent's two rules cannot finish, so the run does not
    saturate at 1.0 and the histories below have something to differ in."""
    out = []
    for i in range(n):
        if i % 3 == 0:
            out.append(Task(id=f"t{i}", prompt=f"  MiXeD Case {i}!  ",
                            meta={"expected": f"mixed case {i}!"}))
        elif i % 3 == 1:
            out.append(Task(id=f"t{i}", prompt=f"  UPPER {i}  ",
                            meta={"expected": f"upper {i}"}))
        else:
            out.append(Task(id=f"t{i}", prompt=f"tricky {i}",
                            meta={"expected": f"IMPOSSIBLE {i}"}))
    return out


def _judge(task, output):
    return SequenceMatcher(None, output, task.meta["expected"]).ratio()


def _inner(**kw):
    kw.setdefault("base_delta", 0.2)
    kw.setdefault("anneal_half_life", 64)
    kw.setdefault("accept_samples", 4000)
    return DefaultAcceptance(**kw)


# -- the rule, directly ------------------------------------------------------

def test_the_cheap_numbers_do_not_reach_the_decision():
    """`base_cheap` and `cand_cheap` are ranking information. Set them to
    nonsense in both directions; the verdict must not move."""
    gate = _inner()
    base = dict(artifact=None, candidate=None, cards=[],
                base_counts=(20.0, 12.0), cand_counts=(26.0, 6.0))

    plain = gate.accept(MergeContext(**base))
    flattering = gate.accept(MergeContext(**base, base_cheap=0.0, cand_cheap=1.0))
    damning = gate.accept(MergeContext(**base, base_cheap=1.0, cand_cheap=0.0))

    assert plain == flattering == damning


def test_the_regression_guard_reads_the_full_set_not_the_sub_sample():
    """The bug this repository shipped once: a four-task sample vetoing a commit
    the full-set Beta test had just approved."""
    gate = _inner()
    # the full set says the candidate is better; the cheap layer says otherwise
    ctx = MergeContext(artifact=None, candidate=None, cards=[],
                       base_counts=(16.0, 16.0), cand_counts=(28.0, 4.0),
                       base_cheap=1.0, cand_cheap=0.0)
    got = gate.accept(ctx)
    assert got.accept, "the cheap layer must not be able to veto"
    assert "regression" not in got.detail


def test_the_reported_rates_are_the_full_set_ones():
    """A refusal a person reads has to name the numbers that decided."""
    gate = _inner()
    got = gate.accept(MergeContext(
        artifact=None, candidate=None, cards=[],
        base_counts=(28.0, 4.0), cand_counts=(16.0, 16.0),
        base_cheap=0.0, cand_cheap=1.0))
    assert not got.accept
    assert "0.875" in got.detail and "0.500" in got.detail, got.detail


# -- and end to end ----------------------------------------------------------

def test_cheap_eval_tasks_does_not_move_a_single_commit():
    """With the tournament off -- the default -- nothing on the commit path
    reads the cheap layer, so its sub-sample size is a no-op for what commits.

    The whole run, not the gate in isolation: anything that let the sub-sample
    reach a cache key, a ranking that fed a merge, or the order of a concurrent
    evaluation would show up here.
    """
    kw = dict(agent=GoodAgent(), rounds=6, n_workers=3, seed=7)
    runs = {n: evolve(_tasks(), _judge, cheap_eval_tasks=n, **kw)
            for n in (2, 8, 20)}

    histories = {n: [r.held_out_reward for r in res.history]
                 for n, res in runs.items()}
    assert len({tuple(h) for h in histories.values()}) == 1, histories
    assert len({res.final_reward for res in runs.values()}) == 1
    assert len({tuple(sorted(res.outcomes().items()))
                for res in runs.values()}) == 1

    trajectory = histories[8]
    assert min(trajectory) < max(trajectory) < 1.0, (
        "premise: the run is mid-climb, so an identical history means something")
