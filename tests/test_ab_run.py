"""The A/B harness must not be able to report a false negative.

`bench/ab_run.py` is the only thing standing between a paid sweep and the
sentence "we tried the rule and it did not help". That sentence is worth writing
only if the rule actually ran, and two things used to make it unsafe:

* the "did it fire" check read `fusion.contested` for **every** rule, which
  counts fusion tournaments and is non-zero whether or not an advantage was ever
  recorded, so three of the four rules were audited by a counter belonging to
  the fourth;
* nothing said, before the money, that `--workers 4` against `min_group=4`
  leaves at most one rollout per round able to carry an advantage at all.

These tests are offline: they check the harness's arithmetic and its refusals,
not a model.
"""

import random

import pytest

from agentdescent.advantage import AdaptiveTrustRegion, GroupAdvantage, TrustRegion
from agentdescent.evolvable import EvidenceCard
from agentdescent.policies import AcceptDecision, MergeContext
from bench.ab_run import _MIN_GROUP, _arm, _Fired, _fired_counts, main


class _Fusion:
    """Enough of `EvolutionResult.fusion_stats()` for the counter under test."""

    def __init__(self, contested=0):
        self.contested = contested


# -- the precondition, before the money --------------------------------------


def test_an_advantage_sweep_below_min_group_is_refused(capsys):
    """Both arms would be the control, and the table would look like a result."""
    code = main(["--rule", "advantage", "--workers", "2", "--plan"])
    assert code == 2
    assert "no rollout can ever carry an advantage" in capsys.readouterr().err


def test_an_advantage_sweep_states_its_signal_ceiling(capsys):
    code = main(["--rule", "advantage", "--workers", "8", "--plan"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"at most {8 - _MIN_GROUP + 1}/8 rollouts" in out


def test_a_thin_advantage_sweep_warns_and_names_the_lever(capsys):
    main(["--rule", "advantage", "--workers", "4", "--plan"])
    err = capsys.readouterr().err
    assert "fewer than half" in err and "--workers 8" in err


@pytest.mark.parametrize("rule", ["trust-region", "stable-distance",
                                  "reflective-fusion"])
def test_the_ceiling_is_only_claimed_for_the_rule_it_bounds(rule, capsys):
    """`min_group` bounds the advantage signal and nothing else."""
    assert main(["--rule", rule, "--plan"]) == 0
    assert "advantage" not in capsys.readouterr().out.lower().replace(
        f"rule     : {rule}", "")


def test_the_ceiling_matches_what_group_advantage_actually_does():
    """The printed bound is arithmetic; this is the bound observed.

    `GroupAdvantage` returns `None` until the group reaches `min_group`, so in a
    round of ``workers`` rollouts sharing one group the first
    ``min_group - 1`` can never carry a value however the rewards fall.
    """
    for workers in (4, 8, 16):
        ga = GroupAdvantage(min_group=_MIN_GROUP)
        rng = random.Random(0)
        key = ga.key(1, "")
        carried = sum(
            1 for _ in range(workers)
            if ga.observe(key, rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])) is not None)
        assert carried <= max(0, workers - _MIN_GROUP + 1)


# -- the firing check, after the run -----------------------------------------


def test_each_rule_is_audited_by_its_own_counter():
    """The bug: `contested` is a fusion counter and was used for every rule.

    A busy fusion tournament must not make a rule that never had a signal look
    like it fired.
    """
    busy_fusion = _Fusion(contested=42)
    silent = _Fired()
    assert _fired_counts("advantage", silent, busy_fusion) == (0, 0)
    assert _fired_counts("stable-distance", silent, busy_fusion) == (0, 0)
    # ...while the rule the counter belongs to still reads it.
    assert _fired_counts("reflective-fusion", None, busy_fusion) == (42, 42)


def test_a_trust_region_that_never_moved_reports_zero():
    """An adaptive knob that held its initial value is a constant."""
    still = AdaptiveTrustRegion()
    assert _fired_counts("trust-region", still, _Fusion())[0] == 0
    moved = AdaptiveTrustRegion()
    moved.history.append(TrustRegion(8, 40_000))
    assert _fired_counts("trust-region", moved, _Fusion())[0] == 1


def _ctx(advantages, stable_distance=0.0):
    """A real `MergeContext`: the policy under test calls `dataclasses.replace`."""
    return MergeContext(
        artifact=None, candidate=None,
        cards=[EvidenceCard(diff=None, base_version={}, touched=[], advantage=a)
               for a in advantages],
        base_counts=(1.0, 1.0), cand_counts=(2.0, 1.0),
        stable_distance=stable_distance)


class _Yes:
    """An inner policy that commits: the real contract is `AcceptDecision`."""

    def accept(self, ctx):
        return AcceptDecision(accept=True, category="committed")


def test_the_advantage_arm_counts_the_cards_that_carried_one():
    """`considered` is every decision, `fired` only those with a signal."""
    kwargs, probe = _arm("advantage", on=True)
    policy = kwargs["policies"].acceptance
    policy.inner = _Yes()
    policy.accept(_ctx([None, None]))
    policy.accept(_ctx([None, 1.5]))
    assert (probe.fired, probe.considered) == (1, 2)


def test_the_stable_distance_arm_counts_decisions_that_had_a_distance():
    """Zero distance is the ordinary state until something is promoted."""
    kwargs, probe = _arm("stable-distance", on=True)
    policy = kwargs["policies"].acceptance
    policy.inner = _Yes()
    policy.accept(_ctx([None], stable_distance=0.0))
    policy.accept(_ctx([None], stable_distance=0.4))
    assert (probe.fired, probe.considered) == (1, 2)


def test_the_control_arm_watches_nothing():
    """Off is the default policies; there is no rule there to have fired."""
    kwargs, watcher = _arm("advantage", on=False)
    assert kwargs == {} and watcher is None
    assert _fired_counts("advantage", watcher, _Fusion(contested=9)) == (0, 0)
