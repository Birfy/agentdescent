"""One definition of the L1/L2 boundary, used by everything that branches on it.

`governance.classify` drew the line at `FAST_MAX = 0.30`. The aggregator's
staleness tolerance re-derived it as `blast_radius > 0.5`, and the audit gate as
`blast_radius >= 0.5`. So an artifact at 0.4 was **L1 by governance** -- the slow,
conservative layer -- and treated as L2 by both mechanisms that decide what that
means: it got the staleness tolerance meant for a cold L2 skill, and no oracle
audit at all.

`evolve()`'s own docstring papered over the gap by recommending 0.2 and 0.6, the
two values where the thresholds happen to agree.
"""

import warnings

import pytest

from agentdescent.aggregator import Aggregator, AggregatorConfig
from agentdescent.evolution import EvolvingArtifact, Task, evolve
from agentdescent.governance import (
    FAST_MAX, FROZEN_IDS, GovernanceError, Layer, classify,
)
from agentdescent.scheduler import AuditScheduler


def _artifact(blast_radius):
    return EvolvingArtifact("art", {}, blast_radius=blast_radius)


class _Card:
    class diff:
        contract_breaking = False


# -- the boundary is one number, in one place -----------------------------------


@pytest.mark.parametrize("blast_radius", [0.31, 0.4, 0.5])
def test_the_gap_between_the_thresholds_is_treated_as_l1_throughout(blast_radius):
    """These are exactly the radii the three thresholds used to disagree about."""
    assert classify(_artifact(blast_radius)) is Layer.L1_SLOW, "premise"

    agg = Aggregator.__new__(Aggregator)
    agg.config = AggregatorConfig(alpha_head=5, alpha_tail=1)
    assert agg._alpha_for(_artifact(blast_radius), _Card()) == 5, (
        "an L1 artifact was given the staleness tolerance meant for a cold L2 one")

    assert AuditScheduler().force_oracle(blast_radius, "art"), (
        "an L1 artifact was not audited; the gate used its own 0.5 threshold")


@pytest.mark.parametrize("blast_radius", [0.0, 0.2, FAST_MAX])
def test_l2_artifacts_keep_the_cheap_treatment(blast_radius):
    assert classify(_artifact(blast_radius)) is Layer.L2_FAST
    agg = Aggregator.__new__(Aggregator)
    agg.config = AggregatorConfig(alpha_head=5, alpha_tail=1)
    assert agg._alpha_for(_artifact(blast_radius), _Card()) == 1
    assert not AuditScheduler().force_oracle(blast_radius, "art"), \
        "a trusted L2 artifact should not spend oracle budget"


def test_there_is_no_second_threshold_constant():
    """`SLOW_MAX = 0.85` existed with a comment describing frozen-layer behaviour.

    `classify` never read it, so 0.31 and 0.99 classified identically and the
    comment documented a rule that did not exist.
    """
    import agentdescent.governance as gov
    assert not hasattr(gov, "SLOW_MAX"), \
        "a threshold nothing reads is worse than no threshold"


# -- L0 is reached by name, and the names are ordinary words --------------------


def test_a_reserved_name_is_refused_before_any_rollout():
    """`oracle` is a plausible name for an evolving judge prompt.

    It used to surface as a GovernanceError on the first round, naming governance
    rather than the actual cause -- the name.
    """
    calls = {"n": 0}

    def run(rendered, task):
        calls["n"] += 1
        return "x"

    with pytest.raises(GovernanceError) as excinfo:
        evolve([Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(8)],
               lambda t, o: 0.0, run=run, propose=lambda *a: None,
               artifact_id="oracle", rounds=3)
    assert calls["n"] == 0, "no rollout should run against a reserved name"
    msg = str(excinfo.value)
    assert "reserved" in msg and "Rename" in msg, \
        "the error must say the problem is the name"


@pytest.mark.parametrize("frozen_id", sorted(FROZEN_IDS))
def test_every_frozen_id_is_refused(frozen_id):
    with pytest.raises(GovernanceError):
        evolve([Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(8)],
               lambda t, o: 0.0, run=lambda r, t: "x", propose=lambda *a: None,
               artifact_id=frozen_id, rounds=1)


def test_a_normal_artifact_id_is_unaffected():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve([Task(id=str(i), prompt="q", meta={"gold": "y"}) for i in range(8)],
                     lambda t, o: 1.0, run=lambda r, t: "x", propose=lambda *a: None,
                     artifact_id="oracle_prompt", rounds=1, held_out_frac=0.5)
    assert res.error is None
