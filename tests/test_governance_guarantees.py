"""The L0 frozen layer is a safety claim, so pin it.

"Without a frozen layer the self-referential loop eventually pollutes itself (a
verifier that learns to pass itself)" is the strongest claim the design makes.
These tests assert the guard actually holds on both engine paths, so it cannot be
loosened by accident.
"""

import pytest

from agentdescent.evolution import AppendRules, EvolvingArtifact, Task, evolve
from agentdescent.governance import (
    FROZEN_IDS,
    GovernanceError,
    Layer,
    assert_mutable,
    classify,
)


class _Agent:
    def solve(self, rendered, task):
        return "x"

    def propose(self, rendered, task, output, reward):
        return "a rule"


def _tasks(n=8):
    return [Task(id=f"t{i}", prompt="q") for i in range(n)]


def test_frozen_ids_are_classified_l0():
    assert FROZEN_IDS, "the frozen set must not be empty"
    for fid in FROZEN_IDS:
        assert classify(EvolvingArtifact(fid, {})) is Layer.L0_FROZEN


def test_assert_mutable_blocks_every_frozen_id():
    for fid in FROZEN_IDS:
        with pytest.raises(GovernanceError):
            assert_mutable(EvolvingArtifact(fid, {}))


def test_evolve_refuses_to_evolve_a_frozen_artifact():
    """End to end: the loop must not mutate an L0 artifact even if asked to."""
    for fid in sorted(FROZEN_IDS):
        with pytest.raises(GovernanceError):
            evolve(_tasks(), lambda t, o: 0.0, agent=_Agent(),
                   strategy=AppendRules(), rounds=2, artifact_id=fid)


def test_async_evolve_also_refuses_a_frozen_artifact():
    from agentdescent.async_evolve import async_evolve

    with pytest.raises(GovernanceError):
        async_evolve(_tasks(), lambda t, o: 0.0, agent=_Agent(),
                     strategy=AppendRules(), n_workers=2, max_seconds=2.0,
                     artifact_id=sorted(FROZEN_IDS)[0])


def test_layer_boundaries_match_the_documented_blast_radii():
    """L2 <= 0.30, L1 in (0.30, 0.85] -- the docs and governance must agree."""
    assert classify(EvolvingArtifact("skill", {}, blast_radius=0.2)) is Layer.L2_FAST
    assert classify(EvolvingArtifact("harness", {}, blast_radius=0.6)) is Layer.L1_SLOW


def test_governance_is_checked_before_any_rollout():
    """A governance violation is a caller error and must fail fast.

    Only the reference aggregator's per-merge guard used to catch an L0 target, so
    nothing was mutated -- but `async_evolve` burned its entire max_seconds budget
    first and then reported the violation through the *backend failure* channel,
    which is both slow and misleading. Both paths now raise up front.
    """
    import time

    from agentdescent.async_evolve import async_evolve

    calls = {"n": 0}

    class _Counting:
        def solve(self, rendered, task):
            calls["n"] += 1
            return "x"

        def propose(self, rendered, task, output, reward):
            return "a rule"

    t0 = time.time()
    with pytest.raises(GovernanceError):
        async_evolve(_tasks(), lambda t, o: 0.0, agent=_Counting(),
                     strategy=AppendRules(), n_workers=2, max_seconds=30.0,
                     artifact_id=sorted(FROZEN_IDS)[0])
    assert time.time() - t0 < 5.0, "should fail before spending the budget"
    assert calls["n"] == 0, "no rollout should run against a frozen artifact"
