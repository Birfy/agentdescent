import pytest

from concordia.domains.router import RouterSkill
from concordia.governance import (
    GovernanceError,
    L1SerialGate,
    Layer,
    assert_mutable,
    classify,
)


def test_blast_radius_layers():
    assert classify(RouterSkill("s", blast_radius=0.1)) is Layer.L2_FAST
    assert classify(RouterSkill("h", blast_radius=0.6)) is Layer.L1_SLOW


def test_named_frozen_artifacts_are_l0():
    assert classify(RouterSkill("oracle", blast_radius=0.1)) is Layer.L0_FROZEN
    assert classify(RouterSkill("safety_constraints", blast_radius=0.1)) is Layer.L0_FROZEN


def test_l0_is_immutable_to_the_loop():
    with pytest.raises(GovernanceError):
        assert_mutable(RouterSkill("audit_budget", blast_radius=0.1))
    # a normal L2 skill is fine.
    assert_mutable(RouterSkill("skill", blast_radius=0.1))


def test_l1_serial_gate_allows_one_at_a_time():
    gate = L1SerialGate()
    assert gate.try_acquire("h1", "d1")
    assert not gate.try_acquire("h2", "d2")  # serialized: at most one L1 in flight
    gate.release("h1")
    assert gate.try_acquire("h2", "d2")
