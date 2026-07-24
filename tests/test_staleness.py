from concordia.staleness import (
    FullStaleness,
    GuardedStaleness,
    ReflectiveStaleness,
    StaleAction,
    get_policy,
)


def test_full_accepts_regardless_of_eta():
    p = FullStaleness()
    assert p.decide(eta=0, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    assert p.decide(eta=99, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    # but never blindly crosses a broken contract.
    assert p.decide(eta=1, alpha=1, contract_breaking=True) is StaleAction.DISCARD


def test_guarded_is_version_gated():
    p = GuardedStaleness()
    assert p.decide(eta=0, alpha=2, contract_breaking=False) is StaleAction.ACCEPT
    assert p.decide(eta=2, alpha=2, contract_breaking=False) is StaleAction.REBASE
    assert p.decide(eta=3, alpha=2, contract_breaking=False) is StaleAction.DISCARD
    assert p.decide(eta=1, alpha=5, contract_breaking=True) is StaleAction.DISCARD


def test_reflective_always_rebases_when_stale():
    p = ReflectiveStaleness()
    assert p.decide(eta=0, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    # ignores the alpha budget: even far-stale diffs get a reflective replay.
    assert p.decide(eta=50, alpha=1, contract_breaking=False) is StaleAction.REBASE
    assert p.decide(eta=2, alpha=1, contract_breaking=True) is StaleAction.DISCARD


def test_registry_lookup():
    assert isinstance(get_policy("full"), FullStaleness)
    assert isinstance(get_policy("guarded"), GuardedStaleness)
    assert isinstance(get_policy("reflective"), ReflectiveStaleness)


def test_unknown_policy_raises():
    try:
        get_policy("nope")
    except ValueError as e:
        assert "unknown staleness policy" in str(e)
    else:
        assert False, "expected ValueError"
