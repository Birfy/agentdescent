import pytest

from agentdescent.domains.router import (
    RouterSkill,
    deserialize_router,
    serialize_router,
)
from agentdescent.evolvable import Diff
from agentdescent.ledger import CASConflict, Ledger


def make_ledger(tmp_path):
    return Ledger(str(tmp_path / "repo"), serialize_router, deserialize_router)


def test_register_and_snapshot(tmp_path):
    led = make_ledger(tmp_path)
    led.register(RouterSkill("s", table={"a": "x"}))
    snap = led.snapshot(Ledger.DEV)
    assert snap.version["s"] == 1
    assert snap.get("s").table == {"a": "x"}


def test_commit_bumps_version(tmp_path):
    led = make_ledger(tmp_path)
    led.register(RouterSkill("s", table={}))
    base = led.head_version(Ledger.DEV)
    new = RouterSkill("s", table={"a": "x"})
    _, v = led.commit(new, base, branch=Ledger.DEV)
    assert v == 2
    assert led.snapshot(Ledger.DEV).get("s").table == {"a": "x"}


def test_cas_conflict_on_stale_base(tmp_path):
    led = make_ledger(tmp_path)
    led.register(RouterSkill("s", table={}))
    stale_base = {"s": 0}  # head is 1, so this is stale
    with pytest.raises(CASConflict):
        led.commit(RouterSkill("s", table={"a": "x"}), stale_base, branch=Ledger.DEV)


def test_atomic_commit_all_or_nothing(tmp_path):
    led = make_ledger(tmp_path)
    led.register(RouterSkill("a", table={}))
    led.register(RouterSkill("b", table={}))
    base = led.head_version(Ledger.DEV)
    _, vv = led.commit_atomic(
        [RouterSkill("a", table={"k": "1"}), RouterSkill("b", table={"k": "2"})],
        base,
    )
    assert vv["a"] == 2 and vv["b"] == 2

    # one stale precondition rolls back the whole transaction.
    bad_base = {"a": led.head_version()["a"], "b": 0}
    with pytest.raises(CASConflict):
        led.commit_atomic(
            [RouterSkill("a", table={"k": "9"}), RouterSkill("b", table={"k": "9"})],
            bad_base,
        )
    # 'a' must not have advanced despite being valid, because the txn aborted.
    assert led.head_version()["a"] == 2


def test_promote_to_stable(tmp_path):
    led = make_ledger(tmp_path)
    led.register(RouterSkill("s", table={}))
    base = led.head_version(Ledger.DEV)
    led.commit(RouterSkill("s", table={"a": "x"}), base, branch=Ledger.DEV)
    assert led.snapshot(Ledger.STABLE).get("s").table == {}  # not yet promoted
    led.promote_to_stable("s")
    assert led.snapshot(Ledger.STABLE).get("s").table == {"a": "x"}


def test_cas_requires_the_base_version_to_declare_the_artifact(tmp_path):
    """A writer that declares no base version must NOT win.

    `base_version.get(aid, head)` made the check unfalsifiable: an omitted entry
    defaulted to the current head, so an empty (or unrelated) vector always
    committed -- exactly the lost update CAS exists to prevent.
    """
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import CASConflict, Ledger

    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v))
    lg.register(EvolvingArtifact("a", {"n": "1"}))
    lg.commit(EvolvingArtifact("a", {"n": "2"}), {"a": 1})

    for bad in ({}, {"other": 7}, {"a": 1}):          # undeclared, unrelated, stale
        try:
            lg.commit(EvolvingArtifact("a", {"n": "99"}), bad)
        except CASConflict:
            continue
        raise AssertionError(f"base_version={bad!r} should have been rejected")

    # the honest writer still succeeds and the value it lost is intact
    lg.commit(EvolvingArtifact("a", {"n": "3"}), {"a": 2})
    assert lg.snapshot(Ledger.DEV).get("a").state == {"n": "3"}


def test_commit_atomic_also_requires_declared_bases(tmp_path):
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import CASConflict, Ledger

    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v))
    lg.register(EvolvingArtifact("a", {"n": "1"}))
    lg.register(EvolvingArtifact("b", {"n": "1"}))
    try:
        lg.commit_atomic([EvolvingArtifact("a", {"n": "9"}),
                          EvolvingArtifact("b", {"n": "9"})], {"a": 1})   # 'b' undeclared
    except CASConflict:
        return
    raise AssertionError("commit_atomic should reject an undeclared artifact base")


def test_reads_do_not_fork_git_when_branch_is_current(tmp_path):
    """Reads used to fork `git checkout` even when already on the branch.

    At ~19 ms a call, serialised on the ledger lock, that capped the whole
    pipeline at ~50 ledger ops/sec no matter how many workers ran.
    """
    import time

    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import Ledger

    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v))
    lg.register(EvolvingArtifact("a", {"n": "1"}))

    lg.head_version(Ledger.DEV)                 # prime the branch tracker
    t0 = time.time()
    for _ in range(100):
        lg.head_version(Ledger.DEV)
    per_call_ms = (time.time() - t0) / 100 * 1000
    assert per_call_ms < 5.0, f"repeat reads still fork git: {per_call_ms:.1f} ms/call"


def test_branch_switching_still_works_with_the_cached_branch(tmp_path):
    """The tracker must not break dual-branch reads or dev->stable promotion."""
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import Ledger

    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v))
    lg.register(EvolvingArtifact("a", {"n": "1"}))
    lg.commit(EvolvingArtifact("a", {"n": "dev2"}), {"a": 1})

    assert lg.snapshot(Ledger.DEV).get("a").state == {"n": "dev2"}
    assert lg.snapshot(Ledger.STABLE).get("a").state == {"n": "1"}   # not yet promoted
    lg.promote_to_stable("a")
    assert lg.snapshot(Ledger.STABLE).get("a").state == {"n": "dev2"}
    assert lg.snapshot(Ledger.DEV).get("a").state == {"n": "dev2"}   # dev intact
