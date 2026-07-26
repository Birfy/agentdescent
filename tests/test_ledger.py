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
