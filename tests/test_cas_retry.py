"""Losing a commit race is ordinary once there is more than one writer.

With one writer it cannot happen: every commit goes through a single merger, so
`CASConflict` is unreachable and the retry below never fires. That is why the
golden trace is unchanged by it.

With several -- separate processes sharing a ledger, which #81 made possible --
somebody committing first is the normal case, and settling the evidence back into
the pool to wait a round is the wrong answer to it.
"""

import multiprocessing as mp
import threading
import time

import pytest

from agentdescent.aggregator import AggregatorConfig
from agentdescent.evolvable import Contract, Diff
from agentdescent.ledger import CASConflict, Ledger

CTX = mp.get_context("spawn")


class Tiny:
    def __init__(self, id="a", state=None, version=1):
        self.id, self.state, self.version = id, dict(state or {}), version
        self.blast_radius = 0.2
        self.contract = Contract(input_schema="task", output_schema="text", major=1)

    def render(self):
        return "\n".join(f"{k}={v}" for k, v in sorted(self.state.items()))

    def apply(self, diff):
        new = dict(self.state)
        for k, v in diff.ops.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        return Tiny(self.id, new, self.version + 1)


def _ledger(path):
    return Ledger(path, lambda a: {"state": dict(a.state)},
                  lambda aid, v, d: Tiny(aid, d.get("state", {}), v))


class _Aggregatorish:
    """Just enough of an Aggregator to exercise `_commit_with_retry`."""

    def __init__(self, ledger, attempts=3, backoff=0.01, meter=None):
        from agentdescent.aggregator import Aggregator
        self.ledger = ledger
        self.config = AggregatorConfig(cas_attempts=attempts, cas_backoff=backoff)
        self.meter = meter
        self._commit_with_retry = Aggregator._commit_with_retry.__get__(self)


def test_one_writer_never_reaches_the_retry(tmp_path):
    """Which is why the golden trace is untouched by this change."""
    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {}))
    try:
        agg = _Aggregatorish(led)
        head = led.head_version(Ledger.DEV)
        art = led.snapshot(Ledger.DEV).get("a")
        diff = Diff(diff_id="d", target="a", ops={"k": "v"})
        assert agg._commit_with_retry("a", art.apply(diff), diff, head) == 2
    finally:
        led.close()


def test_a_losing_commit_rebases_and_lands(tmp_path):
    """The behaviour that changed: a conflict is retried, not abandoned."""
    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {}))
    try:
        art = led.snapshot(Ledger.DEV).get("a")
        stale_head = led.head_version(Ledger.DEV)

        # somebody else commits first, so `stale_head` is now out of date
        led.commit(art.apply(Diff(diff_id="other", target="a", ops={"x": "1"})),
                   base_version=stale_head, branch=Ledger.DEV)

        agg = _Aggregatorish(led)
        diff = Diff(diff_id="ours", target="a", ops={"y": "2"})
        version = agg._commit_with_retry("a", art.apply(diff), diff, stale_head)
        assert version == 3, "the retry did not land"

        state = led.snapshot(Ledger.DEV).get("a").state
        assert state == {"x": "1", "y": "2"}, (
            f"the retry discarded the commit that won the race: {state}")
    finally:
        led.close()


def test_the_retry_rebases_rather_than_resending(tmp_path):
    """The failure mode a naive retry introduces.

    Re-sending the original candidate would commit a state computed against a
    head that no longer exists -- discarding whatever won the race. That is the
    lost update CAS exists to prevent, arriving through the retry meant to
    preserve it."""
    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {"seed": "s"}))
    try:
        art = led.snapshot(Ledger.DEV).get("a")
        stale_head = led.head_version(Ledger.DEV)
        led.commit(art.apply(Diff(diff_id="won", target="a", ops={"won": "yes"})),
                   base_version=stale_head, branch=Ledger.DEV)

        agg = _Aggregatorish(led)
        diff = Diff(diff_id="ours", target="a", ops={"ours": "yes"})
        agg._commit_with_retry("a", art.apply(diff), diff, stale_head)

        state = led.snapshot(Ledger.DEV).get("a").state
        assert state.get("won") == "yes", "the winner's change was overwritten"
        assert state.get("ours") == "yes", "our change did not land"
    finally:
        led.close()


def test_a_duplicate_change_reports_a_conflict_rather_than_a_commit(tmp_path):
    """If the winner applied the same change, there is nothing left to commit --
    and saying "committed" would be a commit that did not happen."""
    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {}))
    try:
        art = led.snapshot(Ledger.DEV).get("a")
        stale_head = led.head_version(Ledger.DEV)
        same = Diff(diff_id="same", target="a", ops={"k": "v"})
        led.commit(art.apply(same), base_version=stale_head, branch=Ledger.DEV)

        agg = _Aggregatorish(led)
        assert agg._commit_with_retry("a", art.apply(same), same, stale_head) is None
    finally:
        led.close()


def test_the_retry_is_bounded(tmp_path):
    """A writer that keeps losing has to give up, or a busy ledger turns one
    commit into an unbounded loop."""
    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {}))
    try:
        agg = _Aggregatorish(led, attempts=2)
        calls = {"n": 0}
        original = led.commit

        def always_conflict(*args, **kwargs):
            calls["n"] += 1
            raise CASConflict("busy")

        led.commit = always_conflict
        head = led.head_version(Ledger.DEV)
        diff = Diff(diff_id="d", target="a", ops={"k": "v"})
        art = led.snapshot(Ledger.DEV).get("a")
        assert agg._commit_with_retry("a", art.apply(diff), diff, head) is None
        assert calls["n"] == 2, f"attempted {calls['n']} times, bound was 2"
        led.commit = original
    finally:
        led.close()


def test_conflicts_are_counted(tmp_path):
    """Contention is invisible otherwise: a retry that succeeds looks exactly
    like a commit that never had to."""
    from agentdescent.metrics import Meter

    led = _ledger(str(tmp_path / "repo"))
    led.register(Tiny("a", {}))
    try:
        meter = Meter()
        agg = _Aggregatorish(led, meter=meter)
        art = led.snapshot(Ledger.DEV).get("a")
        stale = led.head_version(Ledger.DEV)
        led.commit(art.apply(Diff(diff_id="o", target="a", ops={"x": "1"})),
                   base_version=stale, branch=Ledger.DEV)

        diff = Diff(diff_id="ours", target="a", ops={"y": "2"})
        agg._commit_with_retry("a", art.apply(diff), diff, stale)
        assert meter.snapshot().cas_conflicts == 1
    finally:
        led.close()
