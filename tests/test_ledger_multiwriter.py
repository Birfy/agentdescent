"""Two processes, one ledger, and no lost updates.

CAS plus a version vector is already the right primitive; what it did not have
was a critical section that spans processes. `self._lock` is a `threading.Lock`:
it makes one process's workers take turns and says nothing to a second process.
Two of those interleave a read-modify-write of `versions.json` and one commit
disappears -- the lost update CAS exists to prevent, arriving by a route CAS
cannot see, because both writers read the same head and both were right when they
read it.

These use real processes. A threading test would pass against the old code.
"""

import json
import multiprocessing as mp
import os
import time

import pytest

from agentdescent.evolvable import Contract, Diff, Evolvable
from agentdescent.ledger import CASConflict, Ledger

CTX = mp.get_context("spawn")


class Tiny:
    """A minimal Evolvable: the ledger only needs id, version and state."""

    def __init__(self, id="a", state=None, version=1):
        self.id = id
        self.state = dict(state or {})
        self.version = version
        self.blast_radius = 0.2
        self.contract = Contract(input_schema="task", output_schema="text", major=1)

    def render(self):
        return json.dumps(self.state, sort_keys=True)

    def apply(self, diff):
        new = dict(self.state)
        for k, v in diff.ops.items():
            if v is None:
                new.pop(k, None)
            else:
                new[k] = v
        return Tiny(self.id, new, self.version + 1)


def _serialize(a):
    return {"state": dict(a.state)}


def _deserialize(aid, version, data):
    return Tiny(aid, data.get("state", {}), version)


def ledger_at(path):
    return Ledger(path, _serialize, _deserialize)


def _commit_many(repo, worker, attempts, out):
    """One process, hammering the same artifact. Returns commits that stuck."""
    led = ledger_at(repo)
    won = 0
    for i in range(attempts):
        for _ in range(50):                       # bounded rebase-and-retry
            head = led.head_version(Ledger.DEV)
            art = led.snapshot(Ledger.DEV).get("a")
            candidate = art.apply(Diff(diff_id=f"w{worker}-{i}", target="a",
                                       ops={f"k{worker}-{i}": "v"}))
            try:
                led.commit(candidate, base_version=head, branch=Ledger.DEV)
                won += 1
                break
            except CASConflict:
                continue                          # somebody else got there first
    led.close()
    out.put(won)


@pytest.mark.parametrize("processes,attempts", [(4, 5)])
def test_concurrent_processes_do_not_lose_updates(tmp_path, processes, attempts):
    """The core CAS assertion, across processes.

    Final version must equal the number of commits that reported success. Any
    shortfall is a lost update: two writers read the same head, both were right,
    and one of them overwrote the other.
    """
    repo = str(tmp_path / "repo")
    led = ledger_at(repo)
    led.register(Tiny("a", {}))
    led.close()

    out = CTX.Queue()
    procs = [CTX.Process(target=_commit_many, args=(repo, w, attempts, out))
             for w in range(processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, f"a writer died: exitcode={p.exitcode}"

    reported = sum(out.get() for _ in procs)
    led = ledger_at(repo)
    try:
        final = led.head_version(Ledger.DEV)["a"]
        state = led.snapshot(Ledger.DEV).get("a").state
    finally:
        led.close()

    assert reported == processes * attempts, "a writer gave up before finishing"
    assert final == 1 + reported, (
        f"final version {final} but {reported} commits succeeded: "
        f"{1 + reported - final} update(s) were lost")
    assert len(state) == reported, "a committed key is missing from the state"


def test_the_version_file_is_never_seen_half_written(tmp_path):
    """`open(..., 'w')` truncates first and fills after, so a reader in another
    process can see an empty file where the version vector should be. Renaming
    over the target is atomic, so a reader sees the old vector or the new one."""
    repo = str(tmp_path / "repo")
    led = ledger_at(repo)
    led.register(Tiny("a", {}))
    try:
        for i in range(20):
            head = led.head_version(Ledger.DEV)
            art = led.snapshot(Ledger.DEV).get("a")
            led.commit(art.apply(Diff(diff_id=f"d{i}", target="a", ops={f"k{i}": "v"})),
                       base_version=head, branch=Ledger.DEV)
            with open(os.path.join(repo, "versions.json"), encoding="utf-8") as fh:
                json.load(fh)                     # parses every time
    finally:
        led.close()
    assert not [p for p in os.listdir(repo) if p.endswith(".tmp")]


def test_a_stale_base_is_still_refused(tmp_path):
    """The lock serialises writers; it must not make CAS unnecessary. A writer
    holding a version that has moved is told to rebase, exactly as before."""
    repo = str(tmp_path / "repo")
    led = ledger_at(repo)
    led.register(Tiny("a", {}))
    try:
        stale = led.head_version(Ledger.DEV)
        art = led.snapshot(Ledger.DEV).get("a")
        led.commit(art.apply(Diff(diff_id="d1", target="a", ops={"x": "1"})),
                   base_version=stale, branch=Ledger.DEV)
        with pytest.raises(CASConflict):
            led.commit(art.apply(Diff(diff_id="d2", target="a", ops={"y": "2"})),
                       base_version=stale, branch=Ledger.DEV)
    finally:
        led.close()


def test_the_lock_lives_outside_the_working_tree(tmp_path):
    """A lock file in the tree is committed by `add -A` and then blocks the next
    `checkout` as an untracked file that would be overwritten."""
    repo = str(tmp_path / "repo")
    led = ledger_at(repo)
    led.register(Tiny("a", {}))
    try:
        assert not [p for p in os.listdir(repo) if "lock" in p], \
            "the repository lock is inside the working tree"
    finally:
        led.close()
