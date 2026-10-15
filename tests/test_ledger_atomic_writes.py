"""A failed ledger write must leave the working tree exactly as it found it (#265).

The ledger commits with ``git add -A``, so anything a failed operation leaves on
disk -- a torn ``artifacts/<id>.json``, half of an atomic transaction, a temp
sibling -- is swept into the *next*, unrelated commit. A torn artifact file then
makes every later ``snapshot()`` raise.
"""

import os
import subprocess

import pytest

import agentdescent.ledger as ledger_mod
from agentdescent.evolution import EvolvingArtifact
from agentdescent.ledger import Ledger


def make(tmp_path, ids=("a", "b")):
    lg = Ledger(str(tmp_path / "repo"), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v))
    for aid in ids:
        lg.register(EvolvingArtifact(aid, {"n": "1"}))
    return lg


def git_status(lg):
    return subprocess.run(["git", "-C", lg.repo_path, "status", "--porcelain"],
                          capture_output=True, text=True, check=True).stdout


def no_tmp(lg):
    found = []
    for root, _dirs, names in os.walk(lg.repo_path):
        if ".git" in root.split(os.sep):
            continue
        found += [n for n in names if n.endswith(".tmp")]
    return found


def test_an_unencodable_commit_leaves_no_torn_file_for_the_next_commit(tmp_path):
    lg = make(tmp_path)
    with pytest.raises(TypeError):
        lg.commit(EvolvingArtifact("a", {"n": "2", "z": {1, 2}}), {"a": 1})
    assert git_status(lg) == ""
    # An unrelated, valid write must not land a torn a.json.
    lg.commit(EvolvingArtifact("b", {"n": "2"}), {"b": 1})
    snap = lg.snapshot()
    assert snap.get("a").state == {"n": "1"}
    assert snap.get("b").state == {"n": "2"}
    assert snap.version == {"a": 1, "b": 2}
    # The rejected writer can still retry honestly from the same base.
    lg.commit(EvolvingArtifact("a", {"n": "2"}), {"a": 1})
    assert lg.snapshot().get("a").state == {"n": "2"}


def test_commit_atomic_encodes_everything_before_writing_anything(tmp_path):
    lg = make(tmp_path, ids=("a", "b", "c"))
    with pytest.raises(TypeError):
        lg.commit_atomic([EvolvingArtifact("a", {"n": "2"}),
                          EvolvingArtifact("b", {"z": {1}})], {"a": 1, "b": 1})
    assert git_status(lg) == ""
    lg.commit(EvolvingArtifact("c", {"n": "2"}), {"c": 1})
    snap = lg.snapshot()
    assert snap.get("a").state == {"n": "1"}
    assert snap.version == {"a": 1, "b": 1, "c": 2}


def test_commit_atomic_rolls_back_files_already_written_when_a_later_write_fails(
        tmp_path, monkeypatch):
    lg = make(tmp_path, ids=("a", "b", "c"))
    real = ledger_mod._replace_file
    calls = {"n": 0}

    def flaky(path, data):
        calls["n"] += 1
        if calls["n"] == 2:            # a.json is replaced; b.json hits a full disk
            raise OSError(28, "No space left on device")
        real(path, data)

    monkeypatch.setattr(ledger_mod, "_replace_file", flaky)
    with pytest.raises(OSError):
        lg.commit_atomic([EvolvingArtifact("a", {"n": "2"}),
                          EvolvingArtifact("b", {"n": "2"})], {"a": 1, "b": 1})
    monkeypatch.setattr(ledger_mod, "_replace_file", real)
    assert git_status(lg) == ""
    assert no_tmp(lg) == []
    lg.commit(EvolvingArtifact("c", {"n": "2"}), {"c": 1})
    snap = lg.snapshot()
    assert snap.get("a").state == {"n": "1"} and snap.get("b").state == {"n": "1"}
    assert snap.version == {"a": 1, "b": 1, "c": 2}


def test_a_failed_versions_write_leaves_no_tmp_and_no_new_artifact(tmp_path, monkeypatch):
    lg = make(tmp_path, ids=("a",))
    real_replace = os.replace

    def boom(src, dst):
        if dst.endswith("versions.json"):
            raise OSError("read-only")
        real_replace(src, dst)

    monkeypatch.setattr(ledger_mod.os, "replace", boom)
    with pytest.raises(OSError):
        lg.register(EvolvingArtifact("new", {"n": "1"}))
    monkeypatch.setattr(ledger_mod.os, "replace", real_replace)
    assert no_tmp(lg) == []
    assert git_status(lg) == ""
    assert lg.snapshot().version == {"a": 1}
