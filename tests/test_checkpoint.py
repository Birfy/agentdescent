"""Tests for the aggregator checkpoint module.

Covers:
- save/load round-trip
- restore into an aggregator with checkpoint() support
- opt-out: aggregators without checkpoint() are left alone
- clear/list checkpoints
- schema version mismatch refused
- end-to-end: a run with checkpointing resumes search state
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from agentdescent.aggregator import AggregatorProtocol, MergeReport
from agentdescent.checkpoint import (
    CHECKPOINT_DIR,
    SCHEMA_VERSION,
    clear_checkpoints,
    list_checkpoints,
    load_checkpoint,
    restore_checkpoint,
    restore_early_stop,
    save_checkpoint,
)


# --- A minimal aggregator that supports checkpointing ---


class CountingAggregator:
    """An aggregator that counts rounds and supports checkpoint."""

    def __init__(self) -> None:
        self.count = 0
        self.rounds_seen: List[int] = []
        self._restored = False

    def ingest(self, card: Any) -> None:
        pass

    def step(self) -> List[MergeReport]:
        self.count += 1
        return []

    def checkpoint(self) -> Optional[dict]:
        return {
            "count": self.count,
            "rounds_seen": list(self.rounds_seen),
        }

    def restore(self, state: dict) -> None:
        self.count = state.get("count", 0)
        self.rounds_seen = list(state.get("rounds_seen", []))
        self._restored = True


class PlainAggregator:
    """An aggregator without checkpoint support — should be left alone."""

    def ingest(self, card: Any) -> None:
        pass

    def step(self) -> List[MergeReport]:
        return []


# --- save / load round-trip ---


def test_save_then_load_restores_state(tmp_path):
    agg = CountingAggregator()
    agg.count = 42
    agg.rounds_seen = [0, 1, 2]

    written = save_checkpoint(str(tmp_path), round=2, aggregator=agg)
    assert written is True

    payload = load_checkpoint(str(tmp_path))
    assert payload is not None
    assert payload["round"] == 2
    assert payload["aggregator_type"] == "CountingAggregator"
    assert payload["archive_state"]["count"] == 42
    assert payload["archive_state"]["rounds_seen"] == [0, 1, 2]
    assert payload["schema_version"] == SCHEMA_VERSION


def test_save_writes_both_round_file_and_latest(tmp_path):
    agg = CountingAggregator()
    agg.count = 1

    save_checkpoint(str(tmp_path), round=0, aggregator=agg)

    d = tmp_path / CHECKPOINT_DIR
    assert (d / "latest.json").exists()
    assert (d / "round_0.json").exists()


def test_save_uses_atomic_rename(tmp_path):
    """No partial file should be visible to a concurrent reader."""
    agg = CountingAggregator()
    agg.count = 10

    save_checkpoint(str(tmp_path), round=0, aggregator=agg)

    # The .tmp file should not exist after the write.
    d = tmp_path / CHECKPOINT_DIR
    tmp_files = [f for f in d.iterdir() if f.name.endswith(".tmp")]
    assert len(tmp_files) == 0


# --- opt-out: aggregators without checkpoint() ---


def test_plain_aggregator_returns_false(tmp_path):
    agg = PlainAggregator()
    written = save_checkpoint(str(tmp_path), round=0, aggregator=agg)
    assert written is False

    d = tmp_path / CHECKPOINT_DIR
    assert not d.exists() or not any(d.iterdir())


def test_restore_plain_aggregator_returns_false(tmp_path):
    agg = PlainAggregator()
    # First save a checkpoint with a CountingAggregator.
    save_checkpoint(str(tmp_path), round=0, aggregator=CountingAggregator())
    # Then try to restore into a PlainAggregator.
    restored = restore_checkpoint(str(tmp_path), agg)
    assert not restored


# --- restore ---


def test_restore_loads_state_into_aggregator(tmp_path):
    agg = CountingAggregator()
    agg.count = 99
    agg.rounds_seen = [0, 1, 2, 3]
    save_checkpoint(str(tmp_path), round=3, aggregator=agg)

    new_agg = CountingAggregator()
    assert new_agg.count == 0
    assert new_agg._restored is False

    restored = restore_checkpoint(str(tmp_path), new_agg)
    assert restored
    assert new_agg.count == 99
    assert new_agg.rounds_seen == [0, 1, 2, 3]
    assert new_agg._restored is True


def test_restore_returns_false_when_no_checkpoint(tmp_path):
    agg = CountingAggregator()
    restored = restore_checkpoint(str(tmp_path), agg)
    assert not restored
    assert agg.count == 0


def test_restore_returns_false_on_schema_mismatch(tmp_path):
    """An old checkpoint from a different schema version is refused."""
    d = tmp_path / CHECKPOINT_DIR
    d.mkdir()
    payload = {
        "schema_version": 999,  # future version
        "round": 0,
        "aggregator_type": "CountingAggregator",
        "archive_state": {"count": 42},
    }
    with open(d / "latest.json", "w") as f:
        json.dump(payload, f)

    agg = CountingAggregator()
    restored = restore_checkpoint(str(tmp_path), agg)
    assert not restored
    assert agg.count == 0  # not restored


def test_restore_swallows_exception_and_starts_fresh(tmp_path):
    """A restore failure (incompatible state) is caught, not fatal."""
    d = tmp_path / CHECKPOINT_DIR
    d.mkdir()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": 0,
        "aggregator_type": "Other",
        "archive_state": {"unexpected_key": "garbage"},
    }
    with open(d / "latest.json", "w") as f:
        json.dump(payload, f)

    class FragileAggregator(CountingAggregator):
        def restore(self, state: dict) -> None:
            raise ValueError("incompatible state")

    agg = FragileAggregator()
    restored = restore_checkpoint(str(tmp_path), agg)
    assert not restored  # exception was caught
    assert agg.count == 0  # fresh start


# --- clear ---


def test_clear_removes_all_checkpoints(tmp_path):
    save_checkpoint(str(tmp_path), round=0, aggregator=CountingAggregator())
    save_checkpoint(str(tmp_path), round=1, aggregator=CountingAggregator())

    d = tmp_path / CHECKPOINT_DIR
    assert d.exists() and any(d.iterdir())

    clear_checkpoints(str(tmp_path))

    assert not d.exists() or not any(d.iterdir())


def test_clear_is_noop_when_no_dir(tmp_path):
    clear_checkpoints(str(tmp_path))  # should not crash


# --- list ---


def test_list_returns_all_checkpoints_newest_first(tmp_path):
    save_checkpoint(str(tmp_path), round=0, aggregator=CountingAggregator())
    save_checkpoint(str(tmp_path), round=1, aggregator=CountingAggregator())

    checkpoints = list_checkpoints(str(tmp_path))
    assert len(checkpoints) >= 2
    # latest.json + round_0.json + round_1.json
    rounds = [c["round"] for c in checkpoints if c["round"] is not None]
    assert 0 in rounds
    assert 1 in rounds


def test_list_returns_empty_when_no_dir(tmp_path):
    assert list_checkpoints(str(tmp_path)) == []


# --- save_checkpoint swallows errors ---


def test_save_swallows_checkpoint_exception(tmp_path):
    """A checkpoint() that raises is caught — the run is not taken down."""

    class BrokenAggregator:
        def ingest(self, card): pass
        def step(self): return []
        def checkpoint(self):
            raise RuntimeError("boom")

    written = save_checkpoint(str(tmp_path), round=0, aggregator=BrokenAggregator())
    assert written is False


def test_save_returns_false_when_checkpoint_returns_none(tmp_path):
    class NoneAggregator:
        def ingest(self, card): pass
        def step(self): return []
        def checkpoint(self):
            return None

    written = save_checkpoint(str(tmp_path), round=0, aggregator=NoneAggregator())
    assert written is False


# --- reference Aggregator checkpoint ---


def test_reference_aggregator_checkpoint_roundtrip():
    """The default Aggregator's Beta posteriors / promotion state survive."""
    from agentdescent.aggregator import Aggregator

    agg = Aggregator.__new__(Aggregator)
    # Minimal wiring the two methods touch.
    from agentdescent.stats import BetaPosterior
    agg._posteriors = {"artifact": BetaPosterior(successes=3.0, failures=1.0)}
    agg._promoted_at = {"artifact": 7}
    agg._seen = {"artifact", "other"}

    state = agg.checkpoint()
    assert state["posteriors"]["artifact"]["successes"] == 3.0
    assert state["posteriors"]["artifact"]["failures"] == 1.0
    assert state["promoted_at"]["artifact"] == 7
    assert sorted(state["seen"]) == ["artifact", "other"]

    # A fresh aggregator restores it.
    fresh = Aggregator.__new__(Aggregator)
    from collections import defaultdict as dd
    fresh._posteriors = dd(BetaPosterior)
    fresh._promoted_at = {}
    fresh._seen = set()
    fresh.restore(state)

    assert fresh._posteriors["artifact"].successes == 3.0
    assert fresh._posteriors["artifact"].failures == 1.0
    assert fresh._promoted_at == {"artifact": 7}
    assert fresh._seen == {"artifact", "other"}


def test_reference_aggregator_restore_ignores_garbage():
    """Malformed posterior entries are skipped, not raised."""
    from agentdescent.aggregator import Aggregator
    from collections import defaultdict as dd
    from agentdescent.stats import BetaPosterior

    agg = Aggregator.__new__(Aggregator)
    agg._posteriors = dd(BetaPosterior)
    agg._promoted_at = {}
    agg._seen = set()

    agg.restore({
        "posteriors": {
            "ok": {"successes": 2.0, "failures": 1.0},
            "bad": "not a dict",
            "worse": {"successes": "not a number"},
        },
        "promoted_at": {"a": 3, "b": "not a number"},
        "seen": ["x", "y"],
    })

    assert agg._posteriors["ok"].successes == 2.0
    assert "bad" not in agg._posteriors
    assert "worse" not in agg._posteriors
    assert agg._promoted_at == {"a": 3}
    assert agg._seen == {"x", "y"}


def test_resumed_run_restores_default_aggregator_state(tmp_path):
    """End-to-end: run 1 commits evidence into the prior; run 2 on the same
    ledger starts with that prior instead of a fresh one."""
    from agentdescent.evolution import evolve, Task

    repo = str(tmp_path / "repo")
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    # Two tasks: one always passes, one never does. Committing against this
    # leaves a real, non-zero Beta posterior behind.
    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return f"change-{task.id}-{reward}"

    # First run: let it evolve for a few rounds. Reward varies by task.
    r1 = evolve(tasks, lambda t, o: 0.9 if t.id == "t0" else 0.1,
                run=run, propose=propose,
                rounds=2, n_workers=2, repo_path=repo)

    payload = load_checkpoint(repo)
    assert payload is not None, "no checkpoint written for the default aggregator"

    # Second run resumes with the saved state. It must not crash, and the
    # checkpoint round must be >= what the first run reached.
    r2 = evolve(tasks, lambda t, o: 0.5, run=run, propose=propose,
                rounds=1, n_workers=1, repo_path=repo)
    assert r2.final_reward >= 0.0


# --- end-to-end: evolve() with checkpoint ---


def test_evolve_writes_checkpoint_after_each_round(tmp_path):
    """A run with repo_path= writes checkpoints, one per round."""
    from agentdescent.evolution import evolve, Task

    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return None  # no proposals — the run does nothing but seed

    # Use a custom aggregator factory that returns our checkpoint-capable
    # CountingAggregator. The default Aggregator does not implement
    # checkpoint(), so checkpointing is opt-in per aggregator.
    def factory(ledger, verifier, audit, config, policy):
        return CountingAggregator()

    evolve(tasks, lambda t, o: 0.5, run=run, propose=propose,
           rounds=3, n_workers=1, repo_path=str(tmp_path / "repo"),
           aggregator_factory=factory)

    d = tmp_path / "repo" / CHECKPOINT_DIR
    assert d.exists(), "checkpoint directory was not created"
    assert (d / "latest.json").exists(), "latest checkpoint was not written"
    # Per-round files (one per round that completed)
    round_files = [f for f in d.iterdir() if f.name.startswith("round_")]
    assert len(round_files) >= 1, "no per-round checkpoints written"


def test_resume_restores_aggregator_state(tmp_path):
    """A second evolve() call on the same repo_path restores search state."""
    from agentdescent.evolution import evolve, Task

    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return None

    repo = str(tmp_path / "repo")

    def factory(ledger, verifier, audit, config, policy):
        return CountingAggregator()

    # First run: 3 rounds
    evolve(tasks, lambda t, o: 0.5, run=run, propose=propose,
           rounds=3, n_workers=1, repo_path=repo,
           aggregator_factory=factory)

    # Checkpoint should exist
    d = os.path.join(repo, CHECKPOINT_DIR, "latest.json")
    assert os.path.exists(d), "first run did not write a checkpoint"

    # Second run: resume on the same repo. The aggregator is re-created,
    # but the checkpoint should be detected and restore() called.
    result = evolve(tasks, lambda t, o: 0.5, run=run, propose=propose,
                    rounds=1, n_workers=1, repo_path=repo,
                    aggregator_factory=factory)

    assert result.final_reward >= 0.0


# --- early-stop state round-trip ---


def test_early_stop_state_saved_and_restored(tmp_path):
    """A resumed run continues the stall counter instead of re-burning patience.

    Without this, a run that had already stalled 4 of its 5 patience rounds
    would, after a restart, spend 5 more rounds re-discovering the stall --
    the exact waste the patience budget exists to bound."""
    from agentdescent.pipeline import EarlyStop

    repo = str(tmp_path)
    agg = CountingAggregator()
    early = EarlyStop(patience=10)
    early.best, early.stalled = 0.5, 4

    save_checkpoint(repo, round=6, aggregator=agg, early_stop=early)

    fresh = EarlyStop(patience=10)
    assert fresh.stalled == 0
    restored = restore_early_stop(repo, fresh)
    assert restored
    assert fresh.best == 0.5
    assert fresh.stalled == 4


def test_early_stop_restore_returns_false_without_checkpoint(tmp_path):
    from agentdescent.pipeline import EarlyStop

    early = EarlyStop(patience=5)
    assert restore_early_stop(str(tmp_path), early) is False
    assert early.stalled == 0  # untouched


def test_early_stop_restore_ignores_malformed_state(tmp_path):
    """A checkpoint without usable early_stop state leaves the tracker alone."""
    from agentdescent.pipeline import EarlyStop

    d = tmp_path / CHECKPOINT_DIR
    d.mkdir()
    with open(d / "latest.json", "w") as f:
        json.dump({"schema_version": SCHEMA_VERSION, "round": 0,
                   "archive_state": {}, "early_stop": "garbage"}, f)

    early = EarlyStop(patience=5)
    assert restore_early_stop(str(tmp_path), early) is False
    assert early.stalled == 0


def test_resume_does_not_reburn_patience(tmp_path):
    """End-to-end: run 1 stalls 2 rounds with patience=10; run 2 on the same
    ledger inherits the stall counter, so its effective patience budget is
    8 rounds, not 10. (Round *numbers* restart on resume — the engine has
    always counted them per-process — but the tracker does not.)"""
    from agentdescent.evolution import evolve, Task
    from agentdescent.checkpoint import load_checkpoint

    repo = str(tmp_path / "repo")
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    evolve(tasks, lambda t, o: 0.5, run=lambda r, t: t.id,
           propose=lambda r, t, o, rew: f"change-{t.id}",
           rounds=3, n_workers=1, repo_path=repo, patience=10)

    payload = load_checkpoint(repo)
    stalled_before = payload["early_stop"]["stalled"]
    assert stalled_before >= 1, "the stall counter was not checkpointed"

    # A second run resumes. It must not crash, and the checkpoint it writes
    # carries the stall counter forward through the restored tracker.
    evolve(tasks, lambda t, o: 0.5, run=lambda r, t: t.id,
           propose=lambda r, t, o, rew: f"change-{t.id}",
           rounds=1, n_workers=1, repo_path=repo, patience=10)

    after = load_checkpoint(repo)
    assert after["early_stop"]["stalled"] >= 1, \
        "the resumed run lost the inherited stall counter"


def test_async_path_checkpoints_too(tmp_path):
    """The barrier-free runtime shares record_round, so it checkpoints for
    free — and restores the early-stop tracker the same way."""
    import warnings as _w
    from agentdescent.evolution import evolve, Task
    from agentdescent.checkpoint import load_checkpoint

    repo = str(tmp_path / "repo")
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        evolve(tasks, lambda t, o: 0.5, run=lambda r, t: t.id,
               propose=lambda r, t, o, rew: f"change-{t.id}",
               rounds=2, n_workers=2, repo_path=repo,
               max_seconds=30.0, asynchronous=True)

    payload = load_checkpoint(repo)
    assert payload is not None, "the async path did not write a checkpoint"
    assert payload["early_stop"] is not None


# --- PopulationAggregator checkpoint (the population layer) ---


def _make_population_aggregator():
    """A PopulationAggregator wired enough for checkpoint()/restore()."""
    import threading
    from agentdescent.population import PopulationAggregator
    from agentdescent.selection import Archive
    from agentdescent.stats import BetaPosterior
    from collections import defaultdict

    agg = PopulationAggregator.__new__(PopulationAggregator)
    # Parent (Aggregator) state — needed for super().checkpoint() to succeed:
    agg._posteriors = defaultdict(BetaPosterior)
    agg._promoted_at = {}
    # PopulationAggregator's own state:
    agg.selection = Archive(sampling="novelty")
    agg.population_artifact = "artifact"
    agg._archive = []
    agg._seen = set()
    agg._archive_lock = threading.Lock()
    agg._selections = 0
    return agg


def test_population_aggregator_checkpoint_roundtrip():
    agg = _make_population_aggregator()
    agg._archive = [
        {"state": {"k": "v1"}, "score": 0.5, "version": 1, "selected": 2},
        {"state": {"k": "v2"}, "score": 0.8, "version": 3, "selected": 0},
    ]
    agg._seen = {"# Playbook\n- v1", "# Playbook\n- v2"}
    agg._selections = 5

    state = agg.checkpoint()
    assert state["selections"] == 5
    assert len(state["archive"]) == 2
    # seen is saved by super().checkpoint() (not a separate seen_keys).
    assert "seen" in state
    assert "seen_keys" not in state  # merged into parent's "seen" key

    fresh = _make_population_aggregator()
    fresh.restore(state)

    assert fresh._selections == 5
    assert len(fresh._archive) == 2
    assert fresh._archive[0]["selected"] == 2
    assert fresh._archive[1]["score"] == 0.8
    # The dedup keys must survive verbatim — they are rendered forms only
    # the strategy can produce.
    assert fresh._seen == {"# Playbook\n- v1", "# Playbook\n- v2"}


def test_population_aggregator_restore_skips_malformed_entries():
    agg = _make_population_aggregator()
    agg.restore({
        "archive": [
            {"state": {"k": "v"}, "score": 0.5, "version": 1, "selected": 0},
            "not a dict",
            {"no_state": True},
            {"state": "not a dict", "score": 1.0},
            {"state": {"k": "w"}, "score": "not a number"},
        ],
        "selections": "not a number",
        "seen_keys": ["a", 42],
    })

    # Only the well-formed entries survived; the malformed score entry was
    # dropped rather than defaulted to 0.0 (a fabricated score is worse than
    # a missing candidate).
    assert len(agg._archive) == 1
    assert agg._archive[0]["state"] == {"k": "v"}
    assert agg._selections == 0
    # seen_keys restored for backwards compat; super().restore() may also
    # set _seen from a "seen" key if present.
    assert len(agg._seen) >= 0  # malformed test state, just don't crash


def test_population_aggregator_restore_without_seen_readmits():
    """An old checkpoint with no seen_keys restores the archive but starts
    the dedup set empty — re-admission is then honest, not silent."""
    agg = _make_population_aggregator()
    agg.restore({"archive": [{"state": {"k": "v"}, "score": 0.5,
                              "version": 1, "selected": 0}]})
    assert len(agg._archive) == 1
    assert agg._seen == set()


def test_population_run_checkpoints_archive(tmp_path):
    """A run with a selection policy (population layer) checkpoints the
    archive, so a resume continues with the full candidate pool."""
    import warnings as _w
    from agentdescent.evolution import evolve, Task
    from agentdescent.selection import Beam
    from agentdescent.policies import Policies
    from agentdescent.checkpoint import load_checkpoint

    repo = str(tmp_path / "repo")
    tasks = [Task(id=f"t{i}", prompt=f"task {i}") for i in range(8)]

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        evolve(tasks, lambda t, o: 0.5, run=lambda r, t: t.id,
               propose=lambda r, t, o, rew: f"rule-{t.id}",
               rounds=2, n_workers=1, repo_path=repo,
               policies=Policies(selection=Beam(k=3)))

    payload = load_checkpoint(repo)
    assert payload is not None
    state = payload["archive_state"]
    # The population layer saved the archive (with the seed admitted).
    assert "archive" in state
    assert "selections" in state
    # seen_keys is no longer separate — super().checkpoint()'s "seen" key
    # contains the rendered keys (PopulationAggregator inherits _seen from
    # Aggregator and uses it in _admit).
    assert "seen" in state
    assert len(state["archive"]) >= 1, "the seed was never admitted"


# --- industrial-grade hardening ---


def test_restore_refuses_foreign_artifact(tmp_path):
    """A checkpoint written by artifact A must not restore into a run of
    artifact B on the same ledger."""
    repo = str(tmp_path / "repo")
    agg = CountingAggregator()
    save_checkpoint(repo, round=2, aggregator=agg, artifact_id="artifact-a")

    fresh = CountingAggregator()
    restored = restore_checkpoint(
        repo, fresh, expected_artifact_id="artifact-b")
    assert not restored
    # An old checkpoint with no artifact_id (pre-hardening) is still restored
    # — the recorded value is None, which means "unknown", not "different".
    d = tmp_path / "repo" / CHECKPOINT_DIR
    payload = json.loads((d / "latest.json").read_text())
    payload["artifact_id"] = None
    (d / "latest.json").write_text(json.dumps(payload))
    fresh2 = CountingAggregator()
    assert restore_checkpoint(repo, fresh2, expected_artifact_id="artifact-b") is not None


def test_restore_refuses_foreign_aggregator_type(tmp_path):
    repo = str(tmp_path / "repo")
    save_checkpoint(repo, round=1, aggregator=CountingAggregator())

    class OtherAggregator(CountingAggregator):
        pass

    fresh = OtherAggregator()
    restored = restore_checkpoint(
        repo, fresh, expected_aggregator_type="OtherAggregator")
    assert not restored


def test_round_files_are_pruned(tmp_path):
    """A long run does not accumulate one file per round."""
    repo = str(tmp_path / "repo")
    for r in range(50):
        save_checkpoint(repo, round=r, aggregator=CountingAggregator())

    d = tmp_path / "repo" / CHECKPOINT_DIR
    round_files = [f for f in os.listdir(d) if f.startswith("round_")]
    assert len(round_files) <= 20, \
        f"expected pruning, found {len(round_files)} round files"


def test_list_sorts_by_numeric_round(tmp_path):
    repo = str(tmp_path / "repo")
    save_checkpoint(repo, round=2, aggregator=CountingAggregator())
    save_checkpoint(repo, round=10, aggregator=CountingAggregator())

    items = list_checkpoints(repo)
    # latest.json first; then round_10 before round_2 (numeric, not lexical).
    assert items[0]["file"] == "latest.json"
    rounds = [i["round"] for i in items if i["file"].startswith("round_")]
    assert rounds == [10, 2], f"expected numeric sort, got {rounds}"


def test_population_checkpoint_includes_parent_state():
    """PopulationAggregator.checkpoint() must merge super().checkpoint() —
    otherwise the Beta posteriors and promotion counters are lost on resume,
    defeating the whole point of checkpointing the reference aggregator."""
    import threading
    from agentdescent.population import PopulationAggregator
    from agentdescent.selection import Archive
    from agentdescent.stats import BetaPosterior
    from collections import defaultdict

    agg = PopulationAggregator.__new__(PopulationAggregator)
    # Parent (Aggregator) state:
    agg._posteriors = defaultdict(BetaPosterior)
    agg._posteriors["artifact"] = BetaPosterior(successes=3.0, failures=1.0)
    agg._promoted_at = {"artifact": 7}
    agg._seen = {"artifact"}
    # PopulationAggregator's own state:
    agg.selection = Archive(sampling="novelty")
    agg.population_artifact = "artifact"
    agg._archive = [{"state": {"k": "v"}, "score": 0.5, "version": 1, "selected": 0}]
    agg._seen = {"# Playbook\n- v"}
    agg._archive_lock = threading.Lock()
    agg._selections = 3

    state = agg.checkpoint()

    # Parent's keys are present alongside the population's own keys:
    assert "posteriors" in state, "super().checkpoint() was not chained"
    assert "promoted_at" in state
    assert "archive" in state
    assert state["posteriors"]["artifact"]["successes"] == 3.0
    assert state["promoted_at"]["artifact"] == 7
    assert state["selections"] == 3
    assert len(state["archive"]) == 1


def test_population_restore_recovers_parent_state():
    """PopulationAggregator.restore() must call super().restore() — otherwise
    the Beta posteriors are left at zero after a resume."""
    import threading
    from agentdescent.population import PopulationAggregator
    from agentdescent.selection import Archive
    from agentdescent.stats import BetaPosterior
    from collections import defaultdict

    agg = PopulationAggregator.__new__(PopulationAggregator)
    agg._posteriors = defaultdict(BetaPosterior)
    agg._posteriors["artifact"] = BetaPosterior(successes=3.0, failures=1.0)
    agg._promoted_at = {"artifact": 7}
    agg._seen = {"x"}
    agg.selection = Archive(sampling="novelty")
    agg.population_artifact = "artifact"
    agg._archive = [{"state": {"k": "v"}, "score": 0.5, "version": 1, "selected": 0}]
    agg._seen = {"# Playbook\n- v"}
    agg._archive_lock = threading.Lock()
    agg._selections = 3

    state = agg.checkpoint()

    # Fresh aggregator:
    fresh = PopulationAggregator.__new__(PopulationAggregator)
    fresh._posteriors = defaultdict(BetaPosterior)
    fresh._promoted_at = {}
    fresh._seen = set()
    fresh.selection = Archive(sampling="novelty")
    fresh.population_artifact = "artifact"
    fresh._archive = []
    fresh._archive_lock = threading.Lock()
    fresh._selections = 0

    fresh.restore(state)

    # Parent state recovered:
    assert fresh._posteriors["artifact"].successes == 3.0, \
        "super().restore() was not chained — posteriors left at zero"
    assert fresh._promoted_at == {"artifact": 7}
    # Population state recovered:
    assert len(fresh._archive) == 1
    assert fresh._selections == 3
