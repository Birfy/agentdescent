"""Aggregator checkpoint: save and restore in-memory search state across runs.

The ledger persists *artifacts* (the evolving code/skill/prompt) across runs —
that is its job. But an aggregator also holds **search state that is not an
artifact**: PUCT's tree with its visit counts, OpenEvolve's MAP-Elites islands,
DGM's keep-all archive with its novelty counters, ACE's Beta posteriors, GEPA's
Pareto frontier. When ``RunStore.resume()`` re-launches a run on the same ledger,
the aggregator starts fresh — one program (the head), no history, no counts.

This module serialises that search state after every round (in the ``on_round``
hook) and restores it when the aggregator is created (in the factory). The
format is a plain JSON file under ``<repo_path>/checkpoints/``, so it survives
process crashes and is inspectable.

**Nothing here raises.** Every entry point returns a "there is no checkpoint"
value -- ``False``, ``None``, ``[]`` -- when the state will not serialise, the
lock is held by another process, or the disk refuses the write. The floor is
what a run without checkpointing already does (resume the artifact, restart the
search), so a checkpoint problem can only ever cost that difference; letting it
propagate would instead take down the round it fired in, or the resume that was
trying to read it.

``checkpoints/`` is kept out of the ledger's index by
``Ledger._exclude_non_ledger_files``. That is load-bearing, not tidiness:
``Ledger._commit`` runs ``git add -A``, and a committed checkpoint is deleted by
the next ``dev -> stable`` switch and then blocks every switch after it, because
``git checkout`` will not clobber a tracked file the following round rewrote.

**Design.**

The :class:`~agentdescent.aggregator.AggregatorProtocol` gains two *optional*
methods: ``checkpoint() -> dict | None`` and ``restore(state: dict) -> None``.
An aggregator that does not implement them (or returns ``None``) simply opts
out — the run works, the resume just loses search state, which is the current
behaviour. This is a non-breaking extension: every existing aggregator and
custom factory keeps working unchanged.

The wiring is in :func:`~agentdescent.evolution.evolve` and
:func:`~agentdescent.async_evolve.async_evolve`: after each round's
``record_round``, the engine calls ``_save_checkpoint``; in ``_build_engine``,
after the factory creates the aggregator, the engine calls
``_restore_checkpoint``. Both are no-ops when the aggregator does not support
checkpointing or when ``repo_path`` is ``None`` (a scratch run).

**What is saved.**

Each aggregator's ``checkpoint()`` returns a JSON-serialisable dict containing
whatever it needs to resume: the archive, the selection counters, the Beta
posteriors, the Pareto frontier. The format is aggregator-specific and versioned
with a ``"schema_version"`` key, so a future change to the archive structure
can detect and refuse an old checkpoint rather than restoring garbage.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


#: The subdirectory under a ledger's ``repo_path`` where checkpoints live.
CHECKPOINT_DIR = "checkpoints"

#: The filename for the latest checkpoint. Old per-round files are kept for
#: debugging but the resume path reads only this one.
LATEST_FILE = "latest.json"

#: How many per-round files to keep. A long run could otherwise accumulate a
#: file per round; the latest file is always kept, and the per-round history
#: is pruned to this many entries (oldest dropped).
MAX_ROUND_FILES = 20

#: Current checkpoint schema version. Bumped when the dict shape changes;
#: ``_load`` refuses a mismatched version rather than restoring incompatible state.
SCHEMA_VERSION = 1

#: How long a checkpoint lock may be held before a reader gives up. Mirrors the
#: ledger's own lock timeout; a held checkpoint lock is a live writer, and the
#: resume path must not block a run waiting for it.
_LOCK_TIMEOUT = 30.0


#: The lock file guarding checkpoint reads and writes between processes. It
#: lives *inside* ``checkpoints/`` because that directory is the one the ledger
#: excludes from its index (see ``Ledger._exclude_non_ledger_files``): a lock
#: file anywhere else in the working tree would be swept up by ``git add -A``
#: and then block the next checkout. Keeping it here also means checkpoints
#: work on a plain directory, with no git repository at all.
LOCK_FILE = ".lock"


@contextmanager
def _checkpoint_lock(repo_path: str):
    """Exclude concurrent checkpoint writers in *other processes*.

    Without this, a resume racing an old process still running on the same
    ledger could interleave writes and leave ``latest.json`` holding a
    half-history.

    Yields ``True`` when the lock is held and ``False`` when it could not be
    taken -- **it never raises**. Every caller is on a path where "no
    checkpoint" is a supported outcome and a crash is not: a write runs inside
    the round loop, and a read runs at startup and from ``status``. Raising
    here turned the one race this lock exists for -- a resume while the old
    process is still alive -- into a hard failure of the new run *and* of the
    old one's next round, which is strictly worse than the lock-free behaviour
    it replaced.
    """
    from .ledger import _acquire_file_lock, _release_file_lock
    try:
        lock_path = os.path.join(_checkpoint_dir(repo_path), LOCK_FILE)
        # `TimeoutError` is an `OSError`, so a busy lock and an unwritable
        # directory arrive by the same door.
        handle = _acquire_file_lock(lock_path, timeout=_LOCK_TIMEOUT)
    except OSError:
        yield False
        return
    try:
        yield True
    finally:
        try:
            _release_file_lock(handle, lock_path)
        except OSError:
            pass


def _checkpoint_dir(repo_path: str) -> str:
    """The directory checkpoints are written to, created on first use."""
    d = os.path.join(repo_path, CHECKPOINT_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(
    repo_path: str,
    round: int,
    aggregator: Any,
    *,
    round_info: Optional[dict] = None,
    early_stop: Optional[Any] = None,
    artifact_id: Optional[str] = None,
) -> bool:
    """Serialise the aggregator's in-memory state to disk.

    Called from the engine's ``on_round`` hook after every round (or every
    ``checkpoint_interval`` rounds). Returns ``True`` if a checkpoint was
    written, ``False`` if it was not -- the aggregator does not support
    checkpointing, its state does not serialise, the lock was not free, or the
    write failed. **Never raises**: this runs inside the round loop, where the
    worst a missing checkpoint can cost is a resume that starts the search
    fresh, and that is what a run without checkpointing does anyway.

    ``early_stop`` is the run's :class:`~agentdescent.pipeline.EarlyStop`
    tracker. Its ``best``/``stalled`` state is saved alongside the aggregator:
    without it, a resumed run re-burns its patience budget re-discovering the
    stall the previous process had already counted. ``None`` skips it (some
    drivers construct the tracker after the first checkpoint fires).

    ``artifact_id`` is the id of the artifact this aggregator evolves. It is
    recorded in the checkpoint so a resume can refuse a checkpoint that
    belongs to a *different* artifact on the same ledger — restoring search
    state across artifacts would silently seed one search with another's
    history.

    Two files are written: ``round_N.json`` (for debugging / inspection) and
    ``latest.json`` (for the resume path). Both use atomic-rename to avoid a
    truncated read by a concurrent process, and both hold the inter-process
    checkpoint lock so two processes can never interleave writes.
    """
    checkpoint_fn = getattr(aggregator, "checkpoint", None)
    if not callable(checkpoint_fn):
        return False
    try:
        state = checkpoint_fn()
    except Exception:
        # A checkpoint failure must never take the run down. The aggregator's
        # own ``step()`` errors are allowed to (they are a contract violation);
        # checkpoint is a diagnostic path and its failure means "resume will
        # start fresh", which is the current behaviour anyway.
        return False
    if state is None:
        return False

    early = None
    if early_stop is not None:
        early = {
            "best": early_stop.best,
            "stalled": early_stop.stalled,
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": round,
        "artifact_id": artifact_id,
        "aggregator_type": type(aggregator).__name__,
        "archive_state": state,
        "round_info": round_info,
        "early_stop": early,
        "timestamp": time.time(),
    }

    # Serialise before touching the disk, and with no ``default=`` fallback.
    # ``default=str`` looked like belt-and-braces and was the opposite: an
    # aggregator state holding a set or a tuple was written as its ``repr``,
    # the save reported success, and ``restore`` handed the search a candidate
    # whose state had silently changed shape -- which ``PopulationAggregator``
    # then commits to the ledger as the next parent. A state that does not
    # round-trip through JSON has no checkpoint, and says so.
    try:
        blob = json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return False

    try:
        d = _checkpoint_dir(repo_path)
    except OSError:
        return False
    with _checkpoint_lock(repo_path) as locked:
        if not locked:
            return False
        # Write per-round file for inspection, then latest for the resume path.
        round_path = os.path.join(d, f"round_{round}.json")
        latest_path = os.path.join(d, LATEST_FILE)
        for path in (round_path, latest_path):
            tmp = f"{path}.{os.getpid()}.tmp"
            try:
                with open(tmp, "w") as f:
                    f.write(blob)
                os.replace(tmp, path)
            except OSError:
                # A full or read-only disk. Leave no half-written temp file
                # behind, and let the run carry on without a checkpoint.
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return False
        _prune_round_files(d)
    return True


def _prune_round_files(d: str) -> None:
    """Keep the newest ``MAX_ROUND_FILES`` per-round files.

    The latest file is always kept (it is the resume path); the per-round
    history is diagnostic and grows one file per round, so it is bounded.
    Pruned by numeric round order, not lexical — ``round_10.json`` must rank
    above ``round_2.json``.
    """
    rounds = []
    try:
        names = os.listdir(d)
    except OSError:
        return
    for name in names:
        if not name.startswith("round_") or not name.endswith(".json"):
            continue
        try:
            n = int(name[len("round_"):-len(".json")])
        except ValueError:
            continue
        rounds.append((n, name))
    rounds.sort(key=lambda pair: pair[0])
    for _, name in rounds[:-MAX_ROUND_FILES]:
        try:
            os.remove(os.path.join(d, name))
        except OSError:
            pass


def load_checkpoint(repo_path: str) -> Optional[dict]:
    """Read the latest checkpoint from disk, or ``None`` if none exists.

    Called from ``_build_engine`` after the aggregator is created. The
    returned dict is passed to ``aggregator.restore(state)`` if the aggregator
    supports it.

    The inter-process lock is held for the read so a concurrent writer cannot
    swap ``latest.json`` under us mid-read. Atomic rename already prevents a
    truncated file; the lock additionally prevents reading a checkpoint that
    is about to be replaced by a *different* round's write. A lock that is not
    free reads as "no checkpoint" rather than raising: a resume racing a live
    process must start fresh, not die.
    """
    # Checked before the lock, which would otherwise create the directory it
    # puts the lock file in -- a read must leave nothing behind on a path that
    # has no checkpoints, and `status` hands us whatever path it was given.
    if not os.path.isdir(os.path.join(repo_path, CHECKPOINT_DIR)):
        return None
    latest = os.path.join(repo_path, CHECKPOINT_DIR, LATEST_FILE)
    with _checkpoint_lock(repo_path) as locked:
        if not locked:
            return None
        try:
            with open(latest) as f:
                payload = json.load(f)
        except (OSError, ValueError):
            # Missing, unreadable, or not JSON. `json.JSONDecodeError` is a
            # `ValueError`; `FileNotFoundError` an `OSError`.
            return None

    if not isinstance(payload, dict):
        return None
    version = payload.get("schema_version", 0)
    if version != SCHEMA_VERSION:
        # An old checkpoint from a previous schema. Refuse rather than
        # restore incompatible state — the run starts fresh, which is safe.
        return None
    return payload


def restore_checkpoint(
    repo_path: str,
    aggregator: Any,
    *,
    expected_artifact_id: Optional[str] = None,
    expected_aggregator_type: Optional[str] = None,
) -> Optional[dict]:
    """Restore the aggregator's in-memory state from the latest checkpoint.

    Called from ``_build_engine`` after the aggregator is created and before
    the first round. Returns the checkpoint payload (a dict) if state was
    restored, ``None`` if no checkpoint exists or the aggregator does not
    support it. The payload can be passed to :func:`restore_early_stop` to
    avoid a second file read.

    ``expected_artifact_id`` / ``expected_aggregator_type`` make the restore
    refuse a checkpoint that does not belong here. The same ledger path can
    evolve different artifacts, and a changed ``aggregator_factory`` produces a
    different aggregator class — restoring either mismatch would silently seed
    one search with another's history.
    """
    restore_fn = getattr(aggregator, "restore", None)
    if not callable(restore_fn):
        return None
    payload = load_checkpoint(repo_path)
    if payload is None:
        return None
    if expected_artifact_id is not None:
        recorded = payload.get("artifact_id")
        if recorded is not None and recorded != expected_artifact_id:
            # Different artifact on the same ledger — refuse.
            return None
    if expected_aggregator_type is not None:
        recorded = payload.get("aggregator_type")
        if recorded is not None and recorded != expected_aggregator_type:
            # The aggregator implementation changed; its state shape may be
            # incompatible. Refuse rather than half-restore.
            return None
    state = payload.get("archive_state")
    if not isinstance(state, dict):
        return None
    try:
        restore_fn(state)
    except Exception:
        # A restore failure means the checkpoint is incompatible with the
        # current aggregator (e.g. the factory was changed). Start fresh
        # rather than crashing — the run still works, it just has no history.
        return None
    return payload


def restore_early_stop(
    repo_path: str,
    early_stop: Any,
    *,
    payload: Optional[dict] = None,
) -> bool:
    """Restore the run's EarlyStop tracker (best / stalled) from a checkpoint.

    Called from the drivers before the round loop starts. Without this, a
    resumed run forgets how long it had already stalled and re-burns its
    patience budget re-discovering the stall — a real cost on a long run.

    ``payload`` accepts an already-loaded checkpoint dict (from
    :func:`load_checkpoint`), so a driver that has already called
    :func:`restore_checkpoint` (which loads the same file) can pass it through
    instead of reading and parsing it a second time. ``None`` (the default)
    loads it internally.
    """
    if payload is None:
        payload = load_checkpoint(repo_path)
    if payload is None:
        return False
    early = payload.get("early_stop")
    if not isinstance(early, dict):
        return False
    best = early.get("best")
    stalled = early.get("stalled")
    if not isinstance(best, (int, float)) or not isinstance(stalled, int):
        return False
    early_stop.best = float(best)
    early_stop.stalled = int(stalled)
    return True


def clear_checkpoints(repo_path: str) -> None:
    """Remove all checkpoint files.

    Called when a run starts fresh (not a resume) to prevent restoring a
    stale checkpoint from a previous run on the same repo.

    Removes the checkpoints, not the directory holding them: the lock file
    lives in there, and pulling it out from under a concurrent waiter would
    hand two processes what each thinks is the same lock.
    """
    d = os.path.join(repo_path, CHECKPOINT_DIR)
    if not os.path.isdir(d):
        return
    with _checkpoint_lock(repo_path) as locked:
        if not locked:
            return
        try:
            names = os.listdir(d)
        except OSError:
            return
        for name in names:
            if name == LOCK_FILE:
                continue
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass


def list_checkpoints(repo_path: str) -> list:
    """List all checkpoint files with their round numbers, newest first.

    For debugging and the CLI ``status`` command. Sorted by numeric round
    order (``round_10.json`` after ``round_9.json``), with ``latest.json``
    first.
    """
    d = os.path.join(repo_path, CHECKPOINT_DIR)
    if not os.path.isdir(d):
        return []
    out = []
    with _checkpoint_lock(repo_path) as locked:
        if not locked:
            return []
        try:
            names = os.listdir(d)
        except OSError:
            return []
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            out.append({
                "file": name,
                "round": payload.get("round"),
                "aggregator_type": payload.get("aggregator_type"),
                "timestamp": payload.get("timestamp"),
            })
    # latest first, then round files by descending numeric round.
    def key(e):
        if e["file"] == LATEST_FILE:
            return (0, 0)
        r = e["round"]
        return (1, -(r if isinstance(r, int) else 0))
    return sorted(out, key=key)
