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
from typing import Any, Dict, Optional


#: The subdirectory under a ledger's ``repo_path`` where checkpoints live.
CHECKPOINT_DIR = "checkpoints"

#: The filename for the latest checkpoint. Old per-round files are kept for
#: debugging but the resume path reads only this one.
LATEST_FILE = "latest.json"

#: Current checkpoint schema version. Bumped when the dict shape changes;
#: ``_load`` refuses a mismatched version rather than restoring incompatible state.
SCHEMA_VERSION = 1


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
) -> bool:
    """Serialise the aggregator's in-memory state to disk.

    Called from the engine's ``on_round`` hook after every round (or every
    ``checkpoint_interval`` rounds). Returns ``True`` if a checkpoint was
    written, ``False`` if the aggregator does not support checkpointing.

    ``early_stop`` is the run's :class:`~agentdescent.pipeline.EarlyStop`
    tracker. Its ``best``/``stalled`` state is saved alongside the aggregator:
    without it, a resumed run re-burns its patience budget re-discovering the
    stall the previous process had already counted. ``None`` skips it (some
    drivers construct the tracker after the first checkpoint fires).

    Two files are written: ``round_N.json`` (for debugging / inspection) and
    ``latest.json`` (for the resume path). Both use atomic-rename to avoid a
    truncated read by a concurrent process.
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
        "aggregator_type": type(aggregator).__name__,
        "archive_state": state,
        "round_info": round_info,
        "early_stop": early,
        "timestamp": time.time(),
    }

    d = _checkpoint_dir(repo_path)
    # Write per-round file for inspection, then latest for the resume path.
    round_path = os.path.join(d, f"round_{round}.json")
    latest_path = os.path.join(d, LATEST_FILE)
    for path in (round_path, latest_path):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    return True


def load_checkpoint(repo_path: str) -> Optional[dict]:
    """Read the latest checkpoint from disk, or ``None`` if none exists.

    Called from ``_build_engine`` after the aggregator is created. The
    returned dict is passed to ``aggregator.restore(state)`` if the aggregator
    supports it.
    """
    latest = os.path.join(repo_path, CHECKPOINT_DIR, LATEST_FILE)
    try:
        with open(latest) as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    version = payload.get("schema_version", 0)
    if version != SCHEMA_VERSION:
        # An old checkpoint from a previous schema. Refuse rather than
        # restore incompatible state — the run starts fresh, which is safe.
        return None
    return payload


def restore_checkpoint(repo_path: str, aggregator: Any) -> bool:
    """Restore the aggregator's in-memory state from the latest checkpoint.

    Called from ``_build_engine`` after the aggregator is created and before
    the first round. Returns ``True`` if state was restored, ``False`` if no
    checkpoint exists or the aggregator does not support it.
    """
    restore_fn = getattr(aggregator, "restore", None)
    if not callable(restore_fn):
        return False
    payload = load_checkpoint(repo_path)
    if payload is None:
        return False
    state = payload.get("archive_state")
    if not isinstance(state, dict):
        return False
    try:
        restore_fn(state)
    except Exception:
        # A restore failure means the checkpoint is incompatible with the
        # current aggregator (e.g. the factory was changed). Start fresh
        # rather than crashing — the run still works, it just has no history.
        return False
    return True


def restore_early_stop(repo_path: str, early_stop: Any) -> bool:
    """Restore the run's EarlyStop tracker (best / stalled) from a checkpoint.

    Called from the drivers before the round loop starts. Without this, a
    resumed run forgets how long it had already stalled and re-burns its
    patience budget re-discovering the stall — a real cost on a long run.
    Returns ``True`` if the tracker was restored.
    """
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
    """
    d = os.path.join(repo_path, CHECKPOINT_DIR)
    if not os.path.isdir(d):
        return
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def list_checkpoints(repo_path: str) -> list:
    """List all checkpoint files with their round numbers, newest first.

    For debugging and the CLI ``status`` command.
    """
    d = os.path.join(repo_path, CHECKPOINT_DIR)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path) as f:
                payload = json.load(f)
            out.append({
                "file": name,
                "round": payload.get("round"),
                "aggregator_type": payload.get("aggregator_type"),
                "timestamp": payload.get("timestamp"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return out
