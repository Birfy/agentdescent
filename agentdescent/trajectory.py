"""Versioned rollout trajectories -- what actually happened, kept as data.

The engine's *decision* records already exist: the ledger keeps every accepted
artifact, the meter keeps every counter, the audit keeps every probability
sample. What none of them keep is the **rollout itself** -- the rendered
artifact text that was shown to the run provider, the task it was applied to,
the output it produced, the reward that output earned, and the environment and
artifact version the whole thing happened against. An ``EvidenceCard`` carries
``trajectory_refs`` but they are Task *objects* -- pointers into the caller's
own dataset, not a record anyone can re-read after the run.

That gap is what this module fills. A :class:`Trajectory` is one
``(artifact, task, environment)`` evaluation as immutable data, and a
:class:`TrajectoryStore` keeps them durably, keyed by identity so the same
evaluation is written once however many times the run re-computes it.

Why a store and not a column on the ledger? Three properties, each of which the
ledger would break:

* **Derived data.** A trajectory is what one rollout produced, not what any
  artifact is. Committing them into the git history would make every commit's
  diff balloon with transcripts and would *replay* the run's outputs onto every
  branch checkout. The store lives beside the ledger's repo (under
  ``<repo>/trajectories/``) and is excluded from the index, exactly as
  checkpoints are -- it survives, it accumulates, and no ``git add -A`` ever
  sees it.
* **Content addressing.** The identity of an evaluation is its inputs
  ``(rendered, task_id, env_fingerprint)`` -- the same triple the evaluation
  cache keys on. Two states that render identically cannot score differently,
  so two trajectories with the same identity are the same evaluation, however
  many processes computed it. Files are named by that identity, so dedupe falls
  out of the filesystem: write once, and the next writer has nothing to do.
* **No server.** Two processes sharing one run (a supervisor and a worker, a
  resume) need to share trajectories, and a directory of files is the protocol
  two processes can both honour without anyone to broker it -- the same
  reasoning as ``FileCache``.

**Privacy is a knob, not an accident.** The store records the boundary of a
rollout -- what the agent saw, what it said, what that earned -- because those
are the fields process supervision and offline learning need. The boundary
(``rendered``, ``output``, ``reward``) is **always written when recording is
on**; ``AGENTDESCENT_RECORD_TEXT`` (or ``record_text=True``) only adds event
*detail* text -- a model's internal reasoning, a tool payload -- which is
metadata otherwise, never a transcript.

**A failed write must never fail a rollout.** The store is diagnostics on the
hot path. A full disk, a read-only ``repo_path`` or a hostile directory
degrades the recording, not the run: every write is caught, counted in the
meter, and re-raised to nobody.

Consumers today: the worker loops attach ``trace_id``s to the
``EvidenceCard``s they produce (``evolvable.EvidenceCard.trace_ids``), so a
diff carries pointers to the rollouts that justify it. The trajectory-boundary
consumers -- process-supervised reflection, offline re-weighting, audit
replay -- arrive with the increments that need them; this module is the data
they will read.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

__all__ = [
    "PURPOSE_EVAL", "PURPOSE_PROBE", "PURPOSE_ROLLOUT", "PURPOSE_SELF_VERIFY",
    "PURPOSES", "RECORD_TEXT_ENV", "Trajectory", "TrajectoryEvent",
    "TrajectoryStore", "trajectory_key",
]

#: Env var that switches the store to recording event *detail* text. Off by
#: default: provenance records are metadata, not transcripts. Mirrors
#: ``examples/_measure.py``'s ``AGENTDESCENT_RECORD_TEXT`` so the two
#: instrumentation layers agree on what "recording text" means.
RECORD_TEXT_ENV = "AGENTDESCENT_RECORD_TEXT"

#: What kind of work a trajectory records. Consumers filter on these; the
#: engine's call sites pick one and the gate's own evaluations are marked
#: ``eval`` so they can be told apart from worker rollouts even though both run
#: the same ``run`` provider.
PURPOSE_ROLLOUT = "rollout"        # a worker exploring: produced a proposal
PURPOSE_SELF_VERIFY = "self_verify"  # a worker re-running with a diff applied
PURPOSE_EVAL = "eval"              # a gate / held-out measurement
PURPOSE_PROBE = "probe"            # reserved: a probe task in a self-exam loop
PURPOSES = (PURPOSE_ROLLOUT, PURPOSE_SELF_VERIFY, PURPOSE_EVAL, PURPOSE_PROBE)


@dataclass(frozen=True)
class TrajectoryEvent:
    """One structured step inside a rollout.

    The boundary record (what the run provider was shown, what it returned,
    what that earned) lives on the :class:`Trajectory` itself. Events are for
    the *inside* -- the model calls, tool calls and intermediate steps a
    rollout consists of. Nothing in the engine emits them yet; the increments
    that add inside-rollout instrumentation (process-supervised reflection,
    offline re-weighting) will. The schema is here so a consumer can already
    filter and count on ``kind`` without waiting for a producer.

    ``detail`` is the free-text part and the one the privacy knob guards: the
    store drops it unless text recording is on.
    """

    kind: str                 # "model_call" | "tool_call" | "step" | "text"
    source: str               # "run" | "propose" | "solve" | the tool name
    seconds: float = 0.0
    cost_tokens: int = 0
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "seconds": self.seconds,
            "cost_tokens": self.cost_tokens,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrajectoryEvent":
        return cls(
            kind=d["kind"], source=d["source"],
            seconds=float(d.get("seconds", 0.0)),
            cost_tokens=int(d.get("cost_tokens", 0)),
            detail=d.get("detail"),
        )


@dataclass(frozen=True)
class Trajectory:
    """One ``(artifact, task, environment)`` evaluation, as data.

    ``rendered`` is exactly what the run provider saw -- the artifact's
    ``render()``, which is also the evaluation cache's signature, so identity
    and content agree by construction. ``output`` is what the provider
    returned, and ``reward`` the score that output earned.

    ``trace_id`` is filled in by the store on write and is stable across
    processes: ``"<artifact_id>:<sha256(identity)>"``. It is what an
    ``EvidenceCard.trace_ids`` entry means when one is present.
    """

    artifact_id: str
    task_id: str
    env_fingerprint: str
    rendered: str
    purpose: str = PURPOSE_ROLLOUT
    #: JSON-encoded version vector the rollout ran against -- attribution, not
    #: identity: the same rendered text under a different version is the same
    #: evaluation (the cache agrees), so the version is metadata.
    base_version: str = ""
    output: Optional[str] = None
    reward: float = 0.0
    cost_tokens: int = 0
    cost_wallclock: float = 0.0
    lease_id: Optional[str] = None
    worker_id: Optional[str] = None
    events: Tuple[TrajectoryEvent, ...] = ()
    #: Set by the store on write; ``0.0`` before it. Kept on the object so a
    #: caller can see the write time without reading it back.
    ts: float = 0.0
    #: Set when the store capped ``rendered`` (see
    #: :attr:`TrajectoryStore.max_rendered_chars`); a consumer that needs the
    #: full prompt can refuse rather than trust a truncated one.
    rendered_truncated: bool = False
    #: Filled by the store on write; the identity hash. Empty before then.
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "env_fingerprint": self.env_fingerprint,
            "rendered": self.rendered,
            "rendered_truncated": self.rendered_truncated,
            "purpose": self.purpose,
            "base_version": self.base_version,
            "output": self.output,
            "reward": self.reward,
            "cost_tokens": self.cost_tokens,
            "cost_wallclock": self.cost_wallclock,
            "lease_id": self.lease_id,
            "worker_id": self.worker_id,
            "events": [e.to_dict() for e in self.events],
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Trajectory":
        return cls(
            artifact_id=d["artifact_id"],
            task_id=d["task_id"],
            env_fingerprint=d.get("env_fingerprint", ""),
            rendered=d["rendered"],
            purpose=d.get("purpose", PURPOSE_ROLLOUT),
            base_version=d.get("base_version", ""),
            output=d.get("output"),
            reward=float(d.get("reward", 0.0)),
            cost_tokens=int(d.get("cost_tokens", 0)),
            cost_wallclock=float(d.get("cost_wallclock", 0.0)),
            lease_id=d.get("lease_id"),
            worker_id=d.get("worker_id"),
            events=tuple(TrajectoryEvent.from_dict(e) for e in d.get("events", ())),
            ts=float(d.get("ts", 0.0)),
            rendered_truncated=bool(d.get("rendered_truncated", False)),
            trace_id=d.get("trace_id", ""),
        )


def trajectory_key(rendered: str, task_id: str, env_fingerprint: str = "") -> str:
    """The identity of one evaluation: sha256 of the same triple the cache keys.

    ``cache_key`` returns the triple itself because the cache needs the parts
    (it must be able to compare them); a trajectory needs only a filename. The
    hash is over the same three inputs, in the same order, so the two layers
    cannot disagree about what one evaluation *is*.

    Using the full ``rendered`` text means two artifacts that render
    identically share a trajectory even when their internal states differ --
    the exact property that makes the evaluation cache correct, and that makes
    re-running a rollout idempotent.
    """
    digest = hashlib.sha256()
    for part in (rendered, task_id, env_fingerprint):
        digest.update(part.encode("utf-8", "replace"))
        digest.update(b"\x00")           # a separator: no two joined parts collide
    return digest.hexdigest()


def _truthy(value: Optional[bool], default: bool) -> bool:
    return default if value is None else bool(value)


class TrajectoryStore:
    """Durable, content-addressed, thread-safe rollout records.

    One JSON file per trajectory under ``<root>/<key>.json``, in a **flat**
    namespace keyed only by the identity hash -- the artifact id is metadata on
    the record, never part of the path (that is also what keeps a hostile id
    from escaping the root). Writing is: check the file, write a
    uniquely-named sibling, ``os.replace`` over the target -- a reader never
    sees half a file. The exists-check + write pair runs under a per-process
    lock, so within this process the accounting is exact: N racing writers of
    one identity count 1 recorded and N-1 deduped. Across processes the pair is
    still racy (there is no inter-process lock, the same reasoning as
    ``FileCache``), and two processes racing the same identity produce
    byte-identical files, so the outcome stays correct whatever the counters
    say. Everything that can fail with a filesystem is caught and counted
    rather than raised.

    **Privacy is a knob, not an accident.** ``record_text=False`` (the default,
    and the env var's default) drops event ``detail`` text; the boundary fields
    -- rendered, output, reward -- are the record, not the transcript, and are
    **always written when recording is on** (``AGENTDESCENT_RECORD_TEXT`` only
    ever gates event detail, never the boundary). Set it (or
    ``AGENTDESCENT_RECORD_TEXT=1``) to also keep event detail.
    """

    def __init__(
        self,
        root: str,
        *,
        meter: Optional[Any] = None,
        record_text: Optional[bool] = None,
        max_events: int = 2000,
        max_detail_chars: int = 2000,
        max_rendered_chars: Optional[int] = None,
    ) -> None:
        #: Where records land. The engine passes ``<repo_path>/trajectories``,
        #: which the ledger excludes from its index; a caller may use any
        #: directory it owns.
        self.root = root
        self.meter = meter
        self._record_text = _truthy(
            record_text, os.environ.get(RECORD_TEXT_ENV) == "1")
        self._max_events = max(1, int(max_events))
        self._max_detail_chars = max(0, int(max_detail_chars))
        self._max_rendered_chars = (None if max_rendered_chars is None
                                    else max(1, int(max_rendered_chars)))
        self._lock = threading.RLock()

    # -- identity ------------------------------------------------------------

    @property
    def record_text(self) -> bool:
        """Whether event detail text is kept. Read at write time."""
        return self._record_text

    def _path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    # -- writing -------------------------------------------------------------

    def record(self, trajectory: Trajectory) -> Optional[str]:
        """Persist one trajectory; return its ``trace_id``, or ``None`` if the
        identity was already recorded.

        ``None`` is *dedupe*, not failure: the same ``(rendered, task, env)``
        evaluation already exists on disk, so there is nothing to write. A
        write failure is also ``None`` and is counted -- a full disk degrades
        the recording, not the run (the caller records the trace_id only when
        it is returned, so nothing downstream depends on the file having
        landed).

        Thread-safe, and the exists-check + write pair runs under
        ``self._lock``, so two threads in this process racing the same identity
        cannot both pass the check: one writes, the rest dedupe, and the
        counters say exactly that. The lock is only per-process; two *processes*
        racing the same identity can still both write (identical bytes, so the
        outcome is correct), and the counters for that cross-process race may
        over-count recorded -- the trade-off of having no inter-process broker.
        """
        with self._lock:
            key = trajectory_key(
                trajectory.rendered, trajectory.task_id,
                trajectory.env_fingerprint)
            path = self._path(key)
            if os.path.exists(path):
                self._count("trajectories_dedup")
                return None
            payload = self._prepare(trajectory, key)
            t0 = time.time()
            tmp: Optional[str] = None
            try:
                # A sibling named per-thread: two threads in one process would
                # otherwise write the same tmp file and a torn rename could land
                # a partial record. Unique tmp keeps every write whole.
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload.to_dict(), fh, sort_keys=True,
                              default=str, allow_nan=False)
                os.replace(tmp, path)
                self._count("trajectories_recorded")
                self._count("trajectory_write_seconds", time.time() - t0)
                return payload.trace_id
            except OSError:
                self._count("trajectory_write_failures")
                return None
            finally:
                # A failed rename (target directory vanished, cross-device move)
                # leaves the tmp behind; clean it, but never at the cost of the
                # record's own error path.
                if tmp is not None:
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass

    def _prepare(self, trajectory: Trajectory, key: str) -> Trajectory:
        """Enforce the store's caps and privacy before writing.

        ``rendered`` is capped only when the caller asked for a cap; the
        engine does not, because a truncated prompt is a corrupted record. The
        truncation is flagged on the trajectory so a consumer can refuse.
        Events are capped in count, and event ``detail`` text is dropped unless
        text recording is on.
        """
        rendered = trajectory.rendered
        truncated = False
        if self._max_rendered_chars is not None and len(rendered) > self._max_rendered_chars:
            rendered = rendered[: self._max_rendered_chars]
            truncated = True
        events = list(trajectory.events[: self._max_events])
        if not self._record_text:
            events = [
                TrajectoryEvent(
                    kind=e.kind, source=e.source, seconds=e.seconds,
                    cost_tokens=e.cost_tokens, detail=None)
                for e in events
            ]
        elif self._max_detail_chars >= 0:
            events = [
                TrajectoryEvent(
                    kind=e.kind, source=e.source, seconds=e.seconds,
                    cost_tokens=e.cost_tokens,
                    detail=(e.detail[: self._max_detail_chars]
                            if e.detail is not None else None))
                for e in events
            ]
        return Trajectory(
            artifact_id=trajectory.artifact_id,
            task_id=trajectory.task_id,
            env_fingerprint=trajectory.env_fingerprint,
            rendered=rendered,
            purpose=trajectory.purpose,
            base_version=trajectory.base_version,
            output=trajectory.output,
            reward=trajectory.reward,
            cost_tokens=trajectory.cost_tokens,
            cost_wallclock=trajectory.cost_wallclock,
            lease_id=trajectory.lease_id,
            worker_id=trajectory.worker_id,
            events=tuple(events),
            ts=trajectory.ts or time.time(),
            rendered_truncated=truncated,
            trace_id=f"{trajectory.artifact_id}:{key}",
        )

    # -- reading -------------------------------------------------------------

    def get(self, trace_id: str) -> Optional[Trajectory]:
        """Read one trajectory by its ``trace_id`` (``"<artifact_id>:<key>"``).

        ``None`` when no such record exists. Unknown artifact ids and corrupt
        files both read as ``None`` -- a corrupt record is indistinguishable
        from an absent one to anyone but the disk, and refusing loudly would
        make a bad file take a run down with it. Use :meth:`iter_records` if
        you need to see which files are unreadable.
        """
        if ":" not in trace_id:
            return None
        artifact_id, key = trace_id.split(":", 1)
        if not (key and key.isalnum()):
            return None
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as fh:
                return Trajectory.from_dict(json.load(fh))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def iter_records(
        self,
        *,
        artifact_id: Optional[str] = None,
        task_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> Iterator[Trajectory]:
        """Iterate every trajectory, optionally filtered.

        Records are keyed by identity in one flat namespace, so the
        ``artifact_id`` filter is applied after reading -- the file name holds
        the identity, not the artifact, because two artifacts rendering
        identically are one evaluation (the same property the eval cache
        leans on). Filters skip a bad file rather than raising -- this is
        diagnostics, and one corrupt record must not hide the rest.
        ``purpose`` filters on the purpose column; pass one of
        :data:`PURPOSES`.
        """
        try:
            names = sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name), encoding="utf-8") as fh:
                    record = Trajectory.from_dict(json.load(fh))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if artifact_id is not None and record.artifact_id != artifact_id:
                continue
            if task_id is not None and record.task_id != task_id:
                continue
            if purpose is not None and record.purpose != purpose:
                continue
            yield record

    def count(self) -> int:
        """Total records on disk. Best-effort."""
        try:
            if not os.path.isdir(self.root):
                return 0
            return sum(1 for f in os.listdir(self.root) if f.endswith(".json"))
        except OSError:
            return 0

    def path_for(self, trace_id: str) -> Optional[str]:
        """The on-disk path for a ``trace_id``, without reading it.

        ``None`` when the id cannot name a file. Useful for a consumer that
        wants to hand the raw record to something else (an export tool, a
        training pipeline) without round-tripping it through Python.
        """
        if ":" not in trace_id:
            return None
        artifact_id, key = trace_id.split(":", 1)
        if not (key and key.isalnum()):
            return None
        return self._path(key)

    # -- bookkeeping ---------------------------------------------------------

    def _count(self, counter: str, value: float = 1) -> None:
        if self.meter is not None:
            self.meter.add(counter, value)
