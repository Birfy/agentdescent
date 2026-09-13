"""Append-only persistence for audit records.

JSONL, one record per line, and **resolution appends too** rather than rewriting
the line it resolves. Two reasons, and both are about the run that does not
finish cleanly:

* A process killed mid-write damages at most the last line, and a reader that
  skips an unparseable line loses that one record instead of the file. Rewriting
  in place can corrupt a record that was already safely on disk.
* Truth arrives days later, often from another process. Appending needs no lock
  against the writer that is still running, and no read-modify-write race
  between two people resolving different records at the same time.

So a record id may appear several times and **the last occurrence wins**. That
is the whole reconciliation rule.

The file is the durable artifact of an audit, not a cache: it holds the outputs
an experiment still has to be run against. Deleting it discards questions that
have been asked and not yet answered.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .records import SCHEMA_VERSION, AuditRecord, Purpose

#: Marks a JSONL line as a moments snapshot rather than an audit record. Absent
#: on every record ever written, so a file from before this existed still loads.
MOMENTS_KIND = "unlabelled_moments"
#: Snapshot line holding ``artifact_signature -> priority``, drained from the
#: merge path's :class:`~agentdescent.scheduler.AuditScheduler`. Last one wins,
#: like the moments -- a priority is a current opinion, not a measurement, and
#: keeping the history of it would only invite averaging opinions.
PRIORITY_KIND = "audit_priority"

#: The verifier fingerprint last seen by a `VerifierWatch`. Out-of-band
#: like the priorities, and for the same reason: it belongs to the *run*,
#: not to any unit.
WATCH_KIND = "verifier_watch"


class _Welford:
    """Running count, mean and variance of the scores nobody audited.

    Three numbers per stratum, and they are **all** the estimator ever asks of
    the unlabelled half -- :class:`~agentdescent.audit.ppi.Stratum` reduces an
    array to exactly these and then never reads it again. So the alternative to
    this class is persisting every score a run produces, in order to compute a
    correction that needs nine numbers.

    Welford rather than a running sum of squares: the naive form subtracts two
    large nearly-equal numbers and can return a small *negative* variance, which
    propagates as a nan through the interval instead of failing where it
    happened.
    """

    __slots__ = ("n", "mean", "m2")

    def __init__(self, n: int = 0, mean: float = 0.0, m2: float = 0.0) -> None:
        self.n, self.mean, self.m2 = int(n), float(mean), float(m2)

    def observe(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def var(self) -> float:
        """Sample variance (``ddof=1``); 0.0 below two observations."""
        return self.m2 / (self.n - 1) if self.n >= 2 else 0.0

    def to_dict(self):
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d):
        return cls(d["n"], d["mean"], d["m2"])


class AuditStore:
    """Records on disk, indexed in memory.

    ``path=None`` keeps everything in memory, which is what tests and a dry run
    want. Nothing else changes: the same reconciliation and the same assertions
    apply, so a test that passes in memory is testing the code that runs on disk.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: Dict[str, AuditRecord] = {}
        self._order: List[str] = []
        self._corrupt = 0
        #: (verifier_version, stratum) -> running moments of the unlabelled half.
        self._moments: Dict[Tuple[str, str], _Welford] = {}
        #: artifact_signature -> how much the merge path wanted this audited.
        self._priorities: Dict[str, float] = {}
        self._fingerprint: Optional[str] = None
        self._unflushed = 0
        self._lock = threading.RLock()
        if path and os.path.exists(path):
            self.load()

    # -- reading -------------------------------------------------------------

    def load(self) -> None:
        """Re-read the file, last-occurrence-wins.

        A line that will not parse as JSON is **skipped** and counted in
        :attr:`corrupt`: that is a torn write from a killed process, and the
        records before it are fine. A line that parses but carries an unknown
        ``schema_version`` **raises**: that is a different build's file, the
        fields do not mean what this code thinks they mean, and continuing would
        produce numbers rather than an error.
        """
        if not self.path:
            return
        with self._lock:
            self._records.clear()
            self._order.clear()
            self._moments.clear()
            self._priorities.clear()
            self._corrupt = 0
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        self._corrupt += 1
                        continue
                    if payload.get("kind") == PRIORITY_KIND:
                        self._priorities = dict(payload["priorities"])
                        continue
                    if payload.get("kind") == WATCH_KIND:
                        self._fingerprint = payload.get("fingerprint")
                        continue
                    if payload.get("kind") == MOMENTS_KIND:
                        # Last snapshot wins, exactly as for records: a later
                        # line supersedes an earlier one for the same key.
                        self._moments[(payload["verifier_version"],
                                       payload["stratum"])] = _Welford.from_dict(
                                           payload["moments"])
                        continue
                    rec = AuditRecord.from_dict(payload)   # raises on schema drift
                    self._remember(rec)

    @property
    def corrupt(self) -> int:
        """Lines skipped as unparseable on the last :meth:`load`."""
        return self._corrupt

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def __iter__(self) -> Iterator[AuditRecord]:
        return iter(self.all())

    def all(self) -> List[AuditRecord]:
        """Every record, in the order first seen."""
        with self._lock:
            return [self._records[r] for r in self._order]

    def get(self, record_id: str) -> Optional[AuditRecord]:
        with self._lock:
            return self._records.get(record_id)

    # -- writing -------------------------------------------------------------

    def append(self, record: AuditRecord) -> AuditRecord:
        with self._lock:
            self._remember(record)
            self._write(record)
            self._flush_moments()
        return record

    # -- the unlabelled half -------------------------------------------------

    #: Unlabelled observations between snapshots. A crash loses at most this
    #: many, which moves a stratum mean by ~1e-4 on a run of any size. The
    #: alternative is a line per scored unit, which is the file this avoids.
    FLUSH_EVERY = 64

    def observe_unlabelled(self, verifier_version: str, stratum: str,
                           score: float) -> None:
        """Fold one un-audited score into its stratum's running moments.

        Called for every unit the sampler passed over, which is nearly all of
        them, so it has to stay O(1) in both time and space. It does: three
        floats per stratum, whatever the run's length.
        """
        with self._lock:
            key = (verifier_version, stratum)
            acc = self._moments.get(key)
            if acc is None:
                acc = self._moments[key] = _Welford()
            acc.observe(float(score))
            self._unflushed += 1
            if self._unflushed >= self.FLUSH_EVERY:
                self._flush_moments()

    def flush(self) -> None:
        """Persist the moments now. Call it when a run ends."""
        with self._lock:
            self._flush_moments(force=True)

    def unlabelled_moments(self, verifier_version: str):
        """``stratum -> {n, mean, var}`` over the units that were not audited."""
        with self._lock:
            return {stratum: {"n": acc.n, "mean": acc.mean, "var": acc.var}
                    for (version, stratum), acc in self._moments.items()
                    if version == verifier_version}

    @property
    def last_fingerprint(self) -> Optional[str]:
        """The verifier fingerprint a `VerifierWatch` last saw, or ``None``."""
        with self._lock:
            return self._fingerprint

    def remember_fingerprint(self, fingerprint: str) -> None:
        """Persist the fingerprint so a restart can be compared against it.

        A `VerifierWatch` holds its baseline in memory, which was enough while a
        restart meant a new run. `evolve(checkpointing=True)` makes a run span
        one, so "did the verifier change while we were down?" became a question
        the watch could not answer: `check()` on a fresh object has nothing to
        compare against and silently re-baselines.

        The correction was never at risk -- records are keyed by
        `verifier_version`, so the old labels are not applied to the new judge
        and the gate widens on its own. What was lost is the *diagnosis*: it
        presents as "not enough labels yet", which is the confusion
        `VerifierWatch`'s own docstring says a stale rectification causes.
        """
        with self._lock:
            if fingerprint == self._fingerprint:
                return                      # nothing changed; do not grow the file
            self._fingerprint = fingerprint
            self._write_line({"kind": WATCH_KIND, "fingerprint": fingerprint})

    def remember_priorities(self, priorities: Dict[str, float]) -> None:
        """Record what the merge path thought was worth auditing.

        Written as a snapshot rather than merged into the records, because a
        priority belongs to an *artifact* and a record belongs to a unit: one
        number would otherwise be copied onto every unit that artifact produced
        and go stale on the next merge. Persisting it at all is what lets
        :func:`~agentdescent.audit.service.audit_pending` order a queue for a
        person in another process next week -- which is the whole reason the
        merge path's ranking exists and has never been read.
        """
        with self._lock:
            self._priorities.update(
                {str(k): float(v) for k, v in priorities.items()})
            self._write_line({"kind": PRIORITY_KIND,
                              "priorities": dict(self._priorities)})

    @property
    def priorities(self) -> Dict[str, float]:
        """``artifact_signature -> priority``, as last recorded."""
        with self._lock:
            return dict(self._priorities)

    def _flush_moments(self, force: bool = False) -> None:
        if not self.path or (not force and not self._unflushed):
            self._unflushed = 0
            return
        for (version, stratum), acc in self._moments.items():
            self._write_line({"kind": MOMENTS_KIND, "verifier_version": version,
                              "stratum": stratum, "moments": acc.to_dict()})
        self._unflushed = 0

    def resolve(self, record_id: str, oracle_score: float,
                *, at: Optional[float] = None) -> bool:
        """Attach ground truth to a pending record. ``False`` if there was none to attach.

        Refuses to overwrite a resolution. An oracle result is a measurement
        somebody made; silently replacing it with a second one would make the
        calibration set depend on how many times a batch file was replayed, and
        replaying a batch file is exactly what people do when a job half-failed.
        Call :meth:`reopen` first if a result really has to be corrected.
        """
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None:
                return False
            if rec.resolved:
                raise ValueError(
                    f"{record_id} already resolved with oracle_score="
                    f"{rec.oracle_score!r}; refusing to overwrite. Use reopen().")
            updated = _replace(rec, oracle_score=float(oracle_score),
                               resolved_at=time.time() if at is None else at)
            self._remember(updated)
            self._write(updated)
            return True

    def reopen(self, record_id: str) -> bool:
        """Clear a resolution so it can be replaced. For corrections, not for retries."""
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None or not rec.resolved:
                return False
            updated = _replace(rec, oracle_score=None, resolved_at=None)
            self._remember(updated)
            self._write(updated)
            return True

    # -- queries -------------------------------------------------------------

    def pending(self, *, older_than: Optional[float] = None,
                verifier_version: Optional[str] = None) -> List[AuditRecord]:
        """Records still waiting on truth -- the work list for whoever answers.

        ``older_than`` is an absolute timestamp, not an age: pass
        ``time.time() - 86400`` for "dispatched over a day ago".
        """
        out = [r for r in self.all() if not r.resolved]
        if older_than is not None:
            out = [r for r in out if r.dispatched_at < older_than]
        if verifier_version is not None:
            out = [r for r in out if r.verifier_version == verifier_version]
        return out

    def for_calibration(self, verifier_version: str) -> List[AuditRecord]:
        """Resolved CALIBRATION records for one verifier version, and nothing else.

        The assertion is the point of the method. Filtering by
        ``purpose == CALIBRATION`` is one careless edit away from being widened
        to "everything we have labels for", the widened version computes happily,
        and what it computes is a bias estimated partly on the units somebody
        tuned the verifier to agree with. That number is biased *towards zero* --
        it reports the verifier as more honest than it is, which is the
        direction that does no visible damage until the loop has been accepting
        on it for weeks.

        Filtering by ``verifier_version`` carries the same weight for the same
        reason: mixing versions averages the bias of an instrument that exists
        with the bias of one that does not.
        """
        out = [r for r in self.all()
               if r.purpose is Purpose.CALIBRATION
               and r.resolved
               and r.verifier_version == verifier_version]
        assert all(r.purpose is Purpose.CALIBRATION for r in out), (
            "for_calibration returned an IMPROVEMENT record: labels used to edit "
            "the verifier must never enter its calibration set")
        assert all(r.verifier_version == verifier_version for r in out), (
            "for_calibration mixed verifier versions")
        assert all(r.oracle_score is not None for r in out), (
            "for_calibration returned an unresolved record")
        return out

    def for_improvement(self, verifier_version: Optional[str] = None) -> List[AuditRecord]:
        """Resolved IMPROVEMENT records -- the pool you are allowed to look at."""
        out = [r for r in self.all()
               if r.purpose is Purpose.IMPROVEMENT and r.resolved]
        if verifier_version is not None:
            out = [r for r in out if r.verifier_version == verifier_version]
        return out

    def versions(self) -> List[str]:
        """Every ``verifier_version`` seen, in order of first appearance."""
        seen: List[str] = []
        for r in self.all():
            if r.verifier_version not in seen:
                seen.append(r.verifier_version)
        return seen

    # -- internals -----------------------------------------------------------

    def _remember(self, record: AuditRecord) -> None:
        if record.record_id not in self._records:
            self._order.append(record.record_id)
        self._records[record.record_id] = record

    def _write(self, record: AuditRecord) -> None:
        self._write_line(record.to_dict())

    def _write_line(self, payload: Dict[str, Any]) -> None:
        if not self.path:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _replace(rec: AuditRecord, **changes) -> AuditRecord:
    """`dataclasses.replace` without importing it for one call, plus enum repair."""
    payload = rec.to_dict()
    payload.update(changes)
    payload["purpose"] = Purpose(payload["purpose"]) if isinstance(
        payload["purpose"], str) else payload["purpose"]
    payload["schema_version"] = SCHEMA_VERSION
    return AuditRecord(**payload)


def summarise(records: Iterable[AuditRecord]) -> Dict[str, float]:
    """Counts and the raw mean residual. **Not** an estimate of the bias.

    The mean of ``f - Y`` over audited units estimates the bias of the *audited*
    units, which equals the bias of the population only when every unit had the
    same inclusion probability. Stratified or score-dependent sampling breaks
    that, and the correct estimator weights by ``1 / inclusion_prob`` and carries
    an interval. This function is for eyeballing a run, and it is named for what
    it does rather than for what it looks like so that nobody wires it into a
    gate by mistake.
    """
    # Materialise first: `records` may be a generator, and the old version of
    # this counted it *after* the comprehension had drained it, so `n_records`
    # came back 0 for every caller who passed one.
    records = list(records)
    residuals = [r.residual for r in records if r.resolved]
    n = len(residuals)
    return {
        "n_records": len(records),
        "n_resolved": n,
        "mean_residual": (sum(residuals) / n) if n else float("nan"),
    }
