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
from typing import Dict, Iterable, Iterator, List, Optional

from .records import SCHEMA_VERSION, AuditRecord, Purpose


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
        return record

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
        if not self.path:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


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
