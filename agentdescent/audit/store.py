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
import math
import os
import threading
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .records import (SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, AuditRecord,
                      Purpose)

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
        #: (verifier_version, stratum) -> units the tap dropped to keep a task
        #: out of both halves. A count and nothing else: these units have no
        #: recorded score, so they belong in the population frame and nowhere
        #: near the moments PPI borrows from. See `observe_skipped`.
        self._skipped: Dict[Tuple[str, str], int] = {}
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
        records before it are fine. The same goes for a line torn inside a
        multi-byte character (the file is read as bytes and decoded per line, so
        one bad line cannot make the whole store unopenable), a line that parses
        to something other than an object, a side-record missing the keys it
        needs, and a current-schema record whose fields will not construct one.
        A line holding a non-finite number -- the ``NaN`` / ``Infinity`` tokens
        an older build wrote, or a literal like ``1e999`` that overflows -- is
        skipped and counted the same way: folded in, one ``nan`` score would
        poison every residual, moment and interval computed from the store.
        A line that parses but carries an unknown ``schema_version`` **raises**:
        that is a different build's file, the fields do not mean what this code
        thinks they mean, and continuing would produce numbers rather than an
        error. Version 1 lines (pre-``scorer_version``) are migrated on read;
        version 2 is current (#315).
        """
        if not self.path:
            return
        with self._lock:
            self._records.clear()
            self._order.clear()
            self._moments.clear()
            self._skipped.clear()
            self._priorities.clear()
            self._corrupt = 0
            with open(self.path, "rb") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8"),
                                             parse_constant=_refuse_constant,
                                             parse_float=_finite_float)
                    except (UnicodeDecodeError, ValueError):
                        # ValueError covers JSONDecodeError (a torn write) and
                        # a non-finite number in a line an older build wrote.
                        self._corrupt += 1
                        continue
                    if not isinstance(payload, dict):
                        self._corrupt += 1
                        continue
                    if not self._load_line(payload):
                        self._corrupt += 1

    def _load_line(self, payload: Dict[str, Any]) -> bool:
        """Fold one parsed line into memory. ``False`` if it is malformed.

        Raises only on an unknown ``schema_version`` -- see :meth:`load`.
        """
        kind = payload.get("kind")
        try:
            if kind == PRIORITY_KIND:
                self._priorities = {str(k): float(v) for k, v
                                    in dict(payload["priorities"]).items()}
                return True
            if kind == WATCH_KIND:
                self._fingerprint = payload.get("fingerprint")
                return True
            if kind == MOMENTS_KIND:
                # Last snapshot wins, exactly as for records: a later
                # line supersedes an earlier one for the same key.
                key = (payload["verifier_version"], payload["stratum"])
                moments = _Welford.from_dict(payload["moments"])
                # Absent in stores written before the tap could skip.
                skipped = int(payload.get("skipped", 0))
                self._moments[key] = moments
                self._skipped[key] = skipped
                return True
        except (KeyError, TypeError, ValueError, AttributeError):
            return False
        if payload.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            AuditRecord.from_dict(payload)          # raises on schema drift
        # Every supported version (v1 is migrated) goes through the guarded
        # construction below, so a malformed v1 line is one corrupt line rather
        # than an unopenable store.
        try:
            rec = AuditRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return False
        self._remember(rec)
        return True

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
            # Disk first, memory second: a write that raises (a full disk, a
            # read-only file, a non-finite score `allow_nan=False` refuses)
            # must leave the index agreeing with what a fresh load would see.
            self._write(record)
            self._remember(record)
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

        A non-finite score raises ``ValueError``: folded in, it would make the
        stratum's mean and variance ``nan`` for every later read.
        """
        score = _finite_score(score, "score")
        with self._lock:
            key = (verifier_version, stratum)
            acc = self._moments.get(key)
            if acc is None:
                acc = self._moments[key] = _Welford()
            acc.observe(score)
            self._unflushed += 1
            if self._unflushed >= self.FLUSH_EVERY:
                self._flush_moments()

    def observe_skipped(self, verifier_version: str, stratum: str) -> None:
        """Count one unit the tap dropped, without its score.

        A dropped unit is one whose task is audited in *another* stratum, so
        recording its score would put that task in both halves -- which is the
        thing being prevented. But the unit still happened, and the stratum
        weights are a **count of every unit the run scored**, not an estimate
        from a sample: drop it from the frame as well and a 50/50 population
        reads as 83/17 at rates 0.9 and 0.1, which moves the correction rather
        than merely widening it.

        So: in the frame, never in the moments. The mean and variance PPI
        borrows are over units whose scores were actually observed, and stay
        unbiased for the stratum because whether a unit is skipped depends only
        on its task's draw, which is independent of its score.
        """
        with self._lock:
            key = (verifier_version, stratum)
            self._skipped[key] = self._skipped.get(key, 0) + 1
            self._unflushed += 1
            if self._unflushed >= self.FLUSH_EVERY:
                self._flush_moments()

    def flush(self) -> None:
        """Persist the moments now. Call it when a run ends."""
        with self._lock:
            self._flush_moments(force=True)

    def unlabelled_moments(self, verifier_version: str):
        """``stratum -> {n, mean, var, skipped}`` over the units not audited.

        ``n``, ``mean`` and ``var`` describe units whose scores were recorded.
        ``skipped`` counts units that were dropped and have no score: it belongs
        in the population frame and must never be folded into ``n``, which is
        what PPI borrows strength from. A stratum with only skipped units
        appears here too, with ``n`` of zero.
        """
        with self._lock:
            strata = [stratum for (version, stratum)
                      in dict.fromkeys(list(self._moments) + list(self._skipped))
                      if version == verifier_version]
            out = {}
            for stratum in strata:
                acc = self._moments.get((verifier_version, stratum))
                out[stratum] = {
                    "n": acc.n if acc else 0,
                    "mean": acc.mean if acc else 0.0,
                    "var": acc.var if acc else 0.0,
                    "skipped": self._skipped.get((verifier_version, stratum), 0),
                }
            return out

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
            self._write_line({"kind": WATCH_KIND, "fingerprint": fingerprint})
            self._fingerprint = fingerprint

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
            merged = dict(self._priorities)
            merged.update({str(k): float(v) for k, v in priorities.items()})
            self._write_line({"kind": PRIORITY_KIND, "priorities": merged})
            self._priorities = merged

    @property
    def priorities(self) -> Dict[str, float]:
        """``artifact_signature -> priority``, as last recorded."""
        with self._lock:
            return dict(self._priorities)

    def _flush_moments(self, force: bool = False) -> None:
        if not self.path or (not force and not self._unflushed):
            self._unflushed = 0
            return
        for key in dict.fromkeys(list(self._moments) + list(self._skipped)):
            version, stratum = key
            acc = self._moments.get(key) or _Welford()
            self._write_line({"kind": MOMENTS_KIND, "verifier_version": version,
                              "stratum": stratum, "moments": acc.to_dict(),
                              "skipped": self._skipped.get(key, 0)})
        self._unflushed = 0

    def resolve(self, record_id: str, oracle_score: float,
                *, at: Optional[float] = None) -> bool:
        """Attach ground truth to a pending record. ``False`` if there was none to attach.

        Refuses to overwrite a resolution. An oracle result is a measurement
        somebody made; silently replacing it with a second one would make the
        calibration set depend on how many times a batch file was replayed, and
        replaying a batch file is exactly what people do when a job half-failed.
        Call :meth:`reopen` first if a result really has to be corrected.

        A non-finite ``oracle_score`` raises ``ValueError``: it would flow into
        every residual and estimator as ``nan``.
        """
        oracle_score = _finite_score(oracle_score, "oracle_score")
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None:
                return False
            if rec.resolved:
                raise ValueError(
                    f"{record_id} already resolved with oracle_score="
                    f"{rec.oracle_score!r}; refusing to overwrite. Use reopen().")
            updated = _replace(rec, oracle_score=oracle_score,
                               resolved_at=time.time() if at is None else at)
            self._write(updated)
            self._remember(updated)
            return True

    def reopen(self, record_id: str) -> bool:
        """Clear a resolution so it can be replaced. For corrections, not for retries."""
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None or not rec.resolved:
                return False
            updated = _replace(rec, oracle_score=None, resolved_at=None)
            self._write(updated)
            self._remember(updated)
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

    def for_improvement(self, verifier_version: Optional[str] = None,
                        scorer_version: Optional[str] = None) -> List[AuditRecord]:
        """Resolved IMPROVEMENT records -- the pool you are allowed to look at.

        ``verifier_version`` keeps exactly that measurement identity.
        ``scorer_version`` keeps every record of one *underlying scorer*
        whatever correction layer was active when it was written (records that
        carry no scorer stamp fall back to ``verifier_version`` when one is
        given, so an older store still trains)."""
        out = [r for r in self.all()
               if r.purpose is Purpose.IMPROVEMENT and r.resolved]
        if scorer_version is not None:
            out = [r for r in out
                   if r.scorer_version == scorer_version
                   or (not r.scorer_version and verifier_version is not None
                       and r.verifier_version == verifier_version)]
        elif verifier_version is not None:
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
        """Append one JSON line.

        Encoded before the file is touched, with ``allow_nan=False`` so the
        non-standard ``NaN`` / ``Infinity`` tokens never reach disk: a payload
        that cannot be encoded raises with the file unchanged. If the file ends
        in a torn fragment (a process killed mid-line), a newline is written
        first so the fragment stays its own skipped line instead of swallowing
        this one.
        """
        data = (json.dumps(payload, ensure_ascii=False, allow_nan=False)
                + "\n").encode("utf-8")
        if not self.path:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "ab+") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    data = b"\n" + data
            fh.write(data)


def _finite_score(value: Any, name: str) -> float:
    """``float(value)``, refusing ``nan`` / ``inf`` with a ``ValueError``."""
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def _refuse_constant(token: str) -> float:
    """``json`` hook for ``NaN`` / ``Infinity`` / ``-Infinity``: refuse them."""
    raise ValueError(f"non-finite JSON constant {token!r}")


def _finite_float(text: str) -> float:
    """``json`` hook for float literals: refuse one that overflows to ``inf``."""
    x = float(text)
    if not math.isfinite(x):
        raise ValueError(f"non-finite JSON number {text!r}")
    return x


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
