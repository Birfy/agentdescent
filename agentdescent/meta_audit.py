"""Meta-audit: the trust layer for the loop that evolves the rules.

``meta_run`` decides whether an evolved decision rule is better, on numbers it
computed itself: the ``meta_reward`` (an ``auc`` by default) scored on inner
outcomes, the mean gain on validation problems, the transfer ratio. The audit
package verifies the *inner* reward against an oracle; nothing verified the
meta reward -- so a rule that "decides the rules" was the one thing no longer
held to the standard everything below it was held to.

This module is the missing rung. It is ``AuditedReward`` applied one level up:

* a **probability sample** of ``(problem, seed)`` validation pairs is scored by
  both the meta reward and an independent oracle, and the pair is persisted with
  its ``inclusion_prob`` -- the same argument as the audit layer's: without the
  probability the numbers estimate nothing, and the probability cannot be
  reconstructed later;
* the **decision path never sees the oracle** -- ``meta_reward`` scores every
  verdict, the oracle only ever lands in the store -- so auditing can be turned
  on mid-fleet without changing which rules are adopted;
* the **verdict is durable** alongside the sampled evidence that decided it, so
  "why was this rule adopted" is answerable after the run that adopted it is
  gone;
* the **bias is estimable**: over the resolved sample, ``verifier_gain -
  oracle_gain`` is the residual, and :meth:`MetaAuditStore.gain_bias` is its
  Hajek estimate -- the number that says "your meta reward is 0.3 optimistic"
  instead of letting that optimism decide the rules in silence.

The store is JSONL, append-only, and a resolution appends rather than rewriting
-- the exact contract :class:`~agentdescent.audit.store.AuditStore` keeps, for
the same reasons (a killed process damages one line, truth arrives later from
another process, and last occurrence wins).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union,
)

from .audit import new_record_id, verifier_fingerprint
from .evolution import EvolutionResult, Task
from .meta import (
    MetaOutcome, MetaReward, MetaRunReport, Problem, SlotSpec, _mean_gain,
    _require_disjoint, _transfer_from, auc, meta_evolve, meta_validate,
)
from .policies import Policies

__all__ = [
    "MetaAuditRecord",
    "MetaAuditStore",
    "audited_meta_validate",
    "meta_run_audited",
]

SCHEMA_VERSION = 1

#: Marks a JSONL line as a run-level verdict snapshot, not a sampled pair. A
#: verdict belongs to the run; a pair belongs to a measurement, and one line
#: should not be forced to be both.
VERDICT_KIND = "meta_verdict"


@dataclass(frozen=True)
class MetaAuditRecord:
    """One sampled ``(problem, seed)`` before/after pair, awaiting truth.

    The unit of trust at the meta level is the *gain* -- the quantity the
    verdict is built from. ``verifier_gain`` is ``after - before`` under the
    meta reward; ``oracle_gain`` is the same difference under the oracle, and
    is the field an auditor actually cares about. ``None`` means "not yet",
    never "zero", so a pending record cannot be mistaken for a true zero.
    """

    record_id: str
    run_id: str
    slot: str
    problem: str
    seed: int
    #: A digest of the rendered candidate value this outcome was scored for, so
    #: a record can be tied back to the exact rule that produced it.
    value_digest: str
    #: Which verifier (the meta reward) produced ``verifier_gain``. A bias
    #: estimated for one reward says nothing about the next one.
    verifier_version: str
    inclusion_prob: float
    verifier_before: float
    verifier_after: float
    verifier_gain: float
    oracle_before: Optional[float] = None
    oracle_after: Optional[float] = None
    oracle_gain: Optional[float] = None
    #: The verdict this evidence served, stamped after the run decided.
    verdict: str = ""
    dispatched_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    #: The seed the inclusion draw was made from, so a run can be replayed.
    sampler_seed: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise TypeError(f"record_id must be a non-empty str, got {self.record_id!r}")
        if not 0.0 < self.inclusion_prob <= 1.0:
            raise ValueError(
                f"inclusion_prob must be in (0, 1], got {self.inclusion_prob!r}")
        for name, value in (("verifier_before", self.verifier_before),
                            ("verifier_after", self.verifier_after),
                            ("verifier_gain", self.verifier_gain)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if (self.oracle_gain is None) != (self.resolved_at is None):
            raise ValueError(
                "oracle_gain and resolved_at must be set together: "
                f"got {self.oracle_gain!r} / {self.resolved_at!r}")
        if self.oracle_gain is not None and not math.isfinite(self.oracle_gain):
            raise ValueError(f"oracle_gain must be finite, got {self.oracle_gain!r}")

    @property
    def resolved(self) -> bool:
        """Has truth arrived for this pair?"""
        return self.oracle_gain is not None

    @property
    def residual(self) -> Optional[float]:
        """``verifier_gain - oracle_gain``, or ``None`` while unresolved.

        Positive means the meta reward reported the gain higher than the oracle
        did -- the direction that adopts rules which did not actually improve.
        """
        if self.oracle_gain is None:
            return None
        return self.verifier_gain - self.oracle_gain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "slot": self.slot,
            "problem": self.problem,
            "seed": self.seed,
            "value_digest": self.value_digest,
            "verifier_version": self.verifier_version,
            "inclusion_prob": self.inclusion_prob,
            "verifier_before": self.verifier_before,
            "verifier_after": self.verifier_after,
            "verifier_gain": self.verifier_gain,
            "oracle_before": self.oracle_before,
            "oracle_after": self.oracle_after,
            "oracle_gain": self.oracle_gain,
            "verdict": self.verdict,
            "dispatched_at": self.dispatched_at,
            "resolved_at": self.resolved_at,
            "sampler_seed": self.sampler_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetaAuditRecord":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"meta-audit record schema_version={d.get('schema_version')!r}, "
                f"this build reads {SCHEMA_VERSION}; refusing to guess")
        # No silent defaulting: an unknown key is a malformed line, not a record
        # to load under defaults -- the TypeError it raises sends it down the
        # store's corrupt-line path.
        return cls(**d)


class MetaAuditStore:
    """Durable JSONL of sampled meta pairs and verdicts, indexed in memory.

    ``path=None`` keeps everything in memory (tests, dry runs); nothing else
    changes, so a test that passes in memory is testing the code that runs on
    disk. Appends are atomic at the line level: a process killed mid-write
    damages at most the last line, and a reader skips one unparseable line
    rather than losing the file. A resolution appends too, and the last
    occurrence of a ``record_id`` wins -- the whole reconciliation rule.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: Dict[str, MetaAuditRecord] = {}
        self._order: List[str] = []
        self._verdicts: List[Dict[str, Any]] = []
        self._corrupt = 0
        self._lock = threading.RLock()
        if path and os.path.exists(path):
            self.load()

    # -- reading -------------------------------------------------------------

    def load(self) -> None:
        """Re-read the file, last-occurrence-wins for pairs; verdicts in order.

        A line that will not parse as JSON, or parses to something that is not
        a record or a known side-kind, is skipped and counted in :attr:`corrupt`
        -- that is a torn write from a killed process, and the lines around it
        are fine. A line that parses but carries an unknown ``schema_version``
        raises: that is a different build's file and its fields do not mean what
        this code thinks they mean.
        """
        if not self.path:
            return
        with self._lock:
            self._records.clear()
            self._order.clear()
            self._verdicts.clear()
            self._corrupt = 0
            try:
                with open(self.path, "rb") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            payload = json.loads(
                                raw.decode("utf-8"), parse_constant=_refuse_constant)
                        except (UnicodeDecodeError, ValueError):
                            self._corrupt += 1
                            continue
                        if not isinstance(payload, dict):
                            self._corrupt += 1
                            continue
                        if not self._load_line(payload):
                            self._corrupt += 1
            except OSError:
                self._corrupt += 1

    def _load_line(self, payload: Dict[str, Any]) -> bool:
        kind = payload.get("kind")
        if kind == VERDICT_KIND:
            self._verdicts.append({k: v for k, v in payload.items()
                                   if k != "kind"})
            return True
        if kind is not None:
            return False
        if payload.get("schema_version") != SCHEMA_VERSION:
            # Raises on schema drift: fields do not mean what this code thinks.
            MetaAuditRecord.from_dict(payload)
        try:
            rec = MetaAuditRecord.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return False
        if rec.record_id not in self._records:
            self._order.append(rec.record_id)
        self._records[rec.record_id] = rec
        return True

    @property
    def corrupt(self) -> int:
        """Lines skipped as unparseable on the last :meth:`load`."""
        return self._corrupt

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def all(self) -> List[MetaAuditRecord]:
        """Every pair record, in order of first occurrence."""
        with self._lock:
            return [self._records[r] for r in self._order]

    def iter_records(self, *, run_id: Optional[str] = None,
                     resolved_only: bool = False) -> Iterator[MetaAuditRecord]:
        """The pair records, optionally narrowed to one run / resolved only."""
        with self._lock:
            for rid in self._order:
                rec = self._records[rid]
                if run_id is not None and rec.run_id != run_id:
                    continue
                if resolved_only and not rec.resolved:
                    continue
                yield rec

    def get(self, record_id: str) -> Optional[MetaAuditRecord]:
        with self._lock:
            return self._records.get(record_id)

    def pending(self, *, run_id: Optional[str] = None) -> List[MetaAuditRecord]:
        """Pairs still waiting on truth -- the work list for whoever answers."""
        with self._lock:
            out = []
            for rid in self._order:
                rec = self._records[rid]
                if not rec.resolved and (run_id is None or rec.run_id == run_id):
                    out.append(rec)
            return out

    def verdicts(self) -> List[Dict[str, Any]]:
        """Run-level verdict lines, in order."""
        with self._lock:
            return list(self._verdicts)

    # -- writing -------------------------------------------------------------

    def record(self, rec: MetaAuditRecord) -> MetaAuditRecord:
        """Append one sampled pair. Disk first, memory second."""
        with self._lock:
            self._write(rec.to_dict())
            if rec.record_id not in self._records:
                self._order.append(rec.record_id)
            self._records[rec.record_id] = rec
        return rec

    def record_verdict(self, *, run_id: str, slot: str, verdict: str,
                       reason: str, mean_gain: float, transfer: Optional[float],
                       audited_n: int, oracle_gain: Optional[float],
                       value_digest: str) -> None:
        """Persist the run's decision and the audit summary that went with it."""
        line = {"kind": VERDICT_KIND, "run_id": run_id, "slot": slot,
                "verdict": verdict, "reason": reason, "mean_gain": mean_gain,
                "transfer": transfer, "audited_n": audited_n,
                "oracle_gain": oracle_gain, "value_digest": value_digest}
        self._write_line(line)
        with self._lock:
            self._verdicts.append({k: v for k, v in line.items() if k != "kind"})

    def resolve(self, record_id: str, oracle_gain: float,
                *, oracle_before: Optional[float] = None,
                oracle_after: Optional[float] = None,
                at: Optional[float] = None) -> bool:
        """Attach truth to a pending pair. ``False`` if there was none.

        Refuses to overwrite a resolution -- truth is a measurement somebody
        made, and silently replacing it would make the calibration set depend on
        how many times a batch file was replayed.
        """
        if not math.isfinite(float(oracle_gain)):
            raise ValueError(f"oracle_gain must be finite, got {oracle_gain!r}")
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None:
                return False
            if rec.resolved:
                raise ValueError(
                    f"{record_id} already resolved with oracle_gain="
                    f"{rec.oracle_gain!r}; refusing to overwrite")
            updated = MetaAuditRecord(
                record_id=rec.record_id, run_id=rec.run_id, slot=rec.slot,
                problem=rec.problem, seed=rec.seed, value_digest=rec.value_digest,
                verifier_version=rec.verifier_version,
                inclusion_prob=rec.inclusion_prob,
                verifier_before=rec.verifier_before,
                verifier_after=rec.verifier_after,
                verifier_gain=rec.verifier_gain,
                oracle_before=oracle_before,
                oracle_after=oracle_after,
                oracle_gain=float(oracle_gain),
                verdict=rec.verdict,
                dispatched_at=rec.dispatched_at,
                resolved_at=time.time() if at is None else at,
                sampler_seed=rec.sampler_seed, schema_version=SCHEMA_VERSION)
            self._write(updated.to_dict())
            self._records[record_id] = updated
            return True

    def reopen(self, record_id: str) -> bool:
        """Clear a resolution so it can be replaced. For corrections, not retries."""
        with self._lock:
            rec = self._records.get(record_id)
            if rec is None or not rec.resolved:
                return False
            updated = MetaAuditRecord(
                record_id=rec.record_id, run_id=rec.run_id, slot=rec.slot,
                problem=rec.problem, seed=rec.seed, value_digest=rec.value_digest,
                verifier_version=rec.verifier_version,
                inclusion_prob=rec.inclusion_prob,
                verifier_before=rec.verifier_before,
                verifier_after=rec.verifier_after,
                verifier_gain=rec.verifier_gain,
                oracle_before=None, oracle_after=None, oracle_gain=None,
                verdict=rec.verdict, dispatched_at=rec.dispatched_at,
                resolved_at=None, sampler_seed=rec.sampler_seed,
                schema_version=SCHEMA_VERSION)
            self._write(updated.to_dict())
            self._records[record_id] = updated
            return True

    # -- estimation ----------------------------------------------------------

    def gain_bias(self, *, run_id: Optional[str] = None,
                  verifier_version: Optional[str] = None) -> Optional[float]:
        """The honest estimate of how much the meta reward overstates the gain.

        The Hajek (inclusion-probability-weighted) mean of ``verifier_gain -
        oracle_gain`` over the resolved sample. Positive means the meta reward
        is optimistic by that much. ``None`` when no resolved record is
        available -- the sample is the point, and nothing over nothing is not an
        estimate.
        """
        resolved = [r for r in self._resolved(
            run_id=run_id, verifier_version=verifier_version)
            if r.residual is not None]
        if not resolved:
            return None
        total = sum(1.0 / r.inclusion_prob for r in resolved)
        return sum((1.0 / r.inclusion_prob) * r.residual for r in resolved) / total

    def rectified_gain(self, *, run_id: Optional[str] = None,
                       verifier_version: Optional[str] = None) -> Optional[float]:
        """The honest gain estimate: Hajek mean of ``oracle_gain`` over the
        resolved sample. ``None`` when no resolved record is available."""
        resolved = [r for r in self._resolved(
            run_id=run_id, verifier_version=verifier_version)
            if r.oracle_gain is not None]
        if not resolved:
            return None
        total = sum(1.0 / r.inclusion_prob for r in resolved)
        return sum((1.0 / r.inclusion_prob) * r.oracle_gain for r in resolved) / total

    def summarise(self, *, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Counts and raw means. **Not** the estimator -- see :meth:`gain_bias`.
        This is for eyeballing a run: raw unweighted means of the verifier gain,
        the oracle gain and the residual over the records it holds."""
        records = list(self.iter_records(run_id=run_id))
        resolved = [r for r in records if r.resolved]
        return {
            "n_records": len(records),
            "n_resolved": len(resolved),
            "mean_verifier_gain": (sum(r.verifier_gain for r in records) / len(records)
                                   if records else float("nan")),
            "mean_oracle_gain": (sum(r.oracle_gain for r in resolved) / len(resolved)
                                 if resolved else float("nan")),
            "mean_residual": (sum(r.residual for r in resolved) / len(resolved)
                              if resolved else float("nan")),
        }

    def _resolved(self, *, run_id: Optional[str], verifier_version: Optional[str]):
        out = []
        for rec in self.all():
            if run_id is not None and rec.run_id != run_id:
                continue
            if verifier_version is not None \
                    and rec.verifier_version != verifier_version:
                continue
            if rec.resolved:
                out.append(rec)
        return out

    # -- internals -----------------------------------------------------------

    def _write(self, payload: Dict[str, Any]) -> None:
        self._write_line(payload)

    def _write_line(self, payload: Dict[str, Any]) -> None:
        """Append one JSON line. Encoded before the file is touched with
        ``allow_nan=False``; a torn tail keeps its own skipped line."""
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


def _refuse_constant(token: str) -> float:
    raise ValueError(f"non-finite JSON constant {token!r}")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _oracle_score(oracle: MetaReward, outcome: MetaOutcome) -> Optional[float]:
    """``oracle(outcome)`` as a finite float, or ``None`` for "not yet"."""
    value = oracle(outcome)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"oracle returned a non-finite score {value!r}")
    return value


def audited_meta_validate(
    spec: SlotSpec,
    before: str,
    after: str,
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    seeds: Sequence[int],
    meta_reward: MetaReward,
    oracle: MetaReward,
    store: MetaAuditStore,
    slot: str,
    run_id: str,
    sample_rate: float = 0.1,
    seed: int = 0,
    verifier_version: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """:func:`meta_validate`, with a probability sample of pairs sent to the oracle.

    Every ``(problem, seed)`` pair is scored under ``meta_reward`` and feeds the
    returned validation table -- the decision path never sees ``oracle``. With
    probability ``sample_rate`` the pair is scored again under ``oracle`` and
    the before/after record is persisted to ``store`` with its ``inclusion_prob``,
    so the difference the verifier reported and the difference truth reported
    are both on disk, tied to the same ``(problem, seed, value)`` unit.

    ``oracle`` is called only for sampled pairs, on the same two outcomes the
    verifier scored. When it returns a number for both, the record is written
    already resolved (``oracle_gain = oracle(after) - oracle(before)``); when it
    returns ``None`` for either (truth is not available yet), the record is
    written pending, for :meth:`MetaAuditStore.resolve` to attach later.
    """
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate!r}")
    named = (dict(problems) if isinstance(problems, Mapping)
             else {f"p{i}": p for i, p in enumerate(problems)})
    old, new = spec.compile(before), spec.compile(after)
    version = verifier_version or verifier_fingerprint(meta_reward)
    rng = random.Random(seed)
    digest = _digest(after)
    report: Dict[str, Dict[str, float]] = {}
    for name, problem in named.items():
        base, cand, deltas = [], [], []
        for s in seeds:
            out_before = problem(old, s)
            out_after = problem(new, s)
            vb, vc = float(meta_reward(out_before)), float(meta_reward(out_after))
            base.append(vb)
            cand.append(vc)
            deltas.append(vc - vb)
            if rng.random() < sample_rate:
                # the oracle only ever lands in the store: ``base``/``cand``
                # above are already final, so the verdict cannot see it.
                ob, oa = _oracle_score(oracle, out_before), _oracle_score(oracle, out_after)
                resolved = ob is not None and oa is not None
                store.record(MetaAuditRecord(
                    record_id=new_record_id(), run_id=run_id, slot=slot,
                    problem=name, seed=s, value_digest=digest,
                    verifier_version=version, inclusion_prob=sample_rate,
                    verifier_before=vb, verifier_after=vc, verifier_gain=vc - vb,
                    oracle_before=ob, oracle_after=oa,
                    oracle_gain=(oa - ob) if resolved else None,
                    resolved_at=time.time() if resolved else None,
                    sampler_seed=seed))
        report[name] = {
            "n": len(seeds),
            "before": statistics.fmean(base),
            "after": statistics.fmean(cand),
            "gain": statistics.fmean(deltas),
            "gain_sd": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "wins": sum(d > 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
        }
    return report


def meta_run_audited(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    oracle: MetaReward,
    store: MetaAuditStore,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    meta_reward: Optional[MetaReward] = None,
    audit_rate: float = 0.1,
    audit_seed: int = 0,
    run_id: Optional[str] = None,
    seeds: Sequence[int] = (0,),
    val_seeds: Sequence[int] = (1, 2),
    min_gain: float = 0.0,
    transfer_floor: float = 0.0,
    **evolve_kwargs: Any,
) -> Tuple[MetaRunReport, EvolutionResult]:
    """``meta_run`` with an audited validation: sample, oracle, persist, decide.

    The decision is ``meta_run``'s -- evolve on ``problems``, validate on
    ``val_problems`` at ``val_seeds``, and adopt only when the mean gain on
    unseen problems clears ``min_gain`` and the transfer is not below
    ``transfer_floor``. The difference: the validation is
    :func:`audited_meta_validate`, so a ``sample_rate`` fraction of the pairs
    is also scored by ``oracle`` and recorded in ``store``; the verdict is then
    persisted with the audit summary, and ``result.meta`` carries
    ``audited_n`` / ``oracle_gain`` / ``gain_bias`` so a CLI or plugin can show
    "this decision rests on N pairs the oracle checked" without recomputing it.

    ``oracle`` is a :class:`MetaReward` -- a ``MetaOutcome -> float`` that
    scores an inner run with truth, called only on sampled pairs; return
    ``None`` to leave a pair pending for a later :meth:`MetaAuditStore.resolve`. ``meta_reward`` defaults to ``auc`` as in
    ``meta_run``. Governance is the same ``_require_disjoint`` isolation.
    """
    if not val_problems:
        raise ValueError("meta_run_audited() needs val_problems to judge the transfer")
    _require_disjoint(problems, val_problems, seeds, val_seeds)
    score = meta_reward or auc
    result = meta_evolve(
        problems, slot=slot, spec=spec, propose=propose, model=model,
        meta_reward=score, seeds=seeds, **evolve_kwargs)
    seed_value = spec.render(spec.initial())
    value = result.rendered or seed_value
    # a random suffix: two runs in the same second must not share a run_id,
    # or their sampled pairs and verdicts would be pooled under one run.
    rid = run_id or f"meta-audited-{slot}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    train_validation = meta_validate(
        spec, seed_value, value, problems, seeds=seeds, meta_reward=score)
    validation = audited_meta_validate(
        spec, seed_value, value, val_problems, seeds=val_seeds,
        meta_reward=score, oracle=oracle, store=store, slot=slot, run_id=rid,
        sample_rate=audit_rate, seed=audit_seed)

    mean_gain = _mean_gain(validation)
    train_gain = _mean_gain(train_validation)
    transfer = _transfer_from(train_gain, mean_gain)
    if mean_gain <= min_gain:
        verdict, reason = "reject", (
            f"evolved value gained {mean_gain:+.3f} on unseen problems, "
            f"not above min_gain {min_gain:+.3f}")
    elif transfer is not None and transfer < transfer_floor:
        verdict, reason = "reject", (
            f"transfer ratio {transfer:.3f} is below transfer_floor "
            f"{transfer_floor:.3f} (train gain {train_gain:+.3f}, "
            f"val gain {mean_gain:+.3f})")
    else:
        verdict, reason = "adopt", (
            f"mean gain {mean_gain:+.3f} on unseen problems, "
            f"transfer {transfer if transfer is None else round(transfer, 3)}")

    audit_n = sum(1 for _ in store.iter_records(run_id=rid))
    oracle_gain = store.rectified_gain(run_id=rid)
    bias = store.gain_bias(run_id=rid)
    store.record_verdict(run_id=rid, slot=slot, verdict=verdict, reason=reason,
                         mean_gain=mean_gain, transfer=transfer,
                         audited_n=audit_n, oracle_gain=oracle_gain,
                         value_digest=_digest(value))

    report = MetaRunReport(
        slot=slot, seed_value=seed_value, value=value,
        validation=validation, mean_gain=mean_gain, transfer=transfer,
        verdict=verdict, reason=reason)
    try:
        result.meta = {
            **(_report_dict(report)),
            "audited_n": audit_n,
            "oracle_gain": oracle_gain,
            "gain_bias": bias,
            "audit_store": store.path,
        }
    except AttributeError:
        pass
    return report, result


def _report_dict(report: MetaRunReport) -> Dict[str, Any]:
    return {
        "slot": report.slot,
        "seed_value": report.seed_value,
        "value": report.value,
        "validation": report.validation,
        "mean_gain": report.mean_gain,
        "transfer": report.transfer,
        "verdict": report.verdict,
        "reason": report.reason,
    }
