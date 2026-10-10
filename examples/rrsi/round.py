"""Offline, single-round orchestration around the RRSI selection core.

This runner defines the ordering boundary: proposal, pre-evaluation screening,
evaluation of survivors from one immutable base, then round-wide selection. The
screen callback is an integration seam; this module does not detect leakage.
It intentionally does not own a loop, CLI, persistence store, or model client.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .selection import (
    Candidate,
    Measurement,
    SelectionConfig,
    select_round,
)


@dataclass(frozen=True)
class HarnessCandidate:
    """A proposed file-tree variant and the exact base it was derived from."""

    variant: str
    base_version: str
    base_sha256: str
    files: Mapping[str, str]
    components: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RoundBase:
    """The immutable incumbent snapshot passed to the proposal callback."""

    version: str
    files: Mapping[str, str]
    sha256: str


@dataclass(frozen=True)
class ScreeningResult:
    """A deterministic or human/model-provided pre-evaluation gate result."""

    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class CandidateRecord:
    """Serializable outcome for one proposed candidate."""

    variant: str
    base_version: str
    source_base_sha256: Optional[str]
    artifact_sha256: Optional[str]
    components: Tuple[str, ...]
    outcome: str
    reason: str
    measurement: Optional[Measurement] = None
    admissible: bool = False
    decision_reason: str = ""
    delta_score: Optional[float] = None
    delta_cost: Optional[float] = None
    novelty: int = 0
    guards: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "base_version": self.base_version,
            "source_base_sha256": self.source_base_sha256,
            "artifact_sha256": self.artifact_sha256,
            "components": list(self.components),
            "outcome": self.outcome,
            "reason": self.reason,
            "measurement": _measurement_dict(self.measurement),
            "admissible": self.admissible,
            "decision_reason": self.decision_reason,
            "delta_score": self.delta_score,
            "delta_cost": self.delta_cost,
            "novelty": self.novelty,
            "guards": list(self.guards),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        return cls(
            variant=value["variant"],
            base_version=value["base_version"],
            source_base_sha256=value.get("source_base_sha256"),
            artifact_sha256=value.get("artifact_sha256"),
            components=tuple(value["components"]),
            outcome=value["outcome"],
            reason=value["reason"],
            measurement=_measurement_from_dict(value.get("measurement")),
            admissible=value["admissible"],
            decision_reason=value["decision_reason"],
            delta_score=value.get("delta_score"),
            delta_cost=value.get("delta_cost"),
            novelty=value["novelty"],
            guards=tuple(value["guards"]),
        )


@dataclass(frozen=True)
class RoundRecord:
    """JSON round-trip record for a single RRSI selection round."""

    round_index: int
    base_version: str
    base_sha256: str
    incumbent: Measurement
    historical_best: float
    delta: float
    config: SelectionConfig
    incumbent_counts: Mapping[str, int]
    candidates: Tuple[CandidateRecord, ...]
    selected_variant: Optional[str]
    next_best_score: float
    proposal_error: Optional[str] = None

    def __post_init__(self) -> None:
        # A frozen dataclass alone does not freeze a dict supplied by the caller.
        object.__setattr__(
            self, "incumbent_counts", MappingProxyType(dict(self.incumbent_counts))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "round_index": self.round_index,
            "base_version": self.base_version,
            "base_sha256": self.base_sha256,
            "incumbent": _measurement_dict(self.incumbent),
            "historical_best": self.historical_best,
            "delta": self.delta,
            "config": {
                "beta0": self.config.beta0,
                "beta1": self.config.beta1,
                "w_s": self.config.w_s,
                "w_c": self.config.w_c,
                "w_n": self.config.w_n,
            },
            "incumbent_counts": dict(self.incumbent_counts),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_variant": self.selected_variant,
            "next_best_score": self.next_best_score,
            "proposal_error": self.proposal_error,
        }

    def to_json(self) -> str:
        """Return canonical JSON; non-finite values are never silently emitted."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoundRecord":
        if value.get("schema_version") != 1:
            raise ValueError("unsupported RRSI round record schema_version")
        cfg = value["config"]
        incumbent = _measurement_from_dict(value["incumbent"])
        if incumbent is None:
            raise ValueError("round record must include an incumbent measurement")
        return cls(
            round_index=value["round_index"],
            base_version=value["base_version"],
            base_sha256=value["base_sha256"],
            incumbent=incumbent,
            historical_best=value["historical_best"],
            delta=value["delta"],
            config=SelectionConfig(**cfg),
            incumbent_counts=dict(value["incumbent_counts"]),
            candidates=tuple(CandidateRecord.from_dict(c) for c in value["candidates"]),
            selected_variant=value["selected_variant"],
            next_best_score=value["next_best_score"],
            proposal_error=value.get("proposal_error"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "RoundRecord":
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class RoundResult:
    """The selected candidate, if any, plus its durable decision record."""

    winner: Optional[HarnessCandidate]
    record: RoundRecord


def _measurement_dict(value: Optional[Measurement]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return {"score": value.score, "cost": value.cost}


def _measurement_from_dict(value: Optional[Mapping[str, Any]]) -> Optional[Measurement]:
    if value is None:
        return None
    return Measurement(score=value["score"], cost=value.get("cost"))


def _finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_measurement(value: object) -> bool:
    if not isinstance(value, Measurement):
        return False
    score, cost = value.score, value.cost
    return (
        _finite_number(score)
        and 0.0 <= score <= 1.0
        and (cost is None or (_finite_number(cost) and cost >= 0.0))
    )


def _freeze_files(files: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(files, Mapping):
        raise TypeError("harness files must be a mapping of paths to text")
    copied: Dict[str, str] = {}
    for path, content in files.items():
        if not isinstance(path, str) or not isinstance(content, str):
            raise TypeError("harness paths and contents must be strings")
        copied[path] = content
    return MappingProxyType(copied)


def _digest(files: Mapping[str, str]) -> str:
    packed = json.dumps(
        dict(files),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _valid_round_inputs(
    round_index: int,
    base_version: str,
    incumbent: Measurement,
    historical_best: float,
    delta: float,
    config: SelectionConfig,
) -> None:
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 0
    ):
        raise ValueError("round_index must be a non-negative integer")
    if not isinstance(base_version, str) or not base_version:
        raise ValueError("base_version must be a non-empty string")
    if not _valid_measurement(incumbent):
        raise ValueError("incumbent must have a valid finite measurement")
    if (
        not isinstance(historical_best, (int, float))
        or isinstance(historical_best, bool)
        or not _finite_number(historical_best)
        or not 0.0 <= historical_best <= 1.0
    ):
        raise ValueError("historical_best must be finite and in [0, 1]")
    if not _finite_number(delta) or not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be finite and in [0, 1]")
    if not isinstance(config, SelectionConfig):
        raise TypeError("config must be a SelectionConfig")
    weights = (config.beta0, config.beta1, config.w_s, config.w_c, config.w_n)
    if any(not _finite_number(v) or v < 0.0 for v in weights):
        raise ValueError("selection config weights must be finite and non-negative")


def run_round(
    *,
    round_index: int,
    base_version: str,
    base_files: Mapping[str, str],
    incumbent: Measurement,
    historical_best: float,
    delta: float,
    config: SelectionConfig,
    incumbent_counts: Mapping[str, int],
    propose: Callable[[RoundBase], Sequence[HarnessCandidate]],
    screen: Callable[[HarnessCandidate], ScreeningResult],
    evaluate: Callable[[HarnessCandidate], Measurement],
    guard_fn: Optional[
        Callable[[Measurement, Optional[Measurement]], Sequence[str]]
    ] = None,
) -> RoundResult:
    """Run one offline RRSI round.

    ``propose`` is called once with one read-only snapshot, version and digest of
    the incumbent. Each proposed candidate must name that same base version and
    digest. Screening is
    completed before ``evaluate`` is called; only measured, unscreened-veto-free
    candidates are considered by the pinned selection mechanism. Callback errors
    are recorded per candidate and do not suppress other candidates.

    This does not implement leakage detection. ``screen`` is an explicit hook for
    a caller-owned checker and must not be described as a security proof.
    """
    _valid_round_inputs(
        round_index, base_version, incumbent, historical_best, delta, config
    )
    frozen_base = _freeze_files(base_files)
    base_sha = _digest(frozen_base)
    round_base = RoundBase(base_version, frozen_base, base_sha)
    for name, count in incumbent_counts.items():
        if (
            not isinstance(name, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                "incumbent_counts must map component names to non-negative integers"
            )
    counts = dict(incumbent_counts)

    try:
        proposed = list(propose(round_base))
        proposal_error = None
    except Exception as exc:  # captured into the explicit round outcome
        proposed = []
        proposal_error = "{}: {}".format(type(exc).__name__, str(exc))

    names = [
        c.variant
        for c in proposed
        if isinstance(c, HarnessCandidate) and isinstance(c.variant, str)
    ]
    duplicate_names = {name for name, count in Counter(names).items() if count > 1}
    candidates_for_selection: List[Candidate] = []
    candidate_records: List[CandidateRecord] = []
    candidate_objects: Dict[str, HarnessCandidate] = {}

    for index, raw in enumerate(proposed):
        if not isinstance(raw, HarnessCandidate):
            name = "<invalid-{}>".format(index)
            candidate_records.append(
                CandidateRecord(
                    variant=name,
                    base_version=base_version,
                    source_base_sha256=None,
                    artifact_sha256=None,
                    components=(),
                    outcome="proposal_error",
                    reason="proposal callback returned a non-HarnessCandidate",
                )
            )
            candidates_for_selection.append(
                Candidate(name, gate_failure="proposal_error")
            )
            continue

        variant = raw.variant
        components: Tuple[str, ...] = ()
        reason = ""
        if not isinstance(variant, str) or not variant:
            reason = "candidate variant must be non-empty"
        elif variant in duplicate_names:
            reason = "duplicate candidate variant"
        elif not isinstance(raw.base_version, str):
            reason = "candidate base_version must be a string"
        elif raw.base_version != base_version:
            reason = "candidate was derived from a different base version"
        elif raw.base_sha256 != base_sha:
            reason = "candidate was derived from a different base digest"

        try:
            files = _freeze_files(raw.files)
            if isinstance(raw.components, str):
                raise TypeError("candidate components must be a sequence of strings")
            proposed_components = tuple(raw.components)
            if any(not isinstance(c, str) for c in proposed_components):
                raise TypeError("candidate components must be strings")
            components = proposed_components
            candidate = HarnessCandidate(
                variant, raw.base_version, raw.base_sha256, files, components
            )
            artifact_sha = _digest(files)
        except Exception as exc:
            candidate = raw
            artifact_sha = None
            if not reason:
                reason = "{}: {}".format(type(exc).__name__, str(exc))

        safe_variant = (
            variant
            if isinstance(variant, str) and variant
            else "<invalid-{}>".format(index)
        )

        status = "proposed"
        measurement: Optional[Measurement] = None
        if reason:
            status, detail = "proposal_error", reason
            candidate_for_selection = Candidate(
                safe_variant, components=components, gate_failure=detail
            )
        else:
            candidate_objects[safe_variant] = candidate
            try:
                screen_result = screen(candidate)
                if not isinstance(screen_result, ScreeningResult):
                    raise TypeError("screen must return ScreeningResult")
                if not isinstance(screen_result.accepted, bool):
                    raise TypeError("screen.accepted must be bool")
                if not isinstance(screen_result.reason, str):
                    raise TypeError("screen.reason must be a string")
                if not screen_result.accepted:
                    status = "screen_rejected"
                    detail = screen_result.reason or "screen rejected candidate"
                else:
                    status = "screened"
                    detail = screen_result.reason
            except Exception as exc:
                status = "screen_error"
                detail = "{}: {}".format(type(exc).__name__, str(exc))

            if status == "screened":
                try:
                    measured = evaluate(candidate)
                    if not _valid_measurement(measured):
                        status = "invalid_measurement"
                        detail = "evaluator returned an invalid measurement"
                    else:
                        status = "evaluated"
                        # Keep an optional screening note in the durable record.
                        measurement = measured
                except Exception as exc:
                    status = "evaluation_error"
                    detail = "{}: {}".format(type(exc).__name__, str(exc))

            gate_failure = (
                None if status == "evaluated" else "{}: {}".format(status, detail)
            )
            candidate_for_selection = Candidate(
                safe_variant,
                components=candidate.components,
                measurement=measurement,
                gate_failure=gate_failure,
            )

        candidates_for_selection.append(candidate_for_selection)
        candidate_records.append(
            CandidateRecord(
                variant=safe_variant,
                base_version=raw.base_version
                if isinstance(raw.base_version, str)
                else "",
                source_base_sha256=(
                    raw.base_sha256 if isinstance(raw.base_sha256, str) else None
                ),
                artifact_sha256=artifact_sha,
                components=components,
                outcome=status,
                reason=detail,
                measurement=measurement,
            )
        )

    guard_errors: Dict[str, str] = {}

    def checked_guard(
        base: Measurement, measurement: Optional[Measurement]
    ) -> Sequence[str]:
        if guard_fn is None:
            return ()
        # select_round calls this in candidate order. Candidate variants are
        # unique by construction, so they identify any callback failure.
        current = guard_candidate_names[guard_call_index[0]]
        guard_call_index[0] += 1
        try:
            result = guard_fn(base, measurement)
            if isinstance(result, str):
                raise TypeError("guard_fn must return a sequence of strings, not str")
            reasons = tuple(result)
            if any(not isinstance(reason, str) for reason in reasons):
                raise TypeError("guard_fn results must be strings")
            return reasons
        except Exception as exc:
            message = "{}: {}".format(type(exc).__name__, str(exc))
            guard_errors[current] = message
            return ("guard callback error",)

    guard_candidate_names = [
        candidate.variant
        for candidate in candidates_for_selection
        if not candidate.gate_failure and _valid_measurement(candidate.measurement)
    ]
    guard_call_index = [0]
    winner_selection, decisions = select_round(
        candidates_for_selection,
        incumbent,
        historical_best,
        delta,
        config,
        counts,
        guard_fn=checked_guard if guard_fn is not None else None,
    )
    updated_records: List[CandidateRecord] = []
    for result, decision in zip(candidate_records, decisions):
        guard_error = guard_errors.get(result.variant)
        outcome = "guard_error" if guard_error else result.outcome
        reason = guard_error if guard_error else result.reason
        # Selection only updates the candidate's result; pre-evaluation outcomes
        # remain intact and are not replaced by an artificial measurement.
        updated_records.append(
            CandidateRecord(
                variant=result.variant,
                base_version=result.base_version,
                source_base_sha256=result.source_base_sha256,
                artifact_sha256=result.artifact_sha256,
                components=result.components,
                outcome=outcome,
                reason=reason,
                measurement=result.measurement,
                admissible=decision.admissible,
                decision_reason=decision.reason,
                delta_score=decision.delta_score,
                delta_cost=decision.delta_cost,
                novelty=decision.novelty,
                guards=tuple(decision.guards),
            )
        )

    winner = (
        candidate_objects.get(winner_selection.variant)
        if winner_selection is not None
        else None
    )
    selected_score = (
        winner_selection.measurement.score
        if winner_selection is not None and winner_selection.measurement is not None
        else None
    )
    next_best_score = (
        max(historical_best, selected_score)
        if selected_score is not None
        else historical_best
    )
    record = RoundRecord(
        round_index=round_index,
        base_version=base_version,
        base_sha256=base_sha,
        incumbent=incumbent,
        historical_best=historical_best,
        delta=delta,
        config=config,
        incumbent_counts=counts,
        candidates=tuple(updated_records),
        selected_variant=winner.variant if winner is not None else None,
        next_best_score=next_best_score,
        proposal_error=proposal_error,
    )
    return RoundResult(winner=winner, record=record)


__all__ = [
    "CandidateRecord",
    "HarnessCandidate",
    "RoundBase",
    "RoundRecord",
    "RoundResult",
    "ScreeningResult",
    "run_round",
]
