"""Spend verifier compute where it changes the decision.

A single verifier has one failure mode: it is wrong in the same direction on
the cases the loop is optimising. A fixed ensemble has another: paying for
every verifier on every candidate, including the easy ones. This module is the
middle ground:

* **cascade** -- cheap checks (unit tests, static checks, a small judge) run
  first; expensive ones (oracle, large judge) run only while the decision is
  still uncertain and the budget allows;
* **weighted** -- every affordable verifier runs and votes with a weight;
* **majority** / **unanimous** -- every affordable verifier runs and the policy
  is the vote rule. If the budget prevents the full panel, the report is
  incomplete and fails closed.

It is deliberately agnostic about the verifier. A verifier can return a bool, a
score in ``[0, 1]``, or an object with ``score`` / ``passed`` / ``accept``.
`CapabilityReport` works directly; an `EvaluationReport` needs a small adapter
because it has no per-candidate score.

    ensemble = VerifierEnsemble([
        VerifierSpec("unit", unit_test, cost=1),
        VerifierSpec("judge", judge, cost=10, weight=2),
        VerifierSpec("oracle", oracle, cost=100),
    ], pass_threshold=0.6)
    report = ensemble.evaluate(candidate, max_cost=120)
    report.passed, report.confidence, report.disagreement, report.cost

Cascade stopping is intentionally conservative: it stops only when the weighted
score is far from the threshold and at least ``min_votes`` verifiers ran. A tie
or a near-threshold score buys the next verifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "VerifierSpec",
    "VerifierVote",
    "EnsembleReport",
    "VerifierEnsemble",
]

Verifier = Callable[[Any], Any]


@dataclass(frozen=True)
class VerifierSpec:
    """One verifier, its cost unit and its voting weight."""

    name: str
    verify: Verifier
    cost: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("VerifierSpec.name must be a non-empty string")
        if not callable(self.verify):
            raise TypeError("VerifierSpec.verify must be callable")
        for label, value in (("cost", self.cost), ("weight", self.weight)):
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"VerifierSpec.{label} must be finite and >= 0")
        if self.weight == 0:
            raise ValueError("VerifierSpec.weight must be > 0")


@dataclass(frozen=True)
class VerifierVote:
    """A normalised verifier response."""

    name: str
    score: float
    passed: bool
    cost: float
    weight: float
    reason: str = ""
    explicit: bool = False
    error: bool = False


@dataclass(frozen=True)
class EnsembleReport:
    """The votes, the weighted decision and what it cost."""

    votes: Tuple[VerifierVote, ...]
    score: float
    passed: bool
    confidence: float
    disagreement: float
    cost: float
    stopped_early: bool
    complete: bool = True
    errors: int = 0
    detail: str = ""

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(vote.name for vote in self.votes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "cost": self.cost,
            "stopped_early": self.stopped_early,
            "complete": self.complete,
            "errors": self.errors,
            "votes": [
                {"name": v.name, "score": v.score, "passed": v.passed,
                 "cost": v.cost, "weight": v.weight, "reason": v.reason}
                for v in self.votes
            ],
        }


def _finite_score(value: Any, name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"verifier {name!r} returned a non-numeric score") from None
    if not math.isfinite(score):
        raise ValueError(f"verifier {name!r} returned a non-finite score")
    return max(0.0, min(1.0, score))


def _normalise(value: Any, name: str) -> Tuple[float, bool, str, bool]:
    if isinstance(value, bool):
        return (1.0 if value else 0.0), value, "", False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = _finite_score(value, name)
        return score, score >= 0.5, "", False
    if isinstance(value, Mapping):
        score = value.get("score")
        passed = value.get("passed", value.get("accept"))
        reason = str(value.get("reason", ""))
        if passed is not None and not isinstance(passed, bool):
            raise TypeError(f"verifier {name!r} returned a non-bool passed/accept")
        if score is None and passed is not None:
            score = 1.0 if passed else 0.0
        if score is None:
            raise ValueError(f"verifier {name!r} returned a mapping without score/passed")
        score = _finite_score(score, name)
        return score, (passed if passed is not None else score >= 0.5), reason, passed is not None
    for attr in ("score", "confidence"):
        score = getattr(value, attr, None)
        if score is not None:
            score = _finite_score(score, name)
            passed = getattr(value, "passed", getattr(value, "accept", None))
            explicit = isinstance(passed, bool)
            if passed is None:
                passed = score >= 0.5
            return score, bool(passed), str(getattr(value, "reason", "")), explicit
    for attr in ("passed", "accept"):
        passed = getattr(value, attr, None)
        if isinstance(passed, bool):
            return (1.0 if passed else 0.0), passed, str(getattr(value, "reason", "")), True
    raise TypeError(f"verifier {name!r} returned unsupported {type(value).__name__}")


class VerifierEnsemble:
    """A cost-aware ensemble of verifiers.

    ``strategy`` is ``cascade`` (default), ``weighted``, ``majority`` or
    ``unanimous``. ``cascade`` sorts by cost and may stop early; the other
    strategies run every verifier.
    """

    def __init__(self, verifiers: Sequence[VerifierSpec], *,
                 pass_threshold: float = 0.5,
                 min_confidence: float = 0.5,
                 min_votes: int = 2,
                 strategy: str = "cascade") -> None:
        if not verifiers:
            raise ValueError("VerifierEnsemble needs at least one verifier")
        names = [spec.name for spec in verifiers]
        if len(set(names)) != len(names):
            raise ValueError(f"verifier names must be unique: {names}")
        if strategy not in ("cascade", "weighted", "majority", "unanimous"):
            raise ValueError("strategy must be cascade, weighted, majority or unanimous")
        threshold = float(pass_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if isinstance(min_votes, bool) or not isinstance(min_votes, int) or min_votes < 1:
            raise ValueError("min_votes must be an int >= 1")
        self.verifiers = tuple(verifiers)
        self.pass_threshold = threshold
        self.min_confidence = confidence
        self.min_votes = min_votes
        self.strategy = strategy

    def _vote(self, spec: VerifierSpec, candidate: Any) -> VerifierVote:
        try:
            value = spec.verify(candidate)
        except Exception as exc:  # noqa: BLE001 - a verifier failure is a vote
            return VerifierVote(spec.name, 0.0, False, float(spec.cost),
                                float(spec.weight), f"{type(exc).__name__}: {exc}",
                                explicit=True, error=True)
        score, passed, reason, explicit = _normalise(value, spec.name)
        return VerifierVote(spec.name, score, passed, float(spec.cost),
                            float(spec.weight), reason, explicit=explicit)

    def _aggregate(self, votes: Sequence[VerifierVote]) -> float:
        total = sum(vote.weight for vote in votes)
        if total <= 0:
            return 0.0
        return sum(vote.score * vote.weight for vote in votes) / total

    def _decision(self, votes: Sequence[VerifierVote]) -> bool:
        if self.strategy == "unanimous":
            return bool(votes) and all(vote.passed for vote in votes)
        if self.strategy == "majority":
            yes = sum(vote.weight for vote in votes if vote.passed)
            no = sum(vote.weight for vote in votes if not vote.passed)
            return yes > no
        # An explicit fail vetoes a score-based strategy; a verifier that says
        # "reject" must not be outvoted by its own score field.
        if any(vote.explicit and not vote.passed for vote in votes):
            return False
        return self._aggregate(votes) >= self.pass_threshold

    def evaluate(self, candidate: Any, *, max_cost: Optional[float] = None) -> EnsembleReport:
        """Run the strategy and return the decision plus its uncertainty."""
        if max_cost is not None:
            max_cost = float(max_cost)
            if not math.isfinite(max_cost) or max_cost < 0:
                raise ValueError("max_cost must be finite and >= 0")
        ordered = sorted(self.verifiers, key=lambda spec: (spec.cost, spec.name))
        votes: List[VerifierVote] = []
        cost = 0.0
        stopped_early = False
        budget_blocked = False
        for spec in ordered:
            if max_cost is not None and cost + float(spec.cost) > max_cost:
                budget_blocked = True
                stopped_early = True
                if self.strategy == "cascade":
                    break
                continue
            vote = self._vote(spec, candidate)
            votes.append(vote)
            cost += vote.cost
            if self.strategy != "cascade" or len(votes) < self.min_votes:
                continue
            score = self._aggregate(votes)
            confidence = abs(score - self.pass_threshold) / max(
                self.pass_threshold, 1.0 - self.pass_threshold, 1e-9)
            if confidence >= self.min_confidence:
                # A confident decision may stop even if more verifiers exist.
                stopped_early = len(votes) < len(ordered)
                break
        score = self._aggregate(votes)
        passed = self._decision(votes)
        yes = sum(vote.weight for vote in votes if vote.passed)
        no = sum(vote.weight for vote in votes if not vote.passed)
        disagreement = 0.0 if (yes + no) <= 0 else min(yes, no) / (yes + no)
        confidence = abs(score - self.pass_threshold) / max(
            self.pass_threshold, 1.0 - self.pass_threshold, 1e-9)
        complete = not budget_blocked or (self.strategy == "cascade" and not budget_blocked)
        if budget_blocked and self.strategy == "cascade":
            # A budget stop before a confident decision cannot be called complete.
            complete = False
        if self.strategy != "cascade" and budget_blocked:
            complete = False
            passed = False
        errors = sum(1 for vote in votes if vote.error)
        detail = (f"{len(votes)}/{len(ordered)} verifier(s), cost={cost:g}, "
                  f"score={score:.4f}, disagreement={disagreement:.3f}"
                  + ("" if complete else ", incomplete panel"))
        return EnsembleReport(votes=tuple(votes), score=score, passed=passed,
                              confidence=confidence, disagreement=disagreement,
                              cost=cost, stopped_early=stopped_early,
                              complete=complete, errors=errors, detail=detail)

    def __call__(self, candidate: Any, *, max_cost: Optional[float] = None) -> EnsembleReport:
        return self.evaluate(candidate, max_cost=max_cost)
