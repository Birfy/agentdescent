"""Structured multi-agent deliberation with a confidence-weighted judge panel.

Multi-agent debate is not "ask three agents and majority-vote". A useful
protocol needs roles, rounds, evidence, judge confidence and a stopping rule.
This module is that protocol slice:

* :class:`Argument` is one debater's claim + evidence + confidence;
* :class:`JudgeVote` is one judge's winner + confidence + weight;
* :class:`Deliberation` runs rounds, accumulates weighted votes, measures
  agreement, and stops when consensus is reached or the round budget ends.

Argument confidence/evidence are carried for the audit trail and for judges to
read; they are not folded into the tally themselves. Only judge votes (weight ×
confidence) decide the winner and agreement.

The debater/judge callables are model-agnostic. A judge can be an LLM, a
calibrated `self_model`, a human reviewer, or a test double.

    report = Deliberation(DebateConfig(rounds=3, consensus_threshold=0.7)).run(
        topic, {"pro": pro_agent, "con": con_agent}, [judge_a, judge_b])
    report.winner, report.agreement, report.rounds

The report is the audit trail; it does not hide disagreement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Argument",
    "JudgeVote",
    "DebateConfig",
    "DebateRound",
    "DeliberationReport",
    "Deliberation",
]

Debater = Callable[[Any, Sequence["DebateRound"], int], Any]
Judge = Callable[[Any, Sequence["Argument"], int], Any]


def _confidence(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _weight(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("weight must be a number, not bool")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("weight must be finite and > 0")
    return number


@dataclass(frozen=True)
class Argument:
    agent: str
    position: str
    claim: str
    evidence: Tuple[str, ...] = ()
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if not self.agent or not self.position or not self.claim:
            raise ValueError("agent, position and claim are required")
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        object.__setattr__(self, "evidence", tuple(str(item) for item in self.evidence))


@dataclass(frozen=True)
class JudgeVote:
    judge: str
    winner: str
    confidence: float = 0.5
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.judge or not self.winner:
            raise ValueError("judge and winner are required")
        object.__setattr__(self, "confidence", _confidence(self.confidence, "confidence"))
        object.__setattr__(self, "weight", _weight(self.weight))


@dataclass(frozen=True)
class DebateConfig:
    rounds: int = 3
    min_rounds: int = 1
    consensus_threshold: float = 0.7

    def __post_init__(self) -> None:
        for name in ("rounds", "min_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int")
        if self.rounds < 1 or self.min_rounds < 1 or self.min_rounds > self.rounds:
            raise ValueError("need 1 <= min_rounds <= rounds")
        _confidence(self.consensus_threshold, "consensus_threshold")


@dataclass(frozen=True)
class DebateRound:
    index: int
    arguments: Tuple[Argument, ...]
    votes: Tuple[JudgeVote, ...]
    tally: Mapping[str, float]

    @property
    def winner(self) -> str:
        return max(self.tally, key=lambda key: (self.tally[key], key)) if self.tally else ""


@dataclass(frozen=True)
class DeliberationReport:
    winner: str
    agreement: float
    confidence: float
    rounds: Tuple[DebateRound, ...]
    stopped_reason: str
    detail: str = ""

    @property
    def consensus(self) -> bool:
        return self.stopped_reason == "consensus"


def _argument(value: Any, agent: str) -> Argument:
    if isinstance(value, Argument):
        if value.agent != agent:
            raise ValueError(f"debater {agent!r} returned an argument for {value.agent!r}")
        return value
    if isinstance(value, str):
        return Argument(agent, agent, value)
    raise TypeError(f"debater {agent!r} returned {type(value).__name__}")


def _vote(value: Any, judge: str) -> JudgeVote:
    if isinstance(value, JudgeVote):
        if value.judge == judge:
            return value
        return JudgeVote(judge, value.winner, value.confidence, value.weight)
    if isinstance(value, str):
        return JudgeVote(judge, value, 0.5, 1.0)
    if isinstance(value, Mapping):
        winner = value.get("winner")
        if not winner:
            raise ValueError(f"judge {judge!r} returned a mapping without 'winner'")
        return JudgeVote(judge, str(winner),
                         value.get("confidence", 0.5),
                         value.get("weight", 1.0))
    raise TypeError(f"judge {judge!r} returned {type(value).__name__}")


class Deliberation:
    """Run a fixed-role debate and a weighted judge panel."""

    def __init__(self, config: Optional[DebateConfig] = None) -> None:
        self.config = config or DebateConfig()

    def _tally(self, votes: Sequence[JudgeVote]) -> Dict[str, float]:
        tally: Dict[str, float] = {}
        for vote in votes:
            tally[vote.winner] = tally.get(vote.winner, 0.0) + vote.weight * vote.confidence
        return tally

    def run(self, topic: Any, debaters: Mapping[str, Debater],
            judges: Sequence[Judge]) -> DeliberationReport:
        if not debaters:
            raise ValueError("run needs at least one debater")
        if not judges:
            raise ValueError("run needs at least one judge")
        if not all(callable(fn) for fn in debaters.values()) or not all(callable(fn) for fn in judges):
            raise TypeError("debaters and judges must be callable")
        rounds: List[DebateRound] = []
        stopped = "rounds"
        for index in range(1, self.config.rounds + 1):
            arguments = tuple(_argument(fn(topic, tuple(rounds), index), name)
                              for name, fn in debaters.items())
            votes = tuple(_vote(fn(topic, arguments, index), f"judge-{i}")
                          for i, fn in enumerate(judges))
            tally = self._tally(votes)
            round_ = DebateRound(index, arguments, votes, dict(sorted(tally.items())))
            rounds.append(round_)
            total = sum(tally.values())
            agreement = (max(tally.values()) / total) if total > 0 else 0.0
            if index >= self.config.min_rounds and agreement >= self.config.consensus_threshold:
                stopped = "consensus"
                break
        last = rounds[-1]
        total = sum(last.tally.values())
        agreement = (max(last.tally.values()) / total) if total > 0 else 0.0
        winner = last.winner
        confidence = 0.0
        if winner:
            votes = [vote for vote in last.votes if vote.winner == winner]
            total_weight = sum(vote.weight for vote in votes)
            if total_weight > 0:
                confidence = (sum(vote.confidence * vote.weight for vote in votes) /
                              total_weight)
        detail = (f"{len(rounds)} round(s), winner={winner or 'none'}, "
                  f"agreement={agreement:.3f}, confidence={confidence:.3f}")
        return DeliberationReport(winner, agreement, confidence, tuple(rounds),
                                  stopped, detail)
