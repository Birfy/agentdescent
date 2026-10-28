"""A role-based blackboard runtime for multi-agent self-improvement.

`agent_society` evolves a *topology* as an artifact. That is the search
question. This module is the runtime question next to it: given a fixed set of
roles -- proposer, critic, verifier, historian -- run them against a shared
blackboard, enforce a budget, stop on consensus, and record who contributed.

The distinction matters. Evolving the graph finds which agents should exist;
the runtime makes the existing ones cooperate without every caller rebuilding
the loop. It is deliberately model-agnostic: an actor is any callable
``(task, blackboard, history) -> Contribution | str | None``.

    runtime = SocietyRuntime([
        RoleSpec("proposer", "proposer", propose),
        RoleSpec("critic", "critic", critique),
        RoleSpec("verifier", "verifier", verify),
    ], SocietyConfig(max_turns=6, max_cost=20))
    report = runtime.run(task)
    report.consensus, report.role_credit, report.cost

Consensus is deliberately simple: at least one proposal, at least one verifier
vote, and the weighted verifier approval at or above the threshold. A caller
with a numeric evaluator can pass ``score_fn`` instead, and the runtime stops
when the best proposal clears its target. The report's ``role_credit`` is a
normalized approved-weight share; pass ``attribute_score`` to compute a Shapley
attribution through :mod:`agentdescent.credit` when the caller has a replayable
scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import re

from .credit import Edit, CreditReport, shapley_credit

__all__ = [
    "RoleSpec",
    "Contribution",
    "SocietyConfig",
    "SocietyReport",
    "SocietyRuntime",
]

Actor = Callable[[Any, Mapping[str, Any], Sequence["Contribution"]], Any]
ScoreFn = Callable[[Any, Sequence["Contribution"]], float]
AttributeScore = Callable[[Sequence["Contribution"]], float]


@dataclass(frozen=True)
class RoleSpec:
    """One cooperating role and its actor."""

    name: str
    kind: str
    act: Actor
    budget: Optional[int] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("RoleSpec.name must be a non-empty string")
        if self.kind not in ("proposer", "critic", "verifier", "historian"):
            raise ValueError("RoleSpec.kind must be proposer, critic, verifier or historian")
        if not callable(self.act):
            raise TypeError("RoleSpec.act must be callable")
        if self.budget is not None and (
                isinstance(self.budget, bool) or not isinstance(self.budget, int)
                or self.budget < 1):
            raise ValueError("RoleSpec.budget must be an int >= 1 or None")
        if not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ValueError("RoleSpec.weight must be finite and > 0")


@dataclass(frozen=True)
class Contribution:
    """One actor's output on the blackboard."""

    role: str
    kind: str
    content: Any = None
    score: Optional[float] = None
    cost: float = 0.0
    approved: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role or not self.kind:
            raise ValueError("Contribution.role and kind must be non-empty")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("Contribution.score must be finite")
        if not math.isfinite(float(self.cost)) or float(self.cost) < 0:
            raise ValueError("Contribution.cost must be finite and >= 0")


@dataclass(frozen=True)
class SocietyConfig:
    """Bounds and the consensus rule."""

    max_turns: int = 8
    max_cost: Optional[float] = None
    consensus_threshold: float = 1.0
    target_score: Optional[float] = None
    require_verifier: bool = True

    # max_cost is a soft limit: it is checked before and after each actor call,
    # and actors can read board["remaining_cost"] before deciding how much to
    # spend. A single contribution can still cross the limit.

    def __post_init__(self) -> None:
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) \
                or self.max_turns < 1:
            raise ValueError("max_turns must be an int >= 1")
        if self.max_cost is not None:
            cost = float(self.max_cost)
            if not math.isfinite(cost) or cost <= 0:
                raise ValueError("max_cost must be finite and > 0 or None")
        threshold = float(self.consensus_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("consensus_threshold must be in [0, 1]")
        if self.target_score is not None:
            target = float(self.target_score)
            if not math.isfinite(target):
                raise ValueError("target_score must be finite or None")


@dataclass(frozen=True)
class SocietyReport:
    """Contributions, consensus and the role-credit view."""

    contributions: Tuple[Contribution, ...]
    consensus: bool
    score: Optional[float]
    cost: float
    turns: int
    stopped_reason: str
    role_credit: Dict[str, float]
    shapley: Optional[CreditReport] = None
    detail: str = ""

    @property
    def proposals(self) -> Tuple[Contribution, ...]:
        return tuple(c for c in self.contributions if c.kind == "proposal")

    @property
    def verdicts(self) -> Tuple[Contribution, ...]:
        return tuple(c for c in self.contributions if c.kind == "verdict")


_KIND = {"proposer": "proposal", "critic": "critique",
         "verifier": "verdict", "historian": "history"}


def _approval(text: str) -> Optional[bool]:
    words = re.findall(r"[a-z]+", text.strip().lower())
    if not words:
        return None
    first = words[0]
    if first in ("approve", "approved", "accept", "accepted", "yes", "lgtm"):
        return True
    if first in ("reject", "rejected", "deny", "denied", "no", "fail", "failed"):
        return False
    if len(words) > 1 and first in ("looks", "seems"):
        if words[1] in ("good", "fine", "ok", "correct", "right", "approved"):
            return True
        if words[1] in ("bad", "wrong", "incorrect", "broken"):
            return False
    return None


def _normalise(role: RoleSpec, value: Any) -> Optional[Contribution]:
    if value is None:
        return None
    expected = _KIND[role.kind]
    if isinstance(value, Contribution):
        if value.role != role.name:
            raise ValueError(
                f"actor {role.name!r} returned a contribution for role {value.role!r}")
        if value.kind != expected:
            raise ValueError(
                f"actor {role.name!r} returned kind {value.kind!r}, expected {expected!r}")
        return value
    if isinstance(value, str):
        return Contribution(role.name, expected, content=value,
                            approved=(_approval(value) if role.kind == "verifier" else None))
    raise TypeError(
        f"actor {role.name!r} returned {type(value).__name__}; expected "
        "Contribution, str or None")


class SocietyRuntime:
    """Run roles round-robin against a shared blackboard."""

    def __init__(self, roles: Sequence[RoleSpec], config: Optional[SocietyConfig] = None,
                 *, decision_log: Optional[Any] = None) -> None:
        if not roles:
            raise ValueError("SocietyRuntime needs at least one role")
        names = [role.name for role in roles]
        if len(set(names)) != len(names):
            raise ValueError(f"role names must be unique: {names}")
        if config is None:
            config = SocietyConfig()
        if config.require_verifier and not any(role.kind == "verifier" for role in roles):
            raise ValueError("require_verifier=True needs at least one verifier role")
        if not any(role.kind == "proposer" for role in roles):
            raise ValueError("SocietyRuntime needs at least one proposer role")
        if roles[0].kind != "proposer":
            raise ValueError("role order must start with a proposer; verifiers follow")
        self.roles = tuple(roles)
        self.config = config
        self.decision_log = decision_log

    def _score(self, contributions: Sequence[Contribution]) -> Optional[float]:
        proposals = [c for c in contributions if c.kind == "proposal" and c.score is not None]
        if not proposals:
            return None
        return max(float(c.score) for c in proposals)

    def _latest_proposal(self, contributions: Sequence[Contribution]) -> Optional[int]:
        for index in range(len(contributions) - 1, -1, -1):
            if contributions[index].kind == "proposal":
                return index
        return None

    def _consensus(self, contributions: Sequence[Contribution],
                   score: Optional[float] = None) -> bool:
        latest = self._latest_proposal(contributions)
        if latest is None:
            return False
        window = contributions[latest + 1:]
        if any(c.kind == "critique" and c.approved is False for c in window):
            return False
        verdicts = [c for c in window if c.kind == "verdict"]
        if self.config.require_verifier and not verdicts:
            return False
        if not verdicts:
            return False
        weights = {role.name: role.weight for role in self.roles}
        total = sum(weights.get(c.role, 1.0) for c in verdicts)
        yes = sum(weights.get(c.role, 1.0) for c in verdicts if c.approved)
        return total > 0 and yes / total >= self.config.consensus_threshold

    def _role_credit(self, contributions: Sequence[Contribution],
                     consensus: bool) -> Dict[str, float]:
        """Participation share in the current proposal window, not approval count."""
        weights = {role.name: role.weight for role in self.roles}
        window = list(contributions)
        if consensus:
            latest = self._latest_proposal(contributions)
            if latest is not None:
                window = list(contributions[latest:])
        total = sum(weights.get(c.role, 1.0) for c in window)
        if total <= 0:
            return {role.name: 0.0 for role in self.roles}
        out = {role.name: 0.0 for role in self.roles}
        for contribution in window:
            out[contribution.role] = out.get(contribution.role, 0.0) + (
                weights.get(contribution.role, 1.0) / total)
        return out

    def run(self, task: Any, *, blackboard: Optional[Dict[str, Any]] = None,
            score_fn: Optional[ScoreFn] = None,
            attribute_score: Optional[AttributeScore] = None) -> SocietyReport:
        """Run until consensus, budget, or ``max_turns``."""
        board: Dict[str, Any] = {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in dict(blackboard or {}).items()}
        contributions: List[Contribution] = []
        uses = {role.name: 0 for role in self.roles}
        cost = 0.0
        stopped = "max-turns"
        turns = 0
        score: Optional[float] = None
        for turn in range(self.config.max_turns):
            turns = turn + 1
            progressed = False
            for role in self.roles:
                budget = role.budget if role.budget is not None else self.config.max_turns
                if uses[role.name] >= budget:
                    continue
                if self.config.max_cost is not None and cost >= self.config.max_cost:
                    stopped = "budget"
                    break
                if self.config.max_cost is not None:
                    board["remaining_cost"] = max(0.0, self.config.max_cost - cost)
                value = role.act(task, board, tuple(contributions))
                uses[role.name] += 1
                contribution = _normalise(role, value)
                if contribution is None:
                    continue
                progressed = True
                contributions.append(contribution)
                cost += float(contribution.cost)
                board.setdefault("contributions", []).append(contribution)
                if contribution.kind == "proposal":
                    board["latest_proposal"] = contribution.content
                if contribution.kind == "verdict":
                    board.setdefault("verdicts", []).append(contribution)
                if self.config.max_cost is not None and cost >= self.config.max_cost:
                    stopped = "budget"
                    break
            if stopped == "budget":
                break
            score = self._score(contributions)
            if score_fn is not None:
                score = float(score_fn(task, tuple(contributions)))
                if not math.isfinite(score):
                    raise ValueError("score_fn returned a non-finite score")
            if self._consensus(contributions, score):
                stopped = "consensus"
                break
            if self.config.target_score is not None and score is not None \
                    and score >= self.config.target_score:
                stopped = "target"
                break
            if not progressed:
                stopped = "stalled"
                break
        if score is None:
            score = self._score(contributions)
        shapley = None
        if attribute_score is not None and contributions:
            edits = [Edit(c.role + ":" + str(i), payload=c)
                     for i, c in enumerate(contributions)]

            def evaluate_subset(subset: Sequence[Edit]) -> float:
                return float(attribute_score([edit.payload for edit in subset]))

            shapley = shapley_credit(edits, evaluate_subset, permutations=32,
                                     seed=0, exact_limit=6)
        report = SocietyReport(
            contributions=tuple(contributions),
            consensus=stopped in ("consensus", "target"),
            score=score, cost=cost, turns=turns, stopped_reason=stopped,
            role_credit=self._role_credit(contributions, stopped in ("consensus", "target")),
            shapley=shapley,
            detail=(f"{len(contributions)} contribution(s), {turns} turn(s), "
                    f"cost={cost:g}, stopped={stopped}"))
        if self.decision_log is not None:
            try:
                self.decision_log.record(
                    kind="society", artifact_id="society-runtime",
                    accepted=report.consensus, category=stopped,
                    detail=report.detail,
                    evidence={"score": score, "roles": report.role_credit},
                    cost={"society_cost": cost})
            except Exception:  # noqa: BLE001
                pass
        return report
