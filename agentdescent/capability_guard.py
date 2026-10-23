"""A protected-capability gate for recursive self-modification.

An evolution run judges a candidate on the held-out slice of the task it is
improving. That is the right test for *this* capability and the wrong one for a
recursive system: a self-modification that raises the target can silently reduce
a capability the run was not measuring (catastrophic forgetting). The framework
already has the per-run gate (`Policies(acceptance=...)`); what it lacked was a
way to say "and do not regress these other tasks".

:class:`CapabilityGuard` is an :class:`~agentdescent.policies.AcceptancePolicy`
that evaluates the **base artifact and the candidate** on every protected
suite, rejects if any suite falls below its floor (or below the base by more
than its tolerance), and otherwise delegates to the inner acceptance policy.
With no inner policy it keeps the candidate only when the full held-out rate did
not fall.

    Policies(acceptance=CapabilityGuard(
        [CapabilitySuite("tool-use", tasks=tool_tasks, floor=0.85),
         CapabilitySuite("json", tasks=json_tasks, tolerance=0.01)],
        inner=my_acceptance))

The suite tasks use the run's own scorer and agent; a suite is a *task set*, not
a second verifier. That keeps the guard model-agnostic and cheap when the
candidate is already cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from .advantage import _ForwardsInstall
from .policies import AcceptDecision, MergeContext

__all__ = ["CapabilityGuard", "CapabilityReport", "CapabilitySuite"]


@dataclass(frozen=True)
class CapabilitySuite:
    """Tasks a self-modification must not regress, and the bar it must clear."""

    name: str
    tasks: Sequence[Any]
    #: Absolute floor for the candidate's mean reward on this suite.
    floor: float = 0.0
    #: How far below the base artifact the candidate may fall.
    tolerance: float = 0.02


@dataclass(frozen=True)
class CapabilityReport:
    """What the protected suites measured for one candidate."""

    #: ``name -> {"base", "candidate", "floor"}``.
    suites: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Suite names that failed their floor or tolerance.
    regressions: Tuple[str, ...] = ()
    detail: str = ""

    @property
    def passed(self) -> bool:
        return not self.regressions


class CapabilityGuard(_ForwardsInstall):
    """Reject a self-modification that regresses a protected capability.

    ``inner`` is the acceptance policy that would have run without the guard;
    it is consulted only after every suite passes, so the guard can only make
    acceptance stricter. ``inner=None`` is the shipped ``DefaultAcceptance``,
    configured by the aggregator through the forwarded ``configure`` hook -- so
    a guard with no explicit inner is never *looser* than the default.

    Suite scores go through the bound verifier's ``eval_fn`` when the engine
    installed one, and through ``artifact.score(tasks)`` otherwise. A custom
    ``Evolvable`` that provides neither raises :class:`ContractError` naming the
    requirement; ``score`` is not part of the ``Evolvable`` protocol.
    """

    def __init__(self, suites: Sequence[CapabilitySuite], *,
                 inner: Optional[Any] = None) -> None:
        if not suites:
            raise ValueError("CapabilityGuard needs at least one protected suite")
        names = [suite.name for suite in suites]
        if len(set(names)) != len(names):
            raise ValueError(f"CapabilityGuard suite names must be unique: {names}")
        from .defaults import DefaultAcceptance

        self.suites = tuple(suites)
        self.inner = inner if inner is not None else DefaultAcceptance()
        self.verifier: Any = None

    def bind(self, verifier) -> None:
        self.verifier = verifier
        super().bind(verifier)

    def _score(self, artifact: Any, tasks: Sequence[Any]) -> float:
        eval_fn = getattr(self.verifier, "eval_fn", None)
        if callable(eval_fn):
            return float(eval_fn(artifact, list(tasks)))
        score = getattr(artifact, "score", None)
        if not callable(score):
            from .evolvable import ContractError

            raise ContractError(
                "CapabilityGuard needs a bound verifier with eval_fn(artifact, "
                "tasks) or an artifact with score(tasks); this Evolvable has "
                "neither")
        return float(score(list(tasks)))

    def measure(self, artifact: Any) -> Dict[str, float]:
        """Mean reward per protected suite for one artifact."""
        return {suite.name: self._score(artifact, suite.tasks)
                for suite in self.suites}

    def check(self, base: Any, candidate: Any) -> CapabilityReport:
        """Evaluate both artifacts and say which suites the candidate broke."""
        report: Dict[str, Dict[str, float]] = {}
        regressions = []
        for suite in self.suites:
            before = self._score(base, suite.tasks)
            after = self._score(candidate, suite.tasks)
            # The absolute floor is a minimum for a suite already above it; a
            # suite below its floor may climb toward it, but may not fall.
            bar = (max(float(suite.floor), before - float(suite.tolerance))
                   if before >= float(suite.floor)
                   else before - float(suite.tolerance))
            report[suite.name] = {"base": before, "candidate": after,
                                  "bar": bar}
            if after < bar:
                regressions.append(suite.name)
        detail = ""
        if regressions:
            detail = "; ".join(
                f"{name}: {report[name]['candidate']:.3f} < "
                f"{report[name]['bar']:.3f} (base {report[name]['base']:.3f})"
                for name in regressions)
            detail = "protected capability regressed -- " + detail
        return CapabilityReport(suites=report,
                                regressions=tuple(regressions), detail=detail)

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        report = self.check(ctx.artifact, ctx.candidate)
        if not report.passed:
            deltas = [report.suites[name]["candidate"] - report.suites[name]["base"]
                      for name in report.regressions]
            return AcceptDecision(False, "capability-regression", report.detail,
                                  observed_delta=min(deltas) if deltas else 0.0)
        if hasattr(self.inner, "accept"):
            return self.inner.accept(ctx)
        return self.inner(ctx)
