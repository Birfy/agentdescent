"""Compose acceptance guards into one deterministic pipeline.

The framework ships several independent checks -- leakage, capability
regression, verifier/oracle agreement, human approval, the default statistical
gate. Stacking them by hand invites two mistakes: the order is implicit, and the
reason a candidate was refused is lost. ``GuardPipeline`` is the one place the
order lives. It runs each guard in sequence, stops at the first refusal (by
default), and returns the named decision so a report can say which ring fired.

    from agentdescent.defaults import DefaultAcceptance

    def no_gold_in_candidate(ctx):
        ...  # a guard you supply; the shipped guards are separate modules
        return AcceptDecision(True)

    Policies(acceptance=GuardPipeline([
        ("leakage", no_gold_in_candidate),
        ("stats", DefaultAcceptance()),
    ]))

Guards are ordinary :class:`~agentdescent.policies.AcceptancePolicy` objects (or
callables with the same signature). The pipeline is itself an acceptance policy,
so it plugs into every adoption path and the install hooks are forwarded to
every guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

from .advantage import _ForwardsInstall
from .policies import AcceptDecision, MergeContext

__all__ = ["GuardDecision", "GuardPipeline", "GuardPipelineResult"]


@dataclass(frozen=True)
class GuardDecision:
    """One guard's name and decision."""

    name: str
    decision: AcceptDecision


@dataclass(frozen=True)
class GuardPipelineResult:
    """The full ordered trail, including the first refusal when there is one."""

    accepted: bool
    decisions: Tuple[GuardDecision, ...] = ()
    failed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "failed_at": self.failed_at,
            "decisions": [
                {"name": item.name,
                 "accept": item.decision.accept,
                 "category": item.decision.category,
                 "detail": item.decision.detail,
                 "observed_delta": item.decision.observed_delta}
                for item in self.decisions
            ],
        }


class GuardPipeline(_ForwardsInstall):
    """Run guards in order; the first refusal is the pipeline's decision.

    ``collect_all`` keeps evaluating after a refusal so a report can show every
    ring's state; acceptance still requires every guard to pass. An empty
    pipeline accepts (it is the identity gate), which makes it safe to add one
    before deciding which guards to install.
    """

    def __init__(self, guards: Sequence[Tuple[str, Any]], *,
                 collect_all: bool = False) -> None:
        names = [name for name, _ in guards]
        if len(set(names)) != len(names):
            raise ValueError(f"guard names must be unique: {names}")
        self.guards = tuple((str(name), guard) for name, guard in guards)
        self.collect_all = bool(collect_all)

    def evaluate(self, ctx: MergeContext) -> GuardPipelineResult:
        decisions: List[GuardDecision] = []
        failed_at = ""
        for name, guard in self.guards:
            decision = (guard.accept(ctx) if hasattr(guard, "accept")
                        else guard(ctx))
            if not isinstance(decision, AcceptDecision):
                raise TypeError(
                    f"guard {name!r} returned {type(decision).__name__}; an "
                    "acceptance guard must return AcceptDecision")
            decisions.append(GuardDecision(name, decision))
            if not decision.accept:
                failed_at = name
                if not self.collect_all:
                    break
        return GuardPipelineResult(accepted=not failed_at,
                                   decisions=tuple(decisions),
                                   failed_at=failed_at)

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        result = self.evaluate(ctx)
        if result.accepted:
            names = ", ".join(name for name, _ in self.guards) or "none"
            return AcceptDecision(True, detail=f"guards passed ({names})")
        failed = next(item.decision for item in result.decisions
                      if item.name == result.failed_at)
        category = _merge_category(failed.category)
        detail = (f"{result.failed_at}: "
                  f"{failed.detail or 'refused'}")
        return AcceptDecision(
            False, category=category, detail=detail,
            observed_delta=failed.observed_delta)

    def bind(self, verifier) -> None:
        for _, guard in self.guards:
            fn = getattr(guard, "bind", None)
            if callable(fn):
                fn(verifier)

    def configure(self, config) -> None:
        for _, guard in self.guards:
            fn = getattr(guard, "configure", None)
            if callable(fn):
                fn(config)

    def describe(self) -> Tuple[str, ...]:
        """The guard names, in run order -- what a plan/report echoes."""
        return tuple(name for name, _ in self.guards)

    def checkpoint_state(self):
        """Checkpoint every nested policy that opts into the hook."""
        states = {}
        for name, guard in self.guards:
            hook = getattr(guard, "checkpoint_state", None)
            if callable(hook):
                states[name] = hook()
        return states or None

    def restore_state(self, state) -> None:
        """Restore the nested policies that saved state in ``checkpoint_state``."""
        if not isinstance(state, dict):
            return
        for name, guard in self.guards:
            if name not in state:
                continue
            hook = getattr(guard, "restore_state", None)
            if callable(hook):
                hook(state[name])

    def on_screen_reject(self) -> None:
        """Forward the aggregator's screen-reject accounting to every guard."""
        for _, guard in self.guards:
            hook = getattr(guard, "on_screen_reject", None)
            if callable(hook):
                hook()


def _merge_category(category: str) -> str:
    """Keep a real merge category; otherwise use the generic rejection one."""
    from .aggregator import MergeOutcome

    try:
        return MergeOutcome(category).value
    except (TypeError, ValueError):
        return MergeOutcome.BELOW_THRESHOLD.value
