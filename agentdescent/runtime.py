"""A guarded RSI session: compose, inject acceptance, run, record.

The pieces exist separately: ``compose(spec)`` builds the engine call, a guard
pipeline composes acceptance checks, a decision log records the decision and a
kill switch is an operational stop. This module wires them into one call so a
product does not reimplement the order.

``acceptance`` is any ``AcceptancePolicy`` -- typically a
``GuardPipeline`` -- and is injected into the composed ``Policies``. The guard
runs **before** the spec's own acceptance policy, or before the default
statistical gate when the spec has none: the pipeline can make acceptance
stricter, never replace a rule the spec asked for (or the shipped default).
``decision_log`` and ``kill_switch`` are duck-typed (``.record(...)`` and
``.engaged()``/``.reason()``), so this module does not depend on a particular
implementation.

An algorithm that installs an ``aggregator_factory`` owns its merge path; the
injected acceptance is then not consulted, and ``GuardedRun.notes`` says so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

from .aggregator import AggregatorConfig
from .defaults import DefaultAcceptance
from .evolvespec import EvolveSpec, build_agg_config, compose
from .policies import AcceptDecision, Policies

__all__ = ["GuardedRun", "run_guarded"]


class _Chain:
    """Run one acceptance policy, then the spec's own; no imports needed."""

    def __init__(self, first: Any, second: Any) -> None:
        self.first = first
        self.second = second

    def accept(self, ctx):
        decision = (self.first.accept(ctx) if hasattr(self.first, "accept")
                    else self.first(ctx))
        if not isinstance(decision, AcceptDecision):
            raise TypeError("acceptance returned a non-AcceptDecision")
        if not decision.accept:
            return decision
        second = (self.second.accept(ctx) if hasattr(self.second, "accept")
                  else self.second(ctx))
        if not isinstance(second, AcceptDecision):
            raise TypeError("spec acceptance returned a non-AcceptDecision")
        return second

    def bind(self, verifier) -> None:
        for policy in (self.first, self.second):
            fn = getattr(policy, "bind", None)
            if callable(fn):
                fn(verifier)

    def configure(self, config) -> None:
        for policy in (self.first, self.second):
            fn = getattr(policy, "configure", None)
            if callable(fn):
                fn(config)

    def checkpoint_state(self):
        states = {}
        for label, policy in (("first", self.first), ("second", self.second)):
            fn = getattr(policy, "checkpoint_state", None)
            if callable(fn):
                states[label] = fn()
        return states or None

    def restore_state(self, state) -> None:
        if not isinstance(state, dict):
            return
        for label, policy in (("first", self.first), ("second", self.second)):
            if label not in state:
                continue
            fn = getattr(policy, "restore_state", None)
            if callable(fn):
                fn(state[label])

    def on_screen_reject(self) -> None:
        for policy in (self.first, self.second):
            fn = getattr(policy, "on_screen_reject", None)
            if callable(fn):
                fn()

    def describe(self):
        names = []
        for policy in (self.first, self.second):
            describe = getattr(policy, "describe", None)
            if callable(describe):
                names.extend(describe())
            elif getattr(policy, "__name__", None):
                names.append(policy.__name__)
            else:
                names.append(type(policy).__name__)
        return tuple(names)


@dataclass(frozen=True)
class GuardedRun:
    """The evolution result, the recorded decision and the acceptance chain."""

    result: Any
    decision: Any = None
    acceptance: Any = None
    notes: Sequence[str] = ()


def run_guarded(spec: EvolveSpec, *, acceptance: Optional[Any] = None,
                decision_log: Optional[Any] = None,
                kill_switch: Optional[Any] = None,
                **compose_kwargs: Any) -> GuardedRun:
    """Run one spec with an injected acceptance policy and optional records.

    ``compose_kwargs`` go to :func:`~agentdescent.evolvespec.compose` (for
    example ``repo_path``). An engaged kill switch raises before composing.
    """
    if kill_switch is not None and kill_switch.engaged():
        reason = kill_switch.reason() or "operator stop"
        raise RuntimeError(f"kill switch engaged: {reason}")
    comp = compose(spec, **compose_kwargs)
    notes = list(comp.notes)
    base = comp.kwargs.get("policies")
    if not isinstance(base, Policies):
        base = Policies()
    existing = getattr(base, "acceptance", None)
    chain = existing
    if acceptance is not None:
        second = existing
        if second is None:
            cfg = comp.kwargs.get("agg_config")
            if isinstance(cfg, Mapping):
                cfg = build_agg_config(spec, **cfg)
            elif not isinstance(cfg, AggregatorConfig):
                cfg = AggregatorConfig()
            second = DefaultAcceptance.from_config(cfg)
        chain = _Chain(acceptance, second)
    if chain is not None:
        comp.kwargs["policies"] = replace(base, acceptance=chain)
        names = getattr(chain, "describe", None)
        if callable(names):
            notes.append("acceptance: " + ", ".join(names()))
    else:
        notes.append("no acceptance injected")
    if comp.kwargs.get("aggregator_factory") is not None:
        notes.append(
            "algorithm aggregator_factory is installed; the injected acceptance "
            "is not consulted by that merge path")

    result = comp.run()
    decision = None
    if decision_log is not None:
        usage = getattr(result, "usage", None)
        names = getattr(chain, "describe", None)
        outcomes = dict(result.outcomes())
        stop_reason = getattr(result, "stop_reason", "")
        try:
            decision = decision_log.record(
                kind="run", artifact_id=str(comp.kwargs.get(
                    "artifact_id", spec.artifact_id())),
                accepted=outcomes.get("committed", 0) > 0,
                category=str(getattr(stop_reason, "value", stop_reason)),
                detail=str(result.error or ""),
                evidence={
                    "final_reward": float(result.final_reward),
                    "outcomes": outcomes,
                    "rounds": len(result.history),
                    "error": result.error,
                    "acceptance": list(names()) if callable(names) else [],
                },
                cost={
                    "calls": int(getattr(usage, "calls", 0) or 0),
                    "prompt_tokens": int(
                        getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(
                        getattr(usage, "completion_tokens", 0) or 0),
                    "wallclock": float(getattr(result, "wallclock", 0.0) or 0.0),
                })
        except Exception as error:  # noqa: BLE001 - the run already happened
            notes.append(f"decision log failed: {type(error).__name__}: {error}")
    return GuardedRun(result=result, decision=decision, acceptance=chain,
                      notes=tuple(notes))
