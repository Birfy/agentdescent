"""Recursive self-improvement: the loop that moves its own budget is evolved.

``self_improve`` reads the archive, evolves the slots the evidence says are
paying, and moves the next tick's budget with the shipped fixed
``portfolio_report`` rule. ``portfolio_evolve`` evolves that reallocation rule
on layer windows but never runs the loop it governs. This module closes the
recursion: each outer round runs one ``self_improve`` session with the current
rule, then asks ``portfolio_evolve`` to improve the rule on the archive that
session just wrote, and carries an adopted rule into the next round.

The rule starts as ``ReallocationPolicy()`` -- byte-for-byte the shipped fixed
policy -- so a run with no policy proposals behaves like ``self_improve``. An
adopted policy is compiled from the validation gain it had to earn on windows
the policy search never trained on; a rejected one is reported and the previous
rule stays. The report keeps the inner session's tick trail next to the policy
decision, so "did the recursion help?" is answerable without recomputing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .consolidation import ConsolidationStore
from .meta import MetaReward, SlotSpec
from .portfolio_evolve import (ReallocationPolicy, portfolio_evolve,
                               reallocation_spec, windows_from_archive)
from .self_improve import (
    SelfImproveConfig, SelfImproveReport, self_improve,
)

__all__ = ["RecursiveRound", "RecursiveSelfImproveReport",
           "recursive_self_improve"]


@dataclass(frozen=True)
class RecursiveRound:
    """One outer round: the inner self-improvement session and the policy move."""

    index: int
    improve: SelfImproveReport
    #: The rendered reallocation parameters the next round uses (the incumbent
    #: after this round, whether the candidate was adopted or not).
    policy_value: str
    #: The candidate the policy search proposed this round (``""`` when the
    #: archive was too thin to run it).
    policy_candidate: str
    #: ``adopt`` / ``reject`` from ``portfolio_evolve`` (or ``reject`` with the
    #: reason when the archive was too thin to build windows).
    policy_verdict: str
    #: Validation gain over the fixed baseline on unseen windows (``0.0`` when
    #: the policy round could not run).
    policy_gain: float
    reason: str


@dataclass(frozen=True)
class RecursiveSelfImproveReport:
    """The outer session: every round, the final rule, and both gains."""

    rounds: Tuple[RecursiveRound, ...]
    policy: ReallocationPolicy
    improved: bool
    #: Sum of the inner sessions' adopted ``slot:domain`` gains.
    total_gain: float
    #: Sum of the adopted policy rounds' validation gain over the baseline.
    total_policy_gain: float


def recursive_self_improve(
    store: ConsolidationStore,
    config: SelfImproveConfig,
    *,
    propose: Optional[Callable[[str, Any, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    policy_propose: Optional[Callable[[str, Any, str, float], Optional[str]]] = None,
    policy_model: Optional[Any] = None,
    outer_rounds: int = 2,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    meta_reward: Optional[MetaReward] = None,
    **evolve_kwargs: Any,
) -> RecursiveSelfImproveReport:
    """Run ``self_improve`` and ``portfolio_evolve`` alternately.

    ``propose``/``model`` are the inner loop's reflector; ``policy_propose`` /
    ``policy_model`` rewrite the reallocation parameters and are required (a
    slot reflector returns slot values, not a parameter block). ``outer_rounds``
    bounds the recursion. ``evolve_kwargs`` go to the inner ``meta_run``;
    ``policy_kwargs`` go to the ``portfolio_evolve`` search (``rounds``,
    ``seeds``, ``repo_path`` ...), so the two budgets stay separate.

    A policy round that cannot build windows yet (a cold archive) is reported
    as ``reject`` with its reason; it never aborts the outer loop.
    """
    if outer_rounds < 1:
        raise ValueError("outer_rounds must be >= 1")
    if propose is None and model is None:
        raise ValueError("recursive_self_improve() needs propose= or model=")
    if policy_propose is None and policy_model is None:
        raise ValueError(
            "recursive_self_improve() needs policy_propose= or policy_model=; "
            "the reallocation parameters are a different artifact from a slot")
    policy = config.reallocation or ReallocationPolicy(
        floor=config.floor, min_samples=config.min_samples)
    spec = reallocation_spec()
    rendered = spec.render({k: repr(float(v)) for k, v in asdict(policy).items()})
    rounds = []
    improved = False
    total_gain = 0.0
    total_policy_gain = 0.0

    for index in range(outer_rounds):
        inner = self_improve(
            store, replace(config, reallocation=policy), propose=propose,
            model=model, meta_reward=meta_reward, **evolve_kwargs)
        improved = improved or inner.improved
        total_gain += inner.total_gain

        verdict, gain, reason, candidate = "reject", 0.0, "", ""
        kwargs = dict(policy_kwargs or {})
        explicit_windows = "windows" in kwargs or "val_windows" in kwargs
        if explicit_windows:
            can_run = True
        else:
            train_windows, val_windows = windows_from_archive(store)
            can_run = bool(train_windows and val_windows)
        if not can_run:
            reason = ("the archive has fewer than two layer windows; run more "
                      "self-improve ticks (or pass policy_kwargs windows)")
        else:
            report, _ = portfolio_evolve(
                store, propose=policy_propose, model=policy_model,
                baseline=rendered, **kwargs)
            verdict = str(report["verdict"])
            gain = float(report["val_gain"])
            reason = str(report.get("reason", ""))
            candidate = str(report["value"])
            if verdict == "adopt":
                policy = spec.compile(candidate)
                rendered = candidate
                total_policy_gain += gain
        rounds.append(RecursiveRound(
            index=index, improve=inner, policy_value=rendered,
            policy_candidate=candidate, policy_verdict=verdict,
            policy_gain=gain, reason=reason))

    return RecursiveSelfImproveReport(
        rounds=tuple(rounds), policy=policy, improved=improved,
        total_gain=total_gain, total_policy_gain=total_policy_gain)
