"""Pluggable staleness policies (design doc, section 4.2; FlashEvolve 2605.08520).

FlashEvolve's central move is to port asynchronous-RL staleness theory onto
*non-parameter* text/code artifacts, and it defines three escalating policies for
what to do with a diff that was proposed against an out-of-date head:

* **Full**      -- use the stale diff directly (max throughput, min safety).
* **Guarded**   -- version-gated: accept only within a staleness budget ``alpha``,
                   rebasing the middle band; discard beyond it.
* **Reflective**-- always attempt to rebase and *re-verify* (an LLM/cheap replay
                   of the evidence on the current head), discarding only if the
                   improvement no longer holds.

Concordia additionally borrows ROLL Flash's **asynchronous ratio** -- a *global*
lag budget that decides how far any worker's snapshot may drift from head before
it is force-refreshed (see :class:`~concordia.async_runtime.AsyncConfig`).

The aggregator computes ``eta`` and ``alpha`` and asks the policy for one of the
three actions below; the policy is the only thing that changes between async
regimes, so a run can be swapped from Guarded to Reflective without touching the
merge pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class StaleAction(Enum):
    """What the aggregator should do with a (possibly stale) evidence card."""

    ACCEPT = "accept"    # take the diff as-is, no rebase
    REBASE = "rebase"    # rebase onto head and cheaply re-verify; keep iff it holds
    DISCARD = "discard"  # drop the diff, settle its evidence back to the pool


@runtime_checkable
class StalenessPolicy(Protocol):
    name: str

    def decide(self, eta: int, alpha: int, contract_breaking: bool) -> StaleAction: ...


class FullStaleness:
    """Use stale diffs directly regardless of ``eta`` (max throughput).

    Contract-breaking diffs are the one exception even here: a stale
    cross-contract diff cannot be blindly applied, so it is discarded (the cost
    of a cross-contract rebase exceeds re-proposing -- design doc, section 4.2)."""

    name = "full"

    def decide(self, eta: int, alpha: int, contract_breaking: bool) -> StaleAction:
        if contract_breaking and eta > 0:
            return StaleAction.DISCARD
        return StaleAction.ACCEPT


class GuardedStaleness:
    """Version-gated with rebase in the middle band (Concordia's default).

    ``eta == 0`` accepts; ``0 < eta <= alpha`` rebases and re-verifies; beyond
    ``alpha`` discards.  This is the AReaL bounded-staleness discipline expressed
    over diffs."""

    name = "guarded"

    def decide(self, eta: int, alpha: int, contract_breaking: bool) -> StaleAction:
        if eta == 0:
            return StaleAction.ACCEPT
        if contract_breaking:
            return StaleAction.DISCARD  # alpha is forced to 0 for these
        if eta <= alpha:
            return StaleAction.REBASE
        return StaleAction.DISCARD


class ReflectiveStaleness:
    """Always rebase + re-verify; discard only if the improvement no longer holds.

    Ignores the ``alpha`` budget -- every stale diff gets a reflective replay on
    the current head.  Highest recovery of otherwise-wasted proposals, at the
    cost of more cheap-eval work (FlashEvolve's Reflective tier)."""

    name = "reflective"

    def decide(self, eta: int, alpha: int, contract_breaking: bool) -> StaleAction:
        if eta == 0:
            return StaleAction.ACCEPT
        if contract_breaking:
            return StaleAction.DISCARD
        return StaleAction.REBASE


POLICIES = {
    "full": FullStaleness,
    "guarded": GuardedStaleness,
    "reflective": ReflectiveStaleness,
}


def get_policy(name: str) -> StalenessPolicy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(
            f"unknown staleness policy {name!r}; choose from {sorted(POLICIES)}"
        )
