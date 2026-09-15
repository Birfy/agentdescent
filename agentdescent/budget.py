"""Budget governor: graceful degradation before the hard token stop.

``max_tokens`` is a brake: when the spend reaches it the run stops, full stop.
A brake is the floor of cost control, not its ceiling. The last 10% of a token
budget is the most expensive part of the run to waste — the search has already
paid for its exploration, and a round dispatched there may never finish. What a
cost-aware search does instead, and what this module provides, is **degrade
before the wall**:

* at ``soft_floor`` (default 75%) of the budget, stop building fusion
  tournaments. The tournament is an extra held-out sweep of the surviving
  candidates and their fused candidate — pure ranking spend, and the ranking
  can survive without it for the last few rounds;
* at ``hard_floor`` (default 90%), stop the self-verify rollout. That rollout
  doubles the cost of every proposal for a delta the acceptance test folds in
  as a *tie-breaker weight* (``observe_delta`` with a weight of ``min(1,
  4*|delta|)``); skipping it costs ranking precision on the advantage signal
  and nothing on the commit gates, which read the full held-out set either way;
* the **projection** tells the round barrier whether the *next* round fits the
  remaining budget, so a run that cannot afford round N+1 ends at N with a
  clean merge instead of being cut mid-round at the wall.

The three thresholds are one design: every stage of a merge that is optional
ranking spend degrades before anything that decides a commit degrades, and
nothing that decides a commit ever degrades. The gate, the conflict resolution
and the Beta-posterior acceptance run to the last round exactly as they ran in
the first.

All of this is inert without ``max_tokens``: no budget, no governor, no change
to the run. With a budget, the governor changes *how* the tail of the run is
spent, never whether the run's numbers mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


#: Where the fusion tournament stops being built: 75% of the budget spent.
#: The tournament is the most expensive piece of optional ranking (a held-out
#: sweep of every survivor plus the fusion), and it is the piece whose absence
#: costs the least: candidates still commit through the same acceptance gate.
SOFT_FLOOR = 0.75

#: Where the self-verify rollout stops: 90% of the budget spent. This is the
#: last thing to go because it is the cheapest of the optional spends and the
#: one closest to the merge: its delta feeds the acceptance test's
#: ``observe_delta`` tie-breaker. Skipping it at 90% trades that tie-breaker
#: for the run actually finishing its round.
HARD_FLOOR = 0.90


@dataclass
class BudgetGovernor:
    """Tracks token spend against a cap and decides what to degrade.

    Constructed once per run (or not at all, when ``max_tokens`` is ``None``),
    read at the points where optional spend is about to happen. All reads are
    from the merger's thread or between rounds — the same single-threaded
    contract the round barrier already has — so no lock is needed.

    ``spend`` is called by the engine with the meter's cumulative
    ``prompt_tokens + completion_tokens`` once per round, before the barrier
    checks; ``affords`` answers for the *next* round using the per-round
    average of everything seen so far.
    """

    #: The hard cap, in tokens. ``None`` disables the governor entirely —
    #: every method returns the "no budget" answer, which is the run's
    #: behaviour without this class at all.
    max_tokens: Optional[int] = None
    #: Fraction of the budget at which fusion tournaments stop (see SOFT_FLOOR).
    soft_floor: float = SOFT_FLOOR
    #: Fraction of the budget at which self-verify stops (see HARD_FLOOR).
    hard_floor: float = HARD_FLOOR
    #: Cumulative spend, set by ``spend``. Kept as a field so ``affords`` can
    #: be asked without re-passing the number the caller just passed.
    _spent: int = 0
    #: Per-round deltas of the spend, for the projection in ``affords``.
    _round_deltas: list = field(default_factory=list)
    #: The spend at the last ``spend`` call, so a delta can be computed without
    #: the caller tracking the previous value.
    _last: int = 0

    # -- what the engine reports --------------------------------------------

    def spend(self, tokens: int) -> None:
        """Record the cumulative token spend as of this round's barrier.

        Called once per round, before the budget checks; the deltas between
        calls are what ``affords`` projects on. A first call with ``0`` (a run
        whose meter reports no tokens) is fine and leaves the projection
        uninformative rather than wrong.
        """
        self._round_deltas.append(max(0, tokens - self._last))
        self._last = self._spent = tokens

    # -- what the engine asks ------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether the governor has anything to say (a budget was set)."""
        return self.max_tokens is not None

    def _remaining(self) -> Optional[float]:
        if self.max_tokens is None:
            return None
        return self.max_tokens - self._spent

    def over_budget(self) -> bool:
        """The hard stop: the spend has reached the cap."""
        if self.max_tokens is None:
            return False
        return self._spent >= self.max_tokens

    def allow_fusion_tournament(self) -> bool:
        """Whether the extra held-out sweep of a tournament is affordable.

        The first thing to degrade: ranking precision, not a commit gate.
        """
        if self.max_tokens is None:
            return True
        return self._spent < self.max_tokens * self.soft_floor

    def allow_self_verify(self) -> bool:
        """Whether the double-cost verification rollout is affordable.

        The last thing to degrade: its delta feeds the acceptance test's
        tie-breaker, so it survives the soft floor and only dies at the hard
        one.
        """
        if self.max_tokens is None:
            return True
        return self._spent < self.max_tokens * self.hard_floor

    def affords_next_round(self) -> bool:
        """Whether the projection says one more round fits the budget.

        Projects the *average* per-round spend of the rounds seen so far. A
        run with no history (nothing spent yet) always affords a round — a
        budget that cannot afford round one is a ``ValueError`` the caller
        should have raised, not a silent no-op — and a run whose meter reports
        no tokens (``_spent == 0`` after a round) also always affords, because
        the projection has nothing to project with and the token brake will
        never fire anyway.
        """
        if self.max_tokens is None:
            return True
        remaining = self._remaining()
        if remaining <= 0:
            return False
        if not self._round_deltas or self._spent == 0:
            return True
        avg = sum(self._round_deltas) / len(self._round_deltas)
        if avg <= 0:
            return True
        return remaining >= avg

    # -- what the run reports --------------------------------------------------

    def summary(self) -> dict:
        """The governor's state, for the result and the status line.

        Always a dict (possibly all-``None``), so a caller does not have to
        feature-detect — an inactive governor reports ``None`` fields, which
        is the honest answer to "how much of the budget is left" when there
        is no budget.
        """
        if self.max_tokens is None:
            return {"max_tokens": None, "spent": self._spent,
                    "remaining": None, "fusion_degraded": False,
                    "self_verify_degraded": False}
        return {
            "max_tokens": self.max_tokens,
            "spent": self._spent,
            "remaining": max(0, self.max_tokens - self._spent),
            "fusion_degraded": not self.allow_fusion_tournament(),
            "self_verify_degraded": not self.allow_self_verify(),
        }
