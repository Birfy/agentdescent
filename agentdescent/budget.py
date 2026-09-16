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
from typing import Any, Callable, Optional


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
    #: Stop when the run's *measured* return per token has fallen off its own
    #: peak by more than :attr:`efficiency_floor`. Off by default: it is an
    #: economic rule, not a safety one, and a run whose reward only ever rises
    #: late would be cut short by it. Turn it on to spend a budget by value
    #: rather than by exhaustion -- it is what lets a run finish *under* its
    #: budget when the search has stopped paying for itself.
    stop_on_diminishing: bool = False
    #: How far below the run's own peak efficiency counts as diminishing. The
    #: default ``0.25`` means "the last rounds returned less than a quarter of
    #: the best rate this run ever achieved". Self-calibrating: it compares the
    #: run against itself, so there is no magic absolute rate to set (a reward
    #: is in ``[0, 1]`` and a token count is in the millions; their ratio has no
    #: interpretable scale, which is exactly why an absolute floor would be a
    #: guess).
    efficiency_floor: float = 0.25
    #: How many efficiency samples before the rule may fire. Four rounds is the
    #: floor at which "the peak" and "the recent rate" are different claims.
    min_efficiency_samples: int = 4
    #: Cumulative spend, set by ``spend``. Kept as a field so ``affords`` can
    #: be asked without re-passing the number the caller just passed.
    _spent: int = 0
    #: Per-round deltas of the spend, for the projection in ``affords``.
    _round_deltas: list = field(default_factory=list)
    #: The spend at the last ``spend`` call, so a delta can be computed without
    #: the caller tracking the previous value.
    _last: int = 0
    #: Return-per-token for each completed round, from ``observe``.
    _efficiencies: list = field(default_factory=list)
    #: The previous round's reward and spend, so ``observe`` can difference.
    _prev_reward: Optional[float] = None
    _prev_spent: int = 0

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

    def remaining_fraction(self) -> float:
        """Fraction of the token budget still unspent, in ``[0, 1]``.

        What a budget-aware :class:`~agentdescent.selection.SelectionPolicy`
        anneals its exploration on. ``1.0`` when no budget was set -- a run
        without a ceiling never runs short of one -- and clamped to ``[0, 1]``
        so an overshoot (the barrier's up-to-one-round lag) reads as "nothing
        left" rather than a negative fraction that would invert the sign of an
        exploration term.
        """
        if self.max_tokens is None or self.max_tokens <= 0:
            return 1.0
        return max(0.0, min(1.0, (self.max_tokens - self._spent) / self.max_tokens))

    def observe(self, reward: Optional[float]) -> None:
        """Record the reward of the round that just finished.

        Called at the top of each round, after :meth:`spend`, with the reward
        the previous round produced (``None`` on the first round, and on a round
        whose measurement failed). The efficiency it records is that round's
        ``Δreward / Δtokens`` -- the run's own return per token, which
        :meth:`diminishing_returns` compares against its own peak.

        A round that spent no tokens records nothing: ``Δreward / 0`` is not
        infinity, it is a round whose cost was not measured, and letting it into
        the sample would make an unmeasured run look infinitely efficient.
        """
        if reward is None or not isinstance(reward, (int, float)):
            return
        reward = float(reward)
        if self._prev_reward is None:
            self._prev_reward = reward
            self._prev_spent = self._spent
            return
        dtokens = self._spent - self._prev_spent
        dreward = reward - self._prev_reward
        self._prev_reward = reward
        self._prev_spent = self._spent
        if dtokens > 0:
            self._efficiencies.append(dreward / dtokens)

    def diminishing_returns(self) -> bool:
        """Whether the run's return per token has fallen off its own peak.

        The economic stop: keep buying compute while it is still paying, stop
        when it is not -- *even with budget left*, which is the whole point.
        ``max_tokens`` is a ceiling; this is the rule that decides not to spend
        the ceiling.

        Returns ``False`` unless ``stop_on_diminishing`` is set, a budget is in
        force (efficiency needs a token count) and at least
        ``min_efficiency_samples`` rounds have been observed. It also returns
        ``False`` when the run's peak efficiency is not positive: a run that
        never improved has nothing to have declined *from*, and the patience
        counter -- not this -- is the rule for "it never got better".
        """
        if not self.stop_on_diminishing or self.max_tokens is None:
            return False
        samples = self._efficiencies
        if len(samples) < self.min_efficiency_samples:
            return False
        peak = max(samples)
        if peak <= 0:
            return False
        recent = sum(samples[-2:]) / 2.0
        return recent < self.efficiency_floor * peak

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


# ---------------------------------------------------------------------------
# Adaptive per-call thinking budget (o1-style test-time scaling)
# ---------------------------------------------------------------------------


@dataclass
class CallBudget:
    """A mutable per-call max_tokens, set by the engine before each expansion.

    The engine's :class:`BudgetGovernor` decides how many tokens the *next*
    model call should be allowed to spend, and writes that number here. A
    :func:`budgeted_completion` wrapper reads it and delegates to the underlying
    adapter with the right ``max_tokens``.

    The o1 insight, applied to search: a reasoning model's per-call thinking
    budget should be **proportional to the expected value of that expansion** —
    a promising, under-explored parent deserves more room to think than a
    dead-end one the search has already tried five times. The engine knows the
    parent's score and the remaining budget; the adapter knows how to call the
    model; this object is the one place they meet.

    ``base`` is the default the adapter was constructed with. The engine never
    raises ``next`` above ``base`` (a model call that costs more than the
    caller's configured ceiling is a bill the caller did not agree to), and
    never drops it below ``base // 4`` (a reasoning model starved below a
    quarter of its budget returns empty content — measured, see
    :func:`agentdescent.agents.claude`).
    """

    #: The adapter's configured max_tokens — the ceiling the engine never
    #: exceeds.
    base: int = 4096
    #: What the next call should use. Set by the engine before each expansion;
    #: ``None`` means "use ``base``" (the adapter's own default), which is the
    #: state a run without adaptive budgeting stays in.
    next: Optional[int] = None
    #: The floor: a reasoning model below this returns nothing. Measured on
    #: deepseek-v4-flash at 1024: four of eight reflection prompts came back
    #: empty. ``base // 4`` is the conservative reading of that.
    floor: Optional[int] = None

    def __post_init__(self) -> None:
        if self.floor is None:
            self.floor = max(256, self.base // 4)

    def allocate(self, score: float, budget_remaining: float,
                 *, max_boost: float = 1.0) -> None:
        """Set ``next`` for the upcoming expansion.

        The allocation is a product of two factors:

        * **Score factor** (``0.5 + 0.5 * score``): a high-scoring parent
          gets the full ``base``; a zero-scoring one gets half. The parent's
          score is the search's own estimate of how promising it is, so this
          is "spend more thinking where the search says it is worth it."

        * **Budget factor** (``0.5 + 0.5 * budget_remaining``): early in the
          run, when there is budget to exploit what exploration finds, the
          engine allows full thinking; late, when there is not, it tightens.
          This is the same annealing :class:`CostEfficient` uses on the
          exploration bonus, applied to the thinking budget — both are "spend
          more while you still have time to act on it."

        The product ranges from ``base * 0.25`` (dead-end parent, no budget
        left) to ``base`` (promising parent, full budget), never above ``base``
        and never below ``floor``. ``max_boost`` > 1 is allowed but only when
        the caller explicitly sets it — a budget that lets the engine raise
        the per-call ceiling above the adapter's configured ``max_tokens`` is
        a bill the caller has to opt into.
        """
        s = 0.5 + 0.5 * max(0.0, min(1.0, float(score)))
        b = 0.5 + 0.5 * max(0.0, min(1.0, float(budget_remaining)))
        ceiling = int(self.base * max_boost)
        self.next = max(self.floor, min(ceiling, int(self.base * s * b)))


def budgeted_completion(
    factory: Callable[[int], "Completion"],
    budget: CallBudget,
) -> "Completion":
    """Wrap a ``max_tokens -> Completion`` factory into an adaptive Completion.

    The factory is called once per distinct ``max_tokens`` value (cached), so
    a run that stays at the default pays one construction and a run that
    adapts pays one per step it adapts to — not one per call.

    Usage::

        from agentdescent.budget import CallBudget, budgeted_completion
        from agentdescent.agents import claude, LLMAgent

        budget = CallBudget(base=8192)
        agent = LLMAgent(budgeted_completion(
            lambda mt: claude(model="claude-sonnet-4-5", max_tokens=mt, usage=u),
            budget))

        evolve(tasks, reward, agent=agent, max_tokens=500_000,
               call_budget=budget, usage=u)

    Without ``call_budget`` the engine never touches ``budget.next`` and the
    factory is called once with ``budget.base`` — the same cost as constructing
    ``claude(max_tokens=8192)`` directly.
    """

    cache: dict[int, "Completion"] = {}

    def complete(prompt: str) -> str:
        mt = budget.next if budget.next is not None else budget.base
        if mt not in cache:
            cache[mt] = factory(mt)
        return cache[mt](prompt)

    return complete
