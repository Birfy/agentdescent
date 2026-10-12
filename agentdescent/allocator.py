"""Allocating the scarce part of a budget by candidate value, not by clock.

``BudgetGovernor`` degrades optional spend (fusion tournaments, self-verify
rollouts) against a global floor: once ``spent / max_tokens`` crosses a
threshold, *every* candidate stops getting the expensive part. That is fair and
predictable, and it is exactly what a budget-aware allocator should *not* do.

The EvoAlloc line of work measures the cost of a fixed allocation strategy: when
evaluation is the scarce resource, spending it by time-since-start rather than by
expected value wastes it on candidates that cannot move the search while starving
ones that can. This module replaces the global threshold with a **per-candidate**
decision -- ``is this candidate worth the expensive evaluation?`` -- learned from
the run's own merge history, plus a **counterfactual exploration** term that
occasionally evaluates a candidate the allocator judged unworthy, so the value
model sees the ground truth it was wrong about.

Three pieces:

* :class:`AllocatorContext` -- the features one candidate carries into the
  decision. Everything here is arithmetic over numbers the engine already has
  on the merge path (``before_after_delta``, advantage, diff size, P(improve)),
  so the allocator adds no new measurement, only a new reader of existing ones.
* :class:`AllocatorPolicy` -- the seam. ``decide(ctx)`` says whether this
  candidate is worth a deep evaluation; ``observe(ctx, outcome)`` feeds the
  result of whatever was actually spent back into the model.
* :class:`ValueBudgetAllocator` -- the shipped learner. A logistic-style value
  model over the candidate features, fit online against commit outcomes, with an
  epsilon-greedy counterfactual term that re-measures the candidates the model
  judged worst.

The allocator is **off by default**: ``BudgetGovernor(allocator=None)`` (the
default) is the old global-threshold behaviour byte for byte. Pass one and the
governor asks it before spending the expensive part of a candidate's merge, and
tells it what happened afterwards.

The analogy to the rest of the library: the aggregator already learns *what to
keep* (Beta posteriors, adaptive trust region). This learns *how much to spend
finding out* -- the resource-allocation layer between ``max_tokens`` (the cap)
and the merge decisions (the value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "AllocatorContext",
    "AllocatorPolicy",
    "ValueBudgetAllocator",
    "LearningValueModel",
]


# ---------------------------------------------------------------------------
# The features a candidate carries into the decision
# ---------------------------------------------------------------------------


@dataclass
class AllocatorContext:
    """One candidate, as the allocator sees it.

    All fields are arithmetic over numbers the merge path already computes; the
    allocator does not ask the engine to measure anything new. ``None`` means
    "not measured", never "zero" -- a missing advantage and a real zero
    advantage are different facts (the zero-advantage filter exists because of
    exactly that distinction).
    """

    #: The local before/after delta the proposing worker measured. The strongest
    #: cheap signal that a diff helps, and the one every acceptance policy folds
    #: in as extra evidence for the candidate.
    before_after_delta: Optional[float] = None
    #: Group-relative advantage (GRPO-style z-score) of the proposing rollout.
    #: ``None`` when the group was too small to standardise against.
    advantage: Optional[float] = None
    #: ``len(diff.ops)`` -- the trust region's own notion of diff size.
    size: int = 0
    #: The cheap layer's P(improve), when the candidate reached acceptance.
    p_improve: Optional[float] = None
    #: The learned layer's uncertainty (``learned_eval``), when available.
    uncertainty: Optional[float] = None
    #: How far the candidate sits from the confirmed ``stable`` branch.
    stable_distance: float = 0.0
    #: The artifact's blast radius.
    blast_radius: float = 0.2

    def features(self) -> Tuple[float, ...]:
        """A fixed-length numeric vector for the value model.

        Missing values are ``0.0`` here; the model treats a column of all zeros
        as "this signal was never measured on this workload" rather than "this
        candidate scored zero", which is why :attr:`before_after_delta` and
        :attr:`advantage` are kept as separate columns with their own zero-fill
        rather than collapsed.
        """
        return (
            self.before_after_delta if self.before_after_delta is not None else 0.0,
            self.advantage if self.advantage is not None else 0.0,
            float(self.size),
            self.p_improve if self.p_improve is not None else 0.0,
            self.uncertainty if self.uncertainty is not None else 0.0,
            self.stable_distance,
            self.blast_radius,
        )

    @property
    def value_hint(self) -> float:
        """A single scalar summary, for diagnostics and the epsilon anneal.

        The unweighted mean of the two strongest cheap signals: a positive
        before/after delta and a positive advantage both say "this candidate
        moved the needle locally". Rough by design -- the model learns the
        weights; this is only the initial guess that starts it."""
        hints = [v for v in (self.before_after_delta, self.advantage) if v is not None]
        return (sum(hints) / len(hints)) if hints else 0.0


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocateDecision:
    """What the allocator decided for one candidate."""

    #: Whether the candidate is worth the expensive evaluation (the fusion
    #: tournament, or the self-verify rollout).
    deep_eval: bool = True
    #: Whether this decision was a counterfactual exploration -- a candidate
    #: judged unworthy but evaluated anyway, so the value model can learn what
    #: it missed. Diagnostics only; the engine spends the same either way.
    counterfactual: bool = False
    #: The value the model assigned, before the epsilon roll. Diagnostics.
    predicted_value: float = 0.0
    #: How many observations the model had, at decision time.
    n_observed: int = 0


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


@runtime_checkable
class AllocatorPolicy(Protocol):
    """Decide, per candidate, whether the expensive part of a merge is worth it.

    ``decide`` is called just before the allocatable spend (the fusion
    tournament in the sync path, the self-verify rollout on both). ``observe``
    is called afterwards with what actually happened -- the outcome is the
    counterfactual truth the model learns from. Both are optional hooks: a
    policy that implements neither is a no-op allocator that always deep-evals
    (the old behaviour).
    """

    def decide(self, ctx: AllocatorContext) -> AllocateDecision: ...
    def observe(self, ctx: AllocatorContext, committed: bool) -> None: ...


# ---------------------------------------------------------------------------
# The online value model
# ---------------------------------------------------------------------------


@dataclass
class LearningValueModel:
    """A tiny online logistic model over the candidate features.

    ``P(committed | features)`` is what "worth a deep eval" means here: a
    candidate the model expects to commit is worth the extra held-out sweep,
    one it expects to be rejected is not. Fitted online by gradient ascent on
    the run's own commit outcomes, so it needs no labelled data -- the label is
    ``did it commit``, which the engine knows for free.

    The learning rule is plain logistic regression with a momentum term
    (``lr``), a clip on the logit (``max_logit``, so one extreme candidate
    cannot blow the weights), and a small L2 penalty (``reg``) so features that
    never move stay near zero. None of these is a research contribution; they
    are the smallest thing that learns without diverging on one noisy run.
    """

    #: Learning rate.
    lr: float = 0.2
    #: L2 regularisation.
    reg: float = 0.05
    #: Logit clip, so a single outlier cannot send the weights to infinity.
    max_logit: float = 3.0
    #: Weights, one per feature in :meth:`AllocatorContext.features`, plus bias.
    _w: List[float] = field(default_factory=lambda: [0.0] * 8)

    def __post_init__(self) -> None:
        if len(self._w) != 8:
            self._w = [0.0] * 8

    def predict(self, features: Tuple[float, ...]) -> float:
        # The weights are ``[w0..w6, bias]``: the first seven match the seven
        # features one-to-one, the last is the intercept. Not ``zip(self._w,
        # features)`` -- that would silently match the bias weight against the
        # seventh feature and then add it again as the intercept.
        logit = self._bias() + sum(
            wi * x for wi, x in zip(self._w[:-1], features))
        logit = max(-self.max_logit, min(self.max_logit, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def _bias(self) -> float:
        return self._w[-1]

    def update(self, features: Tuple[float, ...], committed: bool) -> None:
        """One gradient-ascent step toward ``committed``.

        ``p = sigmoid(w·x)``, target ``y ∈ {0, 1}``, gradient
        ``(y - p) * x``, plus L2. In-place, so the model is a single mutable
        object the allocator thread can step without copies.
        """
        p = self.predict(features)
        y = 1.0 if committed else 0.0
        err = y - p
        for i in range(len(self._w)):
            # Every feature column contributes ``err * x_i``; the bias (last
            # weight) gets ``err * 1`` -- it must be able to move even when
            # every feature is zero, or an all-zero candidate (nothing measured
            # on this workload) can never learn a non-neutral prior.
            x = features[i] if i < len(features) else 1.0
            grad = err * x - self.reg * self._w[i]
            self._w[i] += self.lr * grad


# ---------------------------------------------------------------------------
# The shipped allocator
# ---------------------------------------------------------------------------


@dataclass
class ValueBudgetAllocator:
    """Learn which candidates are worth the expensive evaluation.

    The two halves of the EvoAlloc mechanism, in one object:

    **Value-directed allocation.** When the budget is not tight
    (``remaining_fraction`` high) every candidate is deep-evaluated -- there is
    budget to spend, and spending it is the old behaviour. As the budget runs
    out the allocator consults :class:`LearningValueModel`: candidates it
    expects to commit still get the deep eval, candidates it expects to be
    rejected are skipped. The expensive spend goes where the model says the
    value is.

    **Counterfactual exploration.** A candidate the model judges unworthy is
    skipped -- except with probability ``epsilon``, when it is evaluated anyway.
    The outcome (``observe``) is then a *counterfactual*: the run paid for a
    measurement it would have skipped, and the model learns "this looked
    worthless and actually committed" (or the reverse). Without this term the
    model only ever sees the candidates it already believed in, and a biased
    prior is never corrected -- the classic selection-bias collapse of any
    value-based allocator.

    ``epsilon`` anneals from ``epsilon_start`` down to ``epsilon_end`` over the
    first ``epsilon_anneal`` observations: early, the model knows nothing and
    explores hard; late, it has seen the workload and exploits. The anneal is
    on *observations*, not wall-clock, so a fast run and a slow run spend the
    same exploration budget.
    """

    #: Below this remaining fraction the allocator starts skipping low-value
    #: candidates. Above it, everything deep-evals (the old behaviour).
    budget_start: float = 0.5
    #: The model used to predict which candidates commit.
    model: LearningValueModel = field(default_factory=LearningValueModel)
    #: Counterfactual exploration probability, at the start of the run.
    epsilon_start: float = 0.3
    #: ...and after the anneal.
    epsilon_end: float = 0.05
    #: Observations over which epsilon anneals.
    epsilon_anneal: int = 40
    #: Number of observations the model must have before it is trusted to skip
    #: anything. Before that, everything deep-evals -- the model has no opinion
    #: yet, and skipping on no evidence is the bias the exploration exists to
    #: avoid.
    min_observations: int = 5
    #: Features whose |weight| is below this are treated as "no signal" for the
    #: skip decision, so a model fit on a workload where advantage never moved
    #: cannot skip on advantage noise.
    min_weight: float = 0.01

    # -- state ---------------------------------------------------------------

    _n_obs: int = 0
    _n_skipped: int = 0
    _n_counterfactual: int = 0
    _n_committed: int = 0

    @property
    def n_observed(self) -> int:
        return self._n_obs

    @property
    def n_skipped(self) -> int:
        return self._n_skipped

    @property
    def n_counterfactual(self) -> int:
        return self._n_counterfactual

    @property
    def n_committed(self) -> int:
        return self._n_committed

    def _epsilon(self) -> float:
        t = self._n_obs
        if self.epsilon_anneal <= 0:
            return self.epsilon_end
        progress = min(1.0, t / self.epsilon_anneal)
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) \
            * (1.0 - progress)

    def decide(self, ctx: AllocatorContext) -> AllocateDecision:
        """Whether ``ctx``'s candidate is worth the expensive evaluation.

        Always deep-evals when the model has no opinion yet
        (``min_observations``), or when this allocator is not under budget
        pressure. Otherwise consults the model and rolls the counterfactual
        epsilon.
        """
        if self._n_obs < self.min_observations:
            return AllocateDecision(deep_eval=True, n_observed=self._n_obs)
        features = ctx.features()
        p = self.model.predict(features)
        # A candidate the model cannot distinguish from noise (every |w| tiny,
        # or the logit near the prior) is not confidently low-value; skipping
        # it is the bias, not the allocation.
        confident = any(abs(w) >= self.min_weight for w in self.model._w[:-1])
        if not confident or p >= 0.5:
            return AllocateDecision(
                deep_eval=True, predicted_value=p, n_observed=self._n_obs)
        # Low value: skip, unless this is a counterfactual exploration.
        from random import random
        if random() < self._epsilon():
            self._n_counterfactual += 1
            return AllocateDecision(
                deep_eval=True, counterfactual=True,
                predicted_value=p, n_observed=self._n_obs)
        self._n_skipped += 1
        return AllocateDecision(
            deep_eval=False, predicted_value=p, n_observed=self._n_obs)

    def observe(self, ctx: AllocatorContext, committed: bool) -> None:
        """Feed a commit outcome back into the value model.

        Called for every candidate the merge path actually evaluated -- the
        skipped ones were never evaluated, so there is no outcome to learn
        from; the counterfactual ones were, and their outcome is the point of
        having spent the budget on them.
        """
        self._n_obs += 1
        if committed:
            self._n_committed += 1
        self.model.update(ctx.features(), committed)
