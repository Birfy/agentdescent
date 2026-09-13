"""Three-layer verifier: rule / learned / full (design doc, section 3.1).

The aggregator uses verification at two very different price points:

* **cheap_eval** (rule + learned) -- run constantly, on every rebase check and
  every candidate in a fusion tournament.  Fast, noisy, no budget.
* **eval_counts / full_eval** -- the same scorer over the *whole* held-out set,
  expensive, and *budgeted* by the
  :class:`~agentdescent.scheduler.AuditScheduler` (design doc, section 5.3).

The learned layer also exposes an *uncertainty*, which feeds the audit priority
``blast_radius * uncertainty / trust``.

.. note:: **The expensive layer is not a second opinion.**

   ``full_eval`` was called ``oracle_eval`` until 0.6, and the old name claimed
   more than the code delivers. Every layer here calls the one ``eval_fn`` the
   caller supplied; they differ in **how many tasks** they score, not in *who is
   scoring*. So the expensive layer bounds the sampling error of a measurement
   and is structurally unable to reveal that the measurement is **biased** --
   if ``eval_fn`` is an agent judging outputs, running it on more tasks makes
   the same judge more confident, not more right.

   For a genuinely independent second source -- a gold answer, a checker, an
   experiment -- see :mod:`agentdescent.audit`, which pairs the two below the
   verifier at the reward level, where the observations can actually be matched
   up.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evolvable import Evolvable, stable_hash

# eval_fn(artifact, tasks) -> accuracy in [0, 1]
EvalFn = Callable[[Evolvable, Sequence], float]


def _identity(artifact) -> str:
    """A stable identity for an artifact, for seeding a per-artifact draw.

    ``render()`` is what the evaluation cache already keys on, so two artifacts
    that the cache calls identical seed identically. Falls back to id/version for
    an `Evolvable` that cannot render, which the protocol does not require.
    """
    try:
        return artifact.render()
    except Exception:  # noqa: BLE001 - an Evolvable need not be renderable
        return f"{getattr(artifact, 'id', '')}:{getattr(artifact, 'version', 0)}"


@dataclass
class VerifierBudget:
    """Budget for full-set evaluations, consumed by :meth:`ThreeLayerVerifier.full_eval`.

    The field names still say ``oracle`` because they are the public spelling of
    ``evolve(oracle_budget=...)``; renaming them would break every caller that
    reads ``result.usage`` or sets the budget, for no gain the method rename has
    not already delivered.
    """

    oracle_calls_remaining: int = 200
    oracle_calls_used: int = 0

    def can_spend(self) -> bool:
        return self.oracle_calls_remaining > 0

    def spend(self) -> None:
        self.oracle_calls_remaining -= 1
        self.oracle_calls_used += 1


@dataclass
class ThreeLayerVerifier:
    """Rule / learned / full backend for the aggregator.

    ``eval_fn`` is the scorer supplied by the domain, and every layer is that one
    function at a different sample size: the rule and learned layers run it on a
    small subset (plus noise, for the learned one), :meth:`eval_counts` and
    :meth:`full_eval` on the whole held-out set.

    It is the domain's *definition* of quality, which is not the same thing as
    ground truth. When ``eval_fn`` is itself a proxy -- an agent judging the
    output -- no layer here can tell, because they all ask it. See
    :mod:`agentdescent.audit`.
    """

    eval_fn: EvalFn
    held_out: Sequence
    #: How many held-out items the *cheap* layers score. Smaller is cheaper and
    #: noisier. Everything that decides a *commit* -- the Beta-posterior acceptance
    #: test and the regression guard beside it -- goes through :meth:`eval_counts`
    #: on the full set, so this trades ranking precision and nothing else.
    rule_subset: int = 8
    learned_noise: float = 0.04
    seed: int = 0
    budget: VerifierBudget = field(default_factory=VerifierBudget)
    _rng: random.Random = field(init=False, repr=False)

    #: Does :meth:`full_eval` score exactly the set :meth:`eval_counts` scores?
    #:
    #: For this class it always does -- both are ``eval_fn(artifact, held_out)``,
    #: which ``docs/verifier.md`` already noted when it explained that an L1 audit
    #: costs no extra model calls. The aggregator reads this and **reuses** the
    #: full-set rates it has already measured instead of buying them again.
    #:
    #: That is not only a saving. Re-buying them goes through :meth:`full_eval`,
    #: which degrades to :meth:`rule_eval` once the budget is gone -- so an
    #: exhausted budget silently turned the audit gate into a *sub-sample* veto,
    #: which is the one thing sub-sampling is documented never to do. See
    #: :meth:`full_eval`.
    #:
    #: Not annotated, so it is a plain class attribute rather than a
    #: constructor field: it is a statement about how this class is written, not
    #: a knob. A verifier whose expensive layer really is an independent
    #: measurement sets it to ``False`` -- and a *subclass* must, because it
    #: inherits this ``True`` and "simply does not define it" is not available
    #: to one. :func:`shares_eval_counts` prefers whichever name the verifier
    #: itself declares over the value it inherited from here.
    full_eval_matches_counts = True

    #: Pre-0.6 name for :attr:`full_eval_matches_counts`. Read through
    #: :func:`shares_eval_counts`, which prefers the new one, so a custom
    #: verifier that set either keeps working.
    oracle_shares_full_set = True

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._subsets: Dict[int, Sequence] = {}

    def _subset(self, k: int) -> Sequence:
        """A **stable** sample of ``k`` held-out items, drawn once per size.

        It used to draw a fresh sample on every call, which is only safe because
        ``evolve()`` pinned ``rule_subset`` to the full set and made the sampling a
        no-op. The moment the cheap layer is genuinely cheap that is a silent
        correctness bug: the aggregator compares candidates against each other with
        it -- ``_resolve_conflicts`` pits two diffs head to head and ``_tournament``
        ranks every candidate -- so a fresh draw per call scores candidate A on
        {1,3,5} and candidate B on {2,4,6} and calls the difference a winner. It
        also defeats the evaluation cache, which memoises per (artifact, task).

        Fixed per size, so a comparison is always like-for-like. Overfitting to the
        sample is bounded by the acceptance test, which never sub-samples.
        """
        if k >= len(self.held_out):
            return self.held_out
        if k not in self._subsets:
            idx = sorted(self._rng.sample(range(len(self.held_out)), k))
            self._subsets[k] = [self.held_out[i] for i in idx]
        return self._subsets[k]

    def rule_eval(self, artifact: Evolvable) -> float:
        """Cheap, deterministic-ish check on a tiny subset."""
        return self.eval_fn(artifact, self._subset(self.rule_subset))

    def learned_eval(self, artifact: Evolvable) -> Tuple[float, float]:
        """Noisy proxy that also returns an uncertainty estimate.

        Returns ``(score, uncertainty)``.  Uncertainty grows when the artifact
        is under-observed; here we approximate it with the noise band, which is
        what the audit scheduler needs to rank oracle spending.

        **The noise is seeded from the artifact**, not drawn from a shared
        stream. Drawing from `self._rng` made a candidate's score depend on how
        many other candidates had been scored before it -- so the same candidate
        got a different number depending on its position in the tournament, and
        scoring two candidates concurrently would have made it depend on which
        thread arrived first. `DefaultAcceptance` already seeds its acceptance
        draw per candidate for the same reason, and this is the other half of
        it: identical artifacts now score identically, whenever and wherever
        they are scored."""
        subset = self._subset(self.rule_subset * 2)
        base = self.eval_fn(artifact, subset)
        rng = random.Random(stable_hash((self.seed, _identity(artifact))) & 0x7FFFFFFF)
        noisy = min(1.0, max(0.0, base + rng.gauss(0.0, self.learned_noise)))
        uncertainty = self.learned_noise + 0.5 / (1 + len(subset))
        return noisy, uncertainty

    def cheap_eval(self, artifact: Evolvable) -> float:
        """The signal used everywhere a budget-free score is needed."""
        rule = self.rule_eval(artifact)
        learned, _ = self.learned_eval(artifact)
        return 0.5 * rule + 0.5 * learned

    def eval_counts(self, artifact: Evolvable,
                    floor: Optional[float] = None) -> Tuple[float, float]:
        """Return (successes, failures) on the full held-out set.

        Feeds the aggregator's Beta-posterior acceptance test with an honest
        sample size (design doc, section 4.4).

        ``floor`` is the rate this artifact has to beat, and it turns the scan
        into :meth:`~agentdescent.evolution.EvolvingArtifact.score_bounded`:
        evaluation stops as soon as the remaining tasks *cannot* lift the mean
        past it, because from there on every additional model call buys a number
        that no longer changes the answer. The counts that come back are then a
        bound rather than a measurement, which is exactly as sound for a test
        asking "is the candidate better" and **not** sound for anything that
        records the rate. See :meth:`score_bounded` for that distinction.

        Only used where a caller has a baseline to hand -- the aggregator has
        one, having just measured the base. ``None`` keeps the full scan, so an
        `Evolvable` that cannot do a bounded one (the protocol does not require
        it) is unaffected.
        """
        n = float(len(self.held_out))
        bounded = getattr(artifact, "score_bounded", None)
        if floor is None or not callable(bounded):
            acc = self.eval_fn(artifact, self.held_out)
        else:
            acc = bounded(self.held_out, floor)
        return acc * n, (1.0 - acc) * n

    def full_eval(self, artifact: Evolvable) -> float:
        """``eval_fn`` on the **whole** held-out set. Consumes audit budget.

        Called ``oracle_eval`` until 0.6, and the rename is a correction rather
        than a tidy-up. The name promised an *independent source of truth* and
        the method delivers the same ``eval_fn`` the cheap layers call, differing
        only in **how many tasks** it looks at -- so it can shrink the variance of
        a measurement and can never reveal that the measurement is biased. Reading
        it as ground truth is what made ``docs/`` claim the loop audits itself
        against something outside itself, which it does not. For a genuinely
        independent second source see :mod:`agentdescent.audit`.

        The budget is a real cap, not a counter: ``eval_fn`` is the caller's
        scorer, so on an LLM workload every call is a full held-out sweep of real
        model calls. Once the budget is exhausted this falls back to the cheap
        layer rather than spending money it was told not to spend.

        **That fallback is a downgrade, so the result must not decide a commit.**
        Past the budget this returns a *sub-sample* score, and the aggregator's
        audit gate used to veto on it: measured, a candidate that doubled the
        full-set rate (0.5 -> 1.0) was reported ``oracle-rejected`` because a
        two-task sample could not see the difference. The merge path no longer
        reaches here at all -- see :attr:`full_eval_matches_counts` -- and a
        substitute whose expensive layer degrades the same way should either set
        that attribute or keep the measurement exact.
        """
        if not self.budget.can_spend():
            return self.rule_eval(artifact)
        self.budget.spend()
        return self.eval_fn(artifact, self.held_out)

    def oracle_eval(self, artifact: Evolvable) -> float:
        """Deprecated alias for :meth:`full_eval`. Removed in 0.7.

        Kept because a caller reaching for it is asking a *reasonable* question
        with the wrong word, and failing on `AttributeError` mid-merge is a bad
        way to find that out.
        """
        warnings.warn(
            "ThreeLayerVerifier.oracle_eval is deprecated, use full_eval. The "
            "old name promised an independent source of truth; the method is "
            "eval_fn on the full held-out set -- the same call eval_counts "
            "makes -- so it bounds sampling error and cannot detect bias. For "
            "an independent second source see agentdescent.audit.",
            DeprecationWarning, stacklevel=2)
        return self.full_eval(artifact)


# -- reading a verifier that may predate the 0.6 rename ----------------------
#
# `VerifierProtocol` is structural, so a verifier is whatever a caller hands the
# engine -- including one written against the pre-0.6 page and carrying only the
# old names. These two read either spelling, so the rename costs a warning
# rather than an `AttributeError` in the middle of somebody's merge.


def full_eval_of(verifier: Any) -> Callable[[Evolvable], float]:
    """The verifier's whole-held-out-set scorer, under whichever name it has.

    Prefers ``full_eval``; falls back to a pre-0.6 ``oracle_eval`` and says so
    once per verifier class, naming the class so the warning points at the file
    that has to change.
    """
    fn = getattr(verifier, "full_eval", None)
    if fn is not None:
        return fn
    legacy = getattr(verifier, "oracle_eval", None)
    if legacy is None:
        raise AttributeError(
            f"{type(verifier).__name__} has neither full_eval nor oracle_eval; "
            "VerifierProtocol needs cheap_eval, learned_eval, eval_counts and "
            "full_eval")
    warnings.warn(
        f"{type(verifier).__name__} defines oracle_eval but not full_eval. The "
        "method was renamed in 0.6 -- oracle_eval is read for now and dropped "
        "in 0.7. Rename it; the behaviour is unchanged.",
        DeprecationWarning, stacklevel=2)
    return legacy


def shares_eval_counts(verifier: Any) -> bool:
    """Does this verifier's full-set scorer return what ``eval_counts`` measured?

    True means the aggregator can reuse rates it has already paid for instead of
    buying the same sweep twice -- and, more importantly, instead of routing a
    commit decision through a call that degrades to a sub-sample once the budget
    is gone. Reads :attr:`~ThreeLayerVerifier.full_eval_matches_counts`, then the
    pre-0.6 ``oracle_shares_full_set``.
    """
    # A *declaration* beats an inherited default. `ThreeLayerVerifier` sets
    # `full_eval_matches_counts = True` as a statement about how that class is
    # written, and a subclass inherits it -- so the escape hatch the attribute's
    # docstring offers ("simply does not define it") does not exist for a
    # subclass, and one that set the pre-0.6 `oracle_shares_full_set = False`
    # was ignored. Its independent expensive layer then went uncalled and the
    # gate degraded to the cheap measurement it existed to cross-check, with no
    # warning.
    if _declares(verifier, "full_eval_matches_counts"):
        return bool(getattr(verifier, "full_eval_matches_counts"))
    if _declares(verifier, "oracle_shares_full_set"):
        return bool(getattr(verifier, "oracle_shares_full_set"))
    return bool(getattr(verifier, "full_eval_matches_counts", False))


def _declares(obj: Any, name: str) -> bool:
    """Did *this* verifier set ``name``, rather than inherit the base default?"""
    if name in vars(obj):
        return True
    for klass in type(obj).__mro__:
        if klass is ThreeLayerVerifier or klass is object:
            continue
        if name in vars(klass):
            return True
    return False
