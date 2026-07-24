"""Three-layer verifier: rule / learned / oracle (design doc, section 3.1).

The aggregator uses verification at two very different price points:

* **cheap_eval** (rule + learned) -- run constantly, on every rebase check and
  every candidate in a fusion tournament.  Fast, noisy, no budget.
* **full_eval / oracle** -- ground truth, expensive, and *budgeted* by the
  :class:`~concordia.scheduler.AuditScheduler` (design doc, section 5.3).

The learned layer also exposes an *uncertainty*, which feeds the audit priority
``blast_radius * uncertainty / trust``.  The oracle is the only source of truth
and, crucially, is part of the frozen L0 layer -- it cannot be evolved
(design doc, section 6, "L0's necessity").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

from .evolvable import Evolvable

# eval_fn(artifact, tasks) -> accuracy in [0, 1]
EvalFn = Callable[[Evolvable, Sequence], float]


@dataclass
class VerifierBudget:
    """Oracle call budget, consumed by :meth:`ThreeLayerVerifier.oracle_eval`."""

    oracle_calls_remaining: int = 200
    oracle_calls_used: int = 0

    def can_spend(self) -> bool:
        return self.oracle_calls_remaining > 0

    def spend(self) -> None:
        self.oracle_calls_remaining -= 1
        self.oracle_calls_used += 1


@dataclass
class ThreeLayerVerifier:
    """Rule / learned / oracle backend for the aggregator.

    ``eval_fn`` is the ground-truth scorer supplied by the domain.  The rule and
    learned layers approximate it cheaply (a small task subset, plus noise for
    the learned layer); the oracle runs it on the full held-out set.
    """

    eval_fn: EvalFn
    held_out: Sequence
    rule_subset: int = 8
    learned_noise: float = 0.04
    seed: int = 0
    budget: VerifierBudget = field(default_factory=VerifierBudget)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _subset(self, k: int) -> Sequence:
        if k >= len(self.held_out):
            return self.held_out
        idx = self._rng.sample(range(len(self.held_out)), k)
        return [self.held_out[i] for i in idx]

    def rule_eval(self, artifact: Evolvable) -> float:
        """Cheap, deterministic-ish check on a tiny subset."""
        return self.eval_fn(artifact, self._subset(self.rule_subset))

    def learned_eval(self, artifact: Evolvable) -> Tuple[float, float]:
        """Noisy proxy that also returns an uncertainty estimate.

        Returns ``(score, uncertainty)``.  Uncertainty grows when the artifact
        is under-observed; here we approximate it with the noise band, which is
        what the audit scheduler needs to rank oracle spending."""
        subset = self._subset(self.rule_subset * 2)
        base = self.eval_fn(artifact, subset)
        noisy = min(1.0, max(0.0, base + self._rng.gauss(0.0, self.learned_noise)))
        uncertainty = self.learned_noise + 0.5 / (1 + len(subset))
        return noisy, uncertainty

    def cheap_eval(self, artifact: Evolvable) -> float:
        """The signal used everywhere a budget-free score is needed."""
        rule = self.rule_eval(artifact)
        learned, _ = self.learned_eval(artifact)
        return 0.5 * rule + 0.5 * learned

    def eval_counts(self, artifact: Evolvable) -> Tuple[float, float]:
        """Return (successes, failures) on the full held-out set.

        Feeds the aggregator's Beta-posterior acceptance test with an honest
        sample size (design doc, section 4.4)."""
        acc = self.eval_fn(artifact, self.held_out)
        n = float(len(self.held_out))
        return acc * n, (1.0 - acc) * n

    def oracle_eval(self, artifact: Evolvable) -> float:
        """Ground truth on the full held-out set. Consumes audit budget."""
        if self.budget.can_spend():
            self.budget.spend()
        return self.eval_fn(artifact, self.held_out)
