"""Statistical acceptance primitives (design doc, section 4.4).

The aggregator does not accept a diff because a point estimate crossed a
threshold; it accepts when the *posterior probability of improvement* exceeds
``1 - delta``.  This gives evidence-starved tail artifacts an automatically more
conservative effective update -- the discrete-space analogue of Adam's
per-parameter adaptive step size.

We keep a Beta(successes, failures) posterior per artifact and, for a candidate
diff, a Beta posterior over its own trials.  ``prob_improvement`` estimates
``P(theta_diff > theta_baseline)`` by Monte-Carlo over the two Beta posteriors
(closed forms exist but MC keeps the code dependency-free and obvious).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """Sample from Beta(alpha, beta) via two Gamma draws (no numpy dependency)."""
    x = rng.gammavariate(alpha, 1.0)
    y = rng.gammavariate(beta, 1.0)
    return x / (x + y)


@dataclass
class BetaPosterior:
    """A Beta(successes + 1, failures + 1) posterior over a success rate."""

    successes: float = 0.0
    failures: float = 0.0

    @property
    def alpha(self) -> float:
        return self.successes + 1.0

    @property
    def beta(self) -> float:
        return self.failures + 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        if success:
            self.successes += weight
        else:
            self.failures += weight

    def observe_delta(self, delta: float) -> None:
        """Fold a signed before/after delta into the posterior.

        Positive delta -> evidence of success, negative -> failure.  The
        magnitude scales the pseudo-count so a large measured improvement counts
        for more than a marginal one."""
        w = min(1.0, abs(delta) * 4.0) + 1e-6
        self.update(delta > 0, weight=w)

    def sample(self, rng: random.Random) -> float:
        return _beta_sample(self.alpha, self.beta, rng)


def prob_improvement(
    candidate: BetaPosterior,
    baseline: BetaPosterior,
    samples: int = 4000,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of ``P(candidate_rate > baseline_rate)``."""
    rng = random.Random(seed)
    wins = 0
    for _ in range(samples):
        if candidate.sample(rng) > baseline.sample(rng):
            wins += 1
    return wins / samples


def annealed_delta(base_delta: float, version: int, half_life: int = 64) -> float:
    """Anneal the acceptance risk ``delta`` with version number (lr decay).

    Early on we accept readily (large delta); as an artifact matures we demand
    higher posterior confidence (design doc, section 4.4, "delta anneals with
    version number")."""
    factor = 0.5 ** (version / max(1, half_life))
    return max(0.01, base_delta * factor)


def difficulty_weight(pass_rate: float, floor: float = 0.0) -> float:
    """`4*p*(1-p)` -- how much learning signal an item still carries.

    Peaks at `p = 0.5` and vanishes at both extremes: something that always
    passes yields no gradient, and neither does something that never passes
    whatever the artifact says (the GRPO zero-advantage argument).

    Lives here because two schedulers need it at two granularities --
    :class:`~agentdescent.sampling.DifficultyWeighted` over *tasks* and
    :class:`~agentdescent.scheduler.TaskScheduler` over task *clusters* -- and
    they had written it out separately, which is one formula and two places for
    it to drift."""
    p = min(1.0, max(0.0, pass_rate))
    return 4.0 * p * (1.0 - p) + floor


def ucb_score(value: float, n_s: float, total: float, c: float = 1.4) -> float:
    """UCB1 acquisition used by the TaskScheduler (design doc, section 5.2).

    ``score(s) = expected_learning_value(s) + c * sqrt(ln T / n_s)``.
    Tail artifacts (small ``n_s``) automatically receive an exploration bonus.
    """
    if n_s <= 0:
        return float("inf")
    return value + c * math.sqrt(math.log(max(2.0, total)) / n_s)
