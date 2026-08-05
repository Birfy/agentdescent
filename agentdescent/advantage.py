"""Three RL *decision rules*, mapped onto diff acceptance. No gradients.

`docs/concepts.md` carries a table of analogies where every row lands on real
code. This module exists to keep it that way while adding three more rows, and
the first thing worth saying is what it deliberately does **not** contain:

**There is no PPO here, and no GRPO.** There are no parameters, no log
probabilities, no policy distribution, and therefore no importance ratio and no
clipping objective. A class named ``PPOAggregator`` that computes none of those
would make the analogy table stop being a map of the code, which is the only
thing it is good for.

What *is* transferable is three decision rules, and two of them already had a
stub here that was never used as a mechanism:

| RL                          | here                                    | before |
|-----------------------------|-----------------------------------------|--------|
| group-relative advantage    | a proposal's reward against its group's | only the zero-advantage *filter*: `sampling.DifficultyWeighted` drops all-pass/all-fail groups. The relative value itself never reached acceptance. |
| trust region / clipping     | the op and character caps on one diff   | constants (`6` ops, `32_000` chars), chosen by guess |
| KL to a reference policy    | distance from the `stable` branch       | did not exist |

## Everything here is off by default

Not conservatism -- the issue this implements sets the bar itself: *an analogy
without an A/B is decoration*. So the **signals** are computed and recorded (they
are arithmetic over numbers the engine already has, and are useful as diagnostics
on their own), while the **policies that act on them** are opt-in. Each has to
earn its place with a measured A/B on a real dataset before anything here is
described as a mechanism rather than an option, and a rule that fails its A/B
should be deleted, with the negative result written down.

    Policies(acceptance=AdvantageAcceptance(base), conflict=AdvantageConflict())
    AggregatorConfig(trust_region_policy=AdaptiveTrustRegion())

See `docs/concepts.md` for the current status of each.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Sequence, Tuple

__all__ = [
    "AdaptiveTrustRegion",
    "AdvantageAcceptance",
    "AdvantageConflict",
    "GroupAdvantage",
    "StableDistanceAcceptance",
    "TrustRegion",
    "state_distance",
]


# ---------------------------------------------------------------------------
# 1. Group-relative advantage (GRPO's one transferable idea)
# ---------------------------------------------------------------------------


class GroupAdvantage:
    """Standardise a rollout's reward against the group it belongs to.

    A group is **one base version and one task cluster**: rollouts that were
    solving comparable problems against comparable artifacts. Within it,
    ``(reward - mean) / std`` is what GRPO calls the advantage, and it is the one
    part of GRPO that needs no policy distribution to compute.

    Why the engine was missing it: acceptance reads a Beta posterior over
    ``(successes, failures)``, which cannot see *how well the other proposals in
    the same group did*. A proposal that passed everything in an easy group and
    one that beat its peers in a hard group arrive at the gate looking identical.

    **Running, not batched.** The statistics accumulate as rollouts land rather
    than being computed at a round barrier, because the barrier-free runtime has
    no barrier -- a batched version would silently do nothing there, which is the
    worse failure. The consequence is that the first members of a group are
    standardised against fewer peers than the last, which is why:

    **A group smaller than ``min_group`` gets ``None``, not zero.** A z-score
    over two samples is noise, and zero is a *meaningful* advantage value ("did
    exactly as well as its peers"), so returning it for "we do not know yet"
    would put noise into the gate wearing the costume of a measurement.

    A group where everything scored the same also gets ``None``: zero variance is
    the zero-advantage case `DifficultyWeighted` already filters on, and dividing
    by it is how a rounding error becomes a large advantage.
    """

    def __init__(self, min_group: int = 4) -> None:
        if min_group < 2:
            raise ValueError("a group needs at least two members to have a spread")
        self.min_group = min_group
        self._n: Dict[str, int] = {}
        self._sum: Dict[str, float] = {}
        self._sq: Dict[str, float] = {}

    @staticmethod
    def key(base_version: int, cluster: str = "") -> str:
        """The group a rollout belongs to. Same base, same cluster."""
        return f"{base_version}:{cluster}"

    def observe(self, key: str, reward: float) -> Optional[float]:
        """Record a reward and return its advantage, or ``None`` if unknown yet.

        The returned value standardises the reward against the group *including*
        itself. Excluding it (a leave-one-out advantage) is defensible too and
        differs by O(1/n); including it is what GRPO's own formulation does.
        """
        self._n[key] = self._n.get(key, 0) + 1
        self._sum[key] = self._sum.get(key, 0.0) + reward
        self._sq[key] = self._sq.get(key, 0.0) + reward * reward
        n = self._n[key]
        if n < self.min_group:
            return None
        mean = self._sum[key] / n
        var = max(0.0, self._sq[key] / n - mean * mean)
        if var <= 1e-12:
            return None                    # zero advantage, the GRPO filter case
        return (reward - mean) / math.sqrt(var)

    def group_size(self, key: str) -> int:
        return self._n.get(key, 0)


class AdvantageAcceptance:
    """Shift the acceptance prior by how well a proposal did against its group.

    Wraps another :class:`~agentdescent.policies.AcceptancePolicy` rather than
    replacing it: the Beta test, the regression guard and their whole recorded
    history stay exactly as they are, and this adds one term. A replacement would
    quietly drop the guards, which is how a codebase forgets what it paid to
    learn.

    ``strength`` is in *pseudo-observations*: an advantage of +1 (one standard
    deviation above its group) adds ``strength`` successes to the candidate's
    prior. Small by default, because the prior is the weakest of the three inputs
    to the decision and a large value would let a lucky group override a measured
    held-out regression.

    Cards without an advantage are skipped, not counted as zero -- see
    :class:`GroupAdvantage`.
    """

    def __init__(self, inner, strength: float = 1.0) -> None:
        if strength < 0:
            raise ValueError("strength must not be negative")
        self.inner = inner
        self.strength = strength

    def accept(self, ctx):
        from dataclasses import replace as _replace

        from .stats import BetaPosterior

        advantages = [c.advantage for c in ctx.cards
                      if getattr(c, "advantage", None) is not None]
        if not advantages:
            return self.inner.accept(ctx)
        mean = sum(advantages) / len(advantages)
        prior = ctx.prior if ctx.prior is not None else BetaPosterior()
        shift = self.strength * mean
        shifted = BetaPosterior(
            successes=max(0.0, prior.successes + max(0.0, shift)),
            failures=max(0.0, prior.failures + max(0.0, -shift)))
        return self.inner.accept(_replace(ctx, prior=shifted))


class AdvantageConflict:
    """Break a contradiction by group-relative advantage, not raw score.

    The default rule keeps whichever of two contradicting diffs scores better on
    the cheap held-out sub-sample. That is a comparison between two *measurements
    of the same thing*, and it is the right rule when both were measured the same
    way. Advantage is the tie-breaker for when they were not: a proposal that beat
    a hard group by a wide margin carries more evidence than one that matched an
    easy group, and the cheap sub-sample cannot see the difference.

    Falls through to the wrapped policy whenever either side has no advantage, or
    the two are within ``margin`` -- so it only fires where it has something the
    default does not.
    """

    def __init__(self, inner, margin: float = 0.5) -> None:
        self.inner = inner
        self.margin = margin

    def resolve(self, artifact, cards):
        from .aggregator import diffs_contradict

        dropped = 0
        kept = []
        for card in cards:
            survivor = card
            while survivor is not None:
                idx = next((i for i, k in enumerate(kept)
                            if diffs_contradict(survivor.diff, k.diff)), None)
                if idx is None:
                    break
                a = getattr(survivor, "advantage", None)
                b = getattr(kept[idx], "advantage", None)
                if a is None or b is None or abs(a - b) < self.margin:
                    # Nothing to add: hand this pair to the wrapped rule by
                    # resolving the two of them on their own.
                    resolved, sub_dropped = self.inner.resolve(
                        artifact, [kept[idx], survivor])
                    dropped += sub_dropped
                    kept.pop(idx)
                    # Whatever the inner rule kept goes back at the same
                    # position, and the loop ends for this card either way. If it
                    # kept *both* -- which the shipped rule never does for a
                    # contradicting pair, but a replacement may -- re-entering
                    # the loop would find the same pair, ask the same question
                    # and get the same answer, forever.
                    kept[idx:idx] = list(resolved)
                    survivor = None
                    break
                dropped += 1
                if a > b:
                    kept.pop(idx)
                else:
                    survivor = None
            if survivor is not None:
                kept.append(survivor)
        return kept, dropped


# ---------------------------------------------------------------------------
# 2. An adaptive trust region (PPO's adaptive KL coefficient)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustRegion:
    """How large one diff may be: operations, and characters."""

    ops: int
    chars: int


class AdaptiveTrustRegion:
    """Widen the diff-size cap while merges land; tighten when they do not.

    The shipped caps are ``6`` ops and ``32_000`` characters. Both are guesses --
    the second exists because a comment records that the op count alone was not
    enough -- and a guess that is right for a rule table is wrong for a file tree
    in both directions: too tight and a legitimate multi-file change can never be
    proposed, too loose and a runaway reflector rewrites the artifact in one step.

    The rule is PPO's adaptive KL coefficient with the terms renamed. Over the
    last ``window`` merges: a high acceptance rate with no reversals means the
    region is not what is limiting progress, so widen it; any reversal -- a
    rollback, or an oracle veto -- means the last widening bought a change that
    should not have landed, so tighten. Reversals are weighted more heavily than
    acceptances because they are the expensive direction.

    Bounded at both ends. Unbounded growth ends with a trust region that does not
    constrain anything, which is the same as not having one.
    """

    def __init__(self, *, initial: TrustRegion = TrustRegion(6, 32_000),
                 minimum: TrustRegion = TrustRegion(1, 2_000),
                 maximum: TrustRegion = TrustRegion(64, 512_000),
                 window: int = 10, widen: float = 1.25, tighten: float = 0.5,
                 accept_rate_to_widen: float = 0.5) -> None:
        self.current = initial
        self.minimum, self.maximum = minimum, maximum
        self.window = window
        self.widen, self.tighten = widen, tighten
        self.accept_rate_to_widen = accept_rate_to_widen
        self._recent: Deque[str] = deque(maxlen=window)
        #: Every value the region has taken, so an A/B can show whether it moved
        #: at all. A knob that adapts to exactly where it started is a knob that
        #: did nothing, and that is indistinguishable from a constant unless it
        #: is recorded.
        self.history: list = [initial]

    def observe(self, outcome: str) -> TrustRegion:
        """Record one merge outcome and return the region for the next merge.

        ``outcome`` is a :class:`~agentdescent.aggregator.MergeOutcome` value --
        ``"committed"``, ``"oracle-rejected"``, ``"below-threshold"``, and so on.
        """
        self._recent.append(outcome)
        if len(self._recent) < self.window:
            return self.current           # not enough evidence to move anything
        reversals = sum(1 for o in self._recent
                        if o in ("oracle-rejected", "rolled-back"))
        accepted = sum(1 for o in self._recent if o == "committed")
        rate = accepted / len(self._recent)
        if reversals:
            factor = self.tighten
        elif rate >= self.accept_rate_to_widen:
            factor = self.widen
        else:
            # Merges are being refused by the *gate*, not by size. Widening would
            # buy bigger proposals for a gate that is already saying no, and
            # tightening would blame the region for something it did not do.
            return self.current
        nxt = TrustRegion(
            ops=int(min(self.maximum.ops,
                        max(self.minimum.ops, round(self.current.ops * factor)))),
            chars=int(min(self.maximum.chars,
                          max(self.minimum.chars,
                              round(self.current.chars * factor)))))
        if nxt != self.current:
            self.current = nxt
            self.history.append(nxt)
            self._recent.clear()          # give the new size a fresh window
        return self.current


# ---------------------------------------------------------------------------
# 3. Distance from `stable` (the KL-to-reference analogy)
# ---------------------------------------------------------------------------


def state_distance(a, b) -> float:
    """Fraction of keys on which two artifact states differ, in ``[0, 1]``.

    Symmetric, and counts a key present in one and missing from the other as a
    difference -- a deletion moves the artifact as much as an edit does. Two
    empty states are at distance zero rather than raising.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return sum(1 for k in keys if a.get(k) != b.get(k)) / len(keys)


class StableDistanceAcceptance:
    """Penalise candidates that drift far from the confirmed branch.

    The KL-to-a-reference-policy analogy, with ``stable`` as the reference. It
    matters most for **L1** artifacts -- a harness, a router, a learned verifier
    -- where the blast radius is large and "how far is this from the thing we know
    works" is real information that today's gate does not use: L1 safety rests
    entirely on the oracle's single veto.

    Implemented as a *required extra margin*, not as a subtraction from the score.
    Scores here are held-out rates that go into a Beta posterior, and quietly
    lowering one would corrupt every downstream number that reads it. Instead a
    candidate far from `stable` has to clear a proportionally higher bar, and when
    it does not, the refusal says so.

    Wraps another policy, and does nothing at all when the context carries no
    stable distance -- so a run whose ledger has no `stable` branch yet behaves
    exactly as before.
    """

    def __init__(self, inner, strength: float = 0.1) -> None:
        if strength < 0:
            raise ValueError("strength must not be negative")
        self.inner = inner
        self.strength = strength

    def accept(self, ctx):
        from dataclasses import replace as _replace

        from .policies import AcceptDecision

        decision = self.inner.accept(ctx)
        distance = getattr(ctx, "stable_distance", 0.0) or 0.0
        if not decision.accept or distance <= 0.0:
            return decision
        required = self.strength * distance
        if decision.observed_delta >= required:
            return decision
        return _replace(
            decision, accept=False, category="below-threshold",
            detail=(f"{decision.detail}; refused by the stable-distance penalty: "
                    f"observed delta {decision.observed_delta:+.4f} < "
                    f"{required:.4f} required at distance {distance:.2f} from "
                    f"stable"))
