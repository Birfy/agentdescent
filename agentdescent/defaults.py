"""The shipped evolution algorithm, as replaceable pieces.

`agentdescent/policies.py` says what a decision *is*; this module is the one the
engine makes by default. Splitting them is what makes "use a different accept
rule" a parameter rather than a rewrite of `Aggregator`.

Every class here is the code that used to live inside `Aggregator._process` and
its neighbours, moved and not rewritten. That is deliberate to the point of
pedantry: each of those rules has a history -- the conflict loop keeps going
after a win because stopping at the first one left mutually contradicting cards
behind, the regression guard reads the full held-out set because reading the
cheap layer made it judge quality from a sub-sample -- and a rewrite is how a
codebase forgets things it has already paid to learn.

The one thing that is new is `AcceptancePolicy` taking a `MergeContext` instead
of reaching into the aggregator's state. That is what lets a replacement be
written without importing `Aggregator` at all.
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple,
)

from .evolvable import Diff, EvidenceCard, Evolvable
from .policies import (
    AcceptDecision, MergeContext, Promotion, ProposalContext,
)
from .stats import annealed_delta, prob_improvement

if TYPE_CHECKING:                                # pragma: no cover
    from .aggregator import MergeReport

__all__ = [
    "DefaultAcceptance",
    "DefaultConflict",
    "DefaultFusion",
    "DefaultPromotion",
    "SingleProposal",
]


class SingleProposal:
    """Adapts today's ``propose(rendered, task, output, reward)`` callable.

    One proposal per rollout, no history, no view of the base version -- exactly
    what the callable can express. A policy that wants k variants replaces this
    one; the contract stopped being the limit, the default did not change."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def propose(self, ctx: ProposalContext) -> Sequence[str]:
        proposal = self._fn(ctx.rendered, ctx.task, ctx.output, ctx.reward)
        return [proposal] if proposal else []


class DefaultConflict:
    """Drop contradicting diffs, keeping whichever scores better (PCGrad-style)."""

    def __init__(self, verifier) -> None:
        self.verifier = verifier

    def resolve(self, artifact: Evolvable,
                cards: Sequence[EvidenceCard]) -> Tuple[List[EvidenceCard], int]:
        from .aggregator import diffs_contradict

        dropped = 0
        kept: List[EvidenceCard] = []
        for card in cards:
            # Resolve against everything already kept, and keep going after a win:
            # a card that displaces one survivor may contradict another (it can
            # touch several keys). Stopping at the first conflict left mutually
            # contradicting cards in `kept`, which then made the tournament's
            # "no contradictions" guard false and silently skipped the fusion.
            survivor: Optional[EvidenceCard] = card
            while survivor is not None:
                idx = next((i for i, k in enumerate(kept)
                            if diffs_contradict(survivor.diff, k.diff)), None)
                if idx is None:
                    break
                # project out the worse of the two on the held-out subset.
                d_score = self.verifier.cheap_eval(artifact.apply(survivor.diff))
                k_score = self.verifier.cheap_eval(artifact.apply(kept[idx].diff))
                dropped += 1
                if d_score > k_score:
                    kept.pop(idx)          # the newcomer wins; re-check the rest
                else:
                    survivor = None        # the newcomer loses; drop it
            if survivor is not None:
                kept.append(survivor)
        return kept, dropped


class DefaultFusion:
    """Run each candidate, plus their fusion, in a cheap held-out tournament."""

    def __init__(self, verifier) -> None:
        self.verifier = verifier

    def select(self, artifact: Evolvable,
               diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        from .aggregator import diffs_contradict, fuse_diffs

        candidates: List[Tuple[Diff, Evolvable, bool]] = []
        for d in diffs:
            candidates.append((d, artifact.apply(d), False))
        # add a fused candidate if the survivors are mutually complementary.
        if len(diffs) > 1 and not any(
            diffs_contradict(a, b) for i, a in enumerate(diffs) for b in diffs[i + 1:]
        ):
            fused = fuse_diffs(diffs)
            candidates.append((fused, artifact.apply(fused), True))

        best = max(candidates, key=lambda c: self.verifier.cheap_eval(c[1]))
        return best


class DefaultAcceptance:
    """Beta posterior on the improvement, plus a guard against measured regression.

    Moved from `Aggregator._process` unchanged, including the parts that are only
    obvious once you know what they fixed:

    * **Both gates read the full held-out set** (`base_counts` / `cand_counts`,
      i.e. `verifier.eval_counts`), never the cheap layer. The regression guard
      used to read `cheap_eval`, which `cheap_eval_tasks` sub-samples -- so a
      four-task sample could veto a commit the full-set Beta test had just
      approved, while the docs promised sub-sampling never affected commit
      safety. `MergeContext` names the two differently so this is awkward to
      write again.
    * **The draw is seeded per candidate**, not per round. A shared seed made the
      ~0.003 Monte-Carlo error identical for every candidate in a round, so on
      knife-edge cases one stream accepted every marginal diff and another
      rejected all of them.
    * **The reported reason names the gate that actually fired.** Reporting the
      posterior either way produced lines like `P(delta>0)=0.97 <= 0.50` when the
      regression guard was the real cause -- self-contradictory to anyone reading
      `outcomes()` to find out why nothing committed, which is its one job.
    """

    def __init__(self, base_delta: float, anneal_half_life: int,
                 accept_samples: int) -> None:
        self.base_delta = base_delta
        self.anneal_half_life = anneal_half_life
        self.accept_samples = accept_samples

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        from .evolvable import stable_hash
        from .stats import BetaPosterior

        prior = ctx.prior if ctx.prior is not None else BetaPosterior()
        base_s, base_f = ctx.base_counts
        cand_s, cand_f = ctx.cand_counts
        base_full = ctx.rate(ctx.base_counts)
        cand_full = ctx.rate(ctx.cand_counts)

        baseline_post = BetaPosterior(prior.successes + base_s, prior.failures + base_f)
        candidate_post = BetaPosterior(prior.successes + cand_s, prior.failures + cand_f)
        # fold each contributing worker's local before/after delta as extra
        # evidence for the candidate (the "gradient magnitude").
        if ctx.diff is not None:
            for card in ctx.cards:
                if set(card.diff.ops) & set(ctx.diff.ops):
                    candidate_post.observe_delta(card.before_after_delta)

        version = getattr(ctx.artifact, "version", 1)
        delta = annealed_delta(self.base_delta, version,
                               half_life=self.anneal_half_life)
        p_improve = prob_improvement(
            candidate_post, baseline_post, samples=self.accept_samples,
            seed=stable_hash((version, getattr(ctx.diff, "diff_id", ""))) & 0x7FFFFFFF)

        regressed = cand_full < base_full
        observed = cand_full - base_full
        if p_improve <= 1.0 - delta or regressed:
            reason = (f"held-out regression {base_full:.3f} -> {cand_full:.3f}"
                      if regressed else
                      f"P(delta>0)={p_improve:.2f} <= {1-delta:.2f}")
            return AcceptDecision(False, "below-threshold", reason,
                                  p_improve=p_improve, observed_delta=observed)
        return AcceptDecision(True, "committed", "", p_improve=p_improve,
                              observed_delta=observed)


class DefaultPromotion:
    """Promote dev -> stable after K **regression-free rounds** on dev.

    Moved verbatim, and this one is worth reading before replacing. The counter
    used to be bumped on the *commit* path, so it measured how many times an
    artifact had **changed** rather than how long it had held up -- the opposite
    of every description of it. The incentive inverted: an artifact that
    converged, which is the success state, stopped committing and so could never
    be promoted, while one that thrashed promoted every K commits. In the shipped
    demo the artifact reached 1.000 held-out accuracy after two commits and then
    sat there for 36 rounds with `stable` stuck at 0.000.

    So one round is one `step()`. A commit restarts the clock -- the new version
    has survived nothing yet -- and so does an oracle rejection, which is a
    measured regression. Everything else is a round survived.

    Artifacts absent from this round's reports still age: not being touched is
    the ordinary way to survive a round, and only counting the ones that were
    looked at would mean a quiet artifact never promotes.
    """

    def __init__(self, promote_after_k: int) -> None:
        self.promote_after_k = promote_after_k
        self._survival: Dict[str, int] = defaultdict(int)

    @property
    def survival(self) -> Dict[str, int]:
        """Rounds survived per artifact -- exposed for diagnostics, not control."""
        return self._survival

    def observe(self, reports: Sequence["MergeReport"]) -> Sequence[Promotion]:
        by_id = {r.artifact_id: r for r in reports}
        out: List[Promotion] = []
        for aid in set(self._survival) | set(by_id):
            rep = by_id.get(aid)
            if rep is not None and (rep.committed_version is not None
                                    or rep.category == "oracle-rejected"):
                self._survival[aid] = 0        # changed, or measurably worse
                continue
            self._survival[aid] += 1
            if self._survival[aid] >= self.promote_after_k:
                out.append(Promotion(aid, f"{self._survival[aid]} rounds survived"))
                self._survival[aid] = 0
        return out
