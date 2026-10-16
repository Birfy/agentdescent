"""Merge diffs that disagree, by asking a model for their union.

`fuse_diffs` is a dictionary update. On different keys that is the whole of
merging; on the *same* key it is last-writer-wins, which is not a merge. So
`DefaultFusion` declines to build a fused candidate once any pair contradicts,
and for an artifact held in **one key** that is every round -- GEPA's
``InstructionSlot`` is the whole instruction under ``"instruction"``, and a full
run of it records ``fusion_stats().contested == 0``.

:class:`ReflectiveFusion` closes that gap the only way text allows. It asks a
model to write one value keeping what each proposal contributed, for the keys the
diffs actually disagree on -- keys they agree on stay the plain union, because a
model asked to merge values that do not disagree can only make them worse.

**There is no tournament.** The union goes straight to the acceptance gate. That
is a deliberate design decision and it is worth being precise about what it
buys and what it costs.

*Buys:* measured on a workload where both paths reach the same final quality,
55 model calls against 114 -- **52% cheaper**. Ranking every candidate was the
single largest cost in a merge, and nothing here ranks anything.

*Costs, and they are not the same size:*

**Acceptable.** The union can commit while being worse than the best single
proposal would have been. It can still never commit a *regression*: the gate
scores it on the full held-out set, runs the Beta test and the regression guard,
and none of that is touched. The bar drops from "beat the best proposal" to
"beat the artifact you started from".

**Load-bearing, and gone.** `best_single_score` is what answers *"does merging
just average the improvements away?"* -- the sharpest objection to this whole
design -- and nothing here scores a single, so there is no answer to record.
Every trial carries ``ranked=False``, `fusion_stats()` reports them as
``unranked`` and keeps them out of the win-rate denominator, and `win_rate` is
`None`. **A union that was never compared has not won anything, and the
statistics cannot pretend otherwise.** Anyone trying to *measure* whether merging
helps needs the shipped `DefaultFusion` on a multi-key artifact instead.

The one path that still ranks is the fallback: when the model cannot be used --
a dead backend, an empty or oversized answer, or one that merely repeats an
input -- the merge falls through to `DefaultFusion` rather than losing the
round's work.

    from agentdescent import Policies, evolve
    from agentdescent.fusion import reflective_merge

    evolve(tasks, reward, agent=agent, n_workers=4,
           policies=Policies(**reflective_merge(completion)))

It ships as a **pair** with :class:`KeepContradictions`, through
:func:`reflective_merge`: conflict resolution runs before fusion, so a fusion
policy installed alone is handed one diff on exactly the workloads it was written
for and correctly declines to merge it with itself.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .defaults import DefaultFusion
from .evolvable import Diff, Evolvable, combined_rationale
from .policies import FusionTrial

__all__ = ["MERGE_PROMPT", "KeepContradictions", "ReflectiveFusion",
           "LearnableFusionPolicy", "reflective_merge"]


#: What the model is asked, and it is asked for a **union of deltas** rather than
#: a rewrite. "Write one version that keeps every improvement" invites a fresh
#: composition that happens to cover the same ground; "work out what each
#: proposal changed relative to CURRENT, then apply all of those changes" is an
#: operation with a checkable result, and it is derivable from exactly what a
#: `FusionPolicy` has -- the current value and the competing ones. Nothing here
#: needs the evidence cards, which fusion does not receive.
MERGE_PROMPT = """\
Several independent improvements were made to the same text, each fixing a
different failure. Produce their UNION.

CURRENT
---
{current}
---

{proposals}
Do this:
1. For each proposal, work out what it CHANGED relative to CURRENT.
2. Output CURRENT with every one of those changes applied together.

Rules:
- Every change introduced by any proposal must survive into your output.
- Introduce nothing that no proposal introduced.
- If two proposals changed the same point incompatibly, keep the more specific one.
- Output only the merged text: no preamble, no explanation, no markers.
"""


class KeepContradictions:
    """A conflict policy that leaves contradicting diffs for fusion to resolve.

    The shipped pipeline resolves conflicts at step 2 and fuses at step 3, so by
    the time a fusion policy runs, the contradicting proposals it would have
    merged **have already been thrown away** -- `DefaultConflict` keeps whichever
    of a contradicting pair scores better and drops the other. On a one-key
    artifact that leaves exactly one candidate, which is why installing
    :class:`ReflectiveFusion` alone changes nothing at all: it is handed a single
    diff and correctly declines to merge it with itself.

    Dropping the loser is the right default when the merge step cannot combine
    two values for one key. It stops being right once it can.

    Safe as a pair, because nothing is lost if synthesis fails: the tournament
    scores every surviving single and takes the best, which is the same candidate
    `DefaultConflict`'s pairwise elimination would have left. The visible
    difference is that ``conflicts_dropped`` becomes 0 -- the contradictions were
    not dropped, they were sent somewhere that can use them.

    Use :func:`reflective_merge` rather than installing this by hand.
    """

    def resolve(self, artifact: Evolvable, cards):
        return list(cards), 0


class ReflectiveFusion:
    """Combine contradicting diffs by asking a model to synthesise their values.

    ``complete`` is any ``prompt -> text`` callable -- the same contract as every
    actor in this package, so a metered adapter, a retrying one, or a stub in a
    test all work unchanged.

    ``verifier`` is normally left unset: the engine constructs it, so a caller
    has no way to pass one in. :class:`~agentdescent.aggregator.Aggregator` calls
    :meth:`bind` for any fusion policy that has it.
    """

    def __init__(self, complete, *, verifier: Any = None,
                 max_chars: int = 8_000, max_proposals: int = 6,
                 validate: Optional[Callable[[Any], Any]] = None) -> None:
        if max_proposals < 2:
            raise ValueError("synthesising needs at least two proposals")
        self.complete = complete
        self.verifier = verifier
        #: The synthesised value is written by a model and reaches the ledger
        #: without passing the trust region, which filters *cards* before the
        #: merge. So it is bounded here; a reflector that echoes its own prompt
        #: would otherwise commit a value the trust region exists to stop.
        self.max_chars = max_chars
        #: Beyond this many competing values the prompt stops being a merge
        #: instruction and becomes a summarisation task with a different failure
        #: mode.
        self.max_proposals = max_proposals
        #: Used only when synthesis fails. It is the one path here that ranks
        #: anything, and it exists so a dead backend costs a ranking pass rather
        #: than the round's work.
        self._fallback: Any = None
        #: Applied to the synthesised value before it is returned; anything it
        #: raises on is refused, and the caller's ranking fallback takes the
        #: round instead.
        #:
        #: Without it an artifact with a *shape* -- Python source behind an AST
        #: gate, JSON behind a schema -- cannot use model merging at all, because
        #: the synthesised value reaches the ledger without passing the trust
        #: region and would then fail at every rollout that reads it. SICA and
        #: Godel Agent declared `reflective=False` for exactly that reason and
        #: paid for it: ranking contested diffs costs evaluations, and a paired
        #: control measured 2.7-2.8x the model calls of a merged run at no gain
        #: in quality.
        self.validate = validate
        #: One record per merge. They carry `ranked=False`, so `fusion_stats()`
        #: reports them as unranked rather than as wins or losses.
        self.trials: List[FusionTrial] = []

    def bind(self, verifier: Any) -> None:
        """Receive the engine's verifier, if the caller did not supply one."""
        from .defaults import DefaultFusion

        if self.verifier is None:
            self.verifier = verifier
        if self._fallback is None:
            self._fallback = DefaultFusion(self.verifier)

    # -- the synthesis ------------------------------------------------------

    @staticmethod
    def _partition(diffs: Sequence[Diff]) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        """Split the union of ops into (agreed, contested).

        A key is *agreed* when every diff that touches it proposes the same
        value -- including the ordinary case of only one diff touching it. That
        is precisely the set `fuse_diffs` handles correctly on its own, and it is
        left alone here: a model asked to merge values that do not disagree can
        only make them worse.
        """
        seen: Dict[str, List[Any]] = {}
        for d in diffs:
            for key, value in d.ops.items():
                bucket = seen.setdefault(key, [])
                if value not in bucket:
                    bucket.append(value)
        agreed = {k: v[0] for k, v in seen.items() if len(v) == 1}
        contested = {k: v for k, v in seen.items() if len(v) > 1}
        return agreed, contested

    def _synthesise(self, current: Any, proposals: Sequence[Any]) -> Optional[str]:
        """One merged value, or ``None`` if anything at all went wrong.

        Every rejection path returns ``None`` rather than raising: this runs on
        the commit path of every round, and a merge that dies because a model was
        slow is worse than a merge that did not improve.
        """
        kept = [p for p in proposals if p is not None][:self.max_proposals]
        if len(kept) < 2:
            return None
        listed = "\n".join(
            f"PROPOSAL {i + 1}\n---\n{p}\n---\n" for i, p in enumerate(kept))
        try:
            merged = self.complete(MERGE_PROMPT.format(
                current="" if current is None else current, proposals=listed))
        except Exception:  # noqa: BLE001 - never break a merge over a backend
            return None
        if not merged:
            return None
        merged = merged.strip()
        if not merged or len(merged) > self.max_chars:
            return None
        # An answer that merely repeats one of its inputs is not a merge. Letting
        # it through would enter the tournament as a duplicate candidate and,
        # on a tie, be indistinguishable from the model having contributed.
        if any(merged == str(p).strip() for p in kept) or merged == str(current or "").strip():
            return None
        if self.validate is not None:
            # The one check `to_diff` would have made and this path skips. A
            # merge that cannot be validated is refused rather than committed:
            # returning None here is already the "fall back to ranking" signal.
            try:
                validated = self.validate(merged)
            except Exception:  # noqa: BLE001 - any rejection is a rejection
                return None
            if not validated:
                return None
            merged = validated if isinstance(validated, str) else merged
        return merged

    def _union_of(self, artifact: Evolvable, diffs: Sequence[Diff]) -> Optional[Diff]:
        """The synthesised union of ``diffs``, or ``None`` if it cannot be built.

        All or nothing across the contested keys: a partial union would commit
        some workers' contributions and silently drop the rest, which looks like
        a successful merge and is not one.
        """
        agreed, contested = self._partition(diffs)
        if not contested:
            return None
        ops: Dict[str, Any] = dict(agreed)
        state = getattr(artifact, "state", {}) or {}
        for key, values in contested.items():
            merged = self._synthesise(state.get(key), values)
            if merged is None:
                return None
            ops[key] = merged
        # The synthesised values never went through the strategy's `to_diff`,
        # so a structural gate there (FileTree validators: e.g. a skill's
        # SKILL.md frontmatter) has not seen them. Ask the strategy directly; a
        # rejection is the same "fall back to ranking" signal as a failed merge.
        strategy = getattr(artifact, "_strategy", None) or getattr(artifact, "strategy", None)
        check = getattr(strategy, "check_candidate", None)
        if callable(check):
            try:
                if check(state, ops) is not None:
                    return None
            except Exception:  # noqa: BLE001 - an unverifiable merge is refused
                return None
        return Diff(
            diff_id=f"synth({'+'.join(sorted(d.diff_id for d in diffs))})",
            target=diffs[0].target, ops=ops,
            contract_breaking=any(d.contract_breaking for d in diffs),
            author="aggregator",
            rationale=combined_rationale(diffs))

    # -- the merge ----------------------------------------------------------

    def select(self, artifact: Evolvable,
               diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        """Build the union and hand it straight to the acceptance gate.

        No ranking happens here at all -- not of the singles, not of the union.
        The gate scores the union on the full held-out set, runs the Beta test
        and the regression guard, and decides. See the class docstring for what
        that gives up.
        """
        if len(diffs) <= 1:
            only = diffs[0]
            self._record(artifact, diffs, "single-candidate", ranked=False)
            return only, artifact.apply(only), False

        union = self._union_of(artifact, diffs)
        if union is not None:
            self._record(artifact, diffs, "", ranked=False)
            return union, artifact.apply(union), True

        # The model could not be used -- a dead backend, an empty or oversized
        # answer, or one that merely repeated an input. Falling through to the
        # shipped tournament costs a ranking pass, and it is the only thing here
        # that ever does; losing the round's work instead would be worse than
        # paying for it on the rare merge where synthesis fails.
        self._record(artifact, diffs, "synthesis-failed", ranked=False)
        return self._fallback.select(artifact, diffs)

    def _record(self, artifact, diffs, reason: str, *, ranked: bool) -> None:
        """One trial per merge, carrying no verdict.

        `best_single_score` is what answers "does merging just average the
        improvements away", and nothing here scores a single -- so these trials
        are `ranked=False` and `fusion_stats()` keeps them out of the win-rate
        denominator. A union that was never compared has not won anything, and
        the statistics must not be able to pretend otherwise.
        """
        self.trials.append(FusionTrial(
            artifact_id=getattr(artifact, "id", ""), n_candidates=len(diffs),
            best_single_score=0.0, baseline_score=0.0,
            winner="synthesized" if not reason else "single",
            reason=reason, ranked=ranked))


def reflective_merge(complete, **kwargs) -> Dict[str, Any]:
    """The two policies model-merging needs, as ``Policies`` keyword arguments.

    They only work together. :class:`ReflectiveFusion` installed on its own is a
    no-op on exactly the workloads it was written for, because conflict
    resolution runs first and hands it one diff -- so this returns the pair and
    the mistake is not available::

        evolve(tasks, reward, agent=agent, n_workers=4,
               policies=Policies(**reflective_merge(completion)))
    """
    return {"fusion": ReflectiveFusion(complete, **kwargs),
            "conflict": KeepContradictions()}


class LearnableFusionPolicy:
    """A fusion policy that learns when merging wins -- and stops paying for
    the ranking it cannot change.

    The tournament (``DefaultFusion(tournament=True)``) answers "does merging
    beat the best single?" by scoring every candidate on the cheap layer, every
    merge. The answer lands in a :class:`~agentdescent.policies.FusionTrial`
    and is then thrown away. This policy learns the answer instead: an online
    logistic over **diff-only features** (zero verifier calls) is trained on
    the tournament's own trials, and once it is confident that the union is the
    right answer for a merge shape, it returns the union **without buying the
    cheap sweep**. When the prediction is uncertain -- or says the tournament
    would change the outcome -- it runs the tournament and observes a fresh
    label.

    This is the allocator's idea (#204) applied to the fusion decision: value-
    directed spending of the ranking cost, learned from the run's own history,
    with the same safety. Three guarantees:

    * **The default is the default.** With no trials the policy is
      :class:`DefaultFusion(tournament=True)` verbatim -- every merge is
      ranked, every trial recorded. Learning changes nothing until it has data.
    * **Counterfactual verification.** ``epsilon`` of the confident skips are
      still run through the tournament, so a wrong prediction is observed
      rather than reinforced (the allocator's counterfactual exploration).
    * **A rollback guard.** A counterfactual merge the model was *confident*
      about skipping but the tournament ranked a single instead breaks the
      trust, and after ``max_wrong_skips`` such events (default one) skipping
      is retired permanently -- a rule that learned to be wrong is disabled,
      not patched. The cost of a false alarm is merely losing the saving; the
      cost of a wrong skip is committing a worse merge.

    Commits are unaffected: the skip returns the same union the non-tournament
    default would have committed, and the acceptance gate still scores it on
    the full held-out set before anything commits.
    """

    def __init__(
        self,
        inner: Optional[Any] = None,
        *,
        tournament: bool = True,
        seed: int = 0,
        min_trials: int = 20,
        uncertainty_threshold: float = 0.15,
        epsilon: float = 0.1,
        max_wrong_skips: int = 1,
        lr: float = 0.2,
        reg: float = 0.05,
        max_logit: float = 3.0,
    ) -> None:
        if not (0.0 < uncertainty_threshold < 1.0):
            raise ValueError("uncertainty_threshold must be in (0, 1)")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if max_wrong_skips < 1:
            raise ValueError("max_wrong_skips must be >= 1")
        if not (isinstance(min_trials, int) and min_trials >= 1):
            raise ValueError("min_trials must be an int >= 1")
        for name, value in (("lr", lr), ("reg", reg), ("max_logit", max_logit)):
            if not (isinstance(value, (int, float)) and math.isfinite(value)
                    and value > 0.0):
                raise ValueError(f"{name} must be a finite number > 0")
        self.inner = inner or DefaultFusion(tournament=tournament)
        #: The engine's budget governor flips this when tokens run low
        #: (``aggregator.fusion_policy.tournament = should``); turning it off
        #: degrades to the union path, no labels collected.
        self.tournament = tournament
        #: The engine reads ``aggregator.fusion_policy.trials`` for
        #: ``EvolutionResult.fusion_stats``. These are the inner tournament's
        #: real measured trials; a skipped merge has none by construction.
        self.trials = self.inner.trials
        self.min_trials = int(min_trials)
        self.threshold = float(uncertainty_threshold)
        self.epsilon = float(epsilon)
        self.max_wrong_skips = int(max_wrong_skips)
        self._rng = random.Random(seed)
        self._model = _FusionModel(n_features=5, lr=lr, reg=reg,
                                   max_logit=max_logit)
        self._union_policy = DefaultFusion(tournament=False)
        #: The budget-degraded union path records unranked trials, and those
        #: must land on the **same** list the engine reads -- otherwise
        #: ``fusion_stats`` goes silent exactly when the governor flipped the
        #: tournament off, which is the one time it should still speak.
        self._union_policy.trials = self.inner.trials
        self._skip_enabled = True
        #: Consecutive counterfactual merges the model would have skipped but
        #: the tournament ranked a single instead (a wrong skip in a row).
        self._wrong_skips = 0
        #: Confident skips that avoided the cheap sweep.
        self.skipped = 0
        #: Merges actually ranked (the labels).
        self.tournament_runs = 0

    # -- engine hooks ---------------------------------------------------------

    def bind(self, verifier: Any) -> None:
        """Forward the verifier to the inner tournament and the union path."""
        self.inner.bind(verifier)
        self._union_policy.bind(verifier)

    def configure(self, config: Any) -> None:
        pass

    # -- the fusion decision --------------------------------------------------

    def select(self, artifact: Evolvable,
               diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        """Learn or predict: skip the ranking sweep when the union is
        confidently predicted to win (a learned logistic, not a proof)."""
        if not self.tournament:
            return self._union_policy.select(artifact, diffs)
        union, _reason = self._union_of(diffs)
        if (self._skip_enabled and self._model.n_obs >= self.min_trials
                and union is not None
                and self._rng.random() >= self.epsilon):
            p = self._model.predict(self._features(diffs, union))
            if p >= 1.0 - self.threshold:
                self.skipped += 1
                return self._union_policy.select(artifact, diffs)
        return self._run_tournament(artifact, diffs)

    def _run_tournament(self, artifact: Evolvable,
                        diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        """Rank the candidates (the label-producing path), then observe."""
        self.tournament_runs += 1
        result = self.inner.select(artifact, diffs)
        fused = bool(result[2])
        union, _reason = self._union_of(diffs)
        self._observe(self._features(diffs, union), fused)
        return result

    def _observe(self, features: Tuple[float, ...], fused: bool) -> None:
        # Snapshot the **pre-update** prediction first. The rollback question
        # is "would the rule that was in force have skipped this merge?" -- and
        # a negative label immediately drags the logistic down, often below the
        # confidence cut in one step, so scoring after the update misses exactly
        # the confident-and-wrong counterfactual the guard exists for.
        predicted_skip = self._model.predict(features) >= 1.0 - self.threshold
        self._model.update(features, fused)
        if self._model.n_obs < self.min_trials:
            return
        # Skipping is right iff the tournament also picked the union
        # (``fused``). The rollback guard is deliberately unforgiving: a
        # counterfactual merge the model was *confident* about skipping but the
        # tournament ranked a single instead breaks the trust, and after
        # ``max_wrong_skips`` such events (default one) skipping is retired
        # permanently. The cost of a false alarm is merely losing the saving;
        # the cost of a wrong skip is committing a worse merge. Conservative is
        # correct here.
        if predicted_skip and not fused:
            self._wrong_skips += 1
            if self._wrong_skips >= self.max_wrong_skips:
                self._skip_enabled = False
        else:
            self._wrong_skips = 0

    # -- features (diff-only, zero verifier calls) -----------------------------

    def _union_of(self, diffs: List[Diff]) -> Tuple[Optional[Diff], str]:
        """The fused candidate, via the inner policy's own rule.

        Deliberately delegated rather than copied: ``DefaultFusion._union_of``
        is the single source of the three-way "nothing-to-fuse" rule, and a
        byte-for-byte copy here would silently drift if that rule ever changed
        again. The inner is the tournament policy the learner wraps; it must
        have a ``_union_of`` (``DefaultFusion`` does)."""
        return self.inner._union_of(diffs)

    @staticmethod
    def _features(diffs: List[Diff], union: Optional[Diff]) -> Tuple[float, ...]:
        """The pre-fusion signal the learner decides on: merge shape, size and
        key overlap, all computable from the diffs alone."""
        import math as _math

        keys = [set(d.ops) for d in diffs]
        union_keys = set().union(*keys) if keys else set()
        inter = set.intersection(*keys) if keys else set()
        overlap = len(inter) / len(union_keys) if union_keys else 0.0
        total_ops = sum(len(d.ops) for d in diffs)
        max_ops = max((len(d.ops) for d in diffs), default=0)
        return (
            _math.log1p(len(diffs)),
            overlap,
            _math.log1p(total_ops),
            _math.log1p(max_ops),
            1.0 if union is not None else 0.0,
        )

    def status(self) -> dict:
        # The skip fraction's denominator is *all* merges decided (skipped or
        # ranked) -- "skips / tournament_runs" would climb past 1.0 on a
        # warm skip-heavy run and read as an occupancy instead of a fraction.
        decided = self.skipped + self.tournament_runs
        return {
            "skipped": self.skipped,
            "tournament_runs": self.tournament_runs,
            "model_observations": self._model.n_obs,
            "skip_enabled": self._skip_enabled,
            "skips_per_merge": (self.skipped / decided
                                if decided else 0.0),
        }


class _FusionModel:
    """A tiny online logistic over the fusion features.

    Deliberately a private twin of the allocator's
    :class:`~agentdescent.allocator.LearningValueModel` rather than a reuse:
    same idea (momentum learning rate, L2, logit clip), different feature count
    and weight layout (bias at ``[-1]`` here), and sharing the allocator's
    hard-coded three-feature class would force this module's five features into
    a layout that does not fit. None of this is a research contribution -- it
    is the smallest thing that learns without diverging on one noisy run."""

    def __init__(self, n_features: int, lr: float = 0.2, reg: float = 0.05,
                 max_logit: float = 3.0) -> None:
        self.n_features = int(n_features)
        self.lr = lr
        self.reg = reg
        self.max_logit = max_logit
        self._w = [0.0] * (n_features + 1)
        self.n_obs = 0

    def predict(self, features: Sequence[float]) -> float:
        logit = self._w[-1] + sum(
            w * x for w, x in zip(self._w[:-1], features))
        logit = max(-self.max_logit, min(self.max_logit, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def update(self, features: Sequence[float], positive: bool) -> None:
        p = self.predict(features)
        err = (1.0 if positive else 0.0) - p
        self._w[-1] += self.lr * err - self.lr * self.reg * self._w[-1]
        for i, x in enumerate(features):
            self._w[i] += self.lr * err * x - self.lr * self.reg * self._w[i]
        self.n_obs += 1
