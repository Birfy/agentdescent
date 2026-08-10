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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evolvable import Diff, Evolvable
from .policies import FusionTrial

__all__ = ["MERGE_PROMPT", "KeepContradictions", "ReflectiveFusion",
           "reflective_merge"]


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
        return Diff(
            diff_id=f"synth({'+'.join(sorted(d.diff_id for d in diffs))})",
            target=diffs[0].target, ops=ops,
            contract_breaking=any(d.contract_breaking for d in diffs),
            author="aggregator")

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
