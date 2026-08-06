"""Fusion that can combine competing values for the *same* key.

`fuse_diffs` is a dictionary update. That is exactly right when two workers
touched different keys -- their improvements are independent and the union keeps
both -- and it is useless when they touched the same one, because the last
writer simply wins. So `DefaultFusion` refuses to fuse at all once any pair
contradicts, and the tournament falls back to picking the best single diff.

For an artifact held in **one key** that is not a corner case, it is every case.
GEPA's ``InstructionSlot`` is the whole instruction under ``"instruction"``, so
any two proposals contradict by construction, no fused candidate is ever built,
and `merge_of_n` degenerates into per-round best-of-N *selection*. Measured:
``fusion_stats().contested == 0`` over every tournament of such a run.

:class:`ReflectiveFusion` closes that gap the only way text allows -- it asks a
model to write one value that keeps what each proposal contributed. Four
properties matter more than the idea:

**It earns its place or it loses.** The synthesised candidate is added to the
same held-out tournament as everything else and has no privilege in it. Ties go
to the incumbent, because `max` keeps the first of equal scores and the
synthesised candidate is appended last -- so it has to *strictly beat* the best
single diff and the plain dictionary fusion. A model that returns something
plausible and worse is rejected by the gate, not by trust.

**It is measured against the model-free fusion, not only against the singles.**
When the diffs agree on every key, both candidates are built and both compete.
Without that, an improvement over the singles could be the union doing the work
while the model took the credit.

**It cannot break a merge.** A failed call, an empty answer, an answer that
merely repeats an input, or one over ``max_chars`` all fall back to exactly what
`DefaultFusion` would have done. Fusion is on the commit path of every round; a
merge that raises because a model was slow is a worse outcome than a merge that
did not improve.

**It costs about 19% more, and the synthesis is not where that goes.** Counted
call-by-call on the HotpotQA configuration (18 rollouts, 3 workers, 20 held-out),
with a stub model so the structure is exact:

===============================  =====  =====
call                             off    on
===============================  =====  =====
agent answering (rollout + gate)    96    112
reflector writing a proposal        39     43
**synthesising the union**           0    **6**
*total*                            135    161
===============================  =====  =====

The synthesis is **3%**. The rest of the increase is
:class:`KeepContradictions`: leaving every contradicting proposal for the
tournament means ranking all N rather than the one `DefaultConflict` would have
left, and each ranking is a cheap-layer sweep.

Worth knowing where the bill actually is before optimising any of this: of those
96 calls only **18 are rollouts**. The other 78 are the gate re-scoring
candidates and rounds -- 58% of the run. Scoring the candidates in parallel was
tried and reverted; it builds a thread pool per tournament, which
`tests/test_evaluator.py` guards against by name, and it would make the run
finish sooner while costing exactly the same.

Pass the same :class:`~agentdescent.agents.Usage` here as to the run and the
totals include all of it.

Off by default, like everything else that has not been A/B'd::

    from agentdescent import Policies, evolve
    from agentdescent.fusion import ReflectiveFusion

    evolve(tasks, reward, agent=agent, n_workers=4,
           policies=Policies(fusion=ReflectiveFusion(completion)))
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evolvable import Diff, Evolvable
from .policies import FusionTrial, MergeContext

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
                 rank_on: str = "cheap", skip_when_dominant: float = 0.0) -> None:
        if max_proposals < 2:
            raise ValueError("synthesising needs at least two proposals")
        if rank_on not in ("cheap", "full"):
            raise ValueError(f"rank_on must be 'cheap' or 'full', not {rank_on!r}")
        if skip_when_dominant < 0:
            raise ValueError("skip_when_dominant must not be negative")
        self.complete = complete
        self.verifier = verifier
        #: The synthesised value is written by a model and reaches the tournament
        #: without passing the trust region, which filters *cards* before the
        #: merge. So it is bounded here too; a reflector that echoes its own
        #: prompt would otherwise commit a value the trust region exists to stop.
        self.max_chars = max_chars
        #: Beyond this many competing values the prompt stops being a merge
        #: instruction and starts being a summarisation task with a different
        #: failure mode. Extra proposals are dropped, best-scoring first kept.
        self.max_proposals = max_proposals
        #: Which layer ranks the tournament. ``"cheap"`` is the shipped
        #: behaviour; ``"full"`` scores every candidate on the whole held-out set
        #: instead.
        #:
        #: The cheap layer exists to be cheap, and on a small held-out set it
        #: barely is. Measured on the HotpotQA configuration -- 20 held-out
        #: tasks, ``rule_subset=8`` -- one `cheap_eval` touches **16 of the 20**
        #: while adding sub-sampling error and a gaussian noise term. That is a
        #: 20% saving in exchange for noise, paid by the one decision that
        #: decides whether a merge survives, and where a tie already loses. The
        #: bigger the held-out set the better the shipped default looks, so this
        #: is a knob and not a new default.
        self.rank_on = rank_on
        #: Skip the synthesis when the best single already beats the runner-up by
        #: this margin. Zero (default) always synthesises.
        #:
        #: A union is worth buying when the proposals are complementary. When one
        #: proposal dominates the field, the merge is unlikely to beat it and
        #: costs a model call plus a whole extra candidate to rank -- on the
        #: commit path of every round. This trades some merges for their cost,
        #: and like everything else here it is off until measured.
        self.skip_when_dominant = skip_when_dominant
        #: Same instrumentation as the shipped policy, so `fusion_stats()` reads
        #: a run of this one without knowing it is different.
        self.trials: List[FusionTrial] = []

    def bind(self, verifier: Any) -> None:
        """Receive the engine's verifier, if the caller did not supply one."""
        if self.verifier is None:
            self.verifier = verifier

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
        return merged

    # -- the tournament -----------------------------------------------------

    def select(self, artifact: Evolvable,
               diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        from .aggregator import diffs_contradict, fuse_diffs

        cheap = (self.verifier.cheap_eval if self.rank_on == "cheap"
                 else lambda a: MergeContext.rate(self.verifier.eval_counts(a)))
        # Applied once, not twice: the old comprehension called `artifact.apply`
        # for the score and again for the tuple, so every candidate was built
        # twice and the object that was scored was not the object kept.
        applied = [artifact.apply(d) for d in diffs]
        # Scored one at a time, on purpose. Scoring them concurrently was tried
        # and reverted: it builds a thread pool per tournament, which
        # `tests/test_evaluator.py` guards against by name (a run once built 83
        # of them), and it multiplies in-flight requests by the candidate count
        # against an endpoint that has already dropped connections under load.
        # It is also the wrong lever -- see the class docstring on cost.
        # (score, diff, candidate, is_fusion, kind)
        scored: List[Tuple[float, Diff, Evolvable, bool, str]] = [
            (cheap(candidate), d, candidate, False, "single")
            for d, candidate in zip(diffs, applied)
        ]
        reason = "single-candidate" if len(diffs) <= 1 else ""
        contradicts = len(diffs) > 1 and any(
            diffs_contradict(a, b)
            for i, a in enumerate(diffs) for b in diffs[i + 1:])

        if len(diffs) > 1 and not contradicts:
            # No disagreement: the union is the merge, and no model is needed.
            plain = fuse_diffs(diffs)
            candidate = artifact.apply(plain)
            scored.append((cheap(candidate), plain, candidate, True, "union"))

        synthesized_score: Optional[float] = None
        contested: Dict[str, List[Any]] = {}
        # One proposal already well clear of the field: a union is for
        # complementary proposals, and buying one here costs a model call and a
        # fourth candidate to rank for a merge that is unlikely to win.
        ranked = sorted((c[0] for c in scored), reverse=True)
        dominant = (len(ranked) > 1 and self.skip_when_dominant > 0
                    and ranked[0] - ranked[1] >= self.skip_when_dominant)
        if len(diffs) > 1 and not dominant:
            agreed, contested = self._partition(diffs)
            ops: Dict[str, Any] = dict(agreed)
            state = getattr(artifact, "state", {}) or {}
            for key, values in contested.items():
                merged = self._synthesise(state.get(key), values)
                if merged is None:
                    # All or nothing. A partial synthesis would commit some
                    # workers' contributions and silently drop the rest, which
                    # looks like a successful merge and is not one.
                    ops = {}
                    break
                ops[key] = merged
            if contested and ops:
                synth = Diff(
                    diff_id=f"synth({'+'.join(sorted(d.diff_id for d in diffs))})",
                    target=diffs[0].target, ops=ops,
                    contract_breaking=any(d.contract_breaking for d in diffs),
                    author="aggregator")
                candidate = artifact.apply(synth)
                synthesized_score = cheap(candidate)
                scored.append((synthesized_score, synth, candidate, True,
                               "synthesized"))

        # `max` keeps the first of equal scores, and every fused candidate is
        # appended after the singles -- so a fusion that merely ties loses. That
        # is the conservative reading, and it is what stops a model-written merge
        # being credited on a held-out set too small to separate anything.
        best = max(scored, key=lambda c: c[0])
        singles = [c[0] for c in scored if not c[3]]
        baseline = cheap(artifact)
        best_single = max(singles) if singles else baseline
        union = next((c[0] for c in scored if c[4] == "union"), None)

        if best[0] <= baseline:
            winner = "neither"
        elif best[4] == "synthesized":
            winner = "synthesized"
        elif best[3]:
            winner = "fused"
        else:
            winner = "single"

        # Why no fused candidate competed, when none did. `synthesis-failed` is
        # the one this class adds: the diffs *did* disagree, a model was asked,
        # and its answer could not be used.
        if any(c[3] for c in scored):
            reason = ""
        elif len(diffs) <= 1:
            reason = "single-candidate"
        elif dominant:
            reason = "dominant-single"
        elif contested:
            reason = "synthesis-failed"
        else:
            reason = "contradiction"

        self.trials.append(FusionTrial(
            artifact_id=getattr(artifact, "id", ""), n_candidates=len(diffs),
            best_single_score=best_single, baseline_score=baseline,
            # The plain union when there was one, else the synthesised score, so
            # `gain` always answers "what did fusing buy over the best single"
            # whichever kind of fusing was available.
            fused_score=union if union is not None else synthesized_score,
            synthesized_score=synthesized_score, winner=winner, reason=reason))
        return best[1], best[2], best[3]


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
