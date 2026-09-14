"""The responsible parent's verdict, as an
:class:`~agentdescent.policies.AcceptancePolicy`.

Upstream, acceptance is not statistical. A parent agent decides whether a
returned contribution is **accepted, rejected, or needs more work**, using the
tests, constraints and integration evidence it has (paper 3.3), and the rule its
prompt states is monotone::

    Partial progress is accepted -- a version is accepted if it improves the
    codebase, even if other parts remain broken.
        -- genesis, apps/evo_git/lib/evo_git/agents/manager.ex:58

The engine's shipped rule is a Beta posterior on held-out improvement. Swapping
one for the other is a change to the **acceptance rule**, which
``docs/port-fidelity.md`` puts in the *must not change* column -- so it is behind
a flag, off by default, and the run prints which rule it used. ACE is the
precedent and the warning: replacing an upstream acceptance rule with this
engine's gate silently emptied an artifact whose entire claim was accumulation.

Two things this cannot do, stated rather than worked around:

* :class:`~agentdescent.policies.AcceptDecision` is a boolean, and its
  ``category`` is a **closed vocabulary** -- the aggregator turns it into a
  :class:`~agentdescent.aggregator.MergeOutcome`, so a policy that invents a name
  raises inside the merge rather than reporting a new reason. "Needs more work"
  therefore commits the partial progress (which is upstream's rule) and leaves a
  request in the :class:`~examples.genesis._world.WorldLog` that the next round's
  manager re-delegates from. The third verdict is reconstructed out of parts the
  engine already has; it is not a third outcome in the engine.
* **The audit gate runs before this policy**, and on an L1 artifact it vetoes any
  candidate with ``oracle_cand <= oracle_base``. That is a strict-improvement rule
  laid over acceptance, and it is incompatible with "partial progress is
  accepted": a formation run's first structural steps move no case at all. The
  port therefore runs the world at L2 and says why (see the entry point);
  governance is deliberately not a seam, so this is a boundary to report, not one
  to route around.
* A parent judging *its own child* happens inside the recursion, before anything
  reaches the ledger (see :mod:`examples.genesis._delegation`). This policy is
  the other judging: the task-level one, on what the whole episode returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentdescent.defaults import DefaultAcceptance
from agentdescent.policies import AcceptDecision, MergeContext

from ._world import WorldLog, normalise

__all__ = ["ParentJudge"]


@dataclass
class ParentJudge:
    """Accept on improvement, refuse on regression, ask again on a tie.

    With ``enabled=False`` every call is forwarded to ``inner`` untouched, so the
    default run is the engine's and a golden comparison against
    :class:`~agentdescent.defaults.DefaultAcceptance` is exact.
    """

    log: Optional[WorldLog] = None
    enabled: bool = False
    #: Below this, a held-out move is a tie rather than progress. Not a
    #: threshold on significance -- the upstream rule has none -- but on
    #: *measurement*: one task flipping on a 12-task split is 0.083, and calling
    #: that "the codebase improved" would accept noise as progress.
    tie: float = 1e-9
    inner: Optional[object] = None
    accepted: int = 0
    rejected: int = 0
    #: Accepted with no case moved -- upstream's "partial progress", and the
    #: number to look at when a run commits a lot and improves little.
    partial: int = 0

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = DefaultAcceptance()

    def configure(self, config) -> None:
        """Fill the inner rule's thresholds from the run's config."""
        configure = getattr(self.inner, "configure", None)
        if callable(configure):
            configure(config)

    def bind(self, verifier) -> None:
        bind = getattr(self.inner, "bind", None)
        if callable(bind):
            bind(verifier)

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        if not self.enabled:
            return self.inner.accept(ctx)

        base = ctx.rate(ctx.base_counts)
        cand = ctx.rate(ctx.cand_counts)
        observed = cand - base

        if not ctx.comparable():
            # Two measurements from different toolchains are not evidence about
            # the change. The engine's own gate carries the same field for the
            # same reason; a parent that ignored it would read an environment
            # difference as progress.
            self.rejected += 1
            return AcceptDecision(False, "below-threshold",
                                  f"incomparable: {ctx.base_env!r} vs {ctx.cand_env!r}",
                                  observed_delta=observed)

        if observed < -self.tie:
            self.rejected += 1
            return AcceptDecision(False, "below-threshold",
                                  f"parent rejects regression {base:.3f} -> {cand:.3f}",
                                  observed_delta=observed)

        if observed > self.tie:
            self.accepted += 1
            return AcceptDecision(True, "committed",
                                  f"parent accepts: {base:.3f} -> {cand:.3f}",
                                  observed_delta=observed)

        # A tie: nothing broke, and no case moved. This is the rule the port
        # exists to run --
        #
        #     Partial progress is accepted -- a version is accepted if it
        #     improves the codebase, even if other parts remain broken.
        #
        # -- and it is what lets a formation run start at all: the first files of
        # an empty repository are structure, and structure moves no test. The
        # parent accepts them *and* sends more work to the subtree they came
        # from, which is the request the next round's manager picks up.
        self.partial += 1
        path = _deepest_touched(ctx)
        if self.log is not None:
            self.log.request_rework(
                path, f"held-out unchanged at {cand:.3f}: the step was structural. "
                      f"Continue in this subtree until a case moves")
        return AcceptDecision(True, "committed",
                              f"parent accepts partial progress at {path or './'} "
                              f"({cand:.3f} unchanged)",
                              observed_delta=observed)

    def stats(self) -> str:
        return (f"parent judge: accepted={self.accepted} rejected={self.rejected} "
                f"partial={self.partial}")


def _deepest_touched(ctx: MergeContext) -> str:
    """The subtree a refusal should be sent back to.

    The common ancestor of the keys the candidate touched: sending the request to
    the root would ask the whole world to try again, and sending it to one file
    would name a path no manager is situated at.
    """
    paths = sorted(getattr(ctx.diff, "ops", {}) or {})
    if not paths:
        return ""
    parts = [normalise(p).split("/")[:-1] for p in paths]
    common = parts[0]
    for other in parts[1:]:
        keep = []
        for a, b in zip(common, other):
            if a != b:
                break
            keep.append(a)
        common = keep
    return "/".join(common)
