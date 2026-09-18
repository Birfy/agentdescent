"""The consumer side of the settled-evidence pool: a proposal policy that sees it.

:mod:`~agentdescent.sampling`'s :class:`~agentdescent.sampling.ReplaySampler` is one
consumer of the pool -- it up-weights tasks whose proposals keep getting discarded.
This module is the other: it hands a *proposal policy* the recently discarded
evidence for the task it is about to propose on, so the policy can avoid re-proposing
what was just thrown away. ``ProposalContext.rejected`` is what carries that evidence,
and its docstring says what this enables: *"'do not re-propose what was just rejected'
is inexpressible"* without a field for it.

The wrapper is deliberately prompt-agnostic: it renders the rejected cards into a
note and appends it to the context's ``output``, and a prompt-based inner policy
that renders ``output`` -- the shipped ``LLMAgent.propose`` does -- naturally sees
"these changes were just discarded for this task, do not repeat them". It never
parses the inner policy's proposal, so it works with any strategy.

``read`` counts the proposals that carried rejection evidence (i.e. the policy was
actually told about a discard), and ``shown`` counts the cards rendered -- two
counters that tell a run whether the mechanism ever fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:                       # runtime-free: only annotations
    from .evolvable import EvidenceCard
    from .policies import ProposalContext

__all__ = ["ReplayAwareProposal"]


def _default_note(card: "EvidenceCard") -> str:
    paths = ", ".join(sorted(card.diff.ops)) or "(no paths)"
    delta = f"{card.before_after_delta:+.3f}"
    version = card.base_version
    return (f"- touched `{paths}` against version `{version}`, "
            f"measured `{delta}` before this discard")


@dataclass
class ReplayAwareProposal:
    """A ``ProposalPolicy`` that tells the inner policy what was just discarded.

    Wraps another :class:`~agentdescent.policies.ProposalPolicy`. When
    ``ctx.rejected`` is non-empty it renders up to ``max_cards`` of the rejected
    evidence and appends it to the context's ``output`` before delegating, so the
    inner policy's prompt carries "these were discarded, do not repeat them".

    ``render`` turns one :class:`~agentdescent.evolvable.EvidenceCard` into the text
    an inner policy should see; the default shows the paths it touched, the version
    it was based on, and the delta it measured.
    """

    policy: object                                  # a ProposalPolicy
    max_cards: int = 3
    render: Callable[["EvidenceCard"], str] = field(
        default=_default_note, repr=False)
    #: Proposals that carried rejection evidence, and cards shown across them.
    read: int = 0
    shown: int = 0

    def propose(self, ctx: "ProposalContext") -> Sequence[str]:
        if not ctx.rejected:
            return self.policy.propose(ctx)
        self.read += 1
        cards = list(ctx.rejected)[:self.max_cards]
        self.shown += len(cards)
        note = "\n\n## Recently discarded for this task\n\n" + "\n\n".join(
            self.render(c) for c in cards)
        ctx = replace(ctx, output=ctx.output + note)
        return self.policy.propose(ctx)
