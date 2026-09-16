"""A synthetic discovery domain with **saturating branches**: the cheap world.

Dreaming is free, but the online rollouts that fill the simulator pool are not,
so the offline stage of this example needs a discovery agent with no model in
it. This is that agent, and the one property it has to get right is the one the
whole method turns on.

**Why branches must saturate.** A replay world contains only what the recording
policy explored, so an alternative policy can reorder, subset and stop -- it can
never reach a direction nobody took. If every extra attempt on a branch bought
more score, the best replay would always be "reveal everything", the recording
policy would win by construction, and dreaming could discover nothing. Real
discovery does not look like that: a direction is worked until it stops paying,
which is what Dream-RSI's own policy prompt spends a section on ("if most
attempts cluster around small variations of one mechanism with flattening
returns, that's a local optimum"). So here each branch has a hidden **ceiling**
and approaches it geometrically. Once a branch is within noise of its ceiling,
further attempts on it are pure cost -- and a policy that notices, and spends
the slots elsewhere, wins on Equation 1 without losing any quality.

**The ceiling is not visible in the first attempt.** A branch's head scores a
plain draw; the ceiling is drawn from the head's *node index* and only starts to
show from the second attempt on. That is deliberate, and it is the appendix's
rule made true rather than merely asserted: *"shallow weak scores are not enough
to discard a branch: deeper attempts can recover."* A policy that closes a
branch on its opening score is throwing away the good ones at the same rate as
the bad.

Two families. ``SOURCE`` is what the policy is evolved on. ``TARGET`` is slower
to reveal a ceiling, fails more often, and spreads its ceilings wider -- it is
never seen by the outer loop, and it is what says whether an evolved policy is a
better exploration policy or a fit to one generator.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agentdescent.dream import Attempt, DiscoveryNode, DiscoveryTree

__all__ = ["Family", "SOURCE", "TARGET", "FAMILIES", "discovery_agent",
           "branch_report"]


@dataclass(frozen=True)
class Family:
    """One generator of discovery worlds."""

    name: str
    #: The range a branch's hidden ceiling is drawn from.
    ceiling_low: float
    ceiling_high: float
    #: How much of the remaining gap one attempt closes. Lower means a branch
    #: takes longer to show what it is worth.
    rate: float
    #: Probability that an attempt produces no score at all -- the repairable
    #: failure of Appendix B.2, not a bad direction.
    p_fail: float
    #: Measurement noise on a scored attempt.
    noise: float
    #: What the initial workspace scores.
    root_score: float


SOURCE = Family("source", ceiling_low=0.25, ceiling_high=0.95, rate=0.55,
                p_fail=0.15, noise=0.03, root_score=0.10)
TARGET = Family("target", ceiling_low=0.15, ceiling_high=0.99, rate=0.35,
                p_fail=0.30, noise=0.06, root_score=0.05)
FAMILIES: Dict[str, Family] = {f.name: f for f in (SOURCE, TARGET)}


def _draw(family: Family, *parts: object) -> float:
    """A stable ``[0, 1)`` draw from the parts, so a world is a function of them.

    ``hash()`` is salted per process, so a world built with it would differ
    between runs and the paired comparison every measurement here rests on
    would stop being paired. SHA-256 of the parts is not.
    """
    payload = "|".join(str(part) for part in (family.name, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _ceiling(family: Family, head: int, seed: int) -> float:
    unit = _draw(family, "ceiling", seed, head)
    return family.ceiling_low + unit * (family.ceiling_high - family.ceiling_low)


def discovery_agent(family: Family, seed: int = 0):
    """A :class:`~agentdescent.dream.Continuation` over ``family``.

    ``(tree, parent, attempt) -> Attempt``. Opening a branch draws an ordinary
    first score; continuing one closes ``rate`` of the gap to that branch's
    hidden ceiling, with noise. Every draw is a function of the family, the
    seed and the node it is about, so two rollouts differ only through the
    ``attempt`` counter :func:`~agentdescent.dream.explore` hands out -- which
    is what makes a world reproducible and a pool of worlds genuinely several.
    """

    def continue_from(tree: DiscoveryTree, parent: DiscoveryNode,
                      attempt: int) -> Attempt:
        rng = random.Random(int(_draw(family, "rng", seed, attempt) * (1 << 32)))
        if rng.random() < family.p_fail:
            return Attempt(None, False,
                           {"fail_class": "compile_other",
                            "error": "the attempt did not build"})
        if parent.parent is None:                      # opening a new branch
            opening = family.root_score + abs(rng.gauss(0.10, 0.08))
            return Attempt(min(1.0, opening), True,
                           {"fail_class": "ok", "kind": "open"})
        head = tree.branch(parent.index)
        ceiling = _ceiling(family, head, seed)
        anchor = _anchor(tree, parent)
        depth = tree.depth(parent.index) + 1
        gap = max(0.0, ceiling - anchor)
        value = ceiling - gap * (1.0 - family.rate)
        value += rng.gauss(0.0, family.noise)
        return Attempt(min(1.0, max(0.0, value)), True,
                       {"fail_class": "ok", "kind": "refine", "step": depth})

    return continue_from


def _anchor(tree: DiscoveryTree, node: DiscoveryNode) -> float:
    """The best score on this node's branch so far -- what a refinement builds on.

    Not ``node.score``: the parent of a refinement is often a *failed* attempt
    (``score is None``), and treating that as "the branch is worth nothing"
    would make one bad build permanently destroy a branch. Appendix B.2 calls
    that out by name -- a repairable failure "must not erase its historical
    successful anchor" -- so the generator does not do it either.
    """
    best: Optional[float] = None
    cursor: Optional[DiscoveryNode] = node
    while cursor is not None:
        if cursor.score is not None:
            best = cursor.score if best is None else max(best, cursor.score)
        cursor = tree.nodes[cursor.parent] if cursor.parent is not None else None
    return 0.0 if best is None else best


def branch_report(tree: DiscoveryTree) -> List[Tuple[int, int, Optional[float]]]:
    """``(branch head, attempts on it, best score on it)`` per branch.

    What a reader needs to see whether a world is worth dreaming in: a world
    whose branches all reached the same score has nothing for a policy to
    choose between.
    """
    rows: Dict[int, Tuple[int, Optional[float]]] = {}
    for node in tree.nodes[1:]:
        head = tree.branch(node.index)
        count, best = rows.get(head, (0, None))
        if node.score is not None:
            best = node.score if best is None else max(best, node.score)
        rows[head] = (count + 1, best)
    return [(head, count, best) for head, (count, best) in sorted(rows.items())]
