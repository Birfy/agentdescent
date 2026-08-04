"""Core data model for AgentDescent.

This module defines the *unit of evolution* (``Evolvable``) and the two
"gradient-like" objects that flow through the system:

* :class:`Diff` -- a proposed change to one or more artifacts (the "gradient").
* :class:`EvidenceCard` -- the metadata attached to a diff (the "gradient
  metadata"): what versions it was derived against, the trajectories that
  justify it, and the locally-measured before/after delta.

The central analogy of the framework (design doc, section 1) is:

    parameter tensor θ            -> library of Evolvable artifacts
    gradient g                    -> Diff + EvidenceCard
    version vector on a diff      -> the base_version it was read against

Everything here is provider-agnostic: an ``Evolvable`` can be a skill, a prompt
template, a harness module, or a learned verifier.  What evolves is decided by
*registration*, not hard-coded (design doc, section 3.2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable


class ContractError(Exception):
    """The caller's own code broke a documented contract.

    Distinct from a *backend* failure (a rate limit, a dead endpoint), which the
    engine absorbs and reports so partial results survive. A contract violation
    makes the run meaningless, so the engines let it propagate."""


def stable_hash(key: Any) -> int:
    """A process-independent hash for seeding and partitioning.

    Python randomises ``hash()`` of ``str``/``bytes`` per process unless
    ``PYTHONHASHSEED`` is pinned, so using it to seed an RNG or assign a section
    makes a ``seed=`` argument meaningless: the same seed gives different results
    on every run. Anything reproducible must hash through here instead.
    """
    return int.from_bytes(
        hashlib.blake2b(repr(key).encode(), digest_size=8).digest(), "big")

# A version vector maps ``artifact_id -> integer version``.  A diff only records
# the artifacts it actually read or wrote, so most vectors are sparse.
VersionVector = Dict[str, int]


def vv_dominates(a: VersionVector, b: VersionVector) -> bool:
    """Return True if ``a`` is at least as new as ``b`` on every shared key.

    A primitive, not part of a path: the engines compare versions through
    :func:`vv_staleness`, which answers *how* stale rather than *whether*."""
    return all(a.get(k, 0) >= v for k, v in b.items())


def vv_staleness(head: VersionVector, base: VersionVector) -> int:
    """Per-diff staleness ``eta`` (design doc, section 4.2).

    ``eta(d) = max over touched artifacts (head_version - base_version)``.

    A staleness of 0 means the diff was proposed against the current head of
    every artifact it touched; larger values mean the world moved on underneath
    the proposal and it must be rebased (or discarded).
    """
    if not base:
        return 0
    return max((head.get(k, 0) - v for k, v in base.items()), default=0)


@dataclass(frozen=True)
class Contract:
    """The externally-visible interface of an artifact.

    ``breaking`` marks a semver-major style change; the Ledger refuses diffs
    that declare a dependency on a superseded major (design doc, section 6,
    "contract mechanism").
    """

    input_schema: str = "any"
    output_schema: str = "any"
    side_effects: Tuple[str, ...] = ()
    major: int = 1

    def is_compatible_with(self, other: "Contract") -> bool:
        return self.major == other.major


@dataclass
class Diff:
    """A proposed change to an artifact's state.

    ``ops`` is an opaque, artifact-specific payload.  For the reference skill
    domain (:mod:`agentdescent.domains.router`) it is a mapping of key -> value
    edits, but the aggregator never inspects it directly -- it always goes
    through the owning :class:`Evolvable`.
    """

    diff_id: str
    target: str  # artifact id
    ops: Dict[str, Any] = field(default_factory=dict)
    contract_breaking: bool = False
    author: str = "unknown"  # which worker produced it

    def size(self) -> int:
        """A crude "number of edited lines" proxy used by the trust-region
        cap (design doc, section 4.4)."""
        return len(self.ops)


@dataclass
class EvidenceCard:
    """The "gradient metadata" carried by every diff (design doc, section 3.3).

    The lifecycle of an evidence card is *longer* than the diff it justifies:
    when a diff is discarded for staleness the card settles back into the
    trajectory pool for later reuse (design doc, section 3.3 / RQ2).
    """

    diff: Diff
    base_version: VersionVector
    touched: List[str]
    # local before/after measurement made by the proposing worker.
    before_after_delta: float = 0.0
    #: The failing work units that justify the diff -- **whatever the owning
    #: artifact's** :meth:`Evolvable.evidence_eval` **can score**. Every producer
    #: in the package puts task *objects* here (``evolution.Task`` /
    #: ``RouterTask``) and both consumers filter for them with ``isinstance``.
    #:
    #: It was annotated ``List[str]``, which is a trap rather than a cosmetic
    #: slip: a custom domain that follows the annotation and stores ids gets an
    #: ``evidence_eval`` over an empty task list, so the staleness policy's
    #: REBASE branch compares ``0.0 <= 0.0`` and keeps *every* rebased diff --
    #: the cheap re-verification silently becomes a no-op, and a diff that makes
    #: the artifact worse survives it.
    trajectory_refs: List[Any] = field(default_factory=list)
    # `version_annotations: List[VersionVector]` used to sit here, described as
    # "per-turn version annotations, used when the ledger hot-updates
    # mid-rollout". Nothing ever wrote one and nothing ever read one -- not a
    # producer in the package, not a consumer, not a test, not a doc page. A field
    # on the object every worker constructs is not free: it is read as a
    # capability, and the mid-rollout hot-update it names does not exist (a worker
    # holds one snapshot for a whole rollout, by construction). It would come back
    # with the code that needs it.
    cost_tokens: int = 0
    cost_wallclock: float = 0.0

    def rebased_onto(self, head: VersionVector) -> "EvidenceCard":
        """Return a copy whose base is advanced to ``head`` for touched keys.

        Used after a successful cheap re-verification during rebase (design
        doc, section 4.2, the ``0 < eta <= alpha`` branch)."""
        new_base = dict(self.base_version)
        for k in self.touched:
            if k in head:
                new_base[k] = head[k]
        return replace(self, base_version=new_base)


@runtime_checkable
class Evolvable(Protocol):
    """The single interface every unit of evolution must satisfy.

    Anything implementing this protocol can be registered into the evolution
    loop.  ``blast_radius`` drives automatic layering into L0/L1/L2 (design
    doc, section 6) -- it is *estimated*, not human-annotated.

    ``evidence_eval`` scores this artifact against the trajectories an evidence
    card carries. It was called ``cheap_eval`` until it collided with
    :meth:`~agentdescent.verifier.ThreeLayerVerifier.cheap_eval`, which takes an
    *artifact* and means something else -- and both are called from the same forty
    lines of the aggregator. ``cheap_eval`` remains as an alias on the shipped
    implementations so existing code keeps working.
    """

    id: str
    version: int
    contract: Contract
    blast_radius: float

    def diff(self, other: "Evolvable") -> Diff: ...
    def apply(self, diff: Diff) -> "Evolvable": ...
    def evidence_eval(self, evidence: EvidenceCard) -> float: ...

    # `full_eval(task_set) -> Mapping[str, float]` used to sit here as a fourth
    # requirement. Nothing ever called it: ground truth reaches the aggregator
    # through the verifier's `eval_fn`, which the *domain* supplies, so the
    # artifact never had to know how to score itself on a task set. Requiring a
    # method the engine does not use is a tax on everyone who implements this
    # protocol, so it is gone; implementations are free to keep one.


# A convenience alias for factory functions that reconstruct an Evolvable from
# its serialized state (used by the git-backed ledger).
Deserializer = Callable[[str, int, str], Evolvable]
