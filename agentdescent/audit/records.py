"""What one audited measurement is, and how a verifier is versioned.

The audit layer exists for one situation: the ``reward`` the loop optimises
against is **not** ground truth. It is an agent judging an output, a learned
scorer, a heuristic -- cheap, available on every rollout, and biased by an
unknown amount. Somewhere there is a truth: a gold answer, a checker, an
experiment someone runs next week. It is too expensive to ask on every rollout
and sometimes too slow to ask at all synchronously.

An :class:`AuditRecord` is one unit where both were (or will be) asked, kept so
the difference can be estimated later. Two fields carry the whole statistical
argument and are the reason this is a persisted record rather than a running
tally:

* ``inclusion_prob`` -- the probability this unit had of being audited. Without
  it the sample is not a probability sample and nothing computed from it
  estimates anything. It cannot be reconstructed after the fact, because the
  policy that drew it may have changed since.
* ``verifier_version`` -- which verifier produced ``verifier_score``. A
  correction estimated for one verifier says nothing about the next one, and
  the failure is silent: the numbers still compute, they are just about a
  measuring instrument that no longer exists.
"""

from __future__ import annotations

import hashlib
import inspect
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

#: Bumped when the on-disk shape of a record changes incompatibly. A reader that
#: meets a record it does not understand must say so rather than guess -- see
#: :meth:`AuditStore.load`.
SCHEMA_VERSION = 1


class Purpose(str, Enum):
    """Which of the two disjoint pools a labelled unit belongs to.

    The split is not bookkeeping. Labels used to *change* the verifier cannot
    also be used to *calibrate* it: the change was chosen to make those very
    units agree, so the residual measured on them is optimistically biased by
    construction, and the resulting correction is confidently wrong rather than
    noisy. Two pools, drawn at random, never mixed.

    :class:`~agentdescent.audit.store.AuditStore` asserts this rather than
    filtering for it, because a query condition is one edit away from being
    dropped and an assertion is not.
    """

    #: Estimating the verifier's bias. Never shown to whoever edits the verifier.
    CALIBRATION = "calibration"
    #: Diagnosing and improving the verifier. Never enters a calibration set.
    IMPROVEMENT = "improvement"


@dataclass(frozen=True)
class AuditRecord:
    """One ``(task, output)`` pair scored by the verifier, awaiting or carrying truth.

    ``output`` is kept because a deferred oracle scores it *later*, possibly in
    another process on another day -- an experiment cannot be run against a
    number. That makes the store the durable artifact of an audit, not a cache,
    and it is why the file is JSONL rather than an in-memory list.
    """

    record_id: str
    task_id: str
    #: The artifact that produced ``output``, as
    #: :meth:`~agentdescent.evolution.EvolvingArtifact._signature` renders it --
    #: the same identity the evaluation cache keys on, so a record can be traced
    #: back to the exact state that was measured.
    artifact_signature: str
    #: What the artifact answered. The oracle scores **this**, not a fresh
    #: rollout: scoring a re-run would fold rollout variance into the residual
    #: and there is no way to take it back out afterwards.
    output: str
    verifier_version: str
    verifier_score: float
    #: ``(0, 1]``. See the module docstring -- this is the field that makes the
    #: sample a probability sample.
    inclusion_prob: float
    purpose: Purpose
    #: Which layer of the sampling design this unit came from. Free-form so a
    #: stratifier can name its own; ``"all"`` when sampling is unstratified.
    stratum: str = "all"
    #: Ground truth, once it arrives. ``None`` means "not yet", never "zero".
    oracle_score: Optional[float] = None
    dispatched_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    #: Seed the inclusion draw was made from, so a run can be replayed and get
    #: the same sample back.
    sampler_seed: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not 0.0 < self.inclusion_prob <= 1.0:
            raise ValueError(
                f"inclusion_prob must be in (0, 1], got {self.inclusion_prob!r}")
        if (self.oracle_score is None) != (self.resolved_at is None):
            raise ValueError(
                "oracle_score and resolved_at must be set together: "
                f"got {self.oracle_score!r} / {self.resolved_at!r}")

    @property
    def resolved(self) -> bool:
        """Has ground truth arrived for this unit?"""
        return self.oracle_score is not None

    @property
    def residual(self) -> Optional[float]:
        """``verifier_score - oracle_score``, or ``None`` while unresolved.

        This is the quantity the whole layer exists to average. Positive means
        the verifier scored the output *higher* than the truth did -- the
        direction that makes a loop accept changes that did not improve
        anything.
        """
        if self.oracle_score is None:
            return None
        return self.verifier_score - self.oracle_score

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["purpose"] = self.purpose.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditRecord":
        d = dict(d)
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"audit record schema_version={version!r}, this build reads "
                f"{SCHEMA_VERSION}. Refusing to guess at the difference.")
        d["purpose"] = Purpose(d["purpose"])
        return cls(**d)


def new_record_id() -> str:
    """A fresh record id. Also the ticket a deferred oracle resolves against."""
    return uuid.uuid4().hex


def output_digest(output: str) -> str:
    """A short stable digest of an output, for logs and for de-duplication."""
    return hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]


def verifier_fingerprint(fn: Callable[..., Any], *, extra: Any = None) -> str:
    """A stable id for the verifier ``fn``, so a correction can be bound to it.

    Nothing in the loop tracks "which verifier is this", and a correction that
    outlives the thing it corrects is worse than no correction: it is applied
    with confidence to an instrument whose bias is now unknown. This derives an
    id from what the verifier *is* -- its qualified name plus, when it can be
    read, its source -- so editing the scorer changes the id and the stale
    correction stops being used instead of quietly going on being used.

    Source is the strongest signal available without asking the caller, and it
    is not always available: a C builtin, a functools.partial, a bound method of
    a class defined in a REPL. When it cannot be read the name alone is used and
    the fingerprint is weaker -- an edit that keeps the name will not be
    noticed. Pass ``extra`` (a version string, a config dict, a model id) to
    make it strong again, and pass it whenever the verifier is an agent: the
    behaviour of an LLM judge changes with its prompt and its model, and neither
    is visible in the source of the function that calls it.
    """
    target = getattr(fn, "__func__", fn)
    parts = [getattr(target, "__module__", "?"),
             getattr(target, "__qualname__", type(target).__name__)]
    try:
        parts.append(inspect.getsource(target))
    except (OSError, TypeError):
        pass
    if extra is not None:
        parts.append(repr(extra))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
