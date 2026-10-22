"""Verified seeding: trust becomes a gate, not a report.

The self-exam (#431) stamps adoptions `passed` / `flagged` / `demoted` and the
meta-audit (#426) measures the oracle-vs-verifier gain bias; the archive keeps
seeding the next search from ``_archived_rules``. This module turns that trust
into a **seeding gate**: ``verified_rules`` returns only the rule records that
meet the caller's bar (exam status at or above ``min_exam``, and — when a
``MetaAuditStore`` is supplied — oracle gain bias within ``bias_threshold``),
and explains, per excluded rule, exactly why it did not qualify. `continuous_meta_run`
and population selection can ask for ``min_exam`` so a rule that failed truth is
not merely reported but is never seeded again.

The default bar is ``min_exam="passed"``: an adoption that the oracle has not
yet certified (``exam is None``) or that failed it once (``flagged``) does not
seed the next search. ``min_exam=None`` keeps the existing behaviour exactly
(demoted rules already excluded by ``_archived_rules``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .consolidation import ConsolidationRecord, ConsolidationStore
from .meta import SlotSpec, _archived_rules

__all__ = ["VerifiedSeed", "verified_rules"]

#: Increasing trust order. ``demoted`` sits below unexamined only for
#: defensive ranking -- ``_archived_rules`` has already dropped those records.
_EXAM_RANK = {"demoted": -1, None: 0, "flagged": 1, "passed": 2}

#: The bars a caller may ask for. ``demoted`` is not one: it is retired upstream.
_BAR_RANK = {None: 0, "flagged": 1, "passed": 2}


@dataclass(frozen=True)
class VerifiedSeed:
    """The archive's rule records that clear the trust bar, and the reasons the
    rest were excluded."""

    rules: Tuple[ConsolidationRecord, ...]
    #: ``run_id -> why it did not seed``.
    excluded: Dict[str, str]


def verified_rules(
    store: ConsolidationStore,
    *,
    slot: str,
    spec: SlotSpec,
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    min_exam: Optional[str] = "passed",
    meta_audit: Optional[Any] = None,
    bias_threshold: float = 0.1,
) -> VerifiedSeed:
    """The archive's seedable rule records, gated by trust.

    Starts from :func:`~agentdescent.meta._archived_rules` (which already
    excludes ``demoted`` rules) and then applies the caller's bar:

    * ``min_exam`` — when set, only rules whose ``exam`` status is **at or
      above** it qualify, ordered ``None < flagged < passed``. So
      ``"passed"`` (the default) means a rule the oracle has not examined, or
      that failed once, does not seed, while ``"flagged"`` keeps both flagged
      and passed rules. Anything other than ``None``/``"flagged"``/``"passed"``
      raises -- a typo must not silently empty the seed set.
    * ``meta_audit`` + ``bias_threshold`` — when a
      :class:`~agentdescent.meta_audit.MetaAuditStore` is supplied, a rule whose
      recorded gain bias for its run exceeds ``bias_threshold`` is excluded
      even if it passed the exam.

    ``excluded`` carries the reason per run, so "this rule did not seed" is
    answerable. ``min_exam=None`` with no audit store reproduces
    ``_archived_rules`` exactly (backwards compatible).
    """
    if min_exam not in _BAR_RANK:
        raise ValueError(
            "min_exam must be None, 'flagged' or 'passed'; "
            f"got {min_exam!r}")
    bar = _BAR_RANK[min_exam]
    rules = _archived_rules(store, slot=slot, spec=spec, config=config,
                            artifact_id=artifact_id)
    kept: List[ConsolidationRecord] = []
    excluded: Dict[str, str] = {}
    for rec in rules:
        if min_exam is not None:
            rank = _EXAM_RANK.get(rec.exam)
            if rank is None:
                raise ValueError(
                    f"unknown exam status {rec.exam!r} on rule {rec.run_id!r}")
            if rank < bar:
                excluded[rec.run_id] = (
                    "rule has not been examined yet" if rec.exam is None
                    else f"exam is {rec.exam!r}, below min_exam {min_exam!r}")
                continue
        if meta_audit is not None:
            bias = meta_audit.gain_bias(run_id=rec.run_id)
            if bias is not None and abs(bias) > bias_threshold:
                excluded[rec.run_id] = (
                    f"gain bias {bias:+.3f} exceeds threshold "
                    f"{bias_threshold:.3f}")
                continue
        kept.append(rec)
    return VerifiedSeed(rules=tuple(kept), excluded=excluded)
