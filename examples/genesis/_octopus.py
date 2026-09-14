"""``git merge --octopus``, as a :class:`~agentdescent.policies.ConflictPolicy`.

The engine's contradiction rule is one line::

    same key, different proposed value        -- aggregator.diffs_contradict

With a file tree the key is a **file path**, so two agents that touch different
functions of the same file contradict, and one of them is dropped on held-out
score. Genesis does not do that. Its parent merges every returned child with
``git merge --octopus`` (``adapters/git.ex:284``), so the same two edits merge
cleanly whenever their *hunks* do not overlap; only a real overlap produces
``{:error, {:conflict, _}}``, and only then is the parent asked to resolve it
(``agent/subagent_processing.ex:527``).

So the difference is granularity, and it is worth having as a policy rather than
as a paragraph: this one three-way merges the contested values against the
artifact's current content, and the cards it reconciles stop contradicting -- at
which point the engine's own fusion unions them exactly as it would for edits to
different files. Only the values that genuinely overlap fall through to the inner
rule, which is the shipped
:class:`~agentdescent.defaults.DefaultConflict` unless a caller passes another.

This is the third arm of the comparison the README's headline row is about
(``keyed union fuses 0 of 48 · reflective merge 42 of 48``): keyed union, textual
three-way, and model-synthesised merge are three answers to the same question,
and only the first two are free.

``git`` is the merge engine because it is the one Genesis uses and because the
ledger already requires it. Without it every contested key is reported as a
conflict and the inner policy decides -- degraded, never wrong.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from agentdescent.defaults import DefaultConflict
from agentdescent.evolvable import EvidenceCard, Evolvable

__all__ = ["OctopusConflict", "three_way", "git_available"]


def git_available() -> bool:
    return shutil.which("git") is not None


def three_way(base: str, ours: str, theirs: str, *, timeout: float = 10.0) -> Optional[str]:
    """``git merge-file`` on three strings. ``None`` means the hunks overlap.

    Identical to what a git merge does to one file, which is the point -- an
    approximation written here would answer a slightly different question from
    the system being ported.
    """
    if ours == theirs:
        return ours
    if base == ours:
        return theirs
    if base == theirs:
        return ours
    if not git_available():
        return None
    tmp = tempfile.mkdtemp(prefix="genesis-merge-")
    try:
        paths = {}
        for name, text in (("ours", ours), ("base", base), ("theirs", theirs)):
            path = os.path.join(tmp, name)
            # A missing trailing newline makes git treat the last line as
            # changed on both sides and conflict where the content agrees.
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text if text.endswith("\n") or not text else text + "\n")
            paths[name] = path
        proc = subprocess.run(
            ["git", "merge-file", "-p", "--quiet", paths["ours"], paths["base"],
             paths["theirs"]],
            capture_output=True, text=True, timeout=timeout)
        # Exit status: 0 clean, >0 the number of conflicts, <0 an error.
        return proc.stdout if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@dataclass
class OctopusConflict:
    """Reconcile contested keys by three-way merge; defer the rest.

    ``inner`` is the rule for values that really overlap. It defaults to the
    engine's own, so turning this policy on changes *only* the outcome for edits
    that upstream would have merged -- which is the property that makes the
    comparison against the default meaningful.
    """

    inner: Optional[object] = None
    #: Contested keys reconciled textually, and keys that fell through.
    merged: int = 0
    conflicted: int = 0

    def __post_init__(self) -> None:
        if self.inner is None:
            self.inner = DefaultConflict()

    def bind(self, verifier) -> None:
        """Forward the engine's verifier to the inner rule (see `install_policy`)."""
        bind = getattr(self.inner, "bind", None)
        if callable(bind):
            bind(verifier)

    def configure(self, config) -> None:
        configure = getattr(self.inner, "configure", None)
        if callable(configure):
            configure(config)

    def resolve(self, artifact: Evolvable,
                cards: Sequence[EvidenceCard]) -> Tuple[List[EvidenceCard], int]:
        cards = list(cards)
        if len(cards) < 2:
            return cards, 0
        state = dict(getattr(artifact, "state", {}) or {})
        proposals: Dict[str, List[int]] = {}
        for i, card in enumerate(cards):
            for key in card.diff.ops:
                proposals.setdefault(key, []).append(i)

        resolved: Dict[str, str] = {}
        for key, owners in proposals.items():
            values = [cards[i].diff.ops[key] for i in owners]
            if len(owners) < 2 or all(v == values[0] for v in values):
                continue                       # not contested, or already agreed
            if any(v is None for v in values):
                self.conflicted += 1           # a delete against an edit
                continue
            base = state.get(key, "")
            folded: Optional[str] = values[0]
            for value in values[1:]:
                folded = three_way(base, folded, value) if folded is not None else None
                if folded is None:
                    break
            if folded is None:
                self.conflicted += 1
            else:
                resolved[key] = folded
                self.merged += 1

        if resolved:
            cards = [self._rewrite(card, resolved) for card in cards]
        # Whatever still contradicts genuinely overlaps: the inner rule decides,
        # exactly as it would have for the whole batch without this policy.
        return self.inner.resolve(artifact, cards)

    @staticmethod
    def _rewrite(card: EvidenceCard, resolved: Dict[str, str]) -> EvidenceCard:
        """A copy of ``card`` carrying the merged value for every reconciled key.

        A copy rather than a mutation: a card outlives the diff it justifies (a
        stale one settles back into the pool), so editing one in place would
        change a record the run may read again later.
        """
        touched = [k for k in card.diff.ops if k in resolved]
        if not touched:
            return card
        ops = dict(card.diff.ops)
        for key in touched:
            ops[key] = resolved[key]
        diff = replace(card.diff, ops=ops,
                       diff_id=f"{card.diff.diff_id}+octopus")
        return replace(card, diff=diff)

    def stats(self) -> str:
        return f"octopus: merged={self.merged} conflicted={self.conflicted}"
