"""Population search on a single-head ledger: how a `SelectionPolicy` runs today.

:mod:`agentdescent.selection` defines *where the next batch starts*; this module
is what makes the answer take effect before the ledger can hold several live
heads. The trick is the one GEPA and DGM each found on their own: keep the
candidate pool **in the aggregator**, and make "selection" a ledger commit that
rewrites the head. The heads are serialised rather than concurrent, so one
`dev` branch is enough.

:class:`PopulationAggregator` subclasses the shipped
:class:`~agentdescent.aggregator.Aggregator` -- staleness, conflict, fusion,
statistical acceptance and promotion all run unchanged -- and adds three things
around it: it archives every distinct committed head with its held-out score,
asks the selection policy which archived candidate the next batch should mutate,
and commits that candidate back to ``dev``. ``finalize`` commits the archive's
best scorer, so a run's final artifact is its best candidate rather than
whichever one it was exploring when the budget ran out.

## Why this lives in the engine

It was written in ``examples/_population.py`` and installed by the port runner,
which meant one `Policies(selection=...)` bundle had three meanings: `evolve()`
refused it, `async_evolve()` dropped it (fixed separately), and the port runner
honoured it through `aggregator_factory=`. A field that means three things is
worse than a field that means nothing, and only one of the three was the
behaviour the docs described.

## The boundary that used to be here

The parent switch is a **state replacement**, and it used to be written as
``head.apply(Diff(ops=target_state))``. That only *sets* keys, so a key the head
had and the target does not survived the switch: on a grow-only key space
(``AppendRules``) the "switch" was a union, and the run silently hill-climbed
instead of exploring. It was invisible in the ports because every declared
policy rode a fixed-key artifact. Moving the layer into the engine makes the
pairing reachable by any caller, so the switch now carries explicit ``None``
deletions for the keys the target does not have -- the sentinel
:meth:`~agentdescent.evolution.EvolvingArtifact.apply` already understood, and
the one :meth:`~agentdescent.evolution.EvolvingArtifact.diff` already emits.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from .aggregator import Aggregator, MergeReport
from .evolvable import Diff
from .ledger import CASConflict, Ledger
from .selection import (
    Candidate, MultiHeadUnsupported, SelectionContext, SelectionPolicy,
)

__all__ = ["PopulationAggregator", "population_factory"]


class PopulationAggregator(Aggregator):
    """The shipped merge pipeline plus an archive and a selection policy."""

    def __init__(self, ledger, verifier, audit, config, staleness_policy=None,
                 *, selection: SelectionPolicy, artifact_id: str, meter=None,
                 conflict=None, fusion=None, acceptance=None, promotion=None):
        super().__init__(ledger, verifier, audit, config,
                         staleness_policy=staleness_policy, meter=meter,
                         conflict=conflict, fusion=fusion, acceptance=acceptance,
                         promotion=promotion)
        self.selection = selection
        self.population_artifact = artifact_id
        self._archive: List[Dict[str, object]] = []
        self._seen: Set[str] = set()
        self._archive_lock = threading.Lock()
        #: Selections made, i.e. the ``round`` handed to the policy. Not under
        #: the archive lock: `step()` is the merger's, one thread, while the
        #: lock guards the archive against the workers that `ingest`.
        self._selections = 0

    # -- the archive ---------------------------------------------------------

    def _admit(self, artifact, version: int) -> None:
        key = artifact.render()
        with self._archive_lock:
            if key in self._seen:
                return
            self._seen.add(key)
        # The gate has just evaluated a committed candidate on the full
        # held-out set, so this is a cache hit, not a new evaluation.
        score = self.verifier.eval_fn(artifact, list(self.verifier.held_out))
        with self._archive_lock:
            self._archive.append({
                "state": dict(getattr(artifact, "state", {}) or {}),
                "score": float(score),
                "version": int(version),
                "selected": 0,
            })

    def _candidates(self) -> List[Candidate]:
        with self._archive_lock:
            return [
                Candidate(artifact_id=self.population_artifact,
                          version=int(entry["version"]),
                          state=dict(entry["state"]),
                          score=float(entry["score"]),
                          selected=int(entry["selected"]))
                for entry in self._archive
            ]

    def _best_state(self) -> Optional[Dict[str, str]]:
        with self._archive_lock:
            if not self._archive:
                return None
            best = max(self._archive, key=lambda entry: entry["score"])
            return dict(best["state"])

    def _commit_state(self, state: Dict[str, str], message: str) -> Optional[int]:
        """Replace the head's state outright, deletions included.

        ``ops`` carries ``None`` for every key the head has and the target does
        not. Without that half the switch is a *union*: `apply` only sets the
        keys it is given, so on a grow-only key space the head kept everything
        it had ever accumulated and "start from candidate C" quietly meant
        "start from C plus the incumbent". See the module docstring.
        """
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.population_artifact)
        if head is None or dict(head.state) == state:
            return None
        ops: Dict[str, Optional[str]] = dict(state)
        ops.update({k: None for k in head.state if k not in state})
        base_vv = {self.population_artifact:
                   snap.version.get(self.population_artifact, 0)}
        candidate = head.apply(Diff(
            diff_id=f"population:{message}", target=self.population_artifact,
            ops=ops, author="population"))
        try:
            _, version = self.ledger.commit(candidate, base_vv,
                                            branch=Ledger.DEV, message=message)
            return version
        except CASConflict:
            return None

    def _offered(self, chosen: Candidate, candidates: List[Candidate]) -> None:
        """Refuse a candidate the archive never offered.

        The engine's own `_check_selection` guards the *no-population* path by
        refusing anything but the head; this is the same rule where the menu is
        longer. Both raise `MultiHeadUnsupported` because both mean the one
        thing: the policy named a starting point this run cannot start from.
        Committing it anyway would write a state no rollout ever produced and no
        gate ever scored.
        """
        target = dict(chosen.state)
        if any(dict(c.state) == target for c in candidates):
            return
        raise MultiHeadUnsupported(
            f"{type(self.selection).__name__}.select() returned a candidate that "
            f"is not in the archive it was given ({len(candidates)} entries). A "
            "selection policy chooses among the candidates in "
            "SelectionContext.candidates; it cannot invent one, because a state "
            "that was never a committed head has never been scored by the gate.")

    # -- AggregatorProtocol --------------------------------------------------

    def step(self) -> List[MergeReport]:
        # Admit the pre-merge head first: on the first step that is the seed,
        # which a post-merge-only admit would lose the moment anything commits
        # over it -- and a population that forgot its seed cannot fall back.
        before = self.ledger.snapshot(Ledger.DEV)
        pre_head = before.get(self.population_artifact)
        if pre_head is not None:
            self._admit(pre_head,
                        before.version.get(self.population_artifact, 0))
        reports = super().step()
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.population_artifact)
        if head is None:
            return reports
        head_version = snap.version.get(self.population_artifact, 0)
        self._admit(head, head_version)
        candidates = self._candidates()
        if len(candidates) < 2:
            return reports
        head_candidate = next(
            (c for c in candidates if dict(c.state) == dict(head.state)),
            candidates[0])
        # ``n=1``: the ledger holds one live head, so a batch has one starting
        # point no matter how many workers are under it. This is the seam that
        # widens when the ledger can hold several -- the policy is already
        # written to answer for ``n``.
        #
        # ``round`` is *which selection this is*, not the engine's round, and it
        # is counted here because only this method knows: `step()` also runs on
        # sweeps that never reach a choice (fewer than two candidates), and
        # counting those would have a beam skip slots it never expanded.
        #
        # It used to be left at its default of 0, and that single omission is
        # what made width inert. A policy asked for one starting point can only
        # rotate on something that changes between calls; the archive and the
        # `selected` counts are two such things, and `MCTS` and `Archive` read
        # them -- but `Beam` and `ParetoFrontier` rank a pool and have no
        # per-candidate state to read, so with a constant round they returned
        # the same entry forever. `Beam(4)` was `Beam(1)`, and `ParetoFrontier`
        # sat on whichever front member was admitted first, usually the seed.
        ctx = SelectionContext(head=head_candidate, candidates=tuple(candidates),
                               round=self._selections, n_workers=1)
        self._selections += 1
        chosen = list(self.selection.select(ctx, 1))
        if not chosen:
            return reports
        self._offered(chosen[0], candidates)
        target = dict(chosen[0].state)
        with self._archive_lock:
            for entry in self._archive:
                if dict(entry["state"]) == target:
                    entry["selected"] = int(entry["selected"]) + 1
                    break
        version = self._commit_state(target, "population: select parent")
        if version is not None:
            reports.append(MergeReport(
                self.population_artifact, None, False, 0, 0, 0, 0, 0.0, version,
                reason=f"population: parent switched (archive={len(candidates)})",
                category="population-select"))
        return reports

    def finalize(self) -> None:
        """Leave the best-scoring candidate on the head, then promote."""
        best = self._best_state()
        if best is not None:
            self._commit_state(best, "population: final best")
        super().finalize()


def population_factory(selection: SelectionPolicy, artifact_id: str, *,
                       meter=None, conflict=None, fusion=None, acceptance=None,
                       promotion=None):
    """The ``aggregator_factory=`` adapter for one run.

    The factory path bypasses the engine's default-aggregator construction, so
    the decision policies that would normally arrive through the ``Policies``
    bundle must travel through here instead -- passing them in the bundle *and*
    a factory would silently drop them, which is the exact failure
    ``require_supported`` exists to prevent on the other path.
    """

    def build(ledger, verifier, audit, config, staleness_policy=None):
        return PopulationAggregator(
            ledger, verifier, audit, config, staleness_policy,
            selection=selection, artifact_id=artifact_id, meter=meter,
            conflict=conflict, fusion=fusion, acceptance=acceptance,
            promotion=promotion)

    return build
