"""Population search on a single-head ledger: the GEPA/DGM pattern, reusable.

The engine's selection seam is honest about its limit: `_check_selection`
refuses any policy that names a starting point other than the live head,
because the ledger holds one `dev` branch ("multi-head support is a separate
change"). GEPA and DGM already solved this inside the sanctioned
`aggregator_factory` exit -- keep the candidate pool *in the aggregator* and
make "selection" a ledger commit that rewrites the head.

:class:`PopulationAggregator` generalises that pattern without forking the
pipeline: it subclasses the shipped :class:`~agentdescent.aggregator.Aggregator`
(staleness, conflict, fusion, statistical acceptance, and promotion all run
unchanged), archives every distinct committed head with its held-out score, and
after each merge step asks a standard
:class:`~agentdescent.selection.SelectionPolicy` -- BinaryTournament, SoftMixed,
``Archive``, ``Beam`` -- to pick the next parent. A pick that differs from the
head is committed back to ``dev``, so the next batch of workers mutates the
selected parent. ``finalize`` commits the archive's best scorer, so a run's
final artifact is its best candidate, not its last exploration target.

Boundary worth stating: a parent switch is a *state overwrite* commit. For
fixed-key artifacts (``SingleSlot``, ``FieldSlots``) that is exact; for
grow-only key spaces it could resurrect deleted keys, so the runner only
installs this when a selection policy is declared -- and the declared policies
all ride fixed-key artifacts.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Set

from agentdescent.aggregator import Aggregator, MergeReport
from agentdescent.evolvable import Diff
from agentdescent.ledger import CASConflict, Ledger
from agentdescent.selection import Candidate, SelectionContext, SelectionPolicy


class PopulationAggregator(Aggregator):
    """The shipped merge pipeline plus an archive and a selection policy."""

    def __init__(self, ledger, verifier, audit, config, staleness_policy=None,
                 *, selection: SelectionPolicy, artifact_id: str,
                 conflict=None, fusion=None, acceptance=None, promotion=None):
        super().__init__(ledger, verifier, audit, config,
                         staleness_policy=staleness_policy, conflict=conflict,
                         fusion=fusion, acceptance=acceptance,
                         promotion=promotion)
        self.selection = selection
        self.population_artifact = artifact_id
        self._archive: List[Dict[str, object]] = []
        self._seen: Set[str] = set()
        self._archive_lock = threading.Lock()

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
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.population_artifact)
        if head is None or dict(head.state) == state:
            return None
        base_vv = {self.population_artifact:
                   snap.version.get(self.population_artifact, 0)}
        candidate = head.apply(Diff(
            diff_id=f"population:{message}", target=self.population_artifact,
            ops=dict(state), author="population"))
        try:
            _, version = self.ledger.commit(candidate, base_vv,
                                            branch=Ledger.DEV, message=message)
            return version
        except CASConflict:
            return None

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
        ctx = SelectionContext(head=head_candidate, candidates=tuple(candidates),
                               n_workers=1)
        chosen = list(self.selection.select(ctx, 1))
        if not chosen:
            return reports
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
                       conflict=None, fusion=None, acceptance=None):
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
            selection=selection, artifact_id=artifact_id,
            conflict=conflict, fusion=fusion, acceptance=acceptance)

    return build
