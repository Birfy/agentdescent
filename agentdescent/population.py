"""Population search: how a `SelectionPolicy` runs, single head or many.

:mod:`agentdescent.selection` defines *where the next batch starts*; this module
is what makes the answer take effect. The candidate pool lives **in the
aggregator**, and "selection" is a ledger commit that moves the head to the
chosen candidate.

:class:`PopulationAggregator` subclasses the shipped
:class:`~agentdescent.aggregator.Aggregator` -- staleness, conflict, fusion,
statistical acceptance and promotion all run unchanged -- and adds three things
around it: it archives every distinct committed head with its held-out score,
asks the selection policy which archived candidate the next batch should mutate,
and commits that candidate back to the right branch.

**Single head (the default).** One ``dev`` branch, all workers start from it,
the selection policy picks one candidate and ``dev`` is rewritten to it. This is
the serialised population that GEPA and DGM each found on their own, and it is
what a caller with no ``n_workers`` gets.

**Multiple live heads (``Ledger.fork``).** When the run has several workers,
``step(n_workers)`` asks the policy for as many candidates as there are workers.
``chosen[0]`` becomes the primary ``dev`` head; each further candidate is
committed to its own ``head/<slot>`` fork, and workers are distributed across
them via :meth:`head_for_worker`. Every branch's head is admitted to the archive
each round, so a worker's improvement on any fork is visible to the next
selection. This is a **beam-style** multi-head: each round re-commits the chosen
candidates to their forks, so a fork's head is reset to its candidate between
rounds and the archive is what carries exploration forward.

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


def _state_size(state: Dict[str, str]) -> int:
    """A candidate's content size, as a deterministic cost proxy.

    The prompt an expansion sends is dominated by the artifact it renders, so
    content size is a first-order proxy for the input tokens that expansion
    costs -- and unlike measured tokens it is attributable to a single candidate
    without racing a shared meter. A cost-aware policy divides its exploration
    bonus by this, so a cheap candidate gets more speculative looks than an
    expensive one at the same rank. It is a *proxy* and the policy treats a
    missing cost as the pool mean, so an imperfect scale cannot turn an
    unmeasured candidate into a free one.
    """
    return sum(len(str(k)) + len(str(v)) for k, v in state.items())


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
        #: Rendered candidates already in the archive, for `_admit`'s dedup.
        #:
        #: **Not** `Aggregator._seen`, which this used to shadow. That set is
        #: the aggregator's record of *artifact ids*: `_known_artifacts` unions
        #: it into the promotion table and `finalize` promotes every member.
        #: Feeding it `artifact.render()` meant `finalize` called
        #: `_promote("# Playbook\n- change-t2\n...")` -- a whole rendered
        #: artifact passed off as an id -- once per distinct candidate. Two
        #: different questions ("which artifacts do I own" and "have I archived
        #: this text") had been sharing one set because both are sets of str.
        self._keys: Set[str] = set()
        self._archive_lock = threading.Lock()
        #: Selections made, i.e. the ``round`` handed to the policy. Not under
        #: the archive lock: `step()` is the merger's, one thread, while the
        #: lock guards the archive against the workers that `ingest`.
        self._selections = 0
        #: Fraction of the run's budget still unspent, in ``[0, 1]``, for the
        #: ``SelectionContext`` a budget-aware policy anneals on. The engine sets
        #: it each round (from its governor); ``1.0`` is the no-budget default,
        #: which is what a run without a token ceiling should report -- it never
        #: runs short of one.
        self.budget_remaining: float = 1.0
        #: The fork names this aggregator has created, in creation order.
        #: ``head_for_worker`` cycles through them.
        self._forks: List[str] = []

    def head_for_worker(self, worker_id: int = 0) -> str:
        """The branch a worker should snapshot.

        Worker 0 always gets ``dev`` (the primary head, chosen[0]). The rest
        cycle through the forks (``head/0``, ``head/1``, ...), so a batch of N
        workers explores up to ``1 + len(self._forks)`` candidates in parallel.
        With no forks created yet, every worker gets ``dev`` (the old
        behaviour)."""
        if worker_id == 0 or not self._forks:
            return Ledger.DEV
        return self._forks[(worker_id - 1) % len(self._forks)]

    # -- the archive ---------------------------------------------------------

    def _admit(self, artifact, version: int) -> None:
        key = artifact.render()
        with self._archive_lock:
            if key in self._keys:
                return
            self._keys.add(key)
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
                          selected=int(entry["selected"]),
                          # A deterministic cost proxy: the artifact's rendered
                          # content size, which dominates the prompt and so the
                          # input tokens of expanding it. It is not the measured
                          # token spend -- that is not attributable to a single
                          # parent from a shared meter under concurrency -- and
                          # the policy treats an unmeasured cost as the pool
                          # mean rather than zero, so a proxy that is wrong in
                          # scale is far less harmful than one that is wrong in
                          # *presence*.
                          cost=float(_state_size(entry["state"])))
                for entry in self._archive
            ]

    def _best_state(self) -> Optional[Dict[str, str]]:
        with self._archive_lock:
            if not self._archive:
                return None
            best = max(self._archive, key=lambda entry: entry["score"])
            return dict(best["state"])

    def _commit_state(self, state: Dict[str, str], message: str,
                      branch: str = Ledger.DEV) -> Optional[int]:
        """Replace the head's state outright, deletions included.

        ``ops`` carries ``None`` for every key the head has and the target does
        not. Without that half the switch is a *union*: `apply` only sets the
        keys it is given, so on a grow-only key space the head kept everything
        it had ever accumulated and "start from candidate C" quietly meant
        "start from C plus the incumbent". See the module docstring.
        """
        snap = self.ledger.snapshot(branch)
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
                                            branch=branch, message=message)
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

    def step(self, n_workers: int = 1) -> List[MergeReport]:
        # Admit the pre-merge head first: on the first step that is the seed,
        # which a post-merge-only admit would lose the moment anything commits
        # over it -- and a population that forgot its seed cannot fall back.
        before = self.ledger.snapshot(Ledger.DEV)
        pre_head = before.get(self.population_artifact)
        if pre_head is not None:
            self._admit(pre_head,
                        before.version.get(self.population_artifact, 0))
        reports = super().step()
        # Admit the head of **every** live branch, not just dev. In a multi-head
        # run, workers commit their proposals back to their own fork, so the
        # fork heads are where new candidates appear. Admitting only dev would
        # throw every fork's exploration away -- the archive would never see a
        # candidate a worker improved, and selection could never choose it.
        # ``dev`` is in ``live_heads()``, so this covers it too; the per-branch
        # admit below is gone.
        any_head = False
        for branch_name in self.ledger.live_heads():
            snap = self.ledger.snapshot(branch_name)
            branch_head = snap.get(self.population_artifact)
            if branch_head is not None:
                self._admit(branch_head,
                            snap.version.get(self.population_artifact, 0))
                any_head = True
        if not any_head:
            return reports
        candidates = self._candidates()
        if len(candidates) < 2:
            return reports
        head = self.ledger.snapshot(Ledger.DEV).get(self.population_artifact)
        if head is None:
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
                               round=self._selections, n_workers=n_workers,
                               budget_remaining=self.budget_remaining)
        self._selections += 1
        # Ask for as many starting points as there are workers, so a multi-head
        # run can explore several candidates in parallel. The ledger's fork
        # mechanism (``head/<slot>``) holds each one as its own branch.
        n_heads = min(len(candidates), n_workers if n_workers else 1)
        chosen = list(self.selection.select(ctx, n_heads))
        if not chosen:
            return reports
        # chosen[0] is the primary head -> dev. The rest become forks, and each
        # fork is **created before** the candidate is committed to it, so a
        # worker that starts from head/1 mid-step never sees a half-written fork.
        self._forks = []
        self._offered(chosen[0], candidates)
        for slot, cand in enumerate(chosen):
            target = dict(cand.state)
            with self._archive_lock:
                for entry in self._archive:
                    if dict(entry["state"]) == target:
                        entry["selected"] = int(entry["selected"]) + 1
                        break
            if slot == 0:
                version = self._commit_state(target, "population: select parent")
            else:
                fork_name = f"head/{slot - 1}"
                self.ledger.fork(fork_name, from_branch=Ledger.DEV)
                self._commit_state(target, f"population: head/{slot - 1}",
                                   branch=fork_name)
                self._forks.append(fork_name)
        if version is not None:
            reports.append(MergeReport(
                self.population_artifact, None, False, 0, 0, 0, 0, 0.0, version,
                reason=f"population: parent switched (archive={len(candidates)}, "
                       f"heads={len(self._forks)})",
                category="population-select"))
        return reports

    def finalize(self) -> None:
        """Leave the best-scoring candidate on the head, then promote.

        Multi-head cleanup: discard all forks so the ledger ends with one
        ``dev`` branch holding the best candidate. A fork that scored higher
        than ``dev`` is committed to ``dev`` first."""
        best = self._best_state()
        if best is not None:
            self._commit_state(best, "population: final best")
        # Discard forks
        for fork_name in list(self._forks):
            try:
                self.ledger.discard_head(fork_name)
            except Exception:  # noqa: BLE001
                pass
        self._forks = []
        super().finalize()

    # -- checkpointing -------------------------------------------------------

    def checkpoint(self) -> Optional[dict]:
        """Serialise the archive and the selection counter.

        The archive is the whole point of a population run: it holds every
        committed candidate with its held-out score and its ``selected``
        count — the novelty term ``Archive(sampling='novelty')`` reads, the
        rotation index ``Beam`` needs. Without this, a resumed run starts an
        archive of one (the head) and every selection policy degenerates:
        ``Beam(4)`` is ``Beam(1)`` again, ``Archive``'s novelty term is gone,
        and ``ParetoFrontier`` has a front of one to sit on.

        The parent ``Aggregator`` state (Beta posteriors, promotion counters,
        the artifact ids in ``_seen``) is merged in via ``super().checkpoint()``.
        ``keys`` is this layer's own: the rendered form of every archived
        candidate, which ``_admit`` dedups on. It is kept separate from the
        parent's ``seen`` because the two answer different questions — see
        ``_keys`` — and folding them together would restore a run whose
        ``finalize`` then promotes rendered artifacts as if they were ids.
        """
        parent = super().checkpoint()
        with self._archive_lock:
            own = {
                "archive": [dict(entry) for entry in self._archive],
                "keys": sorted(self._keys),
                "selections": self._selections,
            }
        if parent is not None:
            own.update(parent)
        return own

    def restore(self, state: dict) -> None:
        """Restore the archive written by :meth:`checkpoint`.

        Entries are validated shape-wise; a malformed entry is skipped rather
        than raised (a partially restored archive still beats an empty one,
        and one bad row must not cost the whole history). The restored
        ``state`` dicts are the candidates' *key spaces* — they are only ever
        compared and committed, never executed, so trusting their shape is
        enough.

        The parent ``Aggregator`` state (Beta posteriors, promotion counters,
        seen artifact ids) is restored via ``super().restore()`` first, so the
        acceptance prior is in place before the population archive is loaded
        on top of it.
        """
        # Restore the parent's state first (posteriors, promoted_at, seen).
        # The parent's ``restore`` reads ``posteriors`` / ``promoted_at`` /
        # ``seen`` keys, which ``checkpoint`` merged in from ``super()``.
        super().restore(state)

        archive = state.get("archive")
        if not isinstance(archive, list):
            return
        restored: List[Dict[str, object]] = []
        for entry in archive:
            if not isinstance(entry, dict):
                continue
            st = entry.get("state")
            if not isinstance(st, dict):
                continue
            try:
                restored.append({
                    "state": dict(st),
                    "score": float(entry.get("score", 0.0)),
                    "version": int(entry.get("version", 0)),
                    "selected": int(entry.get("selected", 0)),
                })
            except (TypeError, ValueError):
                continue
        with self._archive_lock:
            self._archive = restored
            # The dedup keys `_admit` reserves on. `seen_keys` is read as a
            # fallback for a checkpoint written before this layer had a set of
            # its own. Absent both, `_keys` stays empty and the archive can
            # gain a duplicate row for a candidate committed again after the
            # resume -- the keys are `artifact.render()`, which needs the
            # strategy, and the aggregator does not hold one, so they cannot be
            # rebuilt from the archive's states.
            keys = state.get("keys")
            if not isinstance(keys, list):
                keys = state.get("seen_keys")
            if isinstance(keys, list):
                self._keys = {str(k) for k in keys}
            try:
                self._selections = int(state.get("selections", 0))
            except (TypeError, ValueError):
                self._selections = 0


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
