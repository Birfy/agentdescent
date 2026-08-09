"""PromptBreeder's Algorithm 1 as a population layer.

The shared :class:`~examples._population.PopulationAggregator` keeps every
committed head forever and asks a ``SelectionPolicy`` who the next parent is.
Algorithm 1 says three things that is not:

1. the population has a **fixed size** ``N``;
2. the two sampled units are compared on a **random batch of training data**,
   not on the acceptance gate's held-out split;
3. the **loser is replaced** by the mutated copy of the winner -- units die.

(3) is the one that changes the search rather than the bookkeeping. Without it
the population only grows, a weak unit stays eligible for every future
tournament, and the selection pressure the paper's replacement step creates is
absent. This class is the paper's loop mapped onto the engine's batch cycle:

    super().step()      the batch's proposals merge -> one committed child C
    _admit(C)           C enters the population, *overwriting the last loser*
    tournament()        sample two units uniformly, score both on a train batch,
                        commit the winner as the head so the next batch mutates
                        it, and remember the loser as the next slot to overwrite

The population also has to be *readable* by the mutation operators: four of the
paper's nine (EDA, EDA-rank-and-index, lineage, crossover) take the population
itself as input. :class:`PopulationView` is the shared handle -- written here,
read from the port's ``propose``.
"""

from __future__ import annotations

import random
import threading
from typing import Dict, List, Optional, Tuple

from agentdescent.ledger import Ledger

from examples._population import PopulationAggregator


class PopulationView:
    """What the mutation operators are allowed to see of the population.

    Four of PromptBreeder's nine operators are defined over the population --
    EDA over its task-prompts, EDA-rank-and-index over them sorted by fitness,
    lineage over the elites in the order they became elite, and crossover over a
    fitness-weighted draw. A ``propose`` that cannot see the population can only
    implement the other five, which is the shape this port had.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._units: List[Dict[str, object]] = []
        self._elites: List[Dict[str, str]] = []

    def publish(self, units: List[Dict[str, object]],
                elites: List[Dict[str, str]]) -> None:
        with self._lock:
            self._units = [dict(u) for u in units]
            self._elites = [dict(e) for e in elites]

    def units(self) -> List[Dict[str, object]]:
        with self._lock:
            return [dict(u) for u in self._units]

    def task_prompts(self, *, ranked: bool = False) -> List[str]:
        """Deduplicated task-prompts; ranked ascending by fitness on request.

        The paper filters the EDA list for diversity before showing it -- a unit
        whose prompt is already in the list adds nothing to a distribution
        estimate.
        """
        rows = self.units()
        if ranked:
            rows.sort(key=lambda u: float(u["score"]))
        out: List[str] = []
        for unit in rows:
            prompt = str(unit["state"].get("task_prompt", "")).strip()
            if prompt and prompt not in out:
                out.append(prompt)
        return out

    def elite_history(self) -> List[str]:
        with self._lock:
            return [str(e.get("task_prompt", "")).strip() for e in self._elites
                    if str(e.get("task_prompt", "")).strip()]

    def weighted_task_prompt(self, rng: random.Random,
                             exclude: str = "") -> Optional[str]:
        """A task-prompt drawn in proportion to fitness -- the crossover partner."""
        rows = [u for u in self.units()
                if str(u["state"].get("task_prompt", "")).strip()
                and str(u["state"].get("task_prompt", "")).strip() != exclude]
        if not rows:
            return None
        weights = [max(float(u["score"]), 0.0) + 1e-6 for u in rows]
        return str(rng.choices(rows, weights=weights, k=1)[0]["state"]["task_prompt"])


class PromptBreederPopulation(PopulationAggregator):
    """Fixed-size population, train-batch tournament, loser replacement."""

    def __init__(self, ledger, verifier, audit, config, staleness_policy=None,
                 *, selection, artifact_id, conflict=None, fusion=None,
                 acceptance=None, promotion=None, fitness, view: PopulationView,
                 size: int, seed: int = 0, seed_units=(),
                 batch: int = 4):
        super().__init__(ledger, verifier, audit, config, staleness_policy,
                         selection=selection, artifact_id=artifact_id,
                         conflict=conflict, fusion=fusion, acceptance=acceptance,
                         promotion=promotion)
        self._fitness = fitness
        self._batch = max(1, int(batch))
        self._view = view
        self._size = max(2, int(size))
        self._rng = random.Random(seed)
        self._pending_seed_units = list(seed_units)
        #: Index of the unit the last tournament lost, i.e. the slot the next
        #: admitted child overwrites. `None` before the first tournament, when
        #: the population is still filling to `size`.
        self._loser: Optional[int] = None
        self._elites: List[Dict[str, str]] = []
        self._best: float = -1.0

    # -- the population ------------------------------------------------------

    def _admit(self, artifact, version: int) -> None:
        """Enter a committed unit, replacing the last tournament's loser.

        Overrides the shared archive's grow-forever admit. The score is the
        paper's: a batch of *training* data, not the held-out split the gate
        already used -- ranking a population on the gate's own signal would make
        the tournament a second acceptance test rather than a fitness measure.
        """
        state = dict(getattr(artifact, "state", {}) or {})
        key = artifact.render()
        with self._archive_lock:
            if key in self._seen:
                return
            self._seen.add(key)
        score = float(self._fitness(state, self._batch))
        entry = {"state": state, "score": score, "version": int(version),
                 "selected": 0}
        with self._archive_lock:
            if len(self._archive) < self._size or self._loser is None:
                self._archive.append(entry)
            else:
                self._archive[self._loser] = entry
                self._loser = None
            if score > self._best:
                self._best = score
                self._elites.append(dict(state))
        self._publish()

    def snapshot(self):
        """The population as it stands, for the run's own report."""
        with self._archive_lock:
            # 70 characters used to be enough to hide the only thing worth
            # reading: `'Solve monetary story questions and provide your final
            # response in the '` cuts off exactly where the convention would be
            # named or not.
            return [(str(u["state"].get("task_prompt", ""))[:240],
                     float(u["score"])) for u in self._archive]

    def _publish(self) -> None:
        with self._archive_lock:
            units = [dict(u) for u in self._archive]
            elites = [dict(e) for e in self._elites]
        self._view.publish(units, elites)

    # -- Algorithm 1 ---------------------------------------------------------

    # NOT `_tournament`: `Aggregator._tournament(artifact, diffs)` already exists
    # and is what fusion uses to pit two diffs head to head. Overriding it here
    # silently replaced the fusion layer's tournament with this one, and every
    # merge died with "takes 1 positional argument but 3 were given".
    def _replication_event(self) -> Optional[Dict[str, object]]:
        """Sample two units uniformly, score both, return the winner.

        Uniform sampling *without* replacement, and both units re-scored on a
        fresh training batch each time -- the paper re-evaluates rather than
        reusing a cached fitness, because the batch is what makes the comparison
        a sample rather than a verdict.
        """
        with self._archive_lock:
            if len(self._archive) < 2:
                return None
            indices = self._rng.sample(range(len(self._archive)), 2)
            pair = [(i, dict(self._archive[i])) for i in indices]
        scored: List[Tuple[int, Dict[str, object], float]] = []
        for index, unit in pair:
            score = float(self._fitness(dict(unit["state"]), self._batch))
            scored.append((index, unit, score))
        scored.sort(key=lambda row: row[2], reverse=True)
        (win_i, winner, win_s), (lose_i, _, _) = scored[0], scored[1]
        with self._archive_lock:
            self._archive[win_i]["score"] = win_s
            self._archive[win_i]["selected"] = int(
                self._archive[win_i]["selected"]) + 1
            self._loser = lose_i
            if win_s > self._best:
                self._best = win_s
                self._elites.append(dict(winner["state"]))
        self._publish()
        return winner

    def step(self):
        before = self.ledger.snapshot(Ledger.DEV)
        pre_head = before.get(self.population_artifact)
        if pre_head is not None:
            self._admit(pre_head, before.version.get(self.population_artifact, 0))
        # The paper starts from N units, each the problem description mutated by
        # a mutation-prompt and a thinking-style drawn from its hand-designed
        # sets. They are generated once, on the first step, because generating
        # them needs the model and the aggregator only has it from here.
        if self._pending_seed_units:
            pending, self._pending_seed_units = self._pending_seed_units, []
            for state in pending:
                with self._archive_lock:
                    if len(self._archive) >= self._size:
                        break
                    key = "\n\n".join(f"[{k}]\n{v}" for k, v in state.items())
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                score = float(self._fitness(dict(state), self._batch))
                with self._archive_lock:
                    self._archive.append({"state": dict(state), "score": score,
                                          "version": 0, "selected": 0})
            self._publish()

        reports = super(PopulationAggregator, self).step()

        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.population_artifact)
        if head is None:
            return reports
        self._admit(head, snap.version.get(self.population_artifact, 0))

        winner = self._replication_event()
        if winner is None:
            return reports
        target = dict(winner["state"])
        version = self._commit_state(target, "promptbreeder: tournament winner")
        if version is not None:
            from agentdescent.aggregator import MergeReport
            with self._archive_lock:
                size = len(self._archive)
            reports.append(MergeReport(
                self.population_artifact, None, False, 0, 0, 0, 0, 0.0, version,
                reason=(f"promptbreeder: winner bred, loser slot {self._loser} "
                        f"(|P|={size}/{self._size})"),
                category="population-select"))
        return reports


def promptbreeder_population(ctx, *, view: PopulationView, size: int,
                             seed_units=(), batch: int = 4, built=None):
    """The ``aggregator_factory`` for one PromptBreeder run.

    ``built`` is a one-slot list the caller can read the constructed population
    back out of -- the factory is called by the engine, so the port otherwise has
    no handle on the object it configured, and cannot report what it did.
    """

    def build(ledger, verifier, audit, config, staleness_policy=None):
        pop = PromptBreederPopulation(
            ledger, verifier, audit, config, staleness_policy,
            selection=ctx.selection, artifact_id=ctx.artifact_id,
            conflict=ctx.conflict, fusion=ctx.fusion, acceptance=ctx.acceptance,
            fitness=ctx.fitness, view=view, size=size, seed=ctx.seed,
            seed_units=seed_units, batch=batch)
        if built is not None:
            built.append(pop)
        return pop

    return build
