"""The reference synchronous runtime -- now an adapter over ``evolve()``.

This used to be a second implementation of the loop: its own round barrier, its
own worker dispatch, its own snapshot staggering, feeding the shared `Ledger` and
`Aggregator`. `docs/architecture.md` called that "a known wart rather than a
design intent", and issue #93 records what it cost -- two measured fixes that had
to be hand-ported, two early-stop epsilons nobody chose, and three mechanisms the
general engine had to re-derive (and got wrong) because the reference runtime
already had them.

So the loop is gone and this is the adapter that remains: it describes the
reference domain in the vocabulary `evolve()` speaks -- see
:mod:`agentdescent.domains.router` -- runs it, and reports the result in the
shape the experiments read (:class:`RoundStat`).

**What is unchanged:** the public surface, the sequential round barrier
(``max_concurrency=1``, so a run is reproducible), snapshot staggering via
``refresh_interval``, the aggregator config, the oracle budget, and the RQ1 fork
control.

**What differs**, because the general engine's rollout contract genuinely does,
is listed in :mod:`agentdescent.domains.router`: ``before_after_delta`` and
``evidence_eval`` are measured over the whole cluster rather than the failing
subset, and noise is per proposal rather than per worker. Both paths still
converge on the same table, and merge still beats fork -- which is what the
experiments actually claim.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .aggregator import Aggregator, AggregatorConfig
from .domains.router import (
    RouterSkill,
    RouterStrategy,
    TaskUniverse,
    cluster_split_frac,
    cluster_tasks,
    router_propose,
    router_reward,
    router_run,
)
from .evolution import evolve
from .ledger import Ledger


@dataclass
class RoundStat:
    round: int
    dev_accuracy: float
    stable_accuracy: float
    committed: int
    fused: int
    discarded_stale: int
    conflicts_dropped: int
    oracle_used: int


def _read_skill(artifact_id: str, version: int, payload: dict) -> RouterSkill:
    """Read the engine's artifact back as a :class:`RouterSkill`.

    The engine stores a strategy's flat ``{key: value}`` state, and
    :class:`~agentdescent.domains.router.RouterStrategy`'s state **is** the
    keyword table -- so this is a rename, not a conversion."""
    return RouterSkill(artifact_id, payload.get("state", {}), version,
                       payload.get("blast_radius", 0.2))


def _write_skill(skill: RouterSkill) -> dict:
    """Only here so the read-only handle below satisfies `Ledger`'s constructor."""
    return {"state": dict(skill.table), "blast_radius": skill.blast_radius}


class AgentDescent:
    """The merge-based parallel self-evolution system, on the general engine."""

    def __init__(
        self,
        repo_path: str,
        universe: TaskUniverse,
        n_workers: int = 6,
        noise: float = 0.15,
        refresh_interval: int = 2,
        skill_id: str = "mol-router",
        config: Optional[AggregatorConfig] = None,
        oracle_budget: int = 300,
        seed: int = 0,
        staleness_policy=None,
        self_verify: bool = True,
    ) -> None:
        self.universe = universe
        self.skill_id = skill_id
        self.n_workers = n_workers
        self.noise = noise
        self.refresh_interval = refresh_interval
        self.seed = seed
        self.oracle_budget = oracle_budget
        self.config = config or AggregatorConfig()
        self.staleness_policy = staleness_policy
        #: See `AsyncConfig.self_verify`: the reference loop measured its
        #: before/after delta for free, so leaving this on doubles what a
        #: rollout costs. On by default -- the delta is evidence acceptance
        #: reads -- and off for anything measuring throughput.
        self.self_verify = self_verify
        self.train, self.held_out = universe.split()

        self.strategy = RouterStrategy()
        self._train_tasks, self._held_tasks = cluster_tasks(
            universe, n_clusters=max(2, n_workers))
        if len(self._train_tasks) < n_workers:
            # `clusters()` buckets by keyword hash and drops the empty buckets,
            # so asking for n_workers of them can return fewer -- and the shard
            # then hands two workers the same cluster. They roll out the same
            # deterministic tasks and propose diffs that content-address to
            # duplicates: real budget, no extra evidence.
            warnings.warn(
                f"{n_workers} workers but only {len(self._train_tasks)} non-empty "
                f"task clusters, so {n_workers - len(self._train_tasks)} worker(s) "
                "will duplicate another's cluster each round and add no new "
                "evidence. Use more keywords or fewer workers.",
                RuntimeWarning, stacklevel=2)

        #: A read-only handle on the same repository `evolve()` will use, so
        #: `final_accuracy` and the per-round `stable_accuracy` can read a branch
        #: without the engine having to expose its own. The cross-process file
        #: lock is what makes two handles on one repo safe, and is why it exists.
        self.ledger = Ledger(repo_path, _write_skill, _read_skill)
        self._repo_path = repo_path
        #: Captured from the run, so the objects the experiments inspect are the
        #: ones the run actually used.
        self.aggregator: Optional[Aggregator] = None
        self.verifier: Optional[Any] = None
        self.result = None

    # -- reading a branch -----------------------------------------------------

    def _skill_on(self, branch: str) -> Optional[RouterSkill]:
        try:
            return self.ledger.snapshot(branch).get(self.skill_id)
        except Exception:  # noqa: BLE001 -- a diagnostic read must not end a run
            return None

    def _accuracy_on(self, branch: str) -> float:
        skill = self._skill_on(branch)
        return skill.accuracy(self.held_out) if skill else 0.0

    def final_accuracy(self, branch: str = Ledger.DEV) -> float:
        return self._accuracy_on(branch)

    # -- the run --------------------------------------------------------------

    def run(self, rounds: int = 40) -> List[RoundStat]:
        history: List[RoundStat] = []

        def factory(ledger, verifier, audit, config, policy):
            self.verifier = verifier
            self.aggregator = Aggregator(ledger, verifier, audit, config,
                                         staleness_policy=policy)
            return self.aggregator

        def on_round(info) -> None:
            history.append(RoundStat(
                round=info.round,
                dev_accuracy=info.held_out_reward,
                stable_accuracy=self._accuracy_on(Ledger.STABLE),
                committed=info.committed,
                fused=info.fused,
                discarded_stale=info.discarded_stale,
                conflicts_dropped=info.conflicts_dropped,
                oracle_used=(self.verifier.budget.oracle_calls_used
                             if self.verifier is not None else 0),
            ))

        self.result = evolve(
            self._train_tasks + self._held_tasks,
            router_reward,
            run=router_run,
            propose=router_propose(self.universe.gold, noise=self.noise,
                                   seed=self.seed),
            strategy=self.strategy,
            artifact_id=self.skill_id,
            rounds=rounds,
            n_workers=self.n_workers,
            # Sequential on purpose: the reference loop dispatched its workers in
            # order, which is what makes a seeded run reproducible.
            max_concurrency=1,
            refresh_interval=self.refresh_interval,
            held_out_frac=cluster_split_frac(self._train_tasks, self._held_tasks),
            repo_path=self._repo_path,
            agg_config=self.config,
            staleness_policy=self.staleness_policy,
            aggregator_factory=factory,
            oracle_budget=self.oracle_budget,
            self_verify=self.self_verify,
            solved_threshold=0.999,
            seed=self.seed,
            on_round=on_round,
        )
        return history


def run_fork_baseline(
    universe: TaskUniverse,
    n_workers: int = 6,
    noise: float = 0.15,
    rounds: int = 40,
    seed: int = 0,
) -> float:
    """DGM-style archive/fork control: parallel but never merged (RQ1).

    Each worker grows its *own* table from the cluster it happens to see, and the
    forks are never combined. Because a production system cannot ship a random
    fork per user, we score the best single fork -- which still cannot exceed what
    merging all complementary fixes achieves, since no single fork sees every
    keyword.

    Noise stays **per fork** here, unlike the merged path: a fork is one actor
    for its whole run, so there is a worker to attach it to. That is exactly the
    identity the general engine's shared `propose` does not have.
    """
    train_tasks, _ = cluster_tasks(universe, n_clusters=max(2, n_workers))
    _, held_out = universe.split()
    strategy = RouterStrategy()
    best = 0.0
    for i in range(n_workers):
        propose = router_propose(universe.gold,
                                 noise=noise if i % 3 == 0 else 0.0,
                                 seed=seed + i)
        cluster = train_tasks[i % len(train_tasks)]
        state: Dict[str, str] = {}
        for _ in range(rounds):
            rendered = strategy.render(state)
            output = router_run(rendered, cluster)
            proposal = propose(rendered, cluster, output,
                               router_reward(cluster, output))
            if not proposal:
                break
            diff = strategy.to_diff(state, proposal, f"f{i}", 1, f"fork{i}")
            if diff is None:
                break
            # a fork accepts its own proposals without cross-worker merging.
            state = dict(state, **diff.ops)
        best = max(best, RouterSkill(f"fork{i}", state).accuracy(held_out))
    return best
