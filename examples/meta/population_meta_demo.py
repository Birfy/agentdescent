#!/usr/bin/env python3
"""Meta-population: the archive seeds a population, the population promotes.

Session 1 seeds the archive with `continuous_meta_run` (adopt -> one rule
archived). Session 2 runs `meta_population_run`: the archive's best rules each
seed an island, every island searches from its own starting rule against the
same isolated validation set, and the island that gained most is promoted back
to the archive -- so the next call sees a *better* rule at the top.

Offline, deterministic, no key:

    python -m examples.meta.population_meta_demo
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile

from agentdescent import (
    ConsolidationStore,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    continuous_meta_run,
    meta_population_run,
)
from agentdescent.sampling import DifficultyWeighted


def _c_of(rendered: str) -> float:
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    return float(m.group(1)) if m else 0.2


def run(rendered: str, task: Task) -> str:
    return str(task.meta["gold"])


def reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def propose(rendered: str, task: Task, output: str, reward: float) -> str:
    return None


def tasks(prefix: str, n: int = 8) -> list:
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(i)}) for i in range(n)]


def slot() -> ParamSlot:
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def offline_propose(rendered: str, task: Task, output: str, reward: float) -> str:
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def helpful(value, seed):
    # a synthetic landscape where a larger c finds the answer sooner: reward
    # rises with c, so an island starting low has room to gain and an island
    # starting high has little -- which is exactly the promotion signal.
    c = getattr(value, "c", 0.2)
    r = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[r, min(1.0, r + 0.05)], final=r)


def helpful_val(value, seed):
    return helpful(value, seed)


CONFIG = {"dataset": "synthetic-pop", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", default=None,
                        help="archive directory (a tempdir when omitted)")
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--outer-rounds", type=int, default=3)
    args = parser.parse_args()

    store = ConsolidationStore(args.archive or tempfile.mkdtemp(
        prefix="agentdescent-meta-pop-"))
    common = dict(
        slot="task_sampler", spec=slot(), propose=offline_propose,
        tasks_for=lambda rec: tasks(rec.run_id), run=run, reward=reward,
        inner_propose=propose, inner_strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, inner=dict(INNER), seeds=(0, 1, 2, 3), val_seeds=(10, 11),
        rounds=args.outer_rounds, n_workers=1)

    print("=== session 1: seed the archive with continuous_meta_run ===")
    for rid in ("run-1", "run-2"):
        s, _ = continuous_meta_run(
            store=store, run_id=rid, **common,
            repo_path=tempfile.mkdtemp(prefix="agentdescent-cmeta-run-"),
            problems={"ta": helpful, "tb": helpful},
            val_problems={"va": helpful_val})
        print(f"  {rid}: verdict {s.verdict}  value {_c_of(s.value):.2f}  "
              f"archived: {s.archived}")

    print(f"=== session 2: a population of {args.islands} islands ===")
    # train_runs > archived-rule count forces the caller's synthetic landscape
    # as the islands' training set, so the outcomes shown are deterministic
    report, _ = meta_population_run(
        store=store, islands=args.islands, train_runs=3, val_runs=1, **common,
        repo_dir=tempfile.mkdtemp(prefix="agentdescent-meta-pop-repos-"),
        problems={"ta": helpful, "tb": helpful}, val_problems={"va": helpful_val})
    for island in report.islands:
        print(f"  island {island.index}: seeded_from={island.seeded_from} "
              f"value {_c_of(island.value):.2f} gain {island.mean_gain:+.3f} "
              f"-> {island.verdict}")
    print(f"  promoted: island {report.promoted_from} "
          f"({report.promoted_run_id})")
    print(f"  {report.reason}  (archive holds {report.archive_count} rule(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
