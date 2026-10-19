#!/usr/bin/env python3
"""Continuous meta-learning: the archive drives the next session.

`meta_run()` evolves a decision slot and decides; `continuous_meta_run()` makes
that decision durable. Session 1 (empty archive) evolves the UCB exploration
constant of the `task_sampler` on a synthetic landscape that rewards a larger
constant, adopts it, and archives the rule with a problem descriptor. Session 2
on the same archive is **seeded by session 1's rule**, the landscape forces it
to keep climbing, it adopts a strictly better rule, and that lands in the
archive too -- the system's search rule improves with its own history, and the
report says so in both numbers: the gain over the prior it had to beat
(`mean_gain`) and the gain over the fresh seed (`fresh_gain`, the cross-session
number).

Offline, deterministic, no key:

    python -m examples.meta.continuous_meta_demo
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
    evolve_problem,
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


def problem(tasks_, rounds: int):
    return evolve_problem(
        tasks_, reward, slot="task_sampler", run=run, propose=propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def offline_propose(rendered: str, task: Task, output: str, reward: float) -> str:
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def helpful(value, seed):
    # a synthetic landscape: a larger exploration constant finds the answer
    # sooner. Reward rises with c up to a ceiling, and the offline proposer
    # keeps climbing while the reward is below 0.7 -- so a higher prior is a
    # better place to start, which is exactly what the archive hands session 2.
    c = getattr(value, "c", 0.2)
    r = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[r, min(1.0, r + 0.05)], final=r)


CONFIG = {"dataset": "synthetic-cap", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


def helpful_val(value, seed):
    # a distinct object with the same landscape as `helpful`: the isolation
    # guard compares problems by identity, so the validation problem cannot
    # be the same object as a training one.
    return helpful(value, seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", default=None,
                        help="archive directory (a tempdir when omitted)")
    parser.add_argument("--outer-rounds", type=int, default=3)
    args = parser.parse_args()

    store = ConsolidationStore(args.archive or tempfile.mkdtemp(
        prefix="agentdescent-continuous-meta-"))
    common = dict(
        slot="task_sampler", spec=slot(), propose=offline_propose,
        tasks_for=lambda rec: tasks(rec.run_id), run=run, reward=reward,
        inner_propose=propose, inner_strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, inner=dict(INNER), seeds=(0, 1, 2, 3), val_seeds=(10, 11),
        rounds=args.outer_rounds, n_workers=1)

    print("=== session 1: empty archive -> an ordinary meta_run ===")
    s1, _ = continuous_meta_run(
        store=store, run_id="run-1", **common,
        repo_path=tempfile.mkdtemp(prefix="agentdescent-cmeta-run-"),
        problems={"ta": helpful, "tb": helpful}, val_problems={"va": helpful_val})
    print(f"  verdict: {s1.verdict} -- {s1.reason}")
    print(f"  prior:   {_c_of(s1.seed_value):.2f}  ->  value: {_c_of(s1.value):.2f}")
    print(f"  archived: {s1.archived} ({s1.archived_run_id})  mean_gain "
          f"{s1.mean_gain:+.3f}  fresh_gain {s1.fresh_gain:+.3f}")

    print("=== session 2: the archive seeds the next session ===")
    s2, _ = continuous_meta_run(
        store=store, run_id="run-2", train_runs=2, **common,
        repo_path=tempfile.mkdtemp(prefix="agentdescent-cmeta-run-"),
        problems={"ta": helpful, "tb": helpful}, val_problems={"va": helpful_val})
    print(f"  seeded_from: {s2.seeded_from}  prior {_c_of(s2.seed_value):.2f} "
          f"->  value {_c_of(s2.value):.2f}")
    print(f"  verdict: {s2.verdict} -- {s2.reason}")
    print(f"  archived: {s2.archived} ({s2.archived_run_id})  mean_gain "
          f"{s2.mean_gain:+.3f}  fresh_gain {s2.fresh_gain:+.3f}  "
          f"(archive holds {s2.archive_count} rule(s))")

    rec1 = store.find("run-1")
    rec2 = store.find("run-2")
    if rec1 and rec2:
        print(f"  archive progression: {_c_of(rec1.rendered):.2f} -> "
              f"{_c_of(rec2.rendered):.2f}, each with a problem descriptor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
