#!/usr/bin/env python3
"""Meta-evolution: the agent improves its own search rule.

The outer loop evolves one decision slot of the inner search -- here the UCB
exploration constant of the `task_sampler` -- and then **judges the result on
problems the outer loop never saw**. `meta_run()` runs the ordinary engine one
level up, validates the evolved value on unseen problems, and hands back an
adopt/reject verdict with its reason.

Offline, deterministic, no key:

    python -m examples.meta.meta_evolve_demo
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile

from agentdescent import (
    ParamSlot,
    SingleSlot,
    Task,
    evolve_problem,
    meta_run,
)
from agentdescent.sampling import DifficultyWeighted


def _cap(rendered: str) -> int:
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def run(rendered: str, task: Task) -> str:
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def propose(rendered: str, task: Task, output: str, reward: float) -> str:
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"       # jump the capability to the failing task
    return None


def tasks(prefix: str, n: int = 8) -> list:
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def problem(tasks_, rounds: int):
    return evolve_problem(
        tasks_, reward, slot="task_sampler", run=run, propose=propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def slot() -> ParamSlot:
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def offline_propose(rendered: str, task: Task, output: str, reward: float) -> str:
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(m.group(1)) if m else 0.2
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--outer-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = {
        "train-a": problem(tasks("a"), args.rounds),
        "train-b": problem(tasks("b"), args.rounds),
    }
    val = {
        "val-z": problem(tasks("z"), args.rounds),
    }
    report, result = meta_run(
        train, val, slot="task_sampler", spec=slot(),
        propose=offline_propose, seeds=(args.seed, args.seed + 1),
        val_seeds=(20, 21), rounds=args.outer_rounds, n_workers=1,
        repo_path=tempfile.mkdtemp(prefix="agentdescent-meta-"))

    print(f"evolved slot value:\n{report.value}")
    print(f"verdict: {report.verdict} -- {report.reason}")
    for name, row in report.validation.items():
        print(f"  {name}: before={row['before']:.3f} after={row['after']:.3f} "
              f"gain={row['gain']:+.3f}")
    print(f"mean gain: {report.mean_gain:+.3f}  transfer: "
          f"{report.transfer if report.transfer is None else round(report.transfer, 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
