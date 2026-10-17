#!/usr/bin/env python3
"""A/B the persistent-island population engine against the alternatives.

Equal rollout budget, three arms:

* ``single``     -- SingleHead: the engine's untouched path.
* ``multi``      -- CladeThompson, no persistent islands (per-step forks).
* ``islands``    -- CladeThompson + ``population_islands=2`` (persistent).

Reported per arm: time-to-quality, final reward, tokens, rollouts.

Run offline (deterministic, no key):

    python -m bench.population_engine
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time

from agentdescent import Policies, SingleSlot, Task, evolve
from agentdescent.selection import CladeThompson, SingleHead


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    if task.meta.get("a", 0) > _cap(rendered):
        return f"cap:{_cap(rendered) + 1}"
    return None


def propose_rugged(rendered, task, output, reward):
    # a fraction of proposals are dead ends (reset the cap); single-head gets
    # stuck on an unlucky streak, while islands explore several branches
    import random
    if reward >= 0.999:
        return "cap:0"
    if task.meta.get("a", 0) > _cap(rendered):
        if random.random() < 0.3:
            return "cap:0"                  # a dead-end proposal
        return f"cap:{_cap(rendered) + 1}"
    return None


def tasks(prefix, n=8):
    # order puts a large cart in TRAIN so the skill can learn the full range
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def arm(name, *, islands, selection, rounds, repo, rugged=False):
    t0 = time.time()
    result = evolve(tasks(name), reward, run=run,
                    propose=propose_rugged if rugged else propose,
                    strategy=SingleSlot(initial_value="cap:0"),
                    policies=Policies(selection=selection),
                    rounds=rounds, n_workers=3, max_concurrency=3,
                    repo_path=repo, population_islands=islands,
                    target_reward=1.0)
    ttq = result.time_to_quality(1.0)
    return {
        "arm": name,
        "final_reward": round(result.final_reward, 3),
        "time_to_quality": round(ttq, 3) if ttq is not None else None,
        "wallclock": round(time.time() - t0, 2),
        "committed": result.outcomes().get("committed", 0),
        "rollouts": result.rollouts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for seed in range(args.seeds):
        for rugged in (False, True):
            for name, islands, sel in (
                ("single", 0, SingleHead()),
                ("multi", 0, CladeThompson(seed=seed)),
                ("islands", 2, CladeThompson(seed=seed)),
            ):
                repo = tempfile.mkdtemp(prefix=f"agentdescent-{name}-")
                row = arm(name, islands=islands, selection=sel,
                          rounds=args.rounds, repo=repo, rugged=rugged)
                row["seed"] = seed
                row["rugged"] = rugged
                rows.append(row)

    print(f"{'arm':<8} {'land':<7} {'seed':<4} {'ttq':<8} {'final':<7} {'committed':<9}")
    for r in sorted(rows, key=lambda r: (r["rugged"], r["arm"], r["seed"])):
        land = "rugged" if r["rugged"] else "smooth"
        print(f"{r['arm']:<8} {land:<7} {r['seed']:<4} "
              f"{str(r['time_to_quality']):<8} {r['final_reward']:<7} "
              f"{r['committed']:<9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
