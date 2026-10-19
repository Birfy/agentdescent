#!/usr/bin/env python3
"""Harness composition: combine algorithm ports, learn which wins where.

Two deterministic harnesses (a shallow `evolve()` at rounds=4 and a deep one at
rounds=8) run on the same arithmetic domain. `compose_run(mode="all")` records
both outcomes in the archive leaderboard; `select_harness` then routes the same
domain to the harness the recorded evidence says wins; a final
`compose_run(mode="best")` runs only that harness. The composition learns which
algorithm wins where from measured runs, without touching either port.

Offline, deterministic, no key:

    python -m examples.harness.harness_composition_demo
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile

from agentdescent import (
    ConsolidationStore,
    SingleSlot,
    Task,
    compose_run,
    evolve,
    harness,
    harness_leaderboard,
    select_harness,
)


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
        return f"cap:{a}"
    return None


def tasks(prefix: str, n: int = 8) -> list:
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def make_harness(name: str, rounds: int):
    def _run(tasks_, repo):
        return evolve(list(tasks_), reward, run=run, propose=propose,
                      strategy=SingleSlot(initial_value="cap:0"),
                      rounds=rounds, n_workers=2, repo_path=repo,
                      target_reward=1.0)
    return harness(name, _run)


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", default=None,
                        help="archive directory (a tempdir when omitted)")
    args = parser.parse_args()

    store = ConsolidationStore(args.archive or tempfile.mkdtemp(
        prefix="agentdescent-harness-"))
    quick = make_harness("quick", rounds=2)
    deep = make_harness("deep", rounds=8)
    harnesses = [quick, deep]

    print("=== 1. leaderboard training: run both harnesses, record both ===")
    report, _ = compose_run(store, tasks("a"), harnesses, CONFIG, mode="all",
                            repo_dir=tempfile.mkdtemp(prefix="harness-repos-"))
    for r in report.runs:
        print(f"  {r.harness}: reward {r.reward:.3f}  "
              f"time-to-quality {r.time_to_quality}")

    print("=== 2. route the domain by evidence ===")
    chosen, reason = select_harness(store, harnesses, CONFIG)
    print(f"  -> {chosen.name}  ({reason})")
    board = harness_leaderboard(store, CONFIG)
    for name, entry in board.items():
        print(f"  leaderboard {name}: best {entry['best']:.3f} "
              f"over {len(entry['rewards'])} run(s)")

    print("=== 3. production: run only the chosen harness ===")
    report2, _ = compose_run(store, tasks("b"), harnesses, CONFIG, mode="best",
                             repo_dir=tempfile.mkdtemp(prefix="harness-repos-"))
    print(f"  ran: {[r.harness for r in report2.runs]}  "
          f"winner: {report2.winner}  ({report2.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
