#!/usr/bin/env python3
"""Cross-run consolidation: what a run learned survives it.

Run 1 evolves a capability-capped skill on an arithmetic domain and archives
the artifact. Run 2 -- new tasks, same domain -- retrieves the archived skill
through a mini-eval gate and reaches quality earlier than the empty seed.

Offline, deterministic, no key:

    python -m examples.consolidation.consolidation_demo
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass, field

from agentdescent import (
    ConsolidationStore,
    SingleSlot,
    Task,
    evolve,
    seed_from_archive,
)


@dataclass
class Cart:
    """A deterministic arithmetic task the capability-capped skill answers."""
    a: int
    b: int

    @property
    def gold(self) -> str:
        return str(self.a + self.b)


def _cap(rendered: str) -> int:
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def run(rendered: str, task: Task) -> str:
    cart = task.meta["cart"]
    if cart.a <= _cap(rendered) and cart.b <= _cap(rendered):
        return cart.gold
    return "?"


def reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == task.meta["cart"].gold.strip() else 0.0


def propose(rendered: str, task: Task, output: str, reward: float) -> str:
    if reward >= 0.999:
        return "cap:0"
    cart = task.meta["cart"]
    need = max(cart.a, cart.b)
    if need > _cap(rendered):
        return f"cap:{need}"
    return "cap:1"


def tasks(prefix: str, n: int = 8) -> list:
    # order puts a large cart in TRAIN so the skill can learn the full range
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    out = []
    for i, a in enumerate(order):
        cart = Cart(a=a, b=1)
        out.append(Task(id=f"{prefix}{i}", prompt=f"What is {cart.a} + {cart.b}?",
                        meta={"cart": cart}))
    return out


CONFIG = {"dataset": "cart-arithmetic", "scorer": "exact"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--archive", default=None,
                        help="archive directory (a tempdir when omitted)")
    args = parser.parse_args()

    archive_dir = args.archive or tempfile.mkdtemp(prefix="agentdescent-archive-")
    store = ConsolidationStore(archive_dir)

    # --- run 1: learn, then archive the product --------------------------------
    repo1 = tempfile.mkdtemp(prefix="agentdescent-run1-")
    result1 = evolve(tasks("a"), reward, run=run, propose=propose,
                     strategy=SingleSlot(initial_value="cap:0"),
                     rounds=args.rounds, n_workers=2, repo_path=repo1,
                     target_reward=1.0, verbose=True)
    store.archive(result1, run_id="run-1", strategy="SingleSlot", config=CONFIG)
    print(f"run 1 archived: skill={result1.rendered!r} "
          f"reward={result1.final_reward:.3f}")

    # --- run 2: seed from the archive, gated by a mini-eval --------------------
    state, report = seed_from_archive(
        store, tasks("b"), run, reward, strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, sample=6, seed=1)
    print(f"seed: matched={report.matched_run} accepted={report.accepted} "
          f"(seeded={report.seeded_score:.3f} empty={report.empty_score:.3f})")

    repo2s = tempfile.mkdtemp(prefix="agentdescent-seeded-")
    repo2e = tempfile.mkdtemp(prefix="agentdescent-empty-")
    seeded = evolve(tasks("b"), reward, run=run, propose=propose,
                    strategy=SingleSlot(initial_value="cap:0"),
                    initial_state=state, rounds=args.rounds, n_workers=2,
                    repo_path=repo2s, target_reward=1.0)
    empty = evolve(tasks("b"), reward, run=run, propose=propose,
                   strategy=SingleSlot(initial_value="cap:0"),
                   rounds=args.rounds, n_workers=2, repo_path=repo2e,
                   target_reward=1.0)
    st = seeded.time_to_quality(1.0)
    et = empty.time_to_quality(1.0)
    print(f"seeded run: time-to-quality={st if st is not None else 'never':} "
          f"reward={seeded.final_reward:.3f}")
    print(f"empty run : time-to-quality={et if et is not None else 'never':} "
          f"reward={empty.final_reward:.3f}")
    if st is not None and et is not None:
        print(f"cold-start saving: {(1 - st / et) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
