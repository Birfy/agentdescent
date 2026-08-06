"""Template for a faithful self-evolution algorithm port.

Each port owns a directory. Create ``examples/<algorithm>/`` with an
``__init__.py`` and a ``README.md``, copy this file in as
``examples/<algorithm>/<algorithm>_<what_it_evolves>.py``, and replace every
``PORT_*`` / ``DATASET_*`` value. Anything the port needs and nothing else does
-- a sandbox runner, an evaluator, fixtures -- belongs in that directory too.

Keep the upstream algorithm's vocabulary for its iteration flag: ``--rounds``,
``--generations``, ``--iterations``, or ``--steps``. Record every deviation from
released code in ``docs/algo-<algorithm>.md``, and add the port to ``PORTS`` in
``tests/test_example_entrypoints.py`` -- a port outside that table is outside
the whole command-line contract.

The control-flow order is part of the template: ``--dry-run`` returns before
the dataset loader and model adapter, so it is safe on a cold, offline machine.
"""

from __future__ import annotations

import argparse

from agentdescent.agents import Usage
from agentdescent.dataloader import hf_rows
from agentdescent.evolution import AppendRules, LLMAgent, Task, evolve
from agentdescent.rewards import exact_match
from examples._common import (add_standard_args, completion_for, confirm,
                              worker_count)


PORT_NAME = "replace-with-algorithm-name"
PORT_AUTHOR = "TODO: GitHub handle"
UPSTREAM_RELEASED_CODE = "TODO: released repository URL and revision"
DATASET_NAME = "replace-me/dataset"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"

# Pick the representation that matches the released implementation. Alternatives
# include KeyedRules, SingleSlot, and FileTree; do not invent a custom Strategy
# when one of those already expresses the upstream state.
STRATEGY = AppendRules()
REWARD = exact_match()


def load_tasks(limit: int):
    """Load through agentdescent.dataloader; never implement HTTP in a port."""
    if DATASET_NAME.startswith("replace-me/"):
        raise RuntimeError("replace the DATASET_* placeholders before a real run")
    rows = hf_rows(DATASET_NAME, DATASET_SPLIT, config=DATASET_CONFIG, limit=limit)
    return [
        Task(id=str(i), prompt=str(row["prompt"]), meta={"gold": row["gold"]})
        for i, row in enumerate(rows)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, max_seconds_default=30.0)
    # Keep the released implementation's term here; do not standardise it.
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=40)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics: one
    # worker, nothing to merge. Applied to args so the printed plan, the cost
    # estimate and the run cannot disagree about what ran. Without it the port
    # has no baseline, and any speedup it reports is a speedup over nothing.
    args.workers = worker_count(args, args.workers)
    print(f"Algorithm: {PORT_NAME} (port author: {PORT_AUTHOR})")
    print(f"Upstream : {UPSTREAM_RELEASED_CODE}")
    print(f"Dataset  : {DATASET_NAME}")
    print(f"Plan     : model={args.model}, iterations={args.iterations}, "
          f"workers={args.workers}")

    if args.dry_run:
        print("Data     : deferred (dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    tasks = load_tasks(args.limit)
    if len(tasks) < 4:
        raise ValueError("a port needs at least four tasks for train/held-out splitting")
    if not confirm(args):
        return

    usage = Usage()
    completion = completion_for(args, usage=usage)
    result = evolve(
        tasks,
        REWARD,
        agent=LLMAgent(completion),
        strategy=STRATEGY,
        rounds=args.iterations,
        n_workers=args.workers,
        max_concurrency=1 if args.asynchronous else args.workers,
        asynchronous=args.asynchronous,
        async_ratio=args.async_ratio,
        max_seconds=args.max_seconds if args.asynchronous else None,
        seed=args.seed,
        usage=usage,
    )
    print(result.rendered)
    print(f"reward      : {result.final_reward:.3f}")
    print(f"outcomes    : {result.outcomes()}")
    print(f"stop reason : {result.stop_reason}")
    print(f"error       : {result.error or 'none'}")
    print(f"model usage : {usage.summary()}")


if __name__ == "__main__":
    main()
