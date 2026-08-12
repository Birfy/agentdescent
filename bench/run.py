"""Run the configuration matrix and print the table.

    python -m bench.run --config sync-1 --config sync-4 --config async-4 \
                        --budget-calls 400 --seeds 0,1,2

Defaults to the offline router domain: deterministic, no network, seconds. That
is what makes the harness's own self-check possible -- same seed twice must give
the same numbers, and if it does not, nothing the harness measures about anything
else can be believed either.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

from agentdescent import AppendRules, EvolutionResult, Task, evolve, get_policy

from .harness import Config, Measurement, run_config, summarise, to_markdown

#: The configurations the matrix knows how to build. Names are what `--config`
#: takes, so adding one here is the only edit a new comparison needs.
CONFIGS: Dict[str, Config] = {
    "sync-1": Config("sync-1", n_workers=1, eval_concurrency=1),
    "sync-4": Config("sync-4", n_workers=4, eval_concurrency=4),
    "sync-8": Config("sync-8", n_workers=8, eval_concurrency=8),
    "async-4": Config("async-4", n_workers=4, eval_concurrency=4, asynchronous=True),
    "async-8": Config("async-8", n_workers=8, eval_concurrency=8, asynchronous=True),
    # The lag budget is a dimension, not a constant. Reporting one arbitrary
    # value for the barrier-free path while the synchronous one runs at its
    # natural setting is not a comparison between paths, it is a comparison
    # between one path and a misconfiguration of the other.
    "async-4-lag1": Config("async-4-lag1", n_workers=4, eval_concurrency=4,
                           asynchronous=True, async_ratio=1),
    "async-4-lag0": Config("async-4-lag0", n_workers=4, eval_concurrency=4,
                           asynchronous=True, async_ratio=0),
    "async-8-lag1": Config("async-8-lag1", n_workers=8, eval_concurrency=8,
                           asynchronous=True, async_ratio=1),
}

#: The quality bar time- and cost-to-quality are measured against. A bar every
#: configuration clears immediately measures nothing; one nothing clears reports
#: `—` for every row and is equally uninformative.
DEFAULT_TARGET = 0.75


# ---------------------------------------------------------------------------
# The workload: fixed data, fixed semantics, fixed model
# ---------------------------------------------------------------------------


def _dataset(seed: int, n_keywords: int = 16, per_keyword: int = 6):
    """train / val / test over the repository's deterministic router domain.

    Split once per seed and never re-split per configuration: splitting inside
    one would let two configurations see different data and let the difference be
    called parallelism.

    The domain matters as much as the split. An earlier version of this file
    invented one where a rule named a *task*, so nothing learned on train could
    apply to test and every quality column was structurally zero -- a table full
    of honest numbers measuring nothing. Here a rule names a **keyword** shared by
    many tasks, so learning generalises and the test column can move.
    """
    from agentdescent.domains.router import make_task_universe

    universe = make_task_universe(n_keywords=n_keywords,
                                  tasks_per_keyword=per_keyword, seed=seed)
    tasks = [Task(id=f"{t.keyword}-{i}", prompt=t.text,
                  meta={"gold": t.label, "keyword": t.keyword})
             for i, t in enumerate(universe.tasks)]
    rng = random.Random(seed)
    rng.shuffle(tasks)
    cut = int(len(tasks) * 0.7)
    return tasks[:cut], tasks[cut:]


def _run(rendered: str, task: Task) -> str:
    """Deterministic 'agent': answer from the rule table, if it has the keyword."""
    for line in rendered.splitlines():
        if "->" not in line:
            continue
        keyword, _, label = line.partition("->")
        if keyword.strip().lstrip("- ") == task.meta["keyword"]:
            return label.strip()
    return "?"


def _reward(task: Task, output: str) -> float:
    return 1.0 if output == task.meta["gold"] else 0.0


def _propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
    """The rule this failure implies -- about the keyword, not the task."""
    return f"{task.meta['keyword']} -> {task.meta['gold']}"


def workload(config: Config, seed: int) -> EvolutionResult:
    """One run. Only `config` varies; data, actors and semantics do not."""
    train, _test = _dataset(seed)
    return evolve(
        train, _reward, run=_run, propose=_propose, strategy=AppendRules(),
        rounds=30, n_workers=config.n_workers,
        max_concurrency=config.n_workers,
        eval_concurrency=config.eval_concurrency,
        asynchronous=config.asynchronous, async_ratio=config.async_ratio,
        resync_on_commit=config.resync_on_commit,
        staleness_policy=get_policy(config.staleness),
        held_out_frac=0.4, seed=seed, max_seconds=60.0)


def test_quality(seed: int) -> Callable[[EvolutionResult], float]:
    """Score the finished artifact on the split the gate never saw.

    `evolve()` stops when *its* held-out split says so, so reporting that number
    is reporting a training score."""
    def evaluate(result: EvolutionResult) -> float:
        _train, held = _dataset(seed)
        return sum(_reward(t, _run(result.rendered, t)) for t in held) / len(held)
    return evaluate


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def truncate_to_budget(result: EvolutionResult, budget_calls: int) -> EvolutionResult:
    """The run as it stood when the budget ran out.

    Configurations differ in how many calls a round makes, so a budget fixed in
    rounds hands the wider one more model and then reports the extra model as a
    win for parallelism. Truncating on the recorded per-round call count asks all
    of them the same question: where were you after N calls?

    Not a cap applied during the run: stopping mid-round would leave a partial
    round in the history, and the comparison is between completed states.
    """
    if budget_calls <= 0:
        return result
    kept = [h for h in result.history if h.calls <= budget_calls]
    if len(kept) == len(result.history):
        return result
    import dataclasses

    last = kept[-1] if kept else None
    return dataclasses.replace(
        result, history=kept,
        final_reward=last.held_out_reward if last else 0.0,
        rollouts=last.rollouts if last else 0,
        wallclock=last.elapsed_s if last else 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs",
                        choices=sorted(CONFIGS), help="repeatable")
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma-separated; three is the minimum that shows a spread")
    parser.add_argument("--budget-calls", type=int, default=0,
                        help="0 runs each configuration to its own stopping point")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--json", dest="json_out", help="write raw measurements here")
    args = parser.parse_args(argv)

    names = args.configs or ["sync-1", "sync-4", "async-4"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s). This repository has published a "
              "point estimate that moved 4.8 points between two runs of one "
              "configuration; one number per configuration is not a result.",
              file=sys.stderr)

    measurements: List[Measurement] = []
    for name in names:
        config = CONFIGS[name]
        for seed in seeds:
            result = workload(config, seed)
            if args.budget_calls:
                result = truncate_to_budget(result, args.budget_calls)
            measurements.append(Measurement.of(
                name, seed, result, args.target, test_quality(seed)(result)))

    summaries = summarise(measurements)
    print(to_markdown(summaries))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"measurements": [m.__dict__ for m in measurements],
                       "summaries": [s.as_row() for s in summaries]}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
