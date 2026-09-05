"""Evolve a tree search's selection rule, and validate it where it was not evolved.

    "I plugged a tree search into evolve() to solve a problem. Now I want to
    evolve the search algorithm itself."

The algorithm plugs in through :class:`~agentdescent.policies.Policies` seams and
the ``aggregator_factory`` exit -- ERA's flat-PUCT tree is
``EraTree`` + ``FlatPuct`` behind a factory -- and this example turns one of
those seams into the artifact of an **outer** ``evolve()``:

* **artifact**: the source of ``priority(rank, visits, total, prior, depth,
  n_nodes)`` -- the rule that decides which node the tree expands next
  (:mod:`examples.metasearch._policy_source`). The seed is upstream ERA's rule.
* **task**: one search problem -- a seeded landscape instance here, an AlgoTune
  task or a Harbor task (SWE-bench-Science, Terminal-Bench-Science) on the live
  path described in the README.
* **run**: compile the candidate rule, plug it into the *real* ``EraTree`` as
  its ``SelectionPolicy``, run a whole inner search at a fixed expansion
  budget, return the trace.
* **reward**: the inner search's **AUC** -- mean best-so-far over the budget.
  A selection rule cannot make a better program exist; it can only find one
  sooner, and the final best at a fixed budget barely separates rules.
* **propose**: a model reads the rule and the trace (depths expanded, dead
  ends, the curve) and rewrites the rule. Held-out is other instances of the
  same family, and the gate, conflict resolution and merge are the engine's.

Governance is L1 (``blast_radius=0.6``): the artifact is a harness, so every
merge also passes the oracle -- an evolved rule is a change to *how everything
downstream is searched*, not to one answer.

**Validation is a different landscape.** The outer loop never sees ``TARGET``
-- higher-dimensional, ruggeder, deadlier. After the run, the seed rule and the
evolved rule are both scored on fresh instances of ``SOURCE`` (in-distribution)
and ``TARGET`` (transfer), so the report separates "a better search rule" from
"a fit to the landscape it was evolved on". The same two-column read is what a
live run reports on AlgoTune tasks it evolved on versus the science benchmarks
it did not.

Run::

    python -m examples.metasearch.evolve_search_policy --dry-run
    python -m examples.metasearch.evolve_search_policy --provider openai \\
        --model deepseek-v4-flash --rounds 6 --workers 4 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from agentdescent.agents import Completion, Usage
from agentdescent.evolution import EvolutionResult, Task
from agentdescent.meta import (MetaOutcome, Problem, auc, meta_evolve, meta_validate,
                               priority_selection, slot_reflector)
from agentdescent.meta import transfer_ratio as _transfer_ratio

from examples._common import (add_standard_args, budget_kwargs, completion_for,
                              confirm, worker_count)
from examples.metasearch._landscape import FAMILIES, SOURCE, TARGET, Family, search
from examples.metasearch._policy_source import (FUNCTION, SEED_SOURCE,
                                                EvolvedSelection)


ARTIFACT_ID = "search-policy"
DEFAULT_INNER_BUDGET = 24


def build_tasks(family: Family, count: int, *, first_seed: int = 0) -> List[Task]:
    """One task per landscape instance -- the shape `meta_evolve` builds itself
    from a problem and its seeds; kept here for the report and the tests."""
    return [
        Task(id=f"{family.name}:{seed}",
             prompt=f"Search landscape {family.name!r}, instance {seed}: "
                    f"dim={family.dim} step={family.step} p_dead={family.p_dead} "
                    f"ruggedness={family.ruggedness}",
             meta={"family": family.name, "seed": seed})
        for seed in range(first_seed, first_seed + count)
    ]


def landscape_problem(family: Family, budget: int = DEFAULT_INNER_BUDGET) -> Problem:
    """A landscape family as an inner :class:`~agentdescent.meta.Problem`.

    ``(selection policy, seed) -> MetaOutcome``: one whole tree search through
    the real ``EraTree`` with the candidate rule choosing parents. The curve is
    best-so-far after each expansion, so :func:`agentdescent.meta.auc` is the
    mean of it -- how fast the rule found what it found."""

    def problem(policy, seed: int) -> MetaOutcome:
        trace = search(policy, family, seed, budget)
        return MetaOutcome(curve=trace.curve, final=trace.best_score,
                           rollouts=trace.budget,
                           detail={"family": trace.family, "dead_ends": trace.dead_ends,
                                   "expanded_depths": trace.expanded_depths,
                                   "root_score": trace.root_score, "nodes": trace.nodes})

    return problem


def make_run(budget: int = DEFAULT_INNER_BUDGET) -> Callable[[str, Task], str]:
    """The rollout as `meta_evolve` performs it, exposed for the tests."""
    problems = {name: landscape_problem(family, budget) for name, family in FAMILIES.items()}

    def run(rendered: str, task: Task) -> str:
        try:
            policy = EvolvedSelection(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": str(error)}).to_json()
        return problems[task.meta["family"]](policy, int(task.meta["seed"])).to_json()

    return run


def reward(task: Task, output: str) -> float:
    """`auc` of the outcome a rollout wrote; 0 for anything unreadable."""
    try:
        return auc(MetaOutcome.from_json(output))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def era_auc(history: Iterable) -> float:
    """The same reward read off a live ``EraRun``: mean best-so-far held-out
    reward over ``result.history``. Zero when the run recorded no round."""
    return auc(MetaOutcome(curve=[float(getattr(row, "held_out_reward", 0.0))
                                  for row in history]))


def validate(seed_source: str, evolved_source: str, *, seeds: Sequence[int],
             budget: int = DEFAULT_INNER_BUDGET,
             families: Sequence[Family] = (SOURCE, TARGET)) -> Dict[str, Dict[str, float]]:
    """Seed rule against evolved rule on fresh instances of every family.

    ``seeds`` must not overlap the outer run's task seeds: instances the outer
    loop trained or gated on are not a validation of anything."""
    report = meta_validate(priority_selection(), seed_source, evolved_source,
                           {f.name: landscape_problem(f, budget) for f in families},
                           seeds=list(seeds))
    # The example's own column names, for its README and tests.
    return {name: {"n": row["n"], "seed_rule": row["before"], "evolved_rule": row["after"],
                   "gain": row["gain"], "gain_sd": row["gain_sd"],
                   "wins": row["wins"], "losses": row["losses"]}
            for name, row in report.items()}


def transfer_ratio(report: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Target gain over source gain. ``None`` when the source gain is nil."""
    return _transfer_ratio(report, SOURCE.name, TARGET.name)


def format_report(report: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'family':<8} {'n':>3} {'seed':>7} {'evolved':>8} {'gain':>7} {'sd':>6} {'w/l':>5}"]
    for name, row in report.items():
        lines.append(f"{name:<8} {row['n']:>3} {row['seed_rule']:>7.3f} "
                     f"{row['evolved_rule']:>8.3f} {row['gain']:>+7.3f} "
                     f"{row['gain_sd']:>6.3f} {row['wins']:>2}/{row['losses']}")
    ratio = transfer_ratio(report)
    lines.append("transfer ratio (target gain / source gain): "
                 + ("n/a (no source gain)" if ratio is None else f"{ratio:.2f}"))
    return "\n".join(lines)


def run_outer(complete: Completion, *, rounds: int, workers: int, tasks: int,
              seed: int, inner_budget: int, mode: str, max_seconds: float,
              async_ratio: int, usage: Optional[Usage] = None,
              extra: Optional[dict] = None) -> EvolutionResult:
    """The outer loop: `meta_evolve` over the `selection` slot, on `SOURCE`."""
    spec = priority_selection()
    common = dict(
        slot="selection",
        spec=spec,
        propose=slot_reflector(complete, spec),
        seeds=list(range(seed * 1_000, seed * 1_000 + tasks)),
        artifact_id=ARTIFACT_ID,
        n_workers=workers,
        max_concurrency=1 if mode == "serial" else workers,
        held_out_frac=0.4,
        eval_concurrency=max(1, workers),
        seed=seed,
        usage=usage,
        max_seconds=max_seconds,
        **(extra or {}),
    )
    problems = {SOURCE.name: landscape_problem(SOURCE, inner_budget)}
    if mode == "async":
        common.update(asynchronous=True, async_ratio=async_ratio,
                      max_rollouts=common.get("max_rollouts", rounds * workers))
        return meta_evolve(problems, rounds=rounds, **common)
    return meta_evolve(problems, rounds=rounds, **common)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(parser, model_default="deepseek-v4-flash",
                      max_seconds_default=1800.0, include_val_cap=False)
    # An OpenAI-compatible endpoint, and a lag budget of one: a rollout here is
    # a whole inner search, so a three-version lag is many searches stale.
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks", type=int, default=20,
                        help="source landscape instances the outer loop trains and gates on")
    parser.add_argument("--inner-budget", type=int, default=DEFAULT_INNER_BUDGET,
                        help="expansions per inner search")
    parser.add_argument("--validate-seeds", type=int, default=40,
                        help="fresh instances per family for the final report")
    parser.add_argument("--out", default="", help="write the full result here as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workers = worker_count(args, args.workers)
    mode = "serial" if args.serial else "async" if args.asynchronous else "sync"
    plan = (f"metasearch: evolve {FUNCTION}() on {args.tasks} x {SOURCE.name} instances "
            f"(inner budget {args.inner_budget}), rounds={args.rounds} workers={workers} "
            f"mode={mode} blast_radius=0.6 (L1); validate on {args.validate_seeds} fresh "
            f"instances of {SOURCE.name} and {TARGET.name}")
    print(plan)
    if args.dry_run:
        print("[dry-run] no model API was accessed; nothing was fetched -- the inner "
              "domain is synthetic. Seed rule:\n" + SEED_SOURCE)
        return 0
    if not confirm(args):
        return 0
    usage = Usage()
    complete = completion_for(args, usage=usage)
    started = time.monotonic()
    result = run_outer(complete, rounds=args.rounds, workers=workers, tasks=args.tasks,
                       seed=args.seed, inner_budget=args.inner_budget, mode=mode,
                       max_seconds=args.max_seconds, async_ratio=args.async_ratio,
                       usage=usage, extra=budget_kwargs(args))
    if result.error:
        print(f"[error] {result.error}")
    print("[evolved rule]\n" + result.rendered)
    # Fresh seeds, far from the outer run's own instances.
    fresh = range(10_000_000 + args.seed * 1_000, 10_000_000 + args.seed * 1_000 + args.validate_seeds)
    report = validate(SEED_SOURCE, result.rendered, seeds=list(fresh), budget=args.inner_budget)
    print(format_report(report))
    payload = {
        "plan": plan, "seed": args.seed, "mode": mode, "wall_seconds": time.monotonic() - started,
        "evolved_source": result.rendered, "final_reward": result.final_reward,
        "outcomes": result.outcomes(), "rollouts": result.rollouts,
        "validation": report, "transfer_ratio": transfer_ratio(report),
        "usage": {"calls": usage.calls, "input_tokens": usage.input_tokens,
                  "output_tokens": usage.output_tokens},
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print(f"[result saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
