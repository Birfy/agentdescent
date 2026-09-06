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
import statistics
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from agentdescent.agents import Completion, Usage, with_retries
from agentdescent.evolution import EvolutionResult, Task
from agentdescent.meta import (MetaOutcome, Problem, auc, cached_completion,
                               meta_evolve, meta_validate, priority_selection,
                               slot_reflector)
from agentdescent.meta import transfer_ratio as _transfer_ratio

from examples._common import (add_standard_args, budget_kwargs, completion_for,
                              confirm, worker_count)
from examples._measure import usage_dict
from examples.metasearch._landscape import FAMILIES, SOURCE, TARGET, Family, search
from examples.metasearch._policy_source import (FUNCTION, SEED_SOURCE,
                                                EvolvedSelection)


ARTIFACT_ID = "search-policy"

#: Expansions per inner search. 60, not 24, and the difference is measured.
#: Four priority rules over 60 landscape instances, spread between best and
#: worst rule:
#:
#:     budget      12     24     60    120
#:     source   0.130  0.255  0.413  0.474
#:     target   0.023  0.065  0.203  0.349
#:
#: At 24 the transfer family separates rules by 0.065, which is thin enough that
#: a real difference can hide in it; at 60 it is 0.203. (Unlike the task_sampler
#: experiment in `bench/metasearch_slots.py`, no budget here *inverts* the
#: ranking -- a deliberately bad rule, `return -rank`, comes last at every one.)
DEFAULT_INNER_BUDGET = 60

#: What a hand-written rule reaches on this landscape, as the ceiling the search
#: is measured against. Scanned over `c` at budget 60, 80 instances per family:
#: the seed's `c_puct = 1` is too explorative, `c = 0.25` is best on **both**
#: families at +0.021 each, and `c = 2` is worse than the seed by 0.05-0.07. So
#: there is a known, transferable direction here -- explore less -- and a search
#: that finds it should transfer, which is what makes this a fair test.
REFERENCE_RULES: Dict[str, str] = {
    "PUCT c=0.25 (best hand-written)":
        "def priority(rank, visits, total, prior, depth, n_nodes):\n"
        "    return rank + 0.25 * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)\n",
    "greedy (rank only)":
        "def priority(rank, visits, total, prior, depth, n_nodes):\n    return rank\n",
    "worst-first (deliberately bad)":
        "def priority(rank, visits, total, prior, depth, n_nodes):\n    return -rank\n",
}


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


def score_rules(rules: Dict[str, str], *, seeds: Sequence[int], budget: int,
                families: Sequence[Family] = (SOURCE, TARGET)) -> Dict[str, Dict[str, float]]:
    """Mean AUC per family for each named rule -- the reference column.

    Free: an inner search is pure Python, so this costs wall-clock and no model
    calls at all."""
    out: Dict[str, Dict[str, float]] = {}
    for label, source in rules.items():
        policy = EvolvedSelection(source)
        out[label] = {f.name: statistics.fmean(search(policy, f, s, budget).auc for s in seeds)
                      for f in families}
    return out


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


def progress() -> Callable[[object], None]:
    """One line per outer sweep. An outer sweep here is a model call plus
    `tasks` whole inner searches, so a run that reports nothing until its
    summary cannot be told from one that stalled on the endpoint."""

    def on_round(info: object) -> None:
        print(f"[sweep {info.round}] held_out={info.held_out_reward:.3f} "
              f"committed={info.committed} rejected={info.rejected} "
              f"reasons={info.reasons} elapsed={info.elapsed_s:.0f}s "
              f"rollouts={info.rollouts}", flush=True)

    return on_round


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
        on_round=progress(),
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
    parser.add_argument("--tasks", type=int, default=150,
                        help=("source landscape instances the outer loop trains and "
                              "gates on. Large for the same reason --validate-seeds "
                              "is: the model is called once per *rollout*, and "
                              "rollouts are rounds x workers whatever this is, so "
                              "every extra task is free. It buys the gate's held-out "
                              "set (tasks x held_out_frac), and that set is what "
                              "decides every merge -- at 24 tasks it is 9 instances, "
                              "where the paired per-instance sd of 0.04-0.06 puts the "
                              "standard error at 0.013-0.018, wider than the whole "
                              "gain available on this landscape. 150 tasks is 60 "
                              "instances, 0.005-0.008, and costs 0.3s per gate"))
    parser.add_argument("--inner-budget", type=int, default=DEFAULT_INNER_BUDGET,
                        help="expansions per inner search")
    parser.add_argument("--validate-seeds", type=int, default=200,
                        help=("fresh instances per family for the final report. "
                              "Large because an inner search here is pure Python "
                              "and costs nothing -- the model is only the outer "
                              "reflector, so statistics are free and there is no "
                              "reason to report a handful of paired runs"))
    parser.add_argument("--no-reference", action="store_true",
                        help=("skip scoring the hand-written rules. They are what "
                              "says whether the search found the available gain or "
                              "a fraction of it"))
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="default",
                        help=("send `thinking` in the request body. The reflector "
                              "call here is one short function rewrite, and a "
                              "reasoning preamble costs minutes of wall-clock for "
                              "it; `disabled` is accepted by Ark/Volcengine. Part "
                              "of the reported configuration, not just a speed "
                              "knob -- it changes the reply, so it has to be the "
                              "same across arms of a comparison"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--api-timeout", type=float, default=180.0,
                        help="seconds per model call before it is retried")
    parser.add_argument("--out", default="", help="write the full result here as JSON")
    parser.add_argument("--completion-cache", default="",
                        help=("memoise the reflector's prompt -> text here. The "
                              "inner search is already deterministic; this makes "
                              "the *outer* loop resumable, so a run cut short by a "
                              "wall-clock limit continues from where it stopped "
                              "instead of paying for the same proposals again"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workers = worker_count(args, args.workers)
    mode = "serial" if args.serial else "async" if args.asynchronous else "sync"
    plan = (f"metasearch: evolve {FUNCTION}() on {args.tasks} x {SOURCE.name} instances "
            f"(inner budget {args.inner_budget}), rounds={args.rounds} workers={workers} "
            f"mode={mode} blast_radius=0.6 (L1); validate on {args.validate_seeds} fresh "
            f"instances of {SOURCE.name} and {TARGET.name}; "
            f"{args.provider}/{args.model} temperature={args.temperature} "
            f"thinking={args.thinking}")
    print(plan)
    if args.dry_run:
        print("[dry-run] no model API was accessed; nothing was fetched -- the inner "
              "domain is synthetic. Seed rule:\n" + SEED_SOURCE)
        return 0
    if not confirm(args):
        return 0
    # Two meters, not one. The engine wraps `propose` in its own metric and
    # records a zero-token call for every invocation, so a single shared Usage
    # reports `calls` as (API calls + metered wrappers + cache hits) with the
    # tokens of only the first -- 514 "calls" carrying 19k prompt tokens on one
    # run, which reads as a 37-token reflector prompt and is not one. `api` is
    # what the endpoint actually served; `usage` is what the engine spent.
    api = Usage()
    usage = Usage()
    options: Dict[str, object] = {}
    if args.thinking != "default":
        options["thinking"] = {"type": args.thinking}
    complete = with_retries(
        completion_for(args, usage=api, max_tokens=args.max_tokens,
                       timeout=args.api_timeout, temperature=args.temperature,
                       retries=1, **options),
        attempts=4, backoff=3.0)
    if args.completion_cache:
        complete = cached_completion(
            complete, args.completion_cache,
            key_extra=f"{args.model}|{args.temperature}|{args.max_tokens}|{args.thinking}")
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
    reference = {}
    if not args.no_reference:
        reference = score_rules({"evolved": result.rendered, "seed (flat-PUCT c=1)": SEED_SOURCE,
                                 **REFERENCE_RULES},
                                seeds=list(fresh), budget=args.inner_budget)
        base = reference["seed (flat-PUCT c=1)"]
        print(f"\n{'rule':<34} " + "  ".join(f"{f:>8}" for f in base) + "     vs seed")
        for label, per in reference.items():
            deltas = "  ".join(f"{per[f] - base[f]:+8.4f}" for f in base)
            print(f"{label:<34} " + "  ".join(f"{per[f]:>8.4f}" for f in base) + f"   {deltas}")
    payload = {
        "plan": plan, "seed": args.seed, "mode": mode, "wall_seconds": time.monotonic() - started,
        "evolved_source": result.rendered, "final_reward": result.final_reward,
        "outcomes": result.outcomes(), "rollouts": result.rollouts,
        "validation": report, "transfer_ratio": transfer_ratio(report),
        "reference": reference,
        "model": {"provider": args.provider, "model": args.model,
                  "temperature": args.temperature, "max_tokens": args.max_tokens,
                  "thinking": args.thinking, "inner_budget": args.inner_budget},
        "usage": usage_dict(api), "engine_usage": usage_dict(usage),
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print(f"[result saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
