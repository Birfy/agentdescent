"""ERA tree search on AlgoTune -- one tree per task, scored in speedup.

    "How good are language models at coming up with new algorithms? To try to
    answer this, we built a benchmark, AlgoTune, comprised of 154 widely used
    math, physics, and computer science functions. For each function, the goal
    is to write code that produces the same outputs as the original function,
    while being faster."
    -- AlgoTune, arXiv:2507.15887

What the search optimises
-------------------------
A candidate is a single function, ``solve(problem)``, and it is scored by
**how much faster than the task's reference implementation it is** on problems
the task's own ``is_solution`` accepts. The reference is not a strawman: it is
``scipy.linalg.eig``, ``scipy.integrate.solve_ivp`` on an LSODA-class stiff
solver, ``scipy.signal.upfirdn``, ``scipy.spatial.Delaunay`` -- the call a
working scientist already makes. The root node of every tree *is* that
reference, lifted out of its class into a runnable program, so the tree starts
at exactly 1.0x and every gain is measured against the library.

Correctness is not part of the score, it is a precondition of having one: a
solution the checker rejects scores nothing at all, however fast it was. That is
AlgoTune's rule and it is what stops the search from discovering that the fastest
way to compute an SVD is to not compute it.

One tree per task
-----------------
Each task gets its own flat-PUCT tree, its own root, its own held-back shards
and its own result. They are separate searches over separate program spaces --
a factorisation trick found for ``qr_factorization`` is not a node in
``ode_stiff_vanderpol``'s tree and could not be selected there -- so running
them as one tree would be an averaging artefact rather than a search. The
run reports each task's own speedup and, across tasks, the **geometric** mean,
which is what a ratio of times has to be averaged with.

Everything about the search itself is `era_empirical_software.py`: the flat-PUCT
tree, the visit reservation, the staleness handling, the aggregator, the
governance layer. This module supplies a
:class:`~examples.era._era_domain.Domain` per task -- seed program, sandboxed
evaluator, mutation prompt, metric name -- and the command line the other ports
share.

Run
---
    python -m examples.era.era_algotune --dry-run
    python -m examples.era.era_algotune --tasks svd,matrix_exponential \\
        --provider claude --model glm-5.2 --iterations 6 --workers 3 --yes
    python -m examples.era.era_algotune --tasks all --iterations 12 --yes
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Usage, with_retries
from agentdescent.evolution import EvolvingArtifact
from agentdescent.governance import classify

from examples._common import (
    add_standard_args,
    completion_for,
    confirm,
    worker_count,
)
from examples.era._algotune_tasks import UPSTREAM_COMMIT
from examples.era._era_algotune import (
    DEFAULT_TASKS,
    PROBLEM_SECONDS,
    REPEATS,
    TASKS,
    Suite,
    evaluate_source,
    framework_score,
    mutation_prompt,
    prepare_suite,
    repair_prompt,
)
from examples.era._era_domain import Domain
from examples.era._era_support import sandbox_backend, with_intact_replies
from examples.era.era_empirical_software import (
    _require_api_environment,
    _usage_dict,
    _utc_now,
    _write_json,
    run_agentdescent_era,
)


ARTIFACT_ID = "era_program"
DEFAULT_OUTPUT = Path("era-algotune-result.json")


def _make_completion(args: argparse.Namespace, usage: Usage):
    """The sibling ports' completion wiring, calling this module's own import.

    Six lines rather than an import of a neighbour's private helper, for the
    reason `era_hard_integrals` gives: `tests/test_example_entrypoints.py`
    proves a dry-run never crosses an external boundary by replacing **this
    module's** `completion_for` with a tripwire, and a port whose network call
    went through another module's name would pass with the tripwire out of the
    path.
    """
    options: Dict[str, Any] = {}
    if args.thinking != "default":
        options["thinking"] = {"type": args.thinking}
    completion = completion_for(
        args,
        usage=usage,
        max_tokens=args.max_tokens,
        timeout=args.api_timeout,
        temperature=args.temperature,
        # The adapter's own retry is disabled so it does not multiply with the
        # one below: three attempts at 0.5s and 1.0s is right for a dropped
        # connection and useless against a per-minute rate limit.
        retries=1,
        **options,
    )
    # `--repair-attempts N` multiplies model calls by up to N, and this endpoint
    # answers that with 429s: four in twenty-two minutes at N=4, each one killing
    # a worker's whole round (`+0/-0` -- an expansion bought and not spent).
    # Backing off in seconds rather than fractions of one is what a rate limit
    # asks for, and it costs nothing when there is no limit to hit.
    return with_retries(completion, attempts=5, backoff=4.0)


def algotune_domain(
    suite: Suite,
    *,
    candidate_timeout: float = 120.0,
    max_code_length: int = 20_000,
    repeats: int = REPEATS,
    problem_seconds: float = PROBLEM_SECONDS,
    profile: bool = True,
) -> Domain:
    """One AlgoTune task, in the four terms the ERA search needs."""
    return Domain(
        name=f"AlgoTune/{suite.task}, mean speedup over the reference implementation",
        entrypoint="solve",
        metric_key="speedup",
        metric_better="higher",
        initial_program=suite.initial_program,
        initial_summary=f"AlgoTune reference implementation for {suite.task}",
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            repeats=repeats, problem_seconds=problem_seconds,
            max_length=max_code_length, want_profile=profile),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, suite=suite, timeout=candidate_timeout, repeats=repeats,
            ),
        task_prompt=lambda index: (
            f"Time the {suite.task} program against the reference on held-out "
            f"problem set {index}."),
        test_shards=suite.test_range(),
        data_summary={
            "task": suite.task,
            "n": suite.n,
            "published_n": suite.published_n,
            "published_target_time_ms": suite.target_time_ms,
            "published_baseline_ms": suite.published_ms,
            "problems_per_shard": suite.problems,
            "problems_per_test_shard": suite.size(suite.scoring_shards),
            "reported_instances": suite.test_shards * suite.size(suite.scoring_shards),
            "scoring_shards": suite.scoring_shards,
            "test_shards": suite.test_shards,
            "timed_repeats": repeats,
            "line_profile": profile,
            "seed": suite.seed,
        },
    )


def resolve_tasks(selection: str) -> Tuple[str, ...]:
    """Turn ``--tasks`` into task names, refusing an unknown one by name."""
    text = (selection or "").strip()
    if not text or text == "default":
        return DEFAULT_TASKS
    if text == "all":
        return TASKS
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise SystemExit(
            f"unknown AlgoTune task(s): {', '.join(unknown)}. This port runs "
            f"{len(TASKS)} of the 154 -- the ones whose reference needs only "
            f"numpy, scipy and the standard library. Run --list-tasks to see them.")
    return names


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    """The mean a set of speedups actually has.

    A speedup is a ratio, and the arithmetic mean of ratios is not one: 4x on one
    task and 0.25x on another is *no change on average*, and the arithmetic mean
    reports 2.1x. AlgoTune's own headline number over its task set is a geometric
    mean for this reason.
    """
    usable = [float(value) for value in values
              if value is not None and math.isfinite(float(value)) and float(value) > 0]
    if not usable:
        return None
    return math.exp(sum(math.log(value) for value in usable) / len(usable))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=1800.0,
                      eval_concurrency_default=None)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--tasks", default="default",
                        help=("comma-separated AlgoTune task names, `default` "
                              f"for the {len(DEFAULT_TASKS)} spanning the "
                              f"categories, or `all` for every one of the "
                              f"{len(TASKS)} this port can run"))
    parser.add_argument("--list-tasks", action="store_true",
                        help="print the runnable task names and exit")
    parser.add_argument("--staleness", default="guarded",
                        choices=["guarded", "reflective", "full"],
                        help=("what to do with an expansion proposed against a "
                              "head the merger has since moved. The tree is "
                              "append-only, so `full` is the honest default for "
                              "a comparison and `guarded` the conservative one"))
    parser.add_argument("--iterations", type=int, default=6,
                        help=("FUTS expansions per task (upstream's "
                              "num_iterations). Each task gets its own tree"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shards", type=int, default=6,
                        help=("problem sets the search may score against. Half of "
                              "them become the gate's held-out split, and a round "
                              "cannot run more rollouts than it has *train* sets "
                              "-- so this wants to be at least 2x --workers or "
                              "the expansion budget goes unspent"))
    parser.add_argument("--test-shards", type=int, default=3,
                        help="further problem sets the search never sees")
    parser.add_argument("--problems", type=int, default=2,
                        help="problems per scoring set, each one a fresh seed")
    parser.add_argument("--test-problems", type=int, default=0,
                        help=("problems per held-back set; 0 means the same as "
                              "--problems. The scoring sets are paid for on every "
                              "rollout and every gate evaluation, the held-back "
                              "sets twice per task at the end -- so this is the "
                              "cheap way to make the *reported* figure precise. "
                              "AlgoTune reports over 100 test instances, and "
                              "--test-shards 2 --test-problems 50 matches that"))
    parser.add_argument("--no-profile", action="store_true",
                        help=("skip the line_profiler table in the mutation "
                              "prompt. It is upstream's `profile` command and "
                              "costs one extra traced call per evaluation"))
    parser.add_argument("--repair-attempts", type=int, default=1,
                        help=("draws allowed per expansion when the program "
                              "fails: the failure is handed back with the error "
                              "and the model asked to fix it. 1 is upstream ERA, "
                              "where a failed program becomes a node scoring "
                              "-inf that FlatPuct never selects again -- and on "
                              "this benchmark the direction that wins is the one "
                              "whose first attempt usually fails. Each retry "
                              "costs one model call and one scoring-shard "
                              "evaluation"))
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help=("timed runs per program per problem, after a "
                              "discarded warm-up. The metric is the ratio of the "
                              "two minima"))
    parser.add_argument("--size-scale", type=float, default=1.0,
                        help=("multiply upstream's published problem size. A "
                              "wall-clock knob and a difficulty knob at once -- "
                              "a scaled run is not comparable to an unscaled one"))
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--candidate-timeout", type=float, default=120.0,
                        help=("wall-clock for one problem set, reference and "
                              "candidate together. Twice the other ERA tasks' 60s "
                              "because every problem here is timed twice"))
    parser.add_argument("--problem-seconds", type=float, default=PROBLEM_SECONDS,
                        help="wall-clock per problem, inside --candidate-timeout")
    parser.add_argument("--max-code-length", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="disabled")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument(
        "--reply-attempts", type=int, default=4,
        help=("redraws allowed when a reply arrives damaged -- unparsable, or "
              "holding characters Python source cannot hold. A *badly written* "
              "program is never redrawn; it becomes a node scoring -inf, as "
              "upstream requires. 1 disables the guard (see "
              "examples.era._era_support.reply_is_intact)"))
    parser.add_argument("--shutdown-grace", type=float, default=120.0)
    parser.add_argument("--quality-target", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _task_payload(suite: Suite, domain: Domain, run: Any,
                  quality_target: Optional[float]) -> Dict[str, Any]:
    baseline = run.baseline_test_metrics.get("speedup")
    best = run.best_test_metrics.get("speedup")
    return {
        "task": suite.task,
        "status": "completed" if run.result.error is None else "partial",
        "held_out_baseline_speedup": baseline,
        "held_out_best_speedup": best,
        "speedup_gain": domain.gain(baseline, best),
        "nodes": len(run.tree.nodes),
        "wall_seconds": run.wall_seconds,
        "best_program": run.tree.best().program.code,
        "observation": run.summary(quality_target),
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        for name in TASKS:
            print(name)
        return 0
    args.workers = worker_count(args, args.workers)
    if args.budget_rollouts:
        args.iterations = args.budget_rollouts
    if getattr(args, "reflective_merge", False):
        raise SystemExit(
            "--reflective-merge is not supported by the ERA port: a candidate is "
            "a whole program rather than a delta, and fusing a round's "
            "expansions into one would delete tree nodes that FUTS selects from. "
            "Use --staleness {guarded,reflective,full}.")
    tasks = resolve_tasks(args.tasks)
    if args.shards < 4:
        # `evolve()` refuses to split fewer than four tasks into train and
        # held-out, and here one task is one problem set -- so the failure would
        # otherwise arrive once per AlgoTune task, after each one had been
        # fetched and its seed program derived.
        raise SystemExit(
            f"--shards {args.shards} is too few: a shard is one rollout task, and "
            "the engine needs at least 4 to split train from held-out. Raise "
            "--shards, or lower --problems if the run is too slow.")
    train_shards = args.shards - round(args.shards * args.held_out_frac)
    if train_shards < args.workers:
        # A round dispatches one rollout per *train* task, so a worker beyond
        # that count has nothing to be handed and the round stops short of
        # `workers` expansions. Left silent, `--iterations 9 --workers 3` on four
        # shards quietly becomes six expansions and the result file still says
        # nine were budgeted.
        print(f"warning  : {args.shards} shards x held_out_frac="
              f"{args.held_out_frac} leaves {train_shards} train set(s), so each "
              f"round runs {train_shards} rollouts rather than {args.workers} and "
              f"the tree will stop short of {args.iterations} expansions. Raise "
              f"--shards to at least {2 * args.workers}.")
    mode = "async" if args.asynchronous else ("serial" if args.serial else "sync")
    print("Algorithm: ERA Flat UCB tree search (FUTS) on AgentDescent")
    print(f"Task     : AlgoTune -- {len(tasks)} task(s), one tree each, "
          f"mean speedup over the reference implementation")
    print(f"Evaluator: {sandbox_backend() or 'NO SANDBOX -- this run will fail'} "
          f"isolated, reference and candidate timed in the same process, "
          f"min of {args.repeats} runs")
    print(
        f"Plan     : mode={mode}, model={args.model}, iterations={args.iterations}"
        f"/task, workers={args.workers}, c_puct={args.c_puct}, "
        f"temperature={args.temperature}"
    )
    print(f"Repair   : up to {args.repair_attempts} draw(s) per expansion"
          f"{' (upstream ERA: a failure is a -inf node)' if args.repair_attempts <= 1 else ''}")
    print("Prompt   : AlgoTuner's own system message, naming no technique"
          f"{'' if args.no_profile else ' + line profile (upstream `profile`)'}")
    print(f"Tasks    : {', '.join(tasks)}")
    artifact = EvolvingArtifact(ARTIFACT_ID, blast_radius=0.6)
    print(
        f"Governance: generated program blast_radius={artifact.blast_radius} "
        f"-> {classify(artifact).name}"
    )
    if args.dry_run:
        print("[dry-run] no API, task file, or sandbox process was accessed.")
        return 0

    _require_api_environment(args.provider)
    if not confirm(args):
        return 0

    model_usage = Usage()
    damage: Dict[str, int] = {}
    repairs: Dict[str, int] = {}
    complete = with_intact_replies(
        _make_completion(args, model_usage),
        attempts=max(1, args.reply_attempts), counter=damage)

    started = time.monotonic()
    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, task in enumerate(tasks, start=1):
        # One `Usage` per task: the engine's own actor counters are per search,
        # and a shared one would report every task's rollouts against the last.
        actor_usage = Usage()
        try:
            suite = prepare_suite(
                task, seed=args.seed, shards=args.shards,
                test_shards=args.test_shards, problems=args.problems,
                test_problems=args.test_problems, size_scale=args.size_scale)
        except Exception as exc:  # a task that will not load is not a search
            print(f"[{index}/{len(tasks)}] {task}: skipped -- "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failures.append({"task": task, "stage": "prepare",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"\n[{index}/{len(tasks)}] {task}: n={suite.n} "
              f"(upstream {suite.published_n} at {suite.target_time_ms}ms), "
              f"{args.shards} scoring sets of {suite.problems} + "
              f"{args.test_shards} held back of {suite.size(args.shards)} "
              f"({args.test_shards * suite.size(args.shards)} reported instances), "
              f"seed program {len(suite.initial_program)} chars", flush=True)
        domain = algotune_domain(
            suite,
            candidate_timeout=args.candidate_timeout,
            max_code_length=args.max_code_length,
            repeats=args.repeats,
            problem_seconds=args.problem_seconds,
            profile=not args.no_profile,
        )
        try:
            run = run_agentdescent_era(
                complete,
                mode=mode,
                iterations=args.iterations,
                workers=args.workers,
                shards=args.shards,
                test_shards=args.test_shards,
                held_out_frac=args.held_out_frac,
                c_puct=args.c_puct,
                candidate_timeout=args.candidate_timeout,
                max_code_length=args.max_code_length,
                async_ratio=args.async_ratio,
                staleness=args.staleness,
                max_seconds=args.max_seconds,
                shutdown_grace=args.shutdown_grace,
                seed=args.seed,
                usage=actor_usage,
                eval_concurrency=args.eval_concurrency,
                domain=domain,
                repair_attempts=args.repair_attempts,
                repair_prompt=(
                    (lambda program, code, error, attempt, _s=suite:
                     repair_prompt(program, code, error, attempt, suite=_s))
                    if args.repair_attempts > 1 else None),
                repair_counter=repairs,
                verbose=True,
            )
        except Exception as exc:
            # One task's tree is one search. A root that will not run under the
            # sandbox, or a task whose reference the installed numpy rejects,
            # takes that task out of the run and nothing else -- recorded, so a
            # partial result file says which tasks it is missing and why.
            print(f"[{index}/{len(tasks)}] {task}: failed -- "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failures.append({"task": task, "stage": "search",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        entry = _task_payload(suite, domain, run, args.quality_target)
        entry["actor_usage"] = _usage_dict(actor_usage)
        entries.append(entry)
        best_path = args.output.with_name(f"{args.output.stem}-{task}-best.py")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.write_text(run.tree.best().program.code.rstrip() + "\n",
                             encoding="utf-8")
        print(f"[{index}/{len(tasks)}] {task}: held-back speedup "
              f"{entry['held_out_baseline_speedup']} -> "
              f"{entry['held_out_best_speedup']}, nodes={entry['nodes']}, "
              f"wall={entry['wall_seconds']:.1f}s, best -> {best_path}", flush=True)

    wall = time.monotonic() - started
    baseline_mean = geometric_mean(
        [entry["held_out_baseline_speedup"] for entry in entries])
    best_mean = geometric_mean([entry["held_out_best_speedup"] for entry in entries])
    improved = sum(1 for entry in entries
                   if (entry["speedup_gain"] or 0.0) > 0.0)
    payload: Dict[str, Any] = {
        "experiment": "ERA on AgentDescent -- AlgoTune, one tree per task",
        "status": "completed" if entries and not failures else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "benchmark": "AlgoTune (arXiv:2507.15887), oripress/AlgoTune",
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "summary": {
            "tasks_run": len(entries),
            "tasks_failed": len(failures),
            "tasks_improved": improved,
            "geometric_mean_baseline_speedup": baseline_mean,
            "geometric_mean_best_speedup": best_mean,
            "wall_seconds": wall,
        },
        "tasks": entries,
        "failures": failures,
        "model_usage": _usage_dict(model_usage),
        "repair": {
            "attempts_allowed": args.repair_attempts,
            "checked": repairs.get("drawn", 0),
            "failed": repairs.get("failed", 0),
            "repaired": repairs.get("repaired", 0),
            "gave_up": repairs.get("gave_up", 0),
        },
        "reply_damage": {
            "drawn": damage.get("drawn", 0),
            "damaged": damage.get("damaged", 0),
            "attempts_allowed": max(1, args.reply_attempts),
        },
    }
    _write_json(args.output, payload)
    print(
        f"\ncompleted: {len(entries)} task(s), {improved} improved, "
        f"held-back geometric-mean speedup {baseline_mean} -> {best_mean}, "
        f"wall={wall:.1f}s, model_calls={model_usage.calls}, output={args.output}"
    )
    return 0 if entries and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
