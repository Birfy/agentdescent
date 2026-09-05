"""Evolve the ERA tree search's selection rule on AlgoTune, validate on tasks it never saw.

Stage 1 of the meta-evolution plan (`docs/design-meta-evolution.md`, §4): the
inner problem is a whole ERA flat-PUCT search on one AlgoTune task, scored in
speedup over the task's reference implementation; the outer artifact is the
``priority(rank, visits, total, prior, depth, n_nodes)`` rule that decides which
node the tree expands next (`agentdescent.meta.priority_selection`).

Train tasks and validation tasks are disjoint AlgoTune tasks. After the outer
run, the seed rule (upstream ERA's flat PUCT) and the evolved rule are scored on
fresh seeds of *both* sets, and the transfer ratio -- gain on the tasks the rule
never saw over gain on the tasks it was evolved on -- is what the result file is
for. The eight tasks default to the ones `bench/results/era-algotune-model-prior.md`
already measured, so every number here has a baseline beside it.

Cost, before running it: one outer rollout is one whole inner search
(``--iterations`` expansions, each a model call plus a sandboxed timing), and
every held-out outer task is searched again at every gate. With the defaults --
4 train tasks x 2 seeds, 4 outer rounds x 2 workers, 6 expansions -- that is on
the order of a hundred inner searches for the outer run, then 2 rules x 8 tasks
x ``--validate-seeds`` searches for the report. Keep ``--iterations`` small; the
rule is being measured on how fast the curve rises, not on how far it gets.

    python -m bench.metasearch_algotune --dry-run
    python -m bench.metasearch_algotune --provider openai --model deepseek-v4-flash \\
        --rounds 4 --workers 2 --iterations 6 --yes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agentdescent.agents import Usage, with_retries
from agentdescent.meta import (MetaOutcome, Problem, meta_evolve, meta_validate,
                               priority_selection, slot_reflector, transfer_ratio)

from examples._common import add_standard_args, completion_for, confirm, worker_count
from examples.era._era_algotune import TASKS, prepare_suite
from examples.era._era_support import sandbox_backend, with_intact_replies
from examples.era.era_algotune import algotune_domain, geometric_mean
from examples.era.era_empirical_software import run_agentdescent_era


#: The eight tasks of `era-algotune-model-prior.md`, split so that each half
#: spans the same kinds of work (a transform, a factorisation, an FFT, a
#: projection or polynomial root-finder).
TRAIN_TASKS = ("psd_cone_projection", "lu_factorization",
               "fft_cmplx_scipy_fftpack", "polynomial_real")
VALIDATE_TASKS = ("affine_transform_2d", "convolve2d_full_fill",
                  "eigenvectors_complex", "fft_convolution")

DEFAULT_OUTPUT = Path("bench/results/metasearch-algotune.json")


def algotune_problem(task: str, complete: Callable[[str], str], *,
                     iterations: int = 6, workers: int = 1, shards: int = 4,
                     test_shards: int = 2, problems: int = 2, mode: str = "serial",
                     staleness: str = "full", candidate_timeout: float = 120.0,
                     max_seconds: float = 1800.0, suite_seed: int = 0) -> Problem:
    """One AlgoTune task as an inner :class:`~agentdescent.meta.Problem`.

    ``(selection policy, seed) -> MetaOutcome``: a whole ERA search on the task
    with the candidate rule choosing parents. The suite (problem sizes, seed
    program, shards) is prepared once per task; ``seed`` moves the search, not
    the data. ``workers=1`` is upstream ERA's serial loop and the honest inner
    default -- the outer loop already runs ``n_workers`` inner searches at once.
    """
    if task not in TASKS:
        raise ValueError(f"{task!r} is not an AlgoTune task this port can run")
    suite = prepare_suite(task, seed=suite_seed, shards=shards, test_shards=test_shards,
                          problems=problems)
    domain = algotune_domain(suite, candidate_timeout=candidate_timeout, ask_promise=False)

    def problem(policy: Any, seed: int) -> MetaOutcome:
        run = run_agentdescent_era(
            complete, mode=mode, iterations=iterations, workers=workers,
            shards=shards, test_shards=test_shards, seed=seed, domain=domain,
            selection=policy, staleness=staleness, max_seconds=max_seconds,
            candidate_timeout=candidate_timeout, usage=Usage())
        return MetaOutcome.from_result(
            run.result, task=task,
            baseline_speedup=run.baseline_test_metrics.get("speedup"),
            best_speedup=run.best_test_metrics.get("speedup"),
            nodes=len(run.tree.nodes), selection=run.tree.summary().get("selection"))

    return problem


def progress(label: str) -> Callable[[Any], None]:
    """One line per outer sweep. A run that reports nothing until its summary
    cannot be told from a stalled one, and an outer sweep here is minutes of
    inner searches -- the same reason `examples._method_runner` prints one."""

    def on_round(info: Any) -> None:
        print(f"[{label} sweep {info.round}] held_out={info.held_out_reward:.3f} "
              f"committed={info.committed} rejected={info.rejected} "
              f"reasons={info.reasons} elapsed={info.elapsed_s:.0f}s "
              f"rollouts={info.rollouts}", flush=True)

    return on_round


def run_experiment(complete: Callable[[str], str], *,
                   train: Dict[str, Problem], validate: Dict[str, Problem],
                   seeds: Sequence[int], validate_seeds: Sequence[int],
                   rounds: int, workers: int, outer_seed: int = 0,
                   usage: Optional[Usage] = None, max_seconds: Optional[float] = None,
                   max_rollouts: Optional[int] = None,
                   eval_concurrency: Optional[int] = None) -> Dict[str, Any]:
    """The whole experiment over prepared problems; the result file's payload."""
    if set(train) & set(validate):
        raise ValueError(f"train and validate share tasks: {sorted(set(train) & set(validate))}")
    if set(seeds) & set(validate_seeds):
        raise ValueError("validation seeds must not overlap the outer run's seeds")
    spec = priority_selection()
    seed_rule = spec.render(spec.initial())
    started = time.monotonic()
    result = meta_evolve(train, slot="selection", spec=spec,
                         propose=slot_reflector(complete, spec), seeds=list(seeds),
                         rounds=rounds, n_workers=workers, max_concurrency=workers,
                         held_out_frac=0.4,
                         eval_concurrency=eval_concurrency or max(1, workers),
                         max_seconds=max_seconds, max_rollouts=max_rollouts,
                         seed=outer_seed, usage=usage,
                         on_round=progress('algotune'))
    outer_seconds = time.monotonic() - started
    report = meta_validate(spec, seed_rule, result.rendered, {**train, **validate},
                           seeds=list(validate_seeds))
    per_set = {}
    for name, tasks in (("train", train), ("validate", validate)):
        rows = [report[t] for t in tasks]
        per_set[name] = {
            "tasks": list(tasks),
            "seed_rule": sum(r["before"] for r in rows) / len(rows),
            "evolved_rule": sum(r["after"] for r in rows) / len(rows),
            "gain": sum(r["gain"] for r in rows) / len(rows),
            "wins": sum(r["wins"] for r in rows),
            "losses": sum(r["losses"] for r in rows),
        }
    src, tgt = per_set["train"]["gain"], per_set["validate"]["gain"]
    return {
        "evolved_source": result.rendered,
        "seed_source": seed_rule,
        "outer": {"final_reward": result.final_reward, "rollouts": result.rollouts,
                  "outcomes": result.outcomes(), "stop_reason": result.stop_reason,
                  "error": result.error, "seconds": outer_seconds,
                  "rounds": rounds, "workers": workers, "seeds": list(seeds)},
        "validation": report,
        "by_set": per_set,
        "transfer_ratio": (tgt / src) if abs(src) > 1e-9 else None,
        "validate_seeds": list(validate_seeds),
    }


def format_report(payload: Dict[str, Any]) -> str:
    lines = [f"{'task':<26} {'set':<9} {'seed':>7} {'evolved':>8} {'gain':>7} {'w/l':>5}"]
    sets = {t: name for name, row in payload["by_set"].items() for t in row["tasks"]}
    for task, row in payload["validation"].items():
        lines.append(f"{task:<26} {sets.get(task, '?'):<9} {row['before']:>7.3f} "
                     f"{row['after']:>8.3f} {row['gain']:>+7.3f} {row['wins']:>2}/{row['losses']}")
    for name, row in payload["by_set"].items():
        lines.append(f"{name:<26} {'mean':<9} {row['seed_rule']:>7.3f} "
                     f"{row['evolved_rule']:>8.3f} {row['gain']:>+7.3f} {row['wins']:>2}/{row['losses']}")
    ratio = payload["transfer_ratio"]
    lines.append("transfer ratio (validate gain / train gain): "
                 + ("n/a (no train gain)" if ratio is None else f"{ratio:.2f}"))
    return "\n".join(lines)


def _tasks_arg(text: str, default: Sequence[str]) -> List[str]:
    names = [n.strip() for n in (text or "").split(",") if n.strip()] or list(default)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise SystemExit(f"unknown AlgoTune task(s): {', '.join(unknown)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(parser, model_default="deepseek-v4-flash", max_seconds_default=1800.0,
                      eval_concurrency_default=None, include_val_cap=False)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--train-tasks", default=",".join(TRAIN_TASKS))
    parser.add_argument("--validate-tasks", default=",".join(VALIDATE_TASKS))
    parser.add_argument("--seeds", type=int, default=2, help="inner seeds per train task")
    parser.add_argument("--validate-seeds", type=int, default=2,
                        help="fresh seeds per task for the final report")
    parser.add_argument("--rounds", type=int, default=4, help="outer rounds")
    parser.add_argument("--workers", type=int, default=2, help="outer workers")
    parser.add_argument("--iterations", type=int, default=6, help="expansions per inner search")
    parser.add_argument("--inner-workers", type=int, default=1)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--test-shards", type=int, default=2)
    parser.add_argument("--problems", type=int, default=2)
    parser.add_argument("--staleness", default="full", choices=["guarded", "reflective", "full"])
    parser.add_argument("--candidate-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.asynchronous or args.pipelined_gate:
        raise SystemExit("--async / --pipelined-gate are not supported by the "
                         "meta loop; the outer runtime is the synchronous one")
    workers = worker_count(args, args.workers)
    train = _tasks_arg(args.train_tasks, TRAIN_TASKS)
    validate = _tasks_arg(args.validate_tasks, VALIDATE_TASKS)
    seeds = list(range(args.seeds))
    validate_seeds = list(range(1_000, 1_000 + args.validate_seeds))
    inner_per_outer = len(train) * len(seeds)
    print("Algorithm : meta_evolve over the ERA tree search's selection rule (priority_selection)")
    print(f"Train     : {', '.join(train)}  x {len(seeds)} seed(s) = {inner_per_outer} outer tasks")
    print(f"Validate  : {', '.join(validate)}  (+ train) x {len(validate_seeds)} fresh seed(s)")
    print(f"Outer     : rounds={args.rounds} workers={workers} blast_radius=0.6 (L1)")
    print(f"Inner     : {args.iterations} expansions, inner workers={args.inner_workers}, "
          f"shards={args.shards}+{args.test_shards}, staleness={args.staleness}")
    print(f"Evaluator : {sandbox_backend() or 'NO SANDBOX -- this run will fail'}")
    print(f"Model     : {args.provider}/{args.model} temperature={args.temperature}")
    if args.dry_run:
        print("[dry-run] no API, task file or sandbox process was accessed.")
        return 0
    if not confirm(args):
        return 0
    usage = Usage()
    complete = with_intact_replies(
        with_retries(completion_for(args, usage=usage, max_tokens=args.max_tokens,
                                    timeout=args.api_timeout, temperature=args.temperature,
                                    retries=1), attempts=5, backoff=4.0),
        attempts=4)
    inner = dict(iterations=args.iterations, workers=args.inner_workers, shards=args.shards,
                 test_shards=args.test_shards, problems=args.problems, staleness=args.staleness,
                 candidate_timeout=args.candidate_timeout, max_seconds=args.max_seconds,
                 mode="serial" if args.inner_workers == 1 else "sync")
    problems = {task: algotune_problem(task, complete, **inner) for task in train + validate}
    payload = run_experiment(
        complete, train={t: problems[t] for t in train},
        validate={t: problems[t] for t in validate}, seeds=seeds,
        validate_seeds=validate_seeds, rounds=args.rounds, workers=workers,
        outer_seed=args.seed, usage=usage,
        # The inner search gets its own `max_seconds`; this one bounds the outer
        # loop, which is the run the operator is waiting on.
        max_seconds=args.max_seconds, max_rollouts=args.budget_rollouts or None,
        eval_concurrency=args.eval_concurrency)
    payload["config"] = {**inner, "model": args.model, "provider": args.provider,
                         "temperature": args.temperature, "outer_seed": args.seed}
    payload["usage"] = {"calls": usage.calls, "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens}
    print("[evolved rule]\n" + payload["evolved_source"])
    print(format_report(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"[result saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
