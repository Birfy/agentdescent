"""ERA tree search on **LLM-SRBench**, the scientific equation-discovery benchmark.

    "LLM-SRBench, a comprehensive benchmark with 239 challenging problems across
    four scientific domains specifically designed to evaluate LLM-based
    scientific equation discovery methods while preventing trivial memorization"
    -- arXiv:2504.10415 (ICML 2025)

What the search optimises
-------------------------
A candidate is a single function, ``discover(x, y, spec)``, and it is handed one
scientific dataset at a time: a table of samples, the names of the variables, and
the benchmark's own one-paragraph description of what they mean. It returns a
**closed-form equation** as a string, which is parsed rather than executed --
numeric constants, the problem's variables, ``pi``/``e``, the four operators,
powers, and a fixed list of elementary functions. Anything else is rejected, so
the answer is an equation and not a regressor.

Every problem is scored on held-out samples the candidate never sees, by the
benchmark's own metrics: NMSE, and Acc(0.1) -- the share of problems whose worst
relative error stays under 10%. The tree ranks nodes by
``min(12, -log10(NMSE))`` averaged over the problem set, because Acc(0.1) is an
indicator that is flat almost everywhere and raw NMSE is dominated by whichever
problem failed worst.

The baseline node is sequentially thresholded least squares over a fixed
nonlinear library -- SINDy's fitting step (Brunton et al., 2016) without its
domain-chosen library. It is the method a practitioner reaches for before
reaching for an LLM: it recovers several of the synthetic right-hand sides
outright, and its ceiling is its library, which is the headroom the tree search
explores.

Two protocols, and the flag that chooses between them
-----------------------------------------------------
By default this runs **ERA's** protocol: one search writes **one program**, and
that program is run sandboxed over every problem in the category. The model never
sees a sample. It is cheap -- 14 model calls for 129 problems -- and it answers
ERA's question, "can a model write one method that solves a whole scientific
distribution?". It is *not* the benchmark's protocol, so its numbers are not
directly comparable to the paper's tables.

``--per-problem`` runs **the benchmark's** protocol instead: one independent
search per problem, each with its own tree, its own seed program and its own
budget, scored on slices of that problem's own training rows and reported on that
problem's own held-out split. That is what LLM-SR, LaSR and SGA do, so a number
from it belongs in the same column as theirs -- with the budget stated, since
LLM-SR gives each problem 1 000 samples where this gives it a handful of
expansions.

The difference is not a tuning knob. A single program cannot express
``(-A + x1*y1 - x2*y2)/x3``; a per-problem search can propose exactly that
structure and fit it. Both result files record which protocol produced them.

Everything about the search itself is `era_empirical_software.py`: the flat-PUCT
tree, the visit reservation, the staleness handling, the aggregator, the
governance layer. This module supplies only a
:class:`~examples.era._era_domain.Domain` -- seed program, sandboxed evaluator,
mutation prompt, metric name -- and the command line the other ports share.

Run
---
    python -m examples.era.era_llm_srbench --dry-run
    python -m examples.era.era_llm_srbench --provider claude --model glm-5.2 \\
        --iterations 12 --workers 3 --problems 48 --yes
    python -m examples.era.era_llm_srbench --provider claude --model glm-5.2 \\
        --per-problem --dataset lsr_transform --shards 6 --iterations 6 \\
        --workers 3 --problem-concurrency 2 --problem-seconds 8 --yes
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import statistics
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agentdescent.agents import Usage
from agentdescent.evolution import EvolvingArtifact
from agentdescent.governance import classify

from examples._common import (
    add_standard_args,
    completion_for,
    confirm,
    worker_count,
)
from examples.era._era_domain import Domain
from examples.era._era_srbench import (
    BENCHMARK_PAPER,
    GROUPS,
    INITIAL_PROGRAM,
    INITIAL_SUMMARY,
    SEED_PROGRAMS,
    MIRROR_REPO,
    MIRROR_REVISION,
    PROBLEM_SECONDS,
    SUBSETS,
    SrProblem,
    Suite,
    evaluate_source,
    framework_score,
    load_catalogue,
    mutation_prompt,
    per_problem_prompt,
    prepare_problem_suite,
    prepare_suite,
    problem_preview,
    suite_preview,
)
from examples.era._era_srbench_expr import DIGIT_CAP, FUNCTIONS, TOLERANCE
from examples.era._era_support import (
    UPSTREAM_COMMIT,
    sandbox_backend,
    with_intact_replies,
)
from examples.era.era_empirical_software import (
    _require_api_environment,
    _usage_dict,
    _utc_now,
    _write_json,
    run_agentdescent_era,
)


ARTIFACT_ID = "era_program"
DEFAULT_OUTPUT = Path("era-srbench-result.json")

#: The names a returned equation may call, shown to the model exactly as the
#: evaluator holds them.
EQUATION_FUNCTIONS = tuple(sorted(FUNCTIONS))


def _make_completion(args: argparse.Namespace, usage: Usage):
    """The sibling ports' completion wiring, calling this module's own import.

    Six lines rather than an import of a neighbour's private helper, because
    `tests/test_example_entrypoints.py` proves a dry-run never crosses an
    external boundary by replacing **this module's** `completion_for` with a
    tripwire. A port whose network call is made through another module's name
    would pass that test without the tripwire ever being in the path.
    """
    options: Dict[str, Any] = {}
    if args.thinking != "default":
        options["thinking"] = {"type": args.thinking}
    return completion_for(
        args,
        usage=usage,
        max_tokens=args.max_tokens,
        timeout=args.api_timeout,
        temperature=args.temperature,
        **options,
    )


def srbench_domain(
    suite: Suite,
    *,
    candidate_timeout: float = 300.0,
    max_code_length: int = 20_000,
    problem_seconds: float = PROBLEM_SECONDS,
    seed_program: str = "library",
) -> Domain:
    """This task, in the four terms the ERA search needs."""
    preview = suite_preview(suite)
    root_code, root_summary = SEED_PROGRAMS[seed_program]
    return Domain(
        name=("LLM-SRBench scientific equation discovery, mean "
              "min(12, -log10(NMSE)) on held-out samples"),
        entrypoint="discover",
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=root_code,
        initial_summary=root_summary,
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            problem_seconds=problem_seconds, max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, preview=preview, timeout=candidate_timeout,
            problem_seconds=problem_seconds, functions=EQUATION_FUNCTIONS),
        task_prompt=lambda index: (
            f"Discover the governing equation of every problem in held-out "
            f"problem set {index}, to as many correct digits as the time budget "
            f"allows."),
        test_shards=suite.test_range(),
        data_summary={
            "benchmark": "LLM-SRBench",
            "paper": BENCHMARK_PAPER,
            "source": f"{MIRROR_REPO}@{MIRROR_REVISION[:12]}",
            "seed_program": seed_program,
            "subsets": list(suite.subsets),
            "problems_per_subset": suite.counts(),
            "problems_total": len(suite.problems()),
            "problems_per_shard": suite.size(0),
            "scoring_shards": suite.scoring_shards,
            "test_shards": suite.test_shards,
            "train_points": suite.train_points or "all",
            "problem_seconds": problem_seconds,
            "digit_cap": DIGIT_CAP,
            "tolerance": TOLERANCE,
            "seed": suite.seed,
        },
    )


def per_problem_domain(
    suite: Suite,
    problem: SrProblem,
    samples: Dict[str, Any],
    *,
    candidate_timeout: float = 120.0,
    max_code_length: int = 20_000,
    problem_seconds: float = PROBLEM_SECONDS,
    train_points: int = 0,
    seed_program: str = "library",
) -> Domain:
    """One LLM-SRBench problem, in the four terms the ERA search needs.

    The same evaluator, sandbox, grammar and metrics as the whole-category
    domain; what changes is that a shard is a slice of *this problem's* held-out
    training rows, and the single test shard is the benchmark's own id-test (with
    OOD beside it where the category has one).
    """
    preview = problem_preview(problem, samples, train_points=train_points)
    root_code, root_summary = SEED_PROGRAMS[seed_program]
    return Domain(
        name=(f"LLM-SRBench {problem.problem_id}: recover one equation, "
              f"min(12, -log10(NMSE)) on the benchmark's held-out samples"),
        entrypoint="discover",
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=root_code,
        initial_summary=root_summary,
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            problem_seconds=problem_seconds, max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: per_problem_prompt(
            program, preview=preview, timeout=candidate_timeout,
            problem_seconds=problem_seconds, functions=EQUATION_FUNCTIONS),
        task_prompt=lambda index: (
            f"Recover the equation behind {problem.problem_id}, scored on "
            f"held-out slice {index} of its training data."),
        test_shards=suite.test_range(),
        data_summary={
            "benchmark": "LLM-SRBench",
            "paper": BENCHMARK_PAPER,
            "problem_id": problem.problem_id,
            "subset": problem.subset,
            "seed_program": seed_program,
            "input_vars": list(problem.input_vars),
            "fit_rows": problem.train_rows,
            "validation_shards": suite.scoring_shards,
            "test_rows": suite.shard_problems[-1][0].test_rows,
            "ood_rows": suite.shard_problems[-1][0].ood_rows,
            "problem_seconds": problem_seconds,
            "digit_cap": DIGIT_CAP,
            "tolerance": TOLERANCE,
            "seed": suite.seed,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=1800.0,
                      eval_concurrency_default=None)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--staleness", default="guarded",
                        choices=["guarded", "reflective", "full"],
                        help=("what to do with an expansion proposed against a "
                              "head the merger has since moved. The tree is "
                              "append-only, so `full` is the honest default for "
                              "a comparison and `guarded` the conservative one"))
    parser.add_argument("--iterations", type=int, default=6,
                        help=("FUTS expansions in total (upstream's "
                              "num_iterations); with --per-problem, expansions "
                              "*per problem*"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--per-problem", action="store_true",
        help=("run the benchmark's own protocol: one independent search per "
              "problem, scored on slices of that problem's training data and "
              "reported on its own held-out split. Without it, one search "
              "writes one program for the whole category, which is ERA's "
              "protocol and a different experiment"))
    parser.add_argument(
        "--problem-concurrency", type=int, default=1,
        help=("--per-problem only: problems searched at once. Each one already "
              "runs --workers model calls in parallel, so this multiplies the "
              "load on the endpoint"))
    parser.add_argument(
        "--seed-program", default="library", choices=sorted(SEED_PROGRAMS),
        help=("the root node the search starts from. `library` is sparse "
              "regression over a fixed nonlinear basis; `linear` is LLM-SR's "
              "own starting point, a fitted linear model in the raw inputs. On "
              "LSR-Synth the choice is worth 45 points of Acc(0.1) before any "
              "search runs, so a comparison with the paper has to name it"))
    parser.add_argument(
        "--resume", action="store_true",
        help=("--per-problem only: pick up an interrupted sweep from --output, "
              "skipping every problem already in it. Refuses if that file was "
              "written under a different budget"))
    parser.add_argument(
        "--val-frac-per-problem", type=float, default=0.25,
        help=("--per-problem only: share of a problem's training rows held out "
              "of the fit and split into the scoring shards"))
    parser.add_argument("--dataset", default="lsr_synth",
                        choices=sorted(set(GROUPS) | set(SUBSETS)),
                        help=("which of the benchmark's categories to run. "
                              "`lsr_synth` is the four synthetic domains (129 "
                              "problems, the only ones with an OOD split), "
                              "`lsr_transform` the 111 rearranged Feynman "
                              "equations, `all` both"))
    parser.add_argument("--problems", type=int, default=0,
                        help=("cap the number of problems, drawn evenly across "
                              "the chosen subsets; 0 runs all of them. A run "
                              "reports exactly which problems it used"))
    parser.add_argument("--shards", type=int, default=6,
                        help=("problem sets the search may score against; with "
                              "--per-problem, slices of one problem's validation "
                              "pool. Note that `shards * (1 - held_out_frac)` is "
                              "how many rollout tasks exist, and a round can "
                              "only propose one expansion per task -- so "
                              "--shards 4 --workers 3 leaves a worker idle and "
                              "produces two expansions a round, not three"))
    parser.add_argument("--test-shards", type=int, default=2,
                        help="further problem sets the search never sees")
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--candidate-timeout", type=float, default=300.0,
                        help="wall-clock for one whole problem set, in the sandbox")
    parser.add_argument("--problem-seconds", type=float, default=PROBLEM_SECONDS,
                        help=("wall-clock per problem, enforced with SIGALRM "
                              "inside --candidate-timeout. This is half the "
                              "task: given unbounded time the winner is whichever "
                              "method is allowed to search longest"))
    parser.add_argument("--train-points", type=int, default=0,
                        help=("training samples handed to a candidate per problem, "
                              "0 for all of them (4 000 for LSR-Synth, 80 000 for "
                              "LSR-Transform)"))
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


def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if not math.isfinite(number):
        return "inf"
    return f"{number:.4g}"


def _pooled(rows: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows
              if row.get(key) is not None and math.isfinite(float(row[key]))]
    return statistics.median(values) if values else None


def _benchmark_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The benchmark's own numbers over however many problems were searched."""
    if not rows:
        return {"problems": 0}
    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_subset.setdefault(row["subset"], []).append(row)

    def block(group: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        digits = [float(r["digits"]) for r in group]
        ood = [r for r in group if r.get("ood_digits") is not None]
        payload: Dict[str, Any] = {
            "problems": len(group),
            "acc_0.1": sum(int(r["acc"]) for r in group) / len(group),
            "median_nmse": _pooled(group, "nmse"),
            "mean_digits": sum(digits) / len(digits),
            "solved_6_digits": sum(1 for d in digits if d >= 6.0),
        }
        if ood:
            payload["ood_acc_0.1"] = sum(int(r["ood_acc"]) for r in ood) / len(ood)
            payload["ood_mean_digits"] = (
                sum(float(r["ood_digits"]) for r in ood) / len(ood))
        return payload

    return {"overall": block(rows),
            "by_subset": {name: block(group)
                          for name, group in sorted(by_subset.items())}}


def _budget_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    """Everything that changes what a problem's number means.

    A resume that quietly folded a 6-expansion row into a 20-expansion sweep
    would produce a file holding two experiments under one heading, so the
    fingerprint is compared rather than assumed and a mismatch refuses.
    """
    return {
        "dataset": args.dataset,
        "iterations": args.iterations,
        "workers": args.workers,
        "shards": args.shards,
        "held_out_frac": args.held_out_frac,
        "val_frac_per_problem": args.val_frac_per_problem,
        "problem_seconds": args.problem_seconds,
        "candidate_timeout": args.candidate_timeout,
        "train_points": args.train_points,
        "seed_program": args.seed_program,
        "model": args.model,
        "seed": args.seed,
    }


def _load_checkpoint(path: Path,
                     fingerprint: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows already searched under exactly this budget, or nothing.

    A per-problem sweep is hours of independent searches, and a container that
    goes away takes the whole thing with it unless the file on disk is written as
    it goes. This is the other half of that: `--resume` picks the file back up,
    and every problem in it is skipped rather than paid for twice.
    """
    if not path.exists():
        return []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--resume cannot read {path}: {exc}")
    if stored.get("budget") != fingerprint:
        raise SystemExit(
            f"--resume refused: {path} was written under a different budget.\n"
            f"  stored: {stored.get('budget')}\n"
            f"  now   : {fingerprint}\n"
            "Resuming into it would put two experiments in one file; write to a "
            "different --output instead.")
    return list(stored.get("per_problem") or [])


def _one_problem(args: argparse.Namespace, complete, problem: SrProblem,
                 samples: Dict[str, Any], actor_usage: Usage,
                 mode: str) -> Dict[str, Any]:
    """One independent search, for one problem, reported on its own held-out split."""
    suite = prepare_problem_suite(
        problem, samples, seed=args.seed, shards=args.shards,
        val_frac=args.val_frac_per_problem, train_points=args.train_points)
    domain = per_problem_domain(
        suite, problem, samples,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        problem_seconds=args.problem_seconds,
        train_points=args.train_points,
        seed_program=args.seed_program)
    run = run_agentdescent_era(
        complete,
        mode=mode,
        iterations=args.iterations,
        workers=args.workers,
        shards=args.shards,
        test_shards=1,
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
        verbose=False,
    )

    def cell(metrics: Dict[str, Any]) -> Dict[str, Any]:
        detail = (metrics.get("per_problem") or [{}])[0]
        return {
            "digits": detail.get("digits", 0.0),
            "nmse": detail.get("nmse"),
            "acc": detail.get("acc", 0),
            "ood_digits": detail.get("ood_digits"),
            "ood_acc": detail.get("ood_acc"),
            "equation": detail.get("equation", ""),
            "error": detail.get("error", ""),
        }

    best = cell(run.best_test_metrics)
    baseline = cell(run.baseline_test_metrics)
    return {
        "problem_id": problem.problem_id,
        "subset": problem.subset,
        "input_vars": list(problem.input_vars),
        "gt_expression": problem.gt_expression,
        "fit_rows": problem.train_rows,
        "test_rows": suite.shard_problems[-1][0].test_rows,
        "nodes": len(run.tree.nodes),
        "wall_seconds": run.wall_seconds,
        "gate_best": run.tree.best().score,
        "baseline": baseline,
        "best": best,
        **best,
    }


def _checkpoint(args: argparse.Namespace, fingerprint: Dict[str, Any],
                rows: List[Dict[str, Any]], started: float, model_usage: Usage,
                actor_usage: Usage, damage: Dict[str, int], *,
                complete: bool) -> Dict[str, Any]:
    """Write the sweep's file as it stands, and return what was written.

    The same payload whether the sweep has finished or not, so a run killed
    two-thirds of the way through leaves a file that reads exactly like a
    finished one except for ``status`` and the problem count. `_write_json`
    writes through a temporary file and renames, so a reader never sees a half
    file even mid-sweep.
    """
    ordered = sorted(rows, key=lambda row: row["problem_id"])
    payload: Dict[str, Any] = {
        "experiment": ("ERA on AgentDescent -- LLM-SRBench, one search per "
                       "problem (the benchmark's own protocol)"),
        "status": "completed" if complete else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "comparability": (
            "One independent search per problem, scored on slices of that "
            "problem's own training rows and reported on the benchmark's own "
            "held-out split -- the protocol the paper's methods run under. The "
            "budget is not the paper's: LLM-SR gives each problem 1000 samples, "
            f"this gives it {args.iterations} expansions. Symbolic accuracy is "
            "not measured here at all."),
        "budget": fingerprint,
        "config": {key: value for key, value in vars(args).items()
                   if key not in ("output", "yes")},
        "wall_seconds": time.monotonic() - started,
        "baseline": _benchmark_summary([
            {**row["baseline"], "subset": row["subset"]} for row in ordered]),
        "best": _benchmark_summary(ordered),
        "per_problem": ordered,
        "model_usage": _usage_dict(model_usage),
        "actor_usage": _usage_dict(actor_usage),
        "reply_damage": {
            "drawn": damage.get("drawn", 0),
            "damaged": damage.get("damaged", 0),
            "attempts_allowed": max(1, args.reply_attempts),
        },
    }
    _write_json(args.output, payload)
    return payload


def _run_per_problem(args: argparse.Namespace, complete, model_usage: Usage,
                     actor_usage: Usage, damage: Dict[str, int],
                     mode: str) -> int:
    """The benchmark's own protocol: one independent search per problem.

    Each problem gets its own tree, its own seed program, its own budget, and is
    reported on its own held-out split -- which is what makes a number here
    comparable with the paper's tables in a way the whole-category protocol
    never is. The cost is linear in the number of problems, so `--problems`
    caps it and `--problem-concurrency` trades endpoint load for wall-clock.
    """
    catalogue = load_catalogue(args.dataset, problems=args.problems,
                               seed=args.seed)
    counts: Dict[str, int] = {}
    for problem, _samples in catalogue:
        counts[problem.subset] = counts.get(problem.subset, 0) + 1
    print(f"Problems : {len(catalogue)} searched one at a time, from "
          f"{', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}; "
          f"{args.iterations} expansions each, {args.shards} validation shards "
          f"from {int(100 * args.val_frac_per_problem)}% of train, seed={args.seed}")

    fingerprint = _budget_fingerprint(args)
    rows: List[Dict[str, Any]] = (
        _load_checkpoint(args.output, fingerprint) if args.resume else [])
    if rows:
        done = {row["problem_id"] for row in rows}
        catalogue = [pair for pair in catalogue if pair[0].problem_id not in done]
        print(f"Resuming : {len(done)} problems already in {args.output}, "
              f"{len(catalogue)} left to search")
    total = len(rows) + len(catalogue)
    lock = threading.Lock()
    started = time.monotonic()

    def work(index_problem):
        index, (problem, samples) = index_problem
        try:
            row = _one_problem(args, complete, problem, samples, actor_usage, mode)
        except Exception as exc:  # one problem must not take the sweep down
            row = {"problem_id": problem.problem_id, "subset": problem.subset,
                   "gt_expression": problem.gt_expression,
                   "digits": 0.0, "nmse": None, "acc": 0, "equation": "",
                   "error": f"{type(exc).__name__}: {exc}",
                   "baseline": {"digits": 0.0, "acc": 0, "nmse": None},
                   "best": {"digits": 0.0, "acc": 0, "nmse": None}}
        with lock:
            rows.append(row)
            done = len(rows)
            print(f"[{done:3d}/{total}] {row['problem_id']:34s} "
                  f"digits {row['baseline']['digits']:>6} -> {row['digits']:<6} "
                  f"acc {row['baseline']['acc']}->{row['acc']}  "
                  f"({time.monotonic() - started:.0f}s elapsed)"
                  + (f"  [{row['error'][:60]}]" if row.get("error") else ""),
                  flush=True)
            # Written after every problem, not at the end. A sweep at this budget
            # is hours of independent searches and the machine under it is not
            # guaranteed to last that long; without this, an interruption costs
            # every problem already paid for.
            _checkpoint(args, fingerprint, rows, started, model_usage,
                        actor_usage, damage, complete=False)
        return row

    workers = max(1, int(args.problem_concurrency))
    if workers == 1:
        for item in enumerate(catalogue):
            work(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, enumerate(catalogue)))
    rows.sort(key=lambda row: row["problem_id"])

    payload = _checkpoint(args, fingerprint, rows, started, model_usage,
                          actor_usage, damage, complete=True)
    overall_before = payload["baseline"]["overall"]
    overall_after = payload["best"]["overall"]
    print(
        f"completed: {overall_after['problems']} problems, "
        f"Acc(0.1) {_percent(overall_before['acc_0.1'])} -> "
        f"{_percent(overall_after['acc_0.1'])}, "
        f"median NMSE {_number(overall_before['median_nmse'])} -> "
        f"{_number(overall_after['median_nmse'])}, "
        f"mean digits {overall_before['mean_digits']:.3f} -> "
        f"{overall_after['mean_digits']:.3f}, "
        f"wall={payload['wall_seconds']:.0f}s, model_calls={model_usage.calls}, "
        f"output={args.output}"
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.workers = worker_count(args, args.workers)
    if args.budget_rollouts:
        args.iterations = args.budget_rollouts
    if getattr(args, "reflective_merge", False):
        raise SystemExit(
            "--reflective-merge is not supported by the ERA port: a candidate is "
            "a whole program rather than a delta, and fusing a round's "
            "expansions into one would delete tree nodes that FUTS selects from. "
            "Use --staleness {guarded,reflective,full}.")
    mode = "async" if args.asynchronous else ("serial" if args.serial else "sync")
    print("Algorithm: ERA Flat UCB tree search (FUTS) on AgentDescent")
    print(f"Task     : LLM-SRBench equation discovery ({BENCHMARK_PAPER}) -- "
          f"dataset={args.dataset}, mean min({DIGIT_CAP:.0f}, -log10 NMSE) "
          f"on held-out samples")
    print(f"Evaluator: {sandbox_backend() or 'NO SANDBOX -- this run will fail'} "
          f"isolated, {args.problem_seconds:.0f}s per problem, equations parsed "
          f"and never executed")
    protocol = ("one search per problem (the benchmark's own protocol)"
                if args.per_problem
                else "one search for the whole category (ERA's protocol)")
    print(f"Protocol : {protocol}")
    print(
        f"Plan     : mode={mode}, model={args.model}, iterations={args.iterations}"
        + ("/problem" if args.per_problem else "")
        + f", workers={args.workers}, c_puct={args.c_puct}, "
        f"temperature={args.temperature}"
        + (f", problem_concurrency={args.problem_concurrency}"
           if args.per_problem else "")
    )
    artifact = EvolvingArtifact(ARTIFACT_ID, blast_radius=0.6)
    print(
        f"Governance: generated program blast_radius={artifact.blast_radius} "
        f"-> {classify(artifact).name}"
    )
    if args.dry_run:
        print("[dry-run] no API, benchmark download, or sandbox process was accessed.")
        return 0

    _require_api_environment(args.provider)
    if not confirm(args):
        return 0

    model_usage = Usage()
    actor_usage = Usage()
    damage: Dict[str, int] = {}
    complete = with_intact_replies(
        _make_completion(args, model_usage),
        attempts=max(1, args.reply_attempts), counter=damage)
    if args.per_problem:
        return _run_per_problem(args, complete, model_usage, actor_usage, damage,
                                mode)

    suite = prepare_suite(seed=args.seed, shards=args.shards,
                          test_shards=args.test_shards, dataset=args.dataset,
                          problems=args.problems, train_points=args.train_points)
    counts = suite.counts()
    print(f"Problems : {len(suite.problems())} from "
          f"{', '.join(f'{name}={counts[name]}' for name in sorted(counts))}; "
          f"{suite.size(0)} per set, {args.shards} scored + {args.test_shards} "
          f"held back, seed={args.seed}, files under {suite.root}")
    domain = srbench_domain(
        suite,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        problem_seconds=args.problem_seconds,
        seed_program=args.seed_program,
    )
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
        verbose=True,
    )
    payload: Dict[str, Any] = {
        "experiment": "ERA on AgentDescent -- LLM-SRBench equation discovery",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "task": domain.name,
        "comparability": (
            "Same benchmark, splits and metrics as arXiv:2504.10415; different "
            "protocol. The paper's methods see one problem at a time with the "
            "data in context; here the model writes one program that never sees "
            "a sample and is run sandboxed over every problem. Numbers here are "
            "not directly comparable to the paper's tables."),
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "problems": [problem.to_dict() for problem in suite.problems()],
        "observation": run.summary(args.quality_target),
        "model_usage": _usage_dict(model_usage),
        "reply_damage": {
            "drawn": damage.get("drawn", 0),
            "damaged": damage.get("damaged", 0),
            "attempts_allowed": max(1, args.reply_attempts),
        },
    }
    _write_json(args.output, payload)
    best_path = args.output.with_name(f"{args.output.stem}-best.py")
    best_path.write_text(run.tree.best().program.code.rstrip() + "\n", encoding="utf-8")
    baseline = run.baseline_test_metrics
    best = run.best_test_metrics
    print(
        f"completed: held-back mean digits "
        f"{_number(baseline.get('mean_digits'))} -> {_number(best.get('mean_digits'))}, "
        f"Acc(0.1) {_percent(baseline.get('acc_0.1'))} -> {_percent(best.get('acc_0.1'))}, "
        f"median NMSE {_number(baseline.get('median_nmse'))} -> "
        f"{_number(best.get('median_nmse'))}, "
        f"nodes={len(run.tree.nodes)}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls}, output={args.output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
