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

How this differs from the benchmark's own leaderboard
-----------------------------------------------------
LLM-SRBench evaluates LLM-based *searchers* that see one problem at a time, with
the data in context, and propose hypotheses for that problem. This runs ERA's
protocol instead: the model never sees a sample, it writes one program, and that
program is run sandboxed against every problem. Numbers from here are therefore
not directly comparable to the paper's tables -- the same benchmark, splits and
metrics, a different experiment -- and both the result file and
``docs/algo-era.md`` say so.

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
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
    MIRROR_REPO,
    MIRROR_REVISION,
    PROBLEM_SECONDS,
    SUBSETS,
    Suite,
    evaluate_source,
    framework_score,
    mutation_prompt,
    prepare_suite,
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
) -> Domain:
    """This task, in the four terms the ERA search needs."""
    preview = suite_preview(suite)
    return Domain(
        name=("LLM-SRBench scientific equation discovery, mean "
              "min(12, -log10(NMSE)) on held-out samples"),
        entrypoint="discover",
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=INITIAL_PROGRAM,
        initial_summary=INITIAL_SUMMARY,
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
                        help="FUTS expansions in total (upstream's num_iterations)")
    parser.add_argument("--workers", type=int, default=3)
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
                        help="problem sets the search may score against")
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
    print(
        f"Plan     : mode={mode}, model={args.model}, iterations={args.iterations}, "
        f"workers={args.workers}, c_puct={args.c_puct}, temperature={args.temperature}"
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
