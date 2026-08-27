"""ERA tree search on the paper's *numerical solution of integrals* task.

The ERA abstract lists six demonstrations, and five of them are scored against a
leaderboard or a held-out dataset. The sixth is not: "numerical solution of
integrals" is scored against arithmetic. That is the one ported here.

    "ERA also produced expert-level software for geospatial analysis, neural
    activity prediction in zebrafish, and numerical solution of integrals"
    -- arXiv:2509.06503

What the search optimises
-------------------------
A candidate is a single function, ``integrate(f, a, b)``, and it is handed a
**black-box scalar integrand** -- no formula, no parameters, no family name --
over ``[0, 1]``, ``[0, inf)`` or ``(-inf, inf)``. Each problem set holds nine
integrals, one from each of nine difficulty classes: singular at one endpoint,
singular at both, oscillating infinitely often into a corner, spiked over a
width of 1e-7, oscillating without ever decaying, cancelling down to a
ten-thousandth of the integrand's own size, and so on. Every one of them has a
**closed form**, so the score is correct significant digits against an exact
value rather than agreement with a rival integrator.

`scipy.integrate.quad` on defaults is the root node. It is a genuinely strong
baseline -- adaptive Gauss-Kronrod with a documented infinite-range transform --
and it solves several of the nine families to machine precision. It also returns
confident nonsense on others, which is the headroom the tree search explores.

Everything about the search itself is `era_empirical_software.py`: the flat-PUCT
tree, the visit reservation, the staleness handling, the aggregator, the
governance layer. This module supplies only a
:class:`~examples.era._era_domain.Domain` -- seed program, sandboxed evaluator,
mutation prompt, metric name -- and the command line the other ports share.

Run
---
    python -m examples.era.era_hard_integrals --dry-run
    python -m examples.era.era_hard_integrals --provider claude --model glm-5.2 \\
        --iterations 12 --workers 3 --yes
"""

from __future__ import annotations

import argparse
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
from examples.era._era_integrals import DIGIT_CAP, FAMILIES
from examples.era._era_integration import (
    EVAL_BUDGET,
    INITIAL_PROGRAM,
    PROBLEM_SECONDS,
    Suite,
    evaluate_source,
    framework_score,
    mutation_prompt,
    prepare_suite,
    suite_preview,
)
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
DEFAULT_OUTPUT = Path("era-integrals-result.json")


def _make_completion(args: argparse.Namespace, usage: Usage):
    """The sibling port's completion wiring, calling this module's own import.

    Six lines rather than an import of the neighbour's private helper, because
    `tests/test_example_entrypoints.py` proves a dry-run never crosses an
    external boundary by replacing **this module's** `completion_for` with a
    tripwire. A port whose network call is made through another module's name
    would pass that test without the tripwire ever being in the path.

    ``--thinking`` reaches an Anthropic-shaped endpoint and nothing else, as in
    the sibling: left on its default, a reasoning model can spend the whole
    token budget on hidden thinking and return empty visible content, which this
    port would record as a node that scored -inf.
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


def integrals_domain(
    suite: Suite,
    *,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    eval_budget: int = EVAL_BUDGET,
    problem_seconds: float = PROBLEM_SECONDS,
) -> Domain:
    """This task, in the four terms the ERA search needs."""
    preview = suite_preview(suite)
    return Domain(
        name=("nine hard one-dimensional integrals per problem set, "
              "mean correct significant digits"),
        entrypoint="integrate",
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=INITIAL_PROGRAM,
        initial_summary="scipy.integrate.quad on default settings",
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            eval_budget=eval_budget, problem_seconds=problem_seconds,
            max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, preview=preview, timeout=candidate_timeout,
            eval_budget=eval_budget, problem_seconds=problem_seconds),
        task_prompt=lambda index: (
            f"Integrate held-out problem set {index} to as many correct digits "
            f"as the evaluation budget allows."),
        test_shards=suite.test_range(),
        data_summary={
            "families": len(FAMILIES),
            "problems_per_shard": suite.size(0),
            "scoring_shards": suite.scoring_shards,
            "test_shards": suite.test_shards,
            "eval_budget_per_problem": eval_budget,
            "problem_seconds": problem_seconds,
            "digit_cap": DIGIT_CAP,
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
    parser.add_argument("--shards", type=int, default=8,
                        help="problem sets the search may score against")
    parser.add_argument("--test-shards", type=int, default=4,
                        help="further problem sets the search never sees")
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--candidate-timeout", type=float, default=60.0,
                        help="upstream's Sandbox(timeout_seconds=60), per problem set")
    parser.add_argument("--eval-budget", type=int, default=EVAL_BUDGET,
                        help=("calls to the integrand allowed per problem. This is "
                              "half the task: with no cap the winner is whichever "
                              "program is allowed to spend the most"))
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
    print("Task     : numerical solution of integrals -- "
          f"{len(FAMILIES)} difficulty classes, mean correct digits (cap "
          f"{DIGIT_CAP:.0f}), closed-form references")
    print(f"Evaluator: {sandbox_backend() or 'NO SANDBOX -- this run will fail'} "
          f"isolated, {args.eval_budget} integrand calls and "
          f"{args.problem_seconds:.0f}s per problem")
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
        print("[dry-run] no API, problem set, or sandbox process was accessed.")
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
                          test_shards=args.test_shards)
    print(f"Problems : {suite.size(0)} per set, {args.shards} scored + "
          f"{args.test_shards} held back, seed={args.seed}, files under {suite.root}")
    domain = integrals_domain(
        suite,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        eval_budget=args.eval_budget,
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
        "experiment": "ERA on AgentDescent -- numerical solution of integrals",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "task": domain.name,
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
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
    baseline = run.baseline_test_metrics.get("mean_digits")
    best = run.best_test_metrics.get("mean_digits")
    print(
        f"completed: held-back mean digits {baseline} -> {best}, "
        f"nodes={len(run.tree.nodes)}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls}, output={args.output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
