"""ERA tree search on double-precision evaluation of 2F1 -- a task nobody has run.

Not a benchmark with a leaderboard, and deliberately so. The evidence that a run
here means something comes from two places instead:

**The reference is independent of everything being scored.** Every point's value
was computed by mpmath at 30 *and* 60 decimal digits, and points where the two
disagreed above 25 digits were thrown away. mpmath shares no code with SciPy and
none with any candidate, nothing in the sandbox can reach it, and the stress set
with its references is a committed file that anyone can re-derive with
``python -m tools.gen_hyp2f1_stress --check``.

**The baseline is the state of the practice.** Not a strawman written for this
example: ``scipy.special.hyp2f1``, Cephes underneath, the function every
scientist already calls. On the declared distribution it loses more than six
digits on roughly a third of the points -- and the failures are algorithmic, not
a float64 wall, because a textbook Pfaff transformation chosen from
``(a, b, c, z)`` alone recovers a large share of them.

That the problem is hard is not this repository's claim. The standard survey of
the area -- Pearson, Olver & Porter, *Numerical methods for the computation of
the confluent and Gauss hypergeometric functions*, Numerical Algorithms
74:821-866 (2017) -- exists because no single method is reliable across the
parameter space: the Taylor series diverges outside the unit disc, every
transformation has a bad region of its own, the recurrences are unstable in one
direction, and the useful identities cancel.

The search itself is `era_empirical_software.py`, unchanged: the flat-PUCT tree,
the visit reservation, the aggregator, the staleness handling, the governance
layer and the sandbox profile. This module supplies a
:class:`~examples.era._era_domain.Domain` and the command line the other ports
share.

Run
---
    python -m examples.era.era_hypergeometric --dry-run
    python -m examples.era.era_hypergeometric --provider claude --model glm-5.2 \\
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
from examples.era._era_hyp2f1 import (
    DIGIT_CAP,
    INITIAL_PROGRAM,
    SHARD_SECONDS,
    SOLVED_DIGITS,
    Suite,
    evaluate_source,
    framework_score,
    load_suite,
    mutation_prompt,
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
DEFAULT_OUTPUT = Path("era-hyp2f1-result.json")


def _make_completion(args: argparse.Namespace, usage: Usage):
    """This module's own call into ``completion_for``.

    Not an import of the sibling's private helper: the shared contract test
    proves a dry-run never crosses an external boundary by replacing **this
    module's** `completion_for` with a tripwire, and a port whose network call
    is made through another module's name would pass that test with the
    tripwire outside the path.
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


def hypergeometric_domain(
    suite: Suite,
    *,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    shard_seconds: float = SHARD_SECONDS,
) -> Domain:
    """This task, in the four terms the ERA search needs."""
    preview = suite_preview(suite)
    return Domain(
        name=("Gauss hypergeometric 2F1(a, b; c; z) in double precision, "
              "mean correct significant digits against a 25-digit reference"),
        entrypoint="hyp2f1",
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=INITIAL_PROGRAM,
        initial_summary="scipy.special.hyp2f1, called directly",
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            shard_seconds=shard_seconds, max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, preview=preview, timeout=candidate_timeout,
            shard_seconds=shard_seconds),
        task_prompt=lambda index: (
            f"Evaluate held-out point set {index} of the 2F1 stress suite to as "
            f"many correct digits as possible."),
        test_shards=suite.test_range(),
        data_summary={
            "points_per_shard": suite.size(0),
            "scoring_shards": suite.scoring_shards,
            "test_shards": suite.test_shards,
            "digit_cap": DIGIT_CAP,
            "solved_digits": SOLVED_DIGITS,
            "shard_seconds": shard_seconds,
            "reference": suite.metadata.get("reference", {}),
            "distribution": suite.metadata.get("distribution", {}),
            "suite_seed": suite.metadata.get("seed"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=1800.0)
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
                        help="point sets the search may score against")
    parser.add_argument("--test-shards", type=int, default=4,
                        help="further point sets the search never sees")
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--candidate-timeout", type=float, default=60.0,
                        help="upstream's Sandbox(timeout_seconds=60), per point set")
    parser.add_argument("--shard-seconds", type=float, default=SHARD_SECONDS,
                        help="wall-clock for one point set, inside --candidate-timeout")
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
    print("Task     : Gauss hypergeometric 2F1 in double precision -- mean correct "
          f"digits (cap {DIGIT_CAP:.0f}) against mpmath at 25 digits")
    print("Baseline : scipy.special.hyp2f1, called directly")
    print(f"Evaluator: {sandbox_backend() or 'NO SANDBOX -- this run will fail'} "
          f"isolated, {args.shard_seconds:.0f}s per point set")
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
        print("[dry-run] no API, stress set, or sandbox process was accessed.")
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
    suite = load_suite(shards=args.shards, test_shards=args.test_shards)
    reference = suite.metadata.get("reference", {})
    print(f"Points   : {suite.size(0)} per set, {args.shards} scored + "
          f"{args.test_shards} held back; reference {reference.get('library')} "
          f"{reference.get('version')} at {reference.get('precisions_dps')} dps")
    domain = hypergeometric_domain(
        suite,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        shard_seconds=args.shard_seconds,
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
        domain=domain,
        verbose=True,
    )
    payload: Dict[str, Any] = {
        "experiment": "ERA on AgentDescent -- Gauss hypergeometric 2F1",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "task": domain.name,
        "baseline": "scipy.special.hyp2f1",
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "observation": run.summary(args.quality_target),
        "model_usage": _usage_dict(model_usage),
        # What the channel cost, kept beside the result rather than in prose: a
        # reply that arrived damaged is not a program the search evaluated.
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
        f"completed: held-back mean digits {baseline.get('mean_digits')} -> "
        f"{best.get('mean_digits')}, solved {baseline.get('solved')}/"
        f"{baseline.get('points')} -> {best.get('solved')}/{best.get('points')}, "
        f"nodes={len(run.tree.nodes)}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls} "
        f"({damage.get('damaged', 0)} damaged in transit), output={args.output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
