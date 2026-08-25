"""ERA tree search on SciPy entry points a sweep found to be inaccurate.

Where the 2F1 task next door picked its function from the literature, this one
picks by measurement. ``tools/scan_numeric_precision.py`` scores 47 NumPy and
SciPy float64 entry points against mpmath -- each point evaluated at 30 *and*
60 digits and kept only where the two agree -- over parameter ranges declared
before anything was run. The sweep's answer is mostly reassuring: every NumPy
elementary function tested returns 16 correct digits, ``sin`` and ``tan``
included at arguments up to 1e18, and around thirty SciPy entry points sit
above 15 digits. Two do not:

======================  ============  ==============  ==============
target                  mean digits   < 8 digits      < 1 digit
======================  ============  ==============  ==============
scipy.special.pbdv          11.67         17.8%          12.2%
scipy.special.hyperu        14.36          3.0%           2.8%
======================  ============  ==============  ==============

Neither number is a rounding complaint. SciPy's ``pbdv`` returns 4.81e100 at
``v=19.83, x=-29.28`` where the value is 2.46e80, and -2.44e24 at ``v=17.02,
x=-14.61`` where the value is +6.01e15 -- wrong sign, wrong magnitude. SciPy's
``hyperu`` returns ``nan`` on 3% of its declared range, at points such as
``a=-15.82, b=-1.30, x=23.10`` where the function equals 2.45e17 and is
perfectly well-conditioned.

**One function, one tree.** Each ``--function`` is a search of its own: its own
stress set, its own root node, its own flat-PUCT tree and its own result file.
They share the code and nothing else -- no pooled score, no transfer between
them -- because the whole claim being tested is per-function ("can a search
beat SciPy *here*"), and a pooled number would let a large gain on one hide a
regression on the other.

The search itself is `era_empirical_software.py`, unchanged: the flat-PUCT
tree, the visit reservation, the aggregator, the staleness handling, the
governance layer and the sandbox profile.

Run
---
    python -m examples.era.era_special_precision --function pbdv --dry-run
    python -m examples.era.era_special_precision --function hyperu \\
        --provider claude --model glm-5.2 --iterations 12 --workers 3 --yes
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
from examples.era._era_special import (
    DIGIT_CAP,
    SHARD_SECONDS,
    SOLVED_DIGITS,
    TARGETS,
    TARGETS_BY_KEY,
    Suite,
    Target,
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


def default_output(key: str) -> Path:
    return Path(f"era-{key}-result.json")


def _make_completion(args: argparse.Namespace, usage: Usage):
    """This module's own call into ``completion_for``.

    Not an import of a sibling's private helper: the shared contract test proves
    a dry-run never crosses an external boundary by replacing **this module's**
    `completion_for` with a tripwire, and a port whose network call is made
    through another module's name would pass that test with the tripwire outside
    the path.
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


def special_domain(
    target: Target,
    suite: Suite,
    *,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    shard_seconds: float = SHARD_SECONDS,
) -> Domain:
    """This task, in the four terms the ERA search needs."""
    preview = suite_preview(suite)
    return Domain(
        name=target.title,
        entrypoint=suite.entrypoint,
        metric_key="mean_digits",
        metric_better="higher",
        initial_program=target.initial_program,
        initial_summary=f"{target.baseline}, called directly",
        evaluate=lambda code, shard_ids: evaluate_source(
            code, suite=suite, shards=shard_ids, timeout=candidate_timeout,
            shard_seconds=shard_seconds, max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, target=target, suite=suite, preview=preview,
            timeout=candidate_timeout, shard_seconds=shard_seconds),
        task_prompt=lambda index: (
            f"Evaluate held-out point set {index} of the {suite.entrypoint} "
            f"stress suite to as many correct digits as possible."),
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
            "baseline_scan": suite.metadata.get("baseline_scan", {}),
            "suite_seed": suite.metadata.get("seed"),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=1800.0,
                      eval_concurrency_default=None)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--function", default="pbdv", choices=sorted(TARGETS_BY_KEY),
                        help="which SciPy entry point to evolve; one tree per call")
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
    parser.add_argument("--output", type=Path,
                        help="default: era-<function>-result.json")
    parser.add_argument("--list-functions", action="store_true",
                        help="print the available targets and exit")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_functions:
        for target in TARGETS:
            print(f"{target.key:<10} baseline {target.baseline:<24} {target.title}")
        return 0
    target = TARGETS_BY_KEY[args.function]
    output = args.output or default_output(target.key)
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
    print(f"Task     : {target.title}")
    print(f"Baseline : {target.baseline}, called directly")
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
    suite = load_suite(target.key, shards=args.shards, test_shards=args.test_shards)
    reference = suite.metadata.get("reference", {})
    print(f"Points   : {suite.size(0)} per set, {args.shards} scored + "
          f"{args.test_shards} held back; reference {reference.get('library')} "
          f"{reference.get('version')} at {reference.get('precisions_dps')} dps")
    domain = special_domain(
        target,
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
        eval_concurrency=args.eval_concurrency,
        domain=domain,
        verbose=True,
    )
    payload: Dict[str, Any] = {
        "experiment": f"ERA on AgentDescent -- {target.baseline} precision",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "function": target.key,
        "task": domain.name,
        "baseline": target.baseline,
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes", "list_functions")
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
    _write_json(output, payload)
    best_path = output.with_name(f"{output.stem}-best.py")
    best_path.write_text(run.tree.best().program.code.rstrip() + "\n", encoding="utf-8")
    baseline = run.baseline_test_metrics
    best = run.best_test_metrics
    print(
        f"completed: held-back mean digits {baseline.get('mean_digits')} -> "
        f"{best.get('mean_digits')}, solved {baseline.get('solved')}/"
        f"{baseline.get('points')} -> {best.get('solved')}/{best.get('points')}, "
        f"nodes={len(run.tree.nodes)}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls} "
        f"({damage.get('damaged', 0)} damaged in transit), output={output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
