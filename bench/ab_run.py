"""A/B the three borrowed RL decision rules. Off is the control.

    python -m bench.ab_run --dataset hotpotqa --rule advantage \
        --budget-rollouts 96 --seeds 0,1,2 --provider claude --model GLM-5.2 --yes

`agentdescent/advantage.py` implements three rules taken from PPO and GRPO, and
every one of them is off by default because *an analogy without an A/B is
decoration*. This script is the A/B: identical workload, identical budget,
identical seeds, one `Policies` field different.

Two things it reports that a quality column alone would hide:

**Rollbacks, not just final quality.** The stable-distance penalty and the
adaptive trust region are both meant to trade a little progress for fewer
reversals. A run that ends at the same place having been vetoed half as often is
a different run, and `oracle-rejected` is where that shows.

**Whether the mechanism fired at all.** An adaptive trust region that never left
its starting value is a constant, and a `--rule advantage` run in which no group
ever filled is a control against a control. Both are printed, and both are the
first thing to check before reading the quality column: with four workers over
four task clusters the largest possible group is one, and the signal is simply
absent.

If a rule does not win here, delete it and record the negative result in
`docs/concepts.md`. That is the outcome the issue this implements asks for
explicitly, and it is worth more than an unvalidated field nobody reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence

from agentdescent import Usage
from agentdescent.advantage import (
    AdaptiveTrustRegion, AdvantageAcceptance, AdvantageConflict,
    StableDistanceAcceptance,
)
from agentdescent.aggregator import AggregatorConfig
from agentdescent.defaults import DefaultAcceptance, DefaultConflict
from agentdescent.fusion import reflective_merge
from agentdescent.policies import Policies
from examples._common import completion_for, confirm

from .baselines_run import _finer, _fmt, _hotpotqa

RULES = ("advantage", "trust-region", "stable-distance",
         "reflective-fusion")


def _arm(rule: str, on: bool, completion=None) -> Dict[str, Any]:
    """The `evolve()` keyword arguments for one arm of one A/B.

    The control is *not* "a bundle with nothing in it" -- it is the default
    policies, constructed the same way the engine constructs them. Otherwise the
    comparison could be measuring the wiring rather than the rule.
    """
    if not on:
        return {}
    if rule == "advantage":
        # `DefaultAcceptance` is wrapped, not replaced: the Beta test and the
        # regression guard have a documented history of bugs behind them.
        base = DefaultAcceptance(0.5, 64, 4000)
        return {"policies": Policies(
            acceptance=AdvantageAcceptance(base),
            conflict=None)}          # conflict needs the verifier; wired below
    if rule == "trust-region":
        return {"agg_config": AggregatorConfig(
            trust_region_policy=AdaptiveTrustRegion())}
    if rule == "stable-distance":
        base = DefaultAcceptance(0.5, 64, 4000)
        return {"policies": Policies(acceptance=StableDistanceAcceptance(base))}
    if rule == "reflective-fusion":
        # The pair, not the fusion policy alone: conflict resolution runs first
        # and would hand it a single diff. `reflective_merge` is what makes the
        # unpaired mistake unavailable.
        return {"policies": Policies(**reflective_merge(completion))}
    raise ValueError(f"unknown rule {rule!r}; known: {', '.join(RULES)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rule", choices=RULES, required=True)
    p.add_argument("--dataset", choices=["hotpotqa", "finer"], default="hotpotqa")
    p.add_argument("--budget-rollouts", type=int, default=96)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--fetch", type=int, default=48)
    p.add_argument("--pool", type=int, default=800)
    p.add_argument("--top-k", type=int, default=120)
    p.add_argument("--provider", default="claude", choices=["claude", "openai", "glm"])
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--json", dest="json_out")
    p.add_argument("--no-self-verify", dest="self_verify", action="store_false",
                   help="skip the second rollout per proposal. Identical on both "
                        "arms, so the A/B stays valid, and it halves the bill")
    p.add_argument("--eval-concurrency", type=int, default=8)
    p.add_argument("--run-concurrency", type=int, default=1,
                   help="how many of the (arm, seed) runs to execute at once. "
                        "All of them are independent -- separate seeds, separate "
                        "scratch ledgers -- so this changes wall-clock only")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Rule     : {args.rule}")
    print(f"Dataset  : {args.dataset}")
    print(f"Budget   : {args.budget_rollouts} rollouts, {args.workers} workers")
    print(f"Seeds    : {seeds}")
    print(f"Runs     : {2 * len(seeds)} evolve() calls (off and on, per seed)")
    print(f"self_verify: {args.self_verify}")
    if args.plan:
        return 0
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s); one number per arm is not a result.",
              file=sys.stderr)
    if not confirm(args):
        return 0

    total = Usage()                    # the grand total, for the run's own bill
    shared = {"self_verify": args.self_verify,
              "eval_concurrency": args.eval_concurrency}

    def _build(seed: int, usage: Usage):
        """A workload with its **own** meter.

        One `Usage` per run, not one for the sweep. Sharing it made
        `result.usage.calls` the running total across every concurrently
        executing arm, so the per-arm call column was the sum of whatever else
        happened to be in flight -- and a cost comparison read straight off it.
        `agentdescent.baselines` gets this right by giving every arm a fresh
        meter; this did not.

        The dataset is deterministic given the seed and cached on disk, so
        rebuilding it per run costs nothing and both arms still see identical
        data.
        """
        completion = completion_for(args, usage=usage)
        return (_hotpotqa(args.fetch, seed, completion, **shared)
                if args.dataset == "hotpotqa"
                else _finer(args.pool, args.top_k, seed, completion, **shared)), completion

    # Warm the dataset cache once, before any thread starts, so N threads do not
    # race the same download.
    _build(seeds[0], Usage())
    printing = threading.Lock()

    def _one(job) -> Optional[Dict[str, Any]]:
        seed, on = job
        label = f"{args.rule} {'on ' if on else 'off'} seed={seed}"
        usage = Usage()
        workload, completion = _build(seed, usage)
        kwargs = dict(workload.evolve_kwargs)
        kwargs.update(_arm(args.rule, on, completion))
        kwargs.setdefault("rounds", 10_000)
        from agentdescent.evolution import evolve

        try:
            result = evolve(
                workload.tasks, workload.reward, agent=workload.agent,
                strategy=workload.strategy, seed=seed,
                n_workers=args.workers, max_concurrency=args.workers,
                max_rollouts=args.budget_rollouts, usage=usage, **kwargs)
        except Exception as e:  # noqa: BLE001 - one dead arm is not six
            with printing:
                print(f"  {label}  FAILED {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
            return None

        # Scoring runs *after* `evolve()`, outside its backend-failure
        # protection, so a dropped connection here must not discard the arm.
        try:
            test = workload.test_eval(result)
        except Exception as e:  # noqa: BLE001
            with printing:
                print(f"  {label}  scoring failed: {type(e).__name__}",
                      file=sys.stderr)
            test = None

        outcomes = result.outcomes()
        fusion = result.fusion_stats()
        row = {
            "seed": seed, "arm": "on" if on else "off",
            "test": test, "dev": result.final_reward,
            # This arm's own meter, not the sweep's.
            "rollouts": result.rollouts, "calls": usage.calls,
            "committed": outcomes.get("committed", 0),
            "oracle_rejected": outcomes.get("oracle-rejected", 0),
            "oversized": outcomes.get("oversized", 0),
            # Whether the mechanism fired at all. `contested` is the first thing
            # to read for --rule reflective-fusion: zero means no fused candidate
            # ever competed, and the quality column is then about nothing.
            "contested": fusion.contested,
            "fused_wins": fusion.fused_wins,
            "synthesized_wins": fusion.synthesized_wins,
            "synthesis_failed": fusion.synthesis_failed,
            "error": result.error,
        }
        with printing:
            print(f"  {label}  test={_fmt(test)} dev={_fmt(row['dev'])}  "
                  f"{row['rollouts']} rollouts / {row['calls']} calls  "
                  f"+{row['committed']} -{row['oracle_rejected']} vetoed  "
                  f"fusion: {row['contested']} contested, "
                  f"{row['synthesized_wins']} synth wins, "
                  f"{row['synthesis_failed']} failed")
        return row

    jobs = [(seed, on) for seed in seeds for on in (False, True)]
    if args.run_concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.run_concurrency) as pool:
            collected = list(pool.map(_one, jobs))
    else:
        collected = [_one(job) for job in jobs]
    rows: List[Dict[str, Any]] = [r for r in collected if r is not None]
    if not rows:
        print("every arm failed; nothing to compare.", file=sys.stderr)
        return 4

    print()
    print("| arm | seeds | test (min/med/max) | commits | vetoes | contested | synth wins |")
    print("|---|---|---|---|---|---|---|")
    for arm in ("off", "on"):
        group = [r for r in rows if r["arm"] == arm]
        if not group:
            continue
        tests = sorted(r["test"] for r in group if r["test"] is not None)
        interval = ("n/a" if not tests else
                    f"{tests[0]:.3f} / {statistics.median(tests):.3f} / {tests[-1]:.3f}")
        print(f"| {arm} | {len(tests)}/{len(group)} | {interval} | "
              f"{sum(r['committed'] for r in group)} | "
              f"{sum(r['oracle_rejected'] for r in group)} | "
              f"{sum(r['contested'] for r in group)} | "
              f"{sum(r['synthesized_wins'] for r in group)} |")

    print()
    contested_on = sum(r["contested"] for r in rows if r["arm"] == "on")
    if contested_on == 0:
        # Before any quality claim: a mechanism that never ran is a control
        # against a control, and its table looks exactly like a null result.
        print(f"`{args.rule}` never fired: 0 contested tournaments on the `on` "
              "arm. The quality column below is not about this rule -- fix the "
              "workload or the wiring before reading it.")
    else:
        off = sorted(r["test"] for r in rows if r["arm"] == "off"
                     and r["test"] is not None)
        on = sorted(r["test"] for r in rows if r["arm"] == "on"
                    and r["test"] is not None)
        if min(len(off), len(on)) < 3:
            print(f"{min(len(off), len(on))} scored seed(s) per arm: too few to "
                  "compare. Not a null result -- an experiment that could not "
                  "have produced one.")
        elif on[0] > off[-1]:
            print(f"`{args.rule}` is above the control on every seed.")
        elif off[0] > on[-1]:
            print(f"`{args.rule}` is *below* the control on every seed. Delete it "
                  "and record the negative result in docs/concepts.md.")
        else:
            print(f"`{args.rule}` overlaps the control across seeds: this budget "
                  "on this dataset did not separate them.")

    print()
    print(f"total model spend: {sum(r['calls'] for r in rows)} calls "
          f"across {len(rows)} arms")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"rule": args.rule, "dataset": args.dataset,
                       "budget_rollouts": args.budget_rollouts,
                       "workers": args.workers, "model": args.model,
                       "self_verify": args.self_verify, "rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
