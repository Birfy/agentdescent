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
from typing import Any, Dict, List, Optional, Sequence

from agentdescent import Usage
from agentdescent.advantage import (
    AdaptiveTrustRegion, AdvantageAcceptance, AdvantageConflict,
    StableDistanceAcceptance,
)
from agentdescent.aggregator import AggregatorConfig
from agentdescent.defaults import DefaultAcceptance, DefaultConflict
from agentdescent.policies import Policies
from examples._common import completion_for, confirm

from .baselines_run import _finer, _hotpotqa

RULES = ("advantage", "trust-region", "stable-distance")


def _arm(rule: str, on: bool) -> Dict[str, Any]:
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
    print(f"Runs     : {2 * len(seeds)} evolve() calls (off, then on, per seed)")
    if args.plan:
        return 0
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s); one number per arm is not a result.",
              file=sys.stderr)
    if not confirm(args):
        return 0

    usage = Usage()
    completion = completion_for(args, usage=usage)
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        if args.dataset == "hotpotqa":
            workload = _hotpotqa(args.fetch, seed, completion)
        else:
            workload = _finer(args.pool, args.top_k, seed, completion)

        for on in (False, True):
            kwargs = dict(workload.evolve_kwargs)
            kwargs.update(_arm(args.rule, on))
            kwargs.setdefault("rounds", 10_000)
            from agentdescent.evolution import evolve

            result = evolve(
                workload.tasks, workload.reward, agent=workload.agent,
                strategy=workload.strategy, seed=seed,
                n_workers=args.workers, max_concurrency=args.workers,
                max_rollouts=args.budget_rollouts, usage=usage, **kwargs)
            outcomes = result.outcomes()
            rows.append({
                "seed": seed, "arm": "on" if on else "off",
                "test": workload.test_eval(result),
                "dev": result.final_reward,
                "rollouts": result.rollouts, "calls": result.usage.calls,
                "committed": outcomes.get("committed", 0),
                "oracle_rejected": outcomes.get("oracle-rejected", 0),
                "oversized": outcomes.get("oversized", 0),
                "error": result.error,
            })
            print(f"  {args.rule} {'on ' if on else 'off'} seed={seed}  "
                  f"test={rows[-1]['test']:.3f} dev={rows[-1]['dev']:.3f}  "
                  f"{rows[-1]['rollouts']} rollouts / {rows[-1]['calls']} calls  "
                  f"+{rows[-1]['committed']} "
                  f"-{rows[-1]['oracle_rejected']} vetoed")

    print()
    print("| arm | seeds | test (min/med/max) | commits | oracle vetoes | oversized |")
    print("|---|---|---|---|---|---|")
    for arm in ("off", "on"):
        group = [r for r in rows if r["arm"] == arm]
        tests = sorted(r["test"] for r in group)
        print(f"| {arm} | {len(group)} | "
              f"{tests[0]:.3f} / {statistics.median(tests):.3f} / {tests[-1]:.3f} | "
              f"{sum(r['committed'] for r in group)} | "
              f"{sum(r['oracle_rejected'] for r in group)} | "
              f"{sum(r['oversized'] for r in group)} |")

    off = sorted(r["test"] for r in rows if r["arm"] == "off")
    on = sorted(r["test"] for r in rows if r["arm"] == "on")
    print()
    if on[0] > off[-1]:
        print(f"`{args.rule}` is above the control on every seed.")
    elif off[0] > on[-1]:
        print(f"`{args.rule}` is *below* the control on every seed. Delete it and "
              "record the negative result in docs/concepts.md.")
    else:
        print(f"`{args.rule}` overlaps the control across seeds: this budget on "
              "this dataset did not separate them. An unvalidated mechanism is "
              "not a mechanism -- either widen the experiment or drop the rule.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"rule": args.rule, "dataset": args.dataset,
                       "budget_rollouts": args.budget_rollouts,
                       "workers": args.workers, "model": args.model,
                       "rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
