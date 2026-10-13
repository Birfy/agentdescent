"""A/B the value-directed budget allocator. Off is the control.

    python -m bench.ab_allocator --dataset hotpotqa --budget-rollouts 96 \
        --seeds 0,1,2 --provider claude --model GLM-5.2 --yes

`agentdescent/allocator.py` replaces `BudgetGovernor`'s clock-based degradation
(once `spent / max_tokens` crosses a floor, *every* candidate stops getting the
expensive self-verify rollout) with a per-candidate decision learned from the
run's own merge history. Every one of its knobs is off by default because *an
analogy without an A/B is decoration*; this script is that A/B.

Identical workload, identical budget, identical seeds, one argument different:

    off   allocator=None            -- the clock degrades optional spend
    on    allocator=ValueBudgetAllocator()  -- learned per-candidate spend

Two things it reports that a quality column alone would hide:

**Whether the allocator fired at all.** An allocator that never skipped a
candidate (observed > 0 but skipped == 0 and counterfactual == 0) is a control
against a control: it only ever agreed with the clock, and its token column
says nothing about the mechanism. Both are printed and are the first thing to
read before the quality column.

**Tokens, not just quality.** The allocator's promise is *spending the scarce
resource on the right candidates*, which shows up as equal-or-better quality at
fewer tokens (or better quality at the same tokens). The `tokens` column is the
measured spend, not the requested budget, so a run that overshoots the token
cap is visible rather than silently counted as within it.

If the allocator does not win here, delete it and record the negative result in
`docs/budget.md`. That is the outcome this mechanism's own docstring asks for
explicitly, and it is worth more than an unvalidated field nobody reads.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentdescent import Usage
from agentdescent.allocator import ValueBudgetAllocator
from agentdescent.evolution import evolve

from examples._common import completion_for, confirm

from .baselines_run import _finer, _fmt, _hotpotqa


def _arm(on: bool, alloc_kwargs: Optional[Dict[str, Any]] = None,
         seed: int = 0) -> Tuple[Dict[str, Any], Optional[ValueBudgetAllocator]]:
    """The `evolve()` keyword arguments for one arm, and the thing that watched it.

    The control is *not* "an empty allocator" -- it is no allocator at all, the
    engine's default. Otherwise the comparison could be measuring the wiring
    rather than the mechanism. The second return value is the allocator itself
    for the `on` arm (it records every decision and observation) and `None` for
    the control. ``alloc_kwargs`` are passed to the allocator constructor (e.g.
    ``min_observations`` / ``budget_start`` from the command line); ``seed``
    seeds the allocator's counterfactual-exploration RNG so a run's exploration
    is reproducible and isolated from other concurrently running seeds.
    """
    if not on:
        return {}, None
    alloc = ValueBudgetAllocator(seed=seed, **(alloc_kwargs or {}))
    return {"allocator": alloc}, alloc


def _fired_counts(watcher) -> Tuple[int, int]:
    """``(fired, observed)``: whether the allocator ever *changed the spend*.

    `observed` is every merge decision it saw; `fired` counts only decisions
    where the allocator actually *differed* from the clock -- a real skip. A
    counterfactual exploration is a re-measurement of a candidate the allocator
    judged unworthy, and it deep-evals just like the clock would, so it is
    *not* a spend change: counting it as fired would let an allocator that
    never skipped anything pass the control-vs-control safeguard.
    Counterfactuals are reported separately via ``watcher.n_counterfactual``.
    """
    if watcher is None:
        return 0, 0
    return watcher.n_skipped, watcher.n_observed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["hotpotqa", "finer"], default="hotpotqa")
    p.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                   help="HotpotQA data source: the HuggingFace datasets-server "
                        "('hf', default) or the ModelScope mirror ('modelscope') "
                        "for networks where one is reachable and the other is not")
    p.add_argument("--budget-rollouts", type=int, default=96)
    p.add_argument("--min-observations", type=int, default=None,
                   help="allocator `min_observations` (merge decisions before it "
                        "may skip). Lower than the default 5 for workloads whose "
                        "whole run has few merges, so the A/B can actually see "
                        "the allocator act")
    p.add_argument("--budget-start", type=float, default=None,
                   help="allocator `budget_start` -- the remaining fraction below "
                        "which the allocator starts skipping low-value candidates. "
                        "Default 0.5")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--fetch", type=int, default=48)
    p.add_argument("--pool", type=int, default=800)
    p.add_argument("--top-k", type=int, default=120)
    p.add_argument("--provider", default="claude",
                  choices=["claude", "openai", "glm"])
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="token ceiling for both arms (the allocator's own knob "
                        "is `budget_start`; this is the ceiling it spends). "
                        "None means unbounded -- the allocator stays inert "
                        "below its budget pressure and the arms are then the "
                        "same run, so set it to make the A/B mean something.")
    p.add_argument("--json", dest="json_out")
    p.add_argument("--no-self-verify", dest="self_verify", action="store_false",
                   help="skip the second rollout per proposal. Identical on both "
                        "arms, so the A/B stays valid, and it halves the bill")
    p.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                   help="ask an Anthropic-shaped endpoint for no reasoning "
                        "tokens. Applied to *both* arms or they are not "
                        "comparable")
    p.add_argument("--eval-concurrency", type=int, default=8)
    p.add_argument("--run-concurrency", type=int, default=1,
                   help="how many of the (arm, seed) runs to execute at once")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--yes", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Mechanism: value-directed budget allocator")
    print(f"Dataset  : {args.dataset}")
    print(f"Budget   : {args.budget_rollouts} rollouts, {args.workers} workers"
          + (f", max_tokens={args.max_tokens:,}" if args.max_tokens else ""))
    print(f"Seeds    : {seeds}")
    print(f"Runs     : {2 * len(seeds)} evolve() calls (off and on, per seed)")
    print(f"self_verify: {args.self_verify}")
    if args.max_tokens is None:
        print("warning: --max-tokens unset; the allocator stays inert below its "
              "budget pressure, so the two arms are the same run. Set it to "
              "make the A/B mean something.", file=sys.stderr)
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s); one number per arm is not a result.",
              file=sys.stderr)
    if args.plan:
        return 0
    if not confirm(args):
        return 0

    total = Usage()
    shared = {"self_verify": args.self_verify,
              "eval_concurrency": args.eval_concurrency}
    # Optional allocator configuration from the command line (so a workload
    # whose whole run has few merges can still exercise the skip decision).
    alloc_kwargs = {}
    if args.min_observations is not None:
        alloc_kwargs["min_observations"] = args.min_observations
    if args.budget_start is not None:
        alloc_kwargs["budget_start"] = args.budget_start

    def _build(seed: int, usage: Usage):
        completion = completion_for(args, usage=usage)
        return (_hotpotqa(args.fetch, seed, completion, source=args.source, **shared)
                if args.dataset == "hotpotqa"
                else _finer(args.pool, args.top_k, seed, completion, **shared))

    # Warm the dataset cache once, before any thread starts.
    _build(seeds[0], Usage())
    printing = threading.Lock()

    def _one(job) -> Optional[Dict[str, Any]]:
        seed, on = job
        label = f"allocator {'on ' if on else 'off'} seed={seed}"
        usage = Usage()
        try:
            workload = _build(seed, usage)
        except Exception as e:  # noqa: BLE001 - data load failure is one arm's
            # problem, not six
            with printing:
                print(f"  {label}  BUILD FAILED {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
            return {"seed": seed, "arm": "on" if on else "off",
                    "test": None, "dev": None, "rollouts": 0, "calls": 0,
                    "tokens": 0, "committed": 0, "oracle_rejected": 0,
                    "fired": None, "fired_of": None,
                    "skipped": None, "counterfactual": None,
                    "error": f"build failed: {type(e).__name__}: {str(e)[:120]}"}
        kwargs = dict(workload.evolve_kwargs)
        arm_kwargs, watcher = _arm(on, alloc_kwargs, seed=seed)
        kwargs.update(arm_kwargs)
        kwargs.setdefault("rounds", 10_000)

        try:
            result = evolve(
                workload.tasks, workload.reward, agent=workload.agent,
                strategy=workload.strategy, seed=seed,
                n_workers=args.workers, max_concurrency=args.workers,
                max_rollouts=args.budget_rollouts,
                max_tokens=args.max_tokens, usage=usage, **kwargs)
        except Exception as e:  # noqa: BLE001 - one dead arm is not six
            with printing:
                print(f"  {label}  FAILED {type(e).__name__}: {str(e)[:120]}",
                      file=sys.stderr)
            # Keep the arm's accrued usage and mark it failed, so the summary
            # does not silently drop it and compare different surviving seeds
            # on the two arms.
            tokens = usage.prompt_tokens + usage.completion_tokens
            return {"seed": seed, "arm": "on" if on else "off",
                    "test": None, "dev": None, "rollouts": 0,
                    "calls": usage.calls, "tokens": tokens,
                    "committed": 0, "oracle_rejected": 0,
                    "fired": None, "fired_of": None,
                    "skipped": None, "counterfactual": None,
                    "error": f"evolve failed: {type(e).__name__}: {str(e)[:120]}"}

        scoring_error = None
        try:
            test = workload.test_eval(result)
            if test is None:
                scoring_error = "final scoring returned no score"
        except Exception as e:  # noqa: BLE001
            with printing:
                print(f"  {label}  scoring failed: {type(e).__name__}",
                      file=sys.stderr)
            test = None
            scoring_error = (f"final scoring failed: {type(e).__name__}: "
                             f"{str(e)[:120]}")

        if scoring_error == "final scoring returned no score":
            with printing:
                print(f"  {label}  scoring failed: {scoring_error}",
                      file=sys.stderr)

        outcomes = result.outcomes()
        fired, observed = _fired_counts(watcher)
        tokens = usage.prompt_tokens + usage.completion_tokens
        errors = [str(result.error)] if result.error else []
        if scoring_error:
            errors.append(scoring_error)
        row = {
            "seed": seed, "arm": "on" if on else "off",
            "test": test, "dev": result.final_reward,
            "rollouts": result.rollouts, "calls": usage.calls,
            "tokens": tokens,
            "committed": outcomes.get("committed", 0),
            "oracle_rejected": outcomes.get("oracle-rejected", 0),
            "fired": fired, "fired_of": observed,
            "skipped": watcher.n_skipped if watcher else 0,
            "counterfactual": watcher.n_counterfactual if watcher else 0,
            "error": "; ".join(errors) or None,
            "scoring_error": scoring_error,
        }
        with printing:
            print(f"  {label}  test={_fmt(test)} dev={_fmt(row['dev'])}  "
                  f"{row['rollouts']} rollouts / {row['calls']} calls / "
                  f"{tokens:,} tokens  +{row['committed']} "
                  f"-{row['oracle_rejected']} vetoed")
            if watcher:
                print(f"           allocator: {observed} observed, "
                      f"{watcher.n_skipped} skipped, "
                      f"{watcher.n_counterfactual} counterfactual")
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

    # A paired comparison requires both arms to have a run with no error and a
    # final test score. Keep failed runs and their partner's spend in `rows`,
    # but use exactly the measurable pairs for both quality and token summaries.
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    failed = [r for r in rows if r["error"]]
    complete_seeds = {
        seed for seed, pair in by_seed.items()
        if all(arm in pair and not pair[arm]["error"]
               and pair[arm]["test"] is not None for arm in ("off", "on"))
    }
    dropped = sorted(set(by_seed) - complete_seeds)
    failed_tokens = sum(r["tokens"] for r in failed)
    print()
    print(f"arms: {len(rows)} (2 x {len(seeds)} seeds), "
          f"{len(failed)} failed (spent {failed_tokens:,} tokens before "
          f"failing)")
    if dropped:
        print(f"incomplete seed pairs dropped from quality/cost comparison: "
              f"{dropped}")
        excluded_rows = [r for r in rows if r["seed"] not in complete_seeds]
        excluded_failed_tokens = sum(r["tokens"] for r in excluded_rows
                                     if r["error"])
        excluded_partner_tokens = sum(r["tokens"] for r in excluded_rows
                                      if not r["error"])
        print("spend excluded from paired quality/cost but retained: "
              f"{excluded_failed_tokens + excluded_partner_tokens:,} tokens "
              f"({excluded_failed_tokens:,} in failed runs; "
              f"{excluded_partner_tokens:,} in non-failed runs)")
    print()
    if complete_seeds:
        print("| arm | complete seeds | test (min/med/max) | tokens (med) | "
              "commits | vetoes | fired |")
        print("|---|---|---|---|---|---|---|")
        for arm in ("off", "on"):
            group = [by_seed[seed][arm] for seed in sorted(complete_seeds)]
            tests = sorted(r["test"] for r in group)
            interval = (f"{tests[0]:.3f} / {statistics.median(tests):.3f} / "
                        f"{tests[-1]:.3f}")
            med_tokens = statistics.median(r["tokens"] for r in group)
            print(f"| {arm} | {len(group)} | {interval} | {med_tokens:,.0f} | "
                  f"{sum(r['committed'] for r in group)} | "
                  f"{sum(r['oracle_rejected'] for r in group)} | "
                  f"{sum(r['fired'] or 0 for r in group)}/"
                  f"{sum(r['fired_of'] or 0 for r in group)} |")
    else:
        print("No complete successfully scored seed pairs; paired quality/cost "
              "comparison unavailable.")

    fired_on = sum(r["fired"] or 0 for r in rows if r["arm"] == "on")
    observed_on = sum(r["fired_of"] or 0 for r in rows if r["arm"] == "on")
    if complete_seeds and observed_on and fired_on == 0:
        # Before any quality claim: a mechanism that never differed from the
        # clock is a control against a control.
        print(f"\nallocator never fired: it agreed with the clock on all "
              f"{observed_on} decisions (0 skipped). The quality and token "
              "columns above are not about the mechanism -- check that "
              "--max-tokens is tight enough for the allocator's budget_start, "
              "and that self-verify is on (the allocator gates the self-verify "
              "rollout; without it there is nothing to gate).")
    elif complete_seeds and observed_on == 0:
        print(f"\nallocator saw no merge decisions on the `on` arm "
              f"({observed_on} observed). The arms ran different workloads or "
              "nothing was ever evaluated -- the quality column is about "
              "nothing.")
    elif observed_on:
        print(f"\nallocator observed {observed_on} decision(s), but no complete "
              "successfully scored seed pairs survived; no paired result is "
              "available.")

    if args.json_out:
        import json
        with open(args.json_out, "w") as fh:
            json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    return 0 if complete_seeds else 4


if __name__ == "__main__":
    raise SystemExit(main())
