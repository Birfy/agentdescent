"""Run the equal-budget arms on a real dataset and print the table.

    python -m bench.baselines_run --dataset hotpotqa --budget-rollouts 96 \
        --width 4 --seeds 0,1,2 --provider openai --model GLM-5.2 --yes

Three arms -- `serial`, `best_of_n_fork`, `merge_of_n` -- over one
:class:`~agentdescent.baselines.Workload`, so only the execution shape varies.
See `agentdescent/baselines.py` for why this exists: every efficiency number in
this repository is a throughput speedup, and throughput cannot distinguish
merging from sampling-and-selecting because fork-and-select is parallel too.

**The default aggregator, deliberately.** The datasets come from the GEPA and ACE
ports, but their custom optimizers do not: GEPA's Pareto selection and ACE's
grow-and-refine are *search* strategies, and running them here would leave the
comparison unable to say whether a difference came from merging or from the
search. The thing under test is the merge, so the merge is the only thing that
changes. That also means the numbers here are not comparable with those ports'
own results, and are not meant to be.

**The third split.** `evolve()` optimises against its own held-out split and stops
when that split says so, and the fork arm *selects* on it -- so reporting it
would be reporting a training score twice over. `test_eval` scores `ds.test`,
which no gate in any arm ever sees.

Cost is the reason the defaults are small. Three arms x three seeds is nine runs,
and the fork arm is N runs by itself, so `--width 4 --seeds 0,1,2` is 33 runs of
`--budget-rollouts` rollouts each. Print `--plan` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, List, Optional, Sequence

from agentdescent import Usage
from agentdescent.baselines import (
    ArmResult, Budget, Workload, best_of_n_fork, compare, merge_of_n, serial,
    to_markdown,
)
from agentdescent.evolution import EvolutionResult
from examples._common import completion_for, confirm


def _hotpotqa(fetch: int, seed: int, completion) -> Workload:
    """GEPA's dataset and actor; the engine's own optimizer."""
    from examples.gepa import gepa_prompt_evolution as gepa

    ds = gepa.load_dataset(fetch, seed=seed)
    reward = gepa.make_reward()
    agent = gepa.gepa_agent(completion)

    def test_eval(result: EvolutionResult) -> float:
        return gepa.evaluate(agent, result.state.get("instruction", ""), ds.test,
                             reward)

    return Workload(
        tasks=ds.trainval, reward=reward, test_eval=test_eval, agent=agent,
        strategy=gepa.InstructionSlot(),
        evolve_kwargs={
            "initial_state": {"instruction": gepa._SEED_INSTRUCTION},
            "artifact_id": "gepa_prompt", "blast_radius": 0.2,
            "held_out_frac": ds.val_frac, "rounds": 10_000,
        })


def _finer(pool: int, top_k: int, seed: int, completion) -> Workload:
    """ACE's dataset and actor; the engine's own optimizer."""
    from examples.ace import ace_context_evolution as ace

    ds = ace.load_dataset(pool, top_k, seed=seed)
    reward = ace.make_reward()
    agent = ace.ace_agent(completion)

    def test_eval(result: EvolutionResult) -> float:
        return ace.evaluate(agent, result.rendered, ds.test, reward)

    return Workload(
        tasks=ds.trainval, reward=reward, test_eval=test_eval, agent=agent,
        strategy=ace.ACEPlaybook(),
        evolve_kwargs={
            "artifact_id": "ace_playbook", "blast_radius": 0.2,
            "held_out_frac": ds.val_frac, "rounds": 10_000,
        })


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["hotpotqa", "finer"], default="hotpotqa")
    p.add_argument("--budget-rollouts", type=int, default=96)
    p.add_argument("--budget-calls", type=int, default=0,
                   help="0 leaves calls unbounded; see --fixed")
    p.add_argument("--fixed", choices=["rollouts", "calls"], default="rollouts",
                   help="which unit the comparison holds fixed. The other one "
                        "diverges and is printed as a confound -- they cannot "
                        "both be equalised")
    p.add_argument("--width", type=int, default=4,
                   help="N, for both merge-of-N and fork-of-N")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated; three is the minimum that shows a spread")
    p.add_argument("--arms", default="serial,fork,merge")
    p.add_argument("--fetch", type=int, default=48, help="hotpotqa rows")
    p.add_argument("--pool", type=int, default=800, help="finer rows to scan")
    p.add_argument("--top-k", type=int, default=120, help="finer concept cutoff")
    p.add_argument("--provider", default="openai", choices=["claude", "openai", "glm"])
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--json", dest="json_out")
    p.add_argument("--plan", action="store_true",
                   help="print how many runs this would be, and stop")
    p.add_argument("--yes", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    budget = Budget(rollouts=args.budget_rollouts,
                    calls=args.budget_calls or None)

    # A fork arm is N runs, so the run count is not len(arms) x len(seeds) and
    # guessing it wrong is how a "small" sweep turns into an afternoon of API.
    runs = len(seeds) * sum(args.width if a == "fork" else 1 for a in arms)
    print(f"Dataset  : {args.dataset}")
    print(f"Arms     : {', '.join(arms)} (N={args.width})")
    print(f"Budget   : {budget.rollouts} rollouts"
          + (f", {budget.calls} calls" if budget.calls else ", calls unbounded")
          + f"; held fixed in {args.fixed}")
    print(f"Seeds    : {seeds}")
    print(f"Runs     : {runs} evolve() calls, "
          f"~{runs * budget.rollouts} rollouts in total")
    print("Optimizer: the engine's default aggregator -- the port's own search "
          "strategy is deliberately not used, so a difference cannot come from it")
    if args.plan:
        return 0
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s). docs/algo-ace.md records one "
              "configuration moving 4.8 points between two runs; one number per "
              "arm is not a result.", file=sys.stderr)
    if not confirm(args):
        return 0

    usage = Usage()
    completion = completion_for(args, usage=usage)

    results: List[ArmResult] = []
    for seed in seeds:
        # Rebuilt per seed so the split moves with it, and once per seed so every
        # arm at that seed sees byte-identical data.
        if args.dataset == "hotpotqa":
            workload = _hotpotqa(args.fetch, seed, completion)
        else:
            workload = _finer(args.pool, args.top_k, seed, completion)

        builders: dict = {
            "serial": lambda w, **k: serial(w, **k),
            "fork": lambda w, **k: best_of_n_fork(w, args.width, **k),
            "merge": lambda w, **k: merge_of_n(w, args.width, **k),
        }
        for name in arms:
            arm = builders[name](workload, budget=budget, seed=seed)
            print(f"  {arm.arm:<12} seed={seed}  {arm.rollouts} rollouts / "
                  f"{arm.calls} calls  dev={arm.dev_reward:.3f} "
                  f"test={arm.test_reward:.3f}"
                  + (f" oracle={arm.test_oracle:.3f}" if arm.test_oracle else "")
                  + (f"  ERROR {arm.error}" if arm.error else ""))
            results.append(arm)

    comparison = compare(results, fixed=args.fixed)
    print()
    print(to_markdown(comparison))
    print()
    print(f"total model spend: {usage.summary()}")

    if "merge" in arms and "fork" in arms:
        merge_name, fork_name = f"merge-of-{args.width}", f"fork-of-{args.width}"
        if comparison.separates(merge_name, fork_name):
            print(f"\n{merge_name} is above {fork_name} on every seed.")
        else:
            print(f"\n{merge_name} and {fork_name} overlap across seeds: this "
                  "budget on this dataset did not distinguish merging from "
                  "selecting. That is the result, and it belongs in "
                  "docs/results.md as one.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"dataset": args.dataset, "width": args.width,
                       "budget": {"rollouts": budget.rollouts,
                                  "calls": budget.calls},
                       "fixed": args.fixed, "seeds": seeds,
                       "model": args.model, "provider": args.provider,
                       "arms": [r.__dict__ for r in results]},
                      fh, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
