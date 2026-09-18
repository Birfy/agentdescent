"""A/B: DifficultyWeighted vs ReplaySampler at equal budget.

    python -m bench.ab_replay --budget-rollouts 96 --seeds 0,1,2 \\
        --workers 4 --yes

Both arms share the same workload, strategy and seed. The only difference is
``task_sampler=DifficultyWeighted`` vs ``task_sampler=ReplaySampler(temperature=0.3)``.
The hypothesis: ReplaySampler recovers some of the rollouts wasted on stale/
conflicted tasks by steering the search away from them.
"""

from __future__ import annotations
import sys
import threading

from agentdescent.agents import Usage
from agentdescent.baselines import Budget, Workload, compare
from agentdescent.evolution import Task
from agentdescent.strategies import AppendRules
from agentdescent.agents import echo
from agentdescent.evolution import LLMAgent


def _workload(task_sampler) -> Workload:
    tasks = [Task(id=f"t{i}", prompt=f"p{i}") for i in range(16)]
    return Workload(
        tasks=tasks,
        reward=lambda task, out: 1.0 if out and len(out) > 3 else 0.0,
        test_eval=lambda r: r.final_reward or 0.0,
        agent=LLMAgent(echo()),
        strategy=AppendRules(),
        evolve_kwargs=dict(rounds=10_000, task_sampler=task_sampler),
    )


def _run_arm(label, sampler, seed, budget, n_workers):
    """Run one arm and build an ArmResult with a distinct name."""
    from agentdescent.baselines import ArmResult
    usage = Usage()
    wl = _workload(sampler)
    result = wl._evolve(seed=seed, n_workers=n_workers, budget=budget, usage=usage)
    from agentdescent.baselines import ArmResult
    return ArmResult(
        arm=label, seed=seed, width=n_workers,
        rollouts=result.usage.get("rollouts", budget.rollouts) if isinstance(result.usage, dict) else budget.rollouts,
        calls=usage.calls,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        wallclock=usage.seconds,
        wallclock_parallel=usage.seconds,
        dev_reward=result.final_reward,
        test_reward=wl.test_eval(result),
        stop_reason=result.stop_reason,
    )


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="A/B: replay sampler")
    parser.add_argument("--budget-rollouts", type=int, default=96)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    budget = Budget(rollouts=args.budget_rollouts)

    from agentdescent.sampling import DifficultyWeighted, ReplaySampler

    print(f"\nA/B (replay): {args.budget_rollouts} max rollouts, "
          f"{len(seeds)} seeds, {args.workers} workers\n")
    print(f"  control    : DifficultyWeighted")
    print(f"  experiment : ReplaySampler(temperature=0.3)")
    print()

    results = []
    printing = threading.Lock()

    def _one(job):
        seed, label, sampler = job
        try:
            arm = _run_arm(label, sampler, seed, budget, args.workers)
        except Exception as e:  # noqa: BLE001
            with printing:
                print(f"  {label} seed={seed}  FAILED {type(e).__name__}: {str(e)[:120]}")
            return None
        with printing:
            print(f"  {label} seed={seed}  reward={arm.test_reward}  "
                  f"rollouts={arm.rollouts}  calls={arm.calls}")
        return arm

    jobs = [(s, "control" if i < len(seeds) else "replay",
             DifficultyWeighted() if i < len(seeds) else ReplaySampler(temperature=0.3))
            for i, s in enumerate(seeds * 2)]
    # interleave control and replay per seed
    jobs = []
    for s in seeds:
        jobs.append((s, "control", DifficultyWeighted()))
        jobs.append((s, "replay", ReplaySampler(temperature=0.3)))

    for job in jobs:
        r = _one(job)
        if r is not None:
            results.append(r)

    if len(results) >= 2:
        comp = compare(results)
        print(f"\n{comp}\n")
    else:
        print("\nNot enough results to compare.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())