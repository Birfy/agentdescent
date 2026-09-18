"""A/B: DifficultyWeighted vs ReplaySampler at equal budget.

    python -m bench.ab_replay --budget-rollouts 96 --seeds 0,1,2 \\
        --domain stackvm --yes

Both arms share the same workload, strategy and seed. The only difference is
``task_sampler=DifficultyWeighted`` vs ``task_sampler=ReplaySampler(temperature=0.3)``.
"""

from __future__ import annotations
import sys
from agentdescent.evolution import Task
from agentdescent.baselines import Budget, Workload, compare


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-rollouts", type=int, default=96)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    budget = Budget(rollouts=args.budget_rollouts)

    # Use the deterministic comment domain (offline, no model).
    from agentdescent.strategies import AppendRules
    from agentdescent.evolution import reflector, LLMAgent
    from agentdescent.agents import echo

    tasks = [Task(id=f"t{i}", prompt=f"p{i}") for i in range(16)]

    def reward(task, out):
        return 1.0 if out and len(out) > 3 else 0.0

    agent = LLMAgent(echo())
    workload = Workload(
        tasks=tasks, reward=reward, test_eval=lambda r: r.final_reward or 0.0,
        agent=agent, strategy=AppendRules(),
        evolve_kwargs=dict(rounds=10_000))

    from agentdescent.sampling import DifficultyWeighted, ReplaySampler
    kwargs_control = dict(task_sampler=DifficultyWeighted())
    kwargs_experiment = dict(task_sampler=ReplaySampler(temperature=0.3))

    print(f"\nA/B (replay): {args.budget_rollouts} max rollouts, "
          f"{len(seeds)} seeds, {args.workers} workers\n")
    print(f"  control    : DifficultyWeighted")
    print(f"  experiment : ReplaySampler(temperature=0.3)")
    print()

    results = compare(
        [workload],
        {"control": kwargs_control, "replay": kwargs_experiment},
        budget=budget, seeds=seeds, n_workers=args.workers,
    )
    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())