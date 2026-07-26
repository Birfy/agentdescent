"""Tests for the barrier-free async runtime (agentdescent.async_evolve).

Deterministic no-network stub agents; assertions avoid brittle timing by checking
monotonicity (the aggregator never regresses the head) and that the pipeline runs
and commits, rather than exact wall-clock outcomes.
"""

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, Task, evolve


class _Composer:
    """Composes complementary hints: correct once the hint is in the artifact."""

    def solve(self, rendered, task):
        return "yes" if task.meta["hint"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def _tasks(n=18):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"target": "yes", "hint": f"H{i % 3}"})
            for i in range(n)]


REWARD = lambda t, o: 1.0 if o.strip().lower() == "yes" else 0.0


def test_async_evolve_runs_barrier_free_and_improves():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=4, async_ratio=3, max_seconds=8.0, target_reward=1.0,
                     held_out_frac=0.5)
    assert len(r.history) >= 1                       # the merger swept at least once
    assert r.final_reward >= r.history[0].held_out_reward   # head never regresses
    assert r.final_reward >= 0.66                     # composed >= 2 of the 3 hints


def test_async_evolve_tolerates_high_staleness():
    # a large lag budget -> workers propose against stale snapshots; the staleness
    # policy must rebase/discard so the run still completes without error.
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=4, async_ratio=12, max_seconds=8.0, target_reward=1.0,
                     held_out_frac=0.5)
    assert r.final_reward >= r.history[0].held_out_reward
    assert 0.0 <= r.final_reward <= 1.0


def test_async_evolve_stops_at_max_iters():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=30.0, max_iters=6,
                     held_out_frac=0.5)
    # capped early -> few rules committed, but the pipeline produced a valid result.
    assert len(r.state) <= 3
    assert r.final_reward >= 0.0


def test_evolve_asynchronous_flag_delegates_to_async():
    r = evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
               asynchronous=True, n_workers=4, rounds=12, async_ratio=3, max_seconds=8.0,
               held_out_frac=0.5)
    assert r.final_reward >= r.history[0].held_out_reward


def test_async_evolve_works_with_custom_pareto_aggregator():
    from examples.gepa_prompt_evolution import InstructionSlot, pareto_aggregator_factory

    class _Gepa:
        def solve(self, rendered, task):
            return "yes" if task.meta["hint"] in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return (rendered + " " + task.meta["hint"]).strip()

    factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=1)
    async_evolve(_tasks(), REWARD, agent=_Gepa(), strategy=InstructionSlot(),
                 initial_state={"instruction": "start"}, artifact_id="gepa_prompt",
                 n_workers=4, async_ratio=5, max_seconds=8.0, target_reward=1.0,
                 held_out_frac=0.5, aggregator_factory=factory)
    agg = factory.holder["agg"]
    seed_avg = sum(agg.scores[0]) / len(agg.scores[0])
    assert agg.best_avg >= seed_avg                  # illumination held up under async
