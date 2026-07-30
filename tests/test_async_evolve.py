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


class _FlakyAgent:
    """Fails the first ``n_fail`` rollouts, then behaves like _Composer."""

    def __init__(self, n_fail=2):
        self.n_fail, self.calls = n_fail, 0

    def solve(self, rendered, task):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("transient backend failure")
        return "yes" if task.meta["hint"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


class _DeadAgent:
    """Every rollout fails -- simulates credit exhaustion / a dead endpoint."""

    def solve(self, rendered, task):
        raise RuntimeError("backend is down")

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def test_async_survives_transient_backend_errors():
    """A few transient failures must NOT kill the run (they are retried)."""
    r = async_evolve(_tasks(), REWARD, agent=_FlakyAgent(n_fail=2),
                     strategy=AppendRules(), n_workers=2, async_ratio=3,
                     max_seconds=8.0, target_reward=1.0, held_out_frac=0.5)
    assert r.final_reward > 0.0            # recovered and still made progress
    assert len(r.history) >= 1


def test_async_reports_persistent_backend_failure():
    """A dead backend ends the run, but the reason is reported, not swallowed."""
    r = async_evolve(_tasks(), REWARD, agent=_DeadAgent(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=20.0, held_out_frac=0.5)
    assert r.error is not None                  # the failure is surfaced ...
    assert "backend is down" in r.error         # ... with the real cause
    assert r.state == {}                        # nothing bogus was committed


def test_clean_run_reports_no_error():
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=2, async_ratio=3, max_seconds=6.0,
                     target_reward=1.0, held_out_frac=0.5)
    assert r.error is None
