from agentdescent.domains.router import make_task_universe
from agentdescent.orchestrator import AgentDescent, run_fork_baseline


def test_loop_improves_accuracy(tmp_path):
    universe = make_task_universe(seed=7)
    system = AgentDescent(str(tmp_path / "repo"), universe, n_workers=6,
                       noise=0.1, seed=1)
    history = system.run(rounds=40)
    assert history[0].dev_accuracy < history[-1].dev_accuracy
    assert system.final_accuracy() > 0.7


def test_merge_beats_fork(tmp_path):
    # RQ1: at equal worker budget, merging complementary diffs beats the best
    # single fork, because no fork sees every keyword.
    universe = make_task_universe(seed=7)
    system = AgentDescent(str(tmp_path / "repo"), universe, n_workers=6,
                       noise=0.1, seed=1)
    system.run(rounds=40)
    merge_acc = system.final_accuracy()
    fork_acc = run_fork_baseline(universe, n_workers=6, noise=0.1, rounds=40, seed=1)
    assert merge_acc > fork_acc


def test_staleness_is_exercised(tmp_path):
    # with lagging snapshots, some diffs must be rebased or discarded.
    universe = make_task_universe(seed=7)
    system = AgentDescent(str(tmp_path / "repo"), universe, n_workers=6,
                       noise=0.1, refresh_interval=3, seed=1)
    history = system.run(rounds=30)
    assert sum(h.committed for h in history) > 0
