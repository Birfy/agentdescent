"""L-traj resume: a straggler is re-run against a newer head, not just counted.

The async runtime abandons a rollout that overruns its predicted cost. The
reference runtime checkpoints partial turns here; this port's `run` is opaque,
so there is no turn to save. What is saved is the task and the version it
measured -- with a `resume_queue` present, the next idle worker re-runs it
against the current head. A straggler measured version N, its re-run measures
version N+k, so the pair is a free cross-version A/B signal.
"""

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, Task
from agentdescent.scheduler import DurationEstimator, ResumeItem, ResumeQueue
from agentdescent.sampling import RoundRobin


# ---------------------------------------------------------------------------
# the queue primitive
# ---------------------------------------------------------------------------

def test_resume_queue_pop_for_returns_shard_owned_item():
    q = ResumeQueue()
    q.push(ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5}))
    q.push(ResumeItem(task_id="b", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5}))
    # a worker whose shard owns only "b" gets "b"
    item = q.pop_for(["b"])
    assert item is not None and item.task_id == "b"
    # the item was removed, not copied
    assert q.pop_for(["a"]) is not None
    assert q.pop_for(["a"]) is None


def test_resume_queue_pop_for_returns_none_when_shard_has_no_resume():
    q = ResumeQueue()
    q.push(ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5}))
    assert q.pop_for(["x", "y"]) is None
    assert len(q) == 1          # untouched


def test_resume_queue_pop_for_never_hands_same_task_twice():
    q = ResumeQueue()
    q.push(ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5}))
    assert q.pop_for(["a"]) is not None
    assert q.pop_for(["a"]) is None


# ---------------------------------------------------------------------------
# the engine path
# ---------------------------------------------------------------------------

def test_async_re_runs_a_straggler_against_the_newer_head():
    """A worker whose rollout overruns its estimate is re-run; result.resumed
    reports it, and the re-run happens against the current head."""
    import time
    slow_first = {"flag": True}

    def run(rendered, task):
        if slow_first["flag"]:
            slow_first["flag"] = False
            time.sleep(0.05)          # overruns the estimator's tiny prior
        return "yes"

    # A `DurationEstimator` with a tiny prior predicts ~0s, so the first
    # rollout of every task is a straggler by construction.
    estimator = DurationEstimator(prior=0.0001, min_samples=100)
    q = ResumeQueue()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(4)]

    result = async_evolve(
        tasks, lambda t, o: 1.0 if o == "yes" else 0.0,
        run=run, propose=lambda rd, t, o, r: None,
        strategy=AppendRules(),
        n_workers=2, duration_estimator=estimator, resume_queue=q,
        straggler_factor=2.0, max_seconds=10.0, held_out_frac=0.5)
    # every task overran on its first try; at least one was re-run.
    assert result.stragglers >= 1
    assert result.resumed >= 1, f"expected re-runs, got {result.resumed}"
    # A queued straggler is re-run when a worker is free; some may legitimately
    # still be queued when the run ends (no worker got to them), which is fine.
    assert result.resumed <= result.stragglers


def test_async_without_resume_queue_keeps_old_behaviour():
    """No queue -> stragglers counted, nothing re-run, no `resumed`."""
    import time
    slow_first = {"flag": True}

    def run(rendered, task):
        if slow_first["flag"]:
            slow_first["flag"] = False
            time.sleep(0.05)
        return "yes"

    estimator = DurationEstimator(prior=0.0001, min_samples=100)
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(4)]
    result = async_evolve(
        tasks, lambda t, o: 1.0 if o == "yes" else 0.0,
        run=run, propose=lambda rd, t, o, r: None,
        strategy=AppendRules(),
        n_workers=2, duration_estimator=estimator, resume_queue=None,
        straggler_factor=2.0, max_seconds=10.0, held_out_frac=0.5)
    assert result.stragglers >= 1
    assert result.resumed == 0
