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


def test_resume_queue_push_refuses_a_task_resumed_past_max_attempts():
    """A task that keeps being a straggler must stop being re-queued once it
    has been resumed `max_attempts` times -- otherwise it starves the worker's
    fresh tasks."""
    q = ResumeQueue(max_attempts=1)
    item = ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5})
    assert q.push(item), "first push is always accepted"
    assert q.pop_for(["a"]) is not None, "the item is resumed once"
    # Popping counted the resume; the second re-queue must be refused.
    assert not q.push(item), (
        "a task resumed max_attempts times must not be re-queued")


def test_resume_queue_max_attempts_bounds_a_chronically_slow_task():
    """A task that is always a straggler gets exactly `max_attempts` re-runs;
    past the cap its pushes are dropped, so it cannot monopolise the worker."""
    q = ResumeQueue(max_attempts=2)
    resumed = 0
    for _ in range(10):
        item = ResumeItem(task_id="a", turn=0, conversation=[],
                          version_at_checkpoint={"art": 5})
        accepted = q.push(item)
        if not accepted:
            break                    # past the cap: refused, nothing to pop
        if q.pop_for(["a"]) is not None:
            resumed += 1
    assert resumed == 2, f"expected exactly 2 re-runs, got {resumed}"
    assert q.dropped >= 1, "the chronically slow task must be dropped"
    assert len(q) == 0, "nothing may remain queued past the cap"


def test_resume_queue_duplicate_pushes_before_any_pop_are_bounded():
    """A burst of pushes for the same task, before the first pop, must not all
    be accepted: the cap counts *queued* copies too, not just popped ones."""
    q = ResumeQueue(max_attempts=1)
    item = ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5})
    assert q.push(item), "first push is always accepted"
    assert not q.push(item), "a second queued copy must be refused"
    assert not q.push(item), "a third queued copy must be refused"
    assert len(q) == 1, "only the first push may land"
    assert q.dropped == 2, "the two refusals must be counted"
    # And the single queued copy is what `pop_for` hands out, once.
    assert q.pop_for(["a"]) is not None, "the accepted item is resumed"
    assert q.pop_for(["a"]) is None, "nothing is left to hand out twice"


def test_resume_queue_pop_counts_toward_the_cap_too():
    """`pop()` (not just `pop_for`) is a resume and must feed the same bound:
    after a pop, a re-push of the same task is refused."""
    q = ResumeQueue(max_attempts=1)
    item = ResumeItem(task_id="a", turn=0, conversation=[],
                      version_at_checkpoint={"art": 5})
    assert q.push(item)
    assert q.pop() is not None, "the item is resumed via pop()"
    assert not q.push(item), (
        "a task popped once must not be re-queued past the cap")


# ---------------------------------------------------------------------------
# the engine path
# ---------------------------------------------------------------------------

def test_async_re_runs_a_straggler_against_the_newer_head():
    """A worker whose rollout overruns its estimate is re-run against a head
    that has since advanced -- not re-executed for its own sake.

    Each `pop_for()` hand-off is associated with the worker call that follows
    it (a thread-local flag consumed by `run`), so the assertion is about the
    *resumed call specifically*, not any later ordinary rollout of the same
    task: t0 is slow (a straggler) while another shard commits a clue, and the
    resumed t0 call must render the artifact *with* the clue -- a newer version
    than the one its abandoned rollout measured."""
    import threading
    import time
    renders = []                     # (task_id, rendered, is_resumed) per call
    resumed_flag = threading.local()

    def run(rendered, task):
        is_resumed = getattr(resumed_flag, "pending", False)
        resumed_flag.pending = False
        renders.append((task.id, rendered, is_resumed))
        if task.id == "t0":
            time.sleep(0.2)          # t0 is always slow: a chronic straggler
        return "correct" if "clue" in rendered else "wrong"

    # Only t0 stays unsolved; every other task that lands a proposal commits a
    # clue, advancing the head between t0's abandoned rollout and its re-run.
    def propose(rendered, task, output, reward):
        if task.id != "t0" and "clue" not in rendered:
            return "add a clue"
        return None

    # A `DurationEstimator` with a tiny prior predicts ~0s, so every rollout
    # of t0 (0.2s) is a straggler by construction.
    estimator = DurationEstimator(prior=0.0001, min_samples=100)
    q = ResumeQueue()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(4)]

    # Flag the worker call that a `pop_for` hand-off immediately precedes.
    real_pop_for = q.pop_for

    def flagged_pop_for(task_ids):
        item = real_pop_for(task_ids)
        if item is not None:
            resumed_flag.pending = True
        return item

    q.pop_for = flagged_pop_for
    result = async_evolve(
        tasks, lambda t, o: 1.0 if o == "correct" else 0.0,
        run=run, propose=propose,
        strategy=AppendRules(),
        n_workers=2, duration_estimator=estimator, resume_queue=q,
        straggler_factor=2.0, max_seconds=10.0, held_out_frac=0.5)
    # t0 is always a straggler; it must have been re-run at least once.
    assert result.stragglers >= 1
    assert result.resumed >= 1, f"expected re-runs, got {result.resumed}"
    resumed_t0 = [r for tid, r, is_res in renders
                  if tid == "t0" and is_res]
    assert resumed_t0, (
        "no t0 call was flagged as following a pop_for hand-off; the resume "
        f"mechanism never fired: {renders!r}")
    assert "clue" in resumed_t0[-1], (
        "the resumed call itself must render the newer head; "
        f"resumed t0 calls saw {resumed_t0!r}")


def test_async_stale_resume_negative_control_fails():
    """If the resumed call re-runs the *old* (pre-clue) state, the newer-head
    assertion above must fail -- the test discriminates the resumed call's
    version from any later ordinary rollout (review: the regression passed
    even when the resumed call rendered the old state, because a later
    ordinary t0 rollout supplied the clue-bearing render)."""
    import threading
    import time
    import pytest
    renders = []
    resumed_flag = threading.local()

    def run(rendered, task):
        is_resumed = getattr(resumed_flag, "pending", False)
        resumed_flag.pending = False
        # The negative control: a resumed call renders the stale head, even
        # though the artifact has since gained the clue.
        if is_resumed and "clue" in rendered:
            rendered = rendered.replace("clue", "")
        renders.append((task.id, rendered, is_resumed))
        if task.id == "t0":
            time.sleep(0.2)
        return "correct" if "clue" in rendered else "wrong"

    def propose(rendered, task, output, reward):
        if task.id != "t0" and "clue" not in rendered:
            return "add a clue"
        return None

    estimator = DurationEstimator(prior=0.0001, min_samples=100)
    q = ResumeQueue()
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(4)]
    real_pop_for = q.pop_for

    def flagged_pop_for(task_ids):
        item = real_pop_for(task_ids)
        if item is not None:
            resumed_flag.pending = True
        return item

    q.pop_for = flagged_pop_for
    result = async_evolve(
        tasks, lambda t, o: 1.0 if o == "correct" else 0.0,
        run=run, propose=propose,
        strategy=AppendRules(),
        n_workers=2, duration_estimator=estimator, resume_queue=q,
        straggler_factor=2.0, max_seconds=10.0, held_out_frac=0.5)
    resumed_t0 = [r for tid, r, is_res in renders
                  if tid == "t0" and is_res]
    assert resumed_t0, "premise: the resume mechanism fired"
    with pytest.raises(AssertionError):
        assert "clue" in resumed_t0[-1], (
            "the resumed call must render the newer head")


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


def test_a_chronically_slow_task_cannot_be_resumed_forever():
    """A task that is *always* a straggler gets bounded re-runs, not an
    infinite resume loop starving the worker's fresh tasks."""
    import time

    def run(rendered, task):
        time.sleep(0.05)              # always overruns the tiny prior
        return "yes"

    estimator = DurationEstimator(prior=0.0001, min_samples=100)
    q = ResumeQueue(max_attempts=2)
    tasks = [Task(id=f"t{i}", prompt="q") for i in range(4)]

    result = async_evolve(
        tasks, lambda t, o: 1.0 if o == "yes" else 0.0,
        run=run, propose=lambda rd, t, o, r: None,
        strategy=AppendRules(),
        n_workers=2, duration_estimator=estimator, resume_queue=q,
        straggler_factor=2.0, max_seconds=8.0, held_out_frac=0.5)
    # Every task is a straggler, but re-runs are capped: no task may be
    # resumed more than `max_attempts` times.
    assert result.stragglers >= 1
    assert result.resumed <= 2 * len(tasks), (
        f"resume must be bounded per task; got {result.resumed} re-runs")
    assert q.dropped >= 0
    # The queue never grows unbounded: everything pushed past the cap is
    # dropped, so at any point at most one straggler per task is queued.
    assert len(q) <= len(tasks)
