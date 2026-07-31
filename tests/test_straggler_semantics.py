"""`round_timeout` abandons a straggler. What happens to it afterwards?

Python cannot kill a thread, so an abandoned rollout keeps running, and it will
eventually call `aggregator.ingest` with evidence built against the round it
started in. Two things must hold: the process must still be able to exit, and
that late evidence must be judged stale rather than silently applied.
"""
import threading
import time
import warnings

from agentdescent.evolution import SingleSlot, Task, evolve

TASKS = [Task(id=str(i), prompt=str(i), meta={"gold": str(i)}) for i in range(6)]


def test_an_abandoned_straggler_does_not_hold_the_interpreter_open():
    """A ThreadPoolExecutor joins its workers at exit, so `shutdown(wait=False)`
    bounded the round but not the program: a rollout wedged for 600s returned from
    evolve() and then kept the process alive."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import time, warnings
        from agentdescent.evolution import SingleSlot, Task, evolve
        tasks = [Task(id=str(i), prompt=str(i)) for i in range(6)]
        def run(rendered, task):
            if task.id == "0":
                time.sleep(600)          # wedged rollout
            return "x"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evolve(tasks, lambda t, o: 0.5, run=run, propose=lambda *a: None,
                   strategy=SingleSlot(initial_value="v"), rounds=2, n_workers=3,
                   max_concurrency=3, round_timeout=2.0, held_out_frac=0.5)
        print("done")
    """)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "done" in p.stdout


def test_a_straggler_still_bounds_the_round():
    """The wait is capped even though the rollout is not."""
    def run(rendered, task):
        if task.id == "0":
            time.sleep(30)
        return "x"
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolve(TASKS, lambda t, o: 0.5, run=run, propose=lambda *a: None,
               strategy=SingleSlot(initial_value="v"), rounds=2, n_workers=3,
               max_concurrency=3, round_timeout=1.0, held_out_frac=0.5)
    assert time.time() - t0 < 20, "round_timeout did not bound the round"


def test_late_evidence_from_an_abandoned_round_is_judged_stale():
    """The straggler's diff was built against an older artifact version.

    It must go through the staleness filter like any other late diff, not be
    applied as though it were current.
    """
    from agentdescent.aggregator import Aggregator

    seen = {"stale": 0}
    real_filter = Aggregator._staleness_filter

    def spy(self, artifact, head, cards):
        survivors, discarded = real_filter(self, artifact, head, cards)
        seen["stale"] += len(discarded)
        return survivors, discarded

    release = threading.Event()

    def run(rendered, task):
        if task.id == "0":
            release.wait(3.0)        # abandoned, then finishes into a later round
        return "wrong"

    Aggregator._staleness_filter = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = evolve(TASKS, lambda t, o: 0.5, run=run,
                         propose=lambda rendered, t, o, s: f"rule-{t.id}",
                         strategy=SingleSlot(initial_value="v"), rounds=4,
                         n_workers=3, max_concurrency=3, round_timeout=0.5,
                         held_out_frac=0.5)
    finally:
        Aggregator._staleness_filter = real_filter
    # The run survives the straggler; the filter is the thing that saw the cards.
    assert res.error is None
