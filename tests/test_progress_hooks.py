"""Progress reporting for long runs.

An LLM run can take hours; before `on_round` the engine reported nothing at all
until it returned (RoundInfo was only appended to a list, and `verbose` printing
is not something a library caller can route).
"""

import warnings

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, RoundInfo, Task, evolve


class _Agent:
    def solve(self, rendered, task):
        return "yes" if task.meta["h"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["h"]


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt="q", meta={"h": f"H{i % 3}"}) for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def test_on_round_fires_every_round_in_order():
    seen = []
    evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
           rounds=4, on_round=seen.append)
    assert [i.round for i in seen] == [0, 1, 2, 3]
    assert all(isinstance(i, RoundInfo) for i in seen)


def test_on_round_sees_live_progress_not_just_the_end():
    """The callback must receive each round as it happens, with usable fields."""
    seen = []
    res = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                 rounds=5, on_round=lambda i: seen.append((i.round, i.held_out_reward)))
    assert len(seen) == len(res.history)
    assert [r for r, _ in seen] == [h.round for h in res.history]


def test_on_round_exception_does_not_abort_the_run():
    def boom(info):
        raise ZeroDivisionError("callback bug")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                     rounds=3, on_round=boom)
    assert res.error is None                       # the run itself was fine
    assert len(res.history) == 3
    assert any("on_round" in str(x.message) for x in w)


def test_async_on_round_fires_per_sweep():
    seen = []
    async_evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                 n_workers=2, max_seconds=3.0, on_round=seen.append)
    assert seen, "expected at least one merger sweep to report"
    assert all(isinstance(i, RoundInfo) for i in seen)


def test_evolve_forwards_on_round_to_the_async_path():
    seen = []
    evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
           asynchronous=True, n_workers=2, max_seconds=3.0, rounds=3,
           on_round=seen.append)
    assert seen, "on_round must survive the delegation to async_evolve"


def test_a_callback_that_raises_on_the_last_round_is_still_reported():
    """The one case that used to be silent.

    `evolve`'s target-reached path swallowed the exception with a comment saying
    the normal path would report it -- and that path `break`s immediately after,
    so nothing ever did. A callback failing on the final round of a *successful*
    run was the single place the user heard nothing."""
    import warnings as _warnings

    from agentdescent import AppendRules, Task, evolve

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(6)]

    def boom(info):
        raise RuntimeError("callback exploded")

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        evolve(tasks, lambda t, o: 1.0,          # everything scores 1.0
               run=lambda r, t: t.meta["gold"], propose=lambda r, t, o, s: None,
               strategy=AppendRules(), rounds=5, n_workers=1, held_out_frac=0.5,
               target_reward=0.5, on_round=boom, seed=0)
    messages = [str(w.message) for w in caught if "callback exploded" in str(w.message)]
    assert messages, "the callback failed on the target round and nothing said so"
