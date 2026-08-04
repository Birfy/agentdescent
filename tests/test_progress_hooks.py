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


# ---------------------------------------------------------------------------
# What the merge did, per round
# ---------------------------------------------------------------------------


def _contradictory_run(**kw):
    """A workload where workers genuinely disagree about one key."""
    from agentdescent.evolution import KeyedRules

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i % 3)})
             for i in range(16)]
    n = [0]

    def propose(rendered, task, output, score):
        n[0] += 1
        return f"routing: variant {n[0] % 3}"       # same key, competing values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(
            tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
            run=lambda rendered, t: t.meta["gold"] if "variant 1" in rendered else "?",
            propose=propose, strategy=KeyedRules(categories=["routing", "tone"]),
            rounds=6, n_workers=4, max_concurrency=4, held_out_frac=0.5, **kw)


def test_a_round_reports_what_the_merge_did_not_only_its_category():
    """`MergeReport` computes all four and the driver used to discard them, so a
    caller could see *that* nothing committed and never *how* the merge got
    there. The reference runtimes reported them all along."""
    result = _contradictory_run(refresh_interval=2)
    assert result.history

    assert sum(h.considered for h in result.history) > 0, (
        "no round reported any evidence considered")
    assert sum(h.conflicts_dropped for h in result.history) > 0, (
        "workers proposed competing values for one key and nothing counted it")
    # the denominator has to bound the numerator, every round
    for h in result.history:
        assert h.discarded_stale <= h.considered, (
            f"round {h.round}: discarded {h.discarded_stale} of {h.considered}")
        assert h.fused <= h.committed, (
            f"round {h.round}: {h.fused} fused commits but {h.committed} commits")


def test_the_new_round_fields_survive_save_and_load(tmp_path):
    """`load` reads history with `RoundInfo(**h)`, so a field `save` forgets is
    silently zero on the way back."""
    from agentdescent.evolution import EvolutionResult

    result = _contradictory_run(refresh_interval=2)
    path = str(tmp_path / "result.json")
    result.save(path)
    back = EvolutionResult.load(path)

    for field in ("considered", "discarded_stale", "conflicts_dropped", "fused"):
        assert [getattr(h, field) for h in back.history] == \
               [getattr(h, field) for h in result.history], field


def test_both_loops_derive_the_round_the_same_way():
    """`record_round` takes the reports now, so `committed` / `rejected` cannot
    be counted one way in one loop and another way in the other."""
    import inspect

    from agentdescent.evolution import _Engine

    params = inspect.signature(_Engine.record_round).parameters
    assert "reports" in params
    assert "committed" not in params and "rejected" not in params, (
        "a caller can still hand record_round its own tally, which is how the "
        "two loops drifted")
