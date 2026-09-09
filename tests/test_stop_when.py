"""`evolve(stop_when=)`: the caller's own budget, asked between rounds."""

import warnings


from agentdescent import AppendRules, Task, evolve

TASKS = [Task(id=f"t{i}", prompt="q", meta={"gold": f"t{i}"}) for i in range(8)]
REWARD = lambda task, output: 1.0 if output == task.meta["gold"] else 0.0        # noqa: E731
RUN = lambda rendered, task: task.meta["gold"] if task.id in rendered else "?"    # noqa: E731
PROPOSE = lambda rendered, task, output, score: task.id                           # noqa: E731


def test_stop_when_ends_the_run_between_rounds_and_keeps_commits():
    seen = []

    def stop_when(info):
        seen.append(info.round)
        return info.round >= 1                       # stop after the second round

    r = evolve(TASKS, REWARD, run=RUN, propose=PROPOSE, strategy=AppendRules(),
               rounds=10, n_workers=2, held_out_frac=0.5, seed=0, stop_when=stop_when)
    assert r.stop_reason == "stop_when"
    assert seen == [0, 1] and len(r.history) == 2
    assert r.error is None


def test_stop_when_is_asked_after_on_round_with_the_same_info():
    order = []
    r = evolve(TASKS, REWARD, run=RUN, propose=PROPOSE, strategy=AppendRules(),
               rounds=3, n_workers=2, held_out_frac=0.5, seed=0,
               on_round=lambda i: order.append(("on", i.round)),
               stop_when=lambda i: order.append(("stop", i.round)) or False)
    assert order[:2] == [("on", 0), ("stop", 0)]
    assert r.stop_reason in ("rounds", "target_reward")


def test_a_raising_stop_when_is_reported_not_fatal():
    def boom(info):
        raise RuntimeError("budget service down")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = evolve(TASKS, REWARD, run=RUN, propose=PROPOSE, strategy=AppendRules(),
                   rounds=2, n_workers=2, held_out_frac=0.5, seed=0, stop_when=boom)
    assert r.error is None and r.stop_reason != "stop_when"
    assert any("stop_when callback raised" in str(x.message) for x in w)


def test_built_in_stops_take_precedence():
    r = evolve(TASKS, REWARD, run=RUN, propose=PROPOSE, strategy=AppendRules(),
               rounds=10, n_workers=4, held_out_frac=0.5, seed=0, target_reward=0.0,
               stop_when=lambda i: True)
    assert r.stop_reason == "target_reward"


def test_stop_when_reaches_the_async_path():
    r = evolve(TASKS, REWARD, run=RUN, propose=PROPOSE, strategy=AppendRules(),
               asynchronous=True, max_seconds=20, max_rollouts=64, n_workers=2,
               held_out_frac=0.5, seed=0, stop_when=lambda i: True)
    assert r.stop_reason == "stop_when"
    assert r.error is None
