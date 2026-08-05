"""The control arms, and the guards that keep them a control.

These tests do not check that merging wins. Whether it does is an empirical
question and the answer is allowed to be no -- that is stated in the module
docstring and it is the reason the module exists. What is checked here is that
the comparison is *capable* of coming out either way: that the arms share one
workload, spend one budget, and report the spend they actually incurred.

The workload below is deliberately **not** the router domain. There each rule is
an independent row, so two workers' diffs cannot contradict and merging wins by
construction; a test that passed only there would be testing the domain. `Slot`
holds one answer at a time, so two workers proposing different rules genuinely
compete and a merge can be worse than either.
"""

import pytest

from agentdescent.baselines import (
    ArmResult, Budget, Workload, best_of_n_fork, compare, merge_of_n, serial,
    to_markdown,
)
from agentdescent.evolution import EvolutionResult, KeyedRules, Task


# -- a workload where diffs can genuinely contradict -------------------------

KEYS = ("alpha", "beta", "gamma", "delta")


def _tasks(n=24):
    return [Task(id=str(i), prompt=f"q{i}",
                 meta={"key": KEYS[i % len(KEYS)], "gold": f"A{i % len(KEYS)}"})
            for i in range(n)]


def _run(rendered, task):
    """Read the answer `KeyedRules` stored for this task's category, if any.

    `KeyedRules` renders `## <category>` followed by its one current value, so a
    second worker proposing a different value for the same category replaces the
    first rather than adding to it -- which is the property this file needs.
    """
    lines = rendered.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"## {task.meta['key']}" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return "?"


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _propose(rendered, task, output, score):
    return f"{task.meta['key']}: {task.meta['gold']}"


def _test_eval(result: EvolutionResult) -> float:
    """A third split, held out of the run entirely.

    Ids beyond the ones `evolve()` ever saw, so this cannot be the gate's own
    number under another name.
    """
    held = [Task(id=f"t{i}", prompt=f"q{i}",
                 meta={"key": KEYS[i % len(KEYS)], "gold": f"A{i % len(KEYS)}"})
            for i in range(100, 112)]
    return sum(_reward(t, _run(result.rendered, t)) for t in held) / len(held)


def _workload(**extra) -> Workload:
    kwargs = {"held_out_frac": 0.4, "rounds": 10_000, "max_concurrency": 1}
    kwargs.update(extra)
    return Workload(tasks=_tasks(), reward=_reward, test_eval=_test_eval,
                    run=_run, propose=_propose,
                    strategy=KeyedRules(categories=KEYS), evolve_kwargs=kwargs)


BUDGET = Budget(rollouts=48)


# -- the budget --------------------------------------------------------------


def test_budget_splits_both_units():
    b = Budget(rollouts=100, calls=250).split(4)
    assert b == Budget(rollouts=25, calls=62)


def test_splitting_a_call_free_budget_stays_call_free():
    assert Budget(rollouts=100).split(4) == Budget(rollouts=25, calls=None)


def test_a_budget_cannot_be_split_zero_ways():
    with pytest.raises(ValueError):
        Budget(rollouts=10).split(0)


# -- the arms ----------------------------------------------------------------


def test_the_three_arms_spend_the_same_rollouts():
    """Fork's B/N x N must land on merge's B, not on B x N."""
    workload = _workload()
    arms = [serial(workload, budget=BUDGET),
            merge_of_n(workload, 4, budget=BUDGET),
            best_of_n_fork(workload, 4, budget=BUDGET)]
    spends = {a.arm: a.rollouts for a in arms}
    assert len(set(spends.values())) == 1, spends


def test_matching_rollouts_does_not_match_model_spend():
    """Measured, and the reason `Budget` has two fields rather than one.

    Four forks that never talk to each other each start from nothing, so almost
    every rollout of theirs fails and asks for a proposal. Merge-of-4 shares what
    the others already learned, so more of its rollouts solve their task outright
    and never call the proposer. At *identical* rollouts the fork arm here spends
    well over twice the model -- and a table matched only on rollouts would have
    called that equal budget.
    """
    workload = _workload()
    fork = best_of_n_fork(workload, 4, budget=BUDGET)
    merge = merge_of_n(workload, 4, budget=BUDGET)
    assert fork.rollouts == merge.rollouts
    assert fork.calls > merge.calls * 1.5, (fork.calls, merge.calls)

    comparison = compare([fork, merge], fixed="rollouts", tolerance=0.2)
    assert not comparison.unequal, "rollouts were held fixed and did hold"
    assert comparison.confounded, "the call gap has to be reported, not hidden"
    assert "Confound" in to_markdown(comparison)


def test_fixing_calls_instead_moves_the_divergence_rather_than_removing_it():
    """The other half of the finding: the two units trade off, they do not both
    settle. Fixing calls starves the fork arm of rollouts instead."""
    both = Budget(rollouts=10_000, calls=96)
    workload = _workload()
    fork = best_of_n_fork(workload, 4, budget=both)
    merge = merge_of_n(workload, 4, budget=both)
    assert fork.rollouts < merge.rollouts * 0.6, (fork.rollouts, merge.rollouts)

    comparison = compare([fork, merge], fixed="calls", tolerance=0.2)
    assert not comparison.unequal
    assert any(unit == "rollouts" for _, unit, _, _ in comparison.confounded)


def test_the_fixed_unit_must_be_one_of_the_two():
    with pytest.raises(ValueError):
        compare([], fixed="tokens")


def test_an_arm_given_twice_the_budget_is_refused_not_averaged():
    """A table that says 'equal budget' in its caption is not a check."""
    workload = _workload()
    arms = [serial(workload, budget=BUDGET),
            merge_of_n(workload, 4, budget=Budget(rollouts=BUDGET.rollouts * 4))]
    comparison = compare(arms, tolerance=0.2)
    assert comparison.unequal, "doubling one arm's budget must be caught"
    assert "Not an equal-budget comparison" in to_markdown(comparison)


def test_fork_reports_the_selection_it_can_actually_ship():
    """Oracle is an upper bound; selected is the product. Reporting one is a lie
    in one direction or the other."""
    arm = best_of_n_fork(_workload(), 4, budget=BUDGET)
    assert arm.test_oracle is not None
    assert arm.test_oracle >= arm.test_reward, \
        "the best fork on test cannot be worse than the one picked on dev"
    assert len(arm.forks) == 4
    assert arm.test_reward in {f.test_reward for f in arm.forks}


def test_forks_are_not_the_same_run_four_times():
    arm = best_of_n_fork(_workload(), 4, budget=BUDGET)
    assert len({f.seed for f in arm.forks}) == 4


def test_fork_wall_clock_is_reported_both_ways():
    """Summed is what this call took; max is what N machines would take. A
    speedup computed from the wrong one is off by N."""
    arm = best_of_n_fork(_workload(), 4, budget=BUDGET)
    assert arm.wallclock >= arm.wallclock_parallel


def test_single_run_arms_have_one_wall_clock():
    arm = merge_of_n(_workload(), 4, budget=BUDGET)
    assert arm.wallclock == arm.wallclock_parallel


def test_spend_is_measured_not_requested():
    """The engine checks its budget at the round barrier and overshoots. An arm
    that reported the number it asked for would hide exactly that."""
    arm = merge_of_n(_workload(), 4, budget=BUDGET)
    assert arm.rollouts >= BUDGET.rollouts
    assert arm.stop_reason == "max_rollouts"


def test_arms_are_refused_below_one_worker():
    with pytest.raises(ValueError):
        merge_of_n(_workload(), 0, budget=BUDGET)
    with pytest.raises(ValueError):
        best_of_n_fork(_workload(), 0, budget=BUDGET)


# -- reading the result ------------------------------------------------------


def _arm(name, values):
    return [ArmResult(arm=name, seed=i, width=1, rollouts=10, calls=10,
                      prompt_tokens=0, completion_tokens=0, wallclock=1.0,
                      wallclock_parallel=1.0, dev_reward=v, test_reward=v)
            for i, v in enumerate(values)]


def test_separates_requires_no_overlap():
    """The weakest claim a handful of seeds can support, and it must be able to
    say no."""
    clear = compare(_arm("a", [0.8, 0.9, 0.85]) + _arm("b", [0.5, 0.6, 0.55]))
    assert clear.separates("a", "b")

    overlapping = compare(_arm("a", [0.8, 0.6, 0.7]) + _arm("b", [0.5, 0.75, 0.6]))
    assert not overlapping.separates("a", "b"), \
        "overlapping spreads must not be reported as a difference"


def test_one_seed_can_never_separate_anything():
    """With one seed per arm, "no overlap" degenerates to a > b.

    Written without this guard first, and a one-seed pilot duly printed "above on
    every seed" from a single pair of numbers -- the exact overclaim this module
    exists to prevent, made by the thing meant to prevent it.
    """
    one = compare(_arm("a", [0.9]) + _arm("b", [0.1]))
    assert not one.separates("a", "b")
    assert one.underpowered("a", "b")


def test_underpowered_is_distinct_from_finding_nothing():
    """They read the same in a table and mean opposite things about what to do."""
    enough = compare(_arm("a", [0.8, 0.6, 0.7]) + _arm("b", [0.5, 0.75, 0.6]))
    assert not enough.underpowered("a", "b")
    assert not enough.separates("a", "b"), "looked, found no difference"


def test_markdown_flags_a_partial_run_rather_than_averaging_it():
    broken = ArmResult(arm="merge-of-4", seed=0, width=4, rollouts=10, calls=10,
                       prompt_tokens=0, completion_tokens=0, wallclock=1.0,
                       wallclock_parallel=1.0, dev_reward=0.5, test_reward=0.5,
                       error="APIError: 429")
    assert "ended on a failure" in to_markdown(compare([broken]))


def test_markdown_explains_the_oracle_column_when_it_has_one():
    arm = best_of_n_fork(_workload(), 2, budget=Budget(rollouts=24))
    table = to_markdown(compare([arm]))
    assert "fork oracle" in table
    assert "requires the answer to select" in table
