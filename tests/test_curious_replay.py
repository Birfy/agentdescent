"""Curious replay has to be right where the past becomes a weighted training set.

A replayed example carries a curiosity weight -- high while the artifact is
still learning it, low once mastered -- and the evolution spends its budget
where it is curious. The fixtures pin the surfaces that make this honest: the
curiosity weight rises with uncertainty and falls on mastery, replayed examples
accumulate rolling scores, and the run improves more on the still-curious past
than on the mastered one.
"""

import re

import pytest

from agentdescent import (
    CuriousReplayReport,
    ReplayExample,
    SingleSlot,
    curiosity_weight,
    curious_replay_evolve,
)


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, example):
    # the artifact (a cap) solves a replayed task when the cap covers it
    return 1.0 if example.task <= _cap(rendered) else 0.0


def _propose(rendered, task, output, reward):
    if reward < 0.9:
        return f"cap:{_cap(rendered) + 1}"
    return rendered


def _spec():
    return SingleSlot(initial_value="cap:0")


def _propose_for(examples):
    target = max(e.task for e in examples)
    def propose(rendered, task, output, reward):
        if reward >= 0.999:
            return rendered
        return f"cap:{target}"      # jump to the hardest difficulty
    return propose


def test_curiosity_rises_with_uncertainty_and_falls_on_mastery():
    curious = ReplayExample(task=8, scores=[0.0, 1.0])       # still learning
    mastered = ReplayExample(task=1, scores=[1.0, 1.0, 1.0]) # solved
    assert curiosity_weight(curious) > curiosity_weight(mastered)
    assert curiosity_weight(mastered) == pytest.approx(0.05)  # floor
    assert mastered.mastered and not curious.mastered


def test_replay_examples_accumulate_rolling_scores():
    e = ReplayExample(task=5)
    e.record(0.0)
    e.record(1.0)
    assert e.mean == pytest.approx(0.5)
    assert e.uncertainty == pytest.approx(0.5)               # pstdev of [0,1]


def test_curious_replay_evolve_improves_the_curious_past():
    # enough distinct examples for a meaningful held-out split (the engine needs
    # ~4 held-out), and one task per example so the split never duplicates an
    # example into both sides -- the old duplicated-task scheme leaked
    mastered = [ReplayExample(task=d, scores=[1.0, 1.0]) for d in (1, 2, 3, 4)]
    curious = [ReplayExample(task=d, scores=[0.0, 0.0]) for d in (5, 6, 7, 8, 9)]
    report, result = curious_replay_evolve(
        mastered + curious, spec=_spec(), run=_run,
        propose=_propose_for(mastered + curious),
        seeds=(0, 1), rounds=9, n_workers=1, repo_path=None)
    assert isinstance(report, CuriousReplayReport)
    cap = _cap(report.value)
    assert cap >= 6                              # reached the hard (curious) tasks
    assert report.curious_gain > report.mastered_gain
    assert result.meta["curious_gain"] == pytest.approx(report.curious_gain)


def test_cohorts_and_baselines_are_snapshotted_before_the_run():
    # the run records every candidate rollout into the examples; the cohorts and
    # baselines must come from the pre-run state, or the seed's own failing
    # rollouts on a mastered example would flip it into the curious cohort and
    # the mastered_gain would silently read 0 (empty cohort)
    mastered = [ReplayExample(task=d, scores=[1.0, 1.0]) for d in (1, 2)]
    curious = [ReplayExample(task=d, scores=[0.0, 0.0]) for d in (4, 5)]

    def noop(rendered, task, output, reward):
        return rendered

    report, result = curious_replay_evolve(
        mastered + curious, spec=_spec(), run=_run, propose=noop,
        seeds=(0, 1), rounds=1, n_workers=1, repo_path=None)
    assert report.weights == {"0": pytest.approx(0.05), "1": pytest.approx(0.05),
                              "2": pytest.approx(1.0), "3": pytest.approx(1.0)}
    # the mastered cohort is non-empty and its gain uses the pre-run baseline
    assert report.mastered_gain == pytest.approx(-1.0)   # cap:0 fails tasks 1, 2
    assert report.curious_gain == pytest.approx(0.0)


def test_both_cohorts_land_on_both_sides_of_the_split():
    # evolve's positional split holds out the tail; without interleaving the
    # mastered-first ordering would leave the held-out set curious-only, so a
    # candidate that forgets the mastered past is never rejected. Interleaving
    # puts both cohorts in the held-out tail.
    from agentdescent.curious_replay import _interleave
    from agentdescent.evolution import held_out_size
    mastered_idx = [0, 1, 2, 3]            # examples 0..3 are mastered
    curious_idx = [4, 5, 6, 7, 8]
    order = _interleave(curious_idx, mastered_idx)
    cut = len(order) - held_out_size(len(order), 0.4)
    held_out = set(order[cut:])
    assert any(i in held_out for i in mastered_idx)
    assert any(i in held_out for i in curious_idx)


def test_a_forgetting_candidate_is_rejected():
    # a set-valued artifact solves exactly the listed task ids; a candidate
    # that learns the curious past and drops the mastered one must be rejected
    # by the gate -- the mastered held-out examples weigh as much as the
    # curious ones, so forgetting them is not cheap
    def set_run(rendered, example):
        solved = {int(x) for x in rendered.split(":")[1].split(",")}
        return 1.0 if example.task in solved else 0.0

    def set_spec():
        return SingleSlot(initial_value="set:1,2,3,4")

    def forgetting(rendered, task, output, reward):
        return "set:5,6,7,8,9"     # learns the curious past, forgets mastered

    mastered = [ReplayExample(task=d, scores=[1.0, 1.0]) for d in (1, 2, 3, 4)]
    curious = [ReplayExample(task=d, scores=[0.0, 0.0]) for d in (5, 6, 7, 8, 9)]
    report, result = curious_replay_evolve(
        mastered + curious, spec=set_spec(), run=set_run, propose=forgetting,
        seeds=(0, 1), rounds=9, n_workers=1, repo_path=None)
    assert report.mastered_gain == 0.0             # the mastered past is kept
    assert "set:1,2,3,4" in report.value           # forgetting was rejected
    assert "set:5,6,7,8,9" not in report.value


def test_curious_replay_evolve_needs_a_proposer():
    with pytest.raises(ValueError):
        curious_replay_evolve([ReplayExample(task=1)], spec=_spec(),
                              run=_run)
