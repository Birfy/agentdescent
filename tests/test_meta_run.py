"""The meta-evolution product surface has to be right in the places a two-level
loop is dangerous.

meta_run() evolves a decision slot of the inner search, then judges the result
on problems the outer loop never saw. The fixtures pin: the report carries the
compiled value and a verdict, the validation problems are disjoint from the
training set, a value that helps unseen problems is adopted, one that
regresses is rejected, and the report rides the run result so a CLI or plugin
can show the decision.
"""

import re

import pytest

from agentdescent import (
    MetaRunReport,
    ParamSlot,
    SingleSlot,
    Task,
    auc,
    evolve,
    evolve_problem,
    meta_run,
)
from agentdescent.meta import meta_validate, transfer_ratio
from agentdescent.sampling import DifficultyWeighted


# -- a deterministic capability domain -------------------------------------------


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    if task.meta.get("a", 0) > _cap(rendered):
        return f"cap:{_cap(rendered) + 1}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _sampler_slot():
    # the evolved slot: the UCB exploration constant of DifficultyWeighted
    return ParamSlot(
        factory=lambda c: DifficultyWeighted(c=c),
        params={"c": 0.2},
        bounds={"c": (0.05, 1.0)})


def _inner_problem(tasks, *, rounds=5):
    return evolve_problem(
        tasks, _reward, slot="task_sampler", run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def _deterministic_propose(rendered, task, output, reward):
    """An offline hill-climb on the exploration constant: nudge up on a low
    meta-reward (hard tasks carry the signal), keep it otherwise."""
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(m.group(1)) if m else 0.2
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


# -- the product surface ---------------------------------------------------------


def test_meta_run_runs_and_reports(tmp_path):
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b"))}
    val = {"v": _inner_problem(_tasks(prefix="z"))}
    report, result = meta_run(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        propose=_deterministic_propose, seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1, repo_path=str(tmp_path / "repo"))
    assert isinstance(report, MetaRunReport)
    assert report.slot == "task_sampler"
    assert report.value
    assert report.seed_value
    assert set(report.validation.keys()) == {"v"}        # validation only
    assert isinstance(report.mean_gain, float)
    assert report.verdict in ("adopt", "reject")
    assert report.reason
    # the report rides the result so a CLI can show the decision
    assert result.meta == {
        "slot": report.slot, "seed_value": report.seed_value,
        "value": report.value, "validation": report.validation,
        "mean_gain": report.mean_gain, "transfer": report.transfer,
        "verdict": report.verdict, "reason": report.reason}
    # the value compiles to a usable sampler policy
    policy = _sampler_slot().compile(report.value)
    assert hasattr(policy, "pick")


def test_meta_run_requires_validation_problems():
    with pytest.raises(ValueError):
        meta_run({"t1": _inner_problem(_tasks())}, {}, slot="task_sampler",
                 spec=_sampler_slot(), propose=_deterministic_propose)


def test_the_validation_is_isolated_from_training():
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b")),
             "c": _inner_problem(_tasks(prefix="c")),
             "d": _inner_problem(_tasks(prefix="d"))}
    val = {"e": _inner_problem(_tasks(prefix="e")),
           "f": _inner_problem(_tasks(prefix="f"))}
    report, _ = meta_run(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        propose=_deterministic_propose, seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1)
    # the validation problems are disjoint from the training ones by name
    assert set(report.validation.keys()) == {"e", "f"}
    assert not (set(report.validation.keys()) & set(train.keys()))


def test_adopt_and_reject_verdicts():
    # a value that clearly helps unseen problems is adopted; one that regresses
    # is rejected. Drive the verdict by constructing the report decision path
    # directly through meta_run's helpers on a controlled outcome set.
    from agentdescent.meta import MetaOutcome

    slot = _sampler_slot()
    def helping(value, seed):
        # the evolved value (c > 0.2, nudged up by the offline proposer) beats
        # the seed (c == 0.2)
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.5, 1.0], final=1.0)
        return MetaOutcome(curve=[0.2, 0.3], final=0.3)

    def harming(value, seed):
        # the evolved value is worse than the seed on this unseen problem
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.0], final=0.0)
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)

    # Distinct callables with the same helping behavior: #356 forbids sharing
    # the same Problem object across train and val even when names differ.
    def helping_val(value, seed):
        return helping(value, seed)

    # helping on unseen -> adopt
    report, _ = meta_run(
        {"a": helping, "b": helping}, {"v": helping_val}, slot="task_sampler",
        spec=slot, propose=_deterministic_propose, seeds=(0, 1),
        val_seeds=(10, 11), rounds=1, n_workers=1)
    assert report.verdict == "adopt"
    assert report.mean_gain >= 0.0

    # harming on unseen (regression) -> reject
    report2, _ = meta_run(
        {"a": helping, "b": helping}, {"v": harming}, slot="task_sampler",
        spec=slot, propose=_deterministic_propose, seeds=(0, 1),
        val_seeds=(10, 11), rounds=1, n_workers=1)
    assert report2.verdict == "reject"
    assert "not above min_gain" in report2.reason or "below" in report2.reason


def test_transfer_ratio_contract():
    report = {
        "a": {"gain": 0.2, "n": 4, "before": 0.4, "after": 0.6},
        "b": {"gain": 0.1, "n": 4, "before": 0.4, "after": 0.5},
    }
    assert transfer_ratio(report, "a", "b") == pytest.approx(0.5)
    assert transfer_ratio({"a": {"gain": 0.0}}, "a", "b") is None


def test_transfer_detects_overfit_that_does_not_transfer():
    """Positive val gain that fails to transfer must reject; genuine transfer adopts.

    ``transfer`` is mean val gain / mean train gain. A buggy implementation that
    sets ``transfer = mean_gain`` cannot satisfy both branches below: the
    overfit case has ``mean_gain`` below the floor (so the bug would also
    reject — caught by asserting ``transfer != mean_gain`` and the ratio),
    and the transferring case has ``mean_gain`` below the floor while the
    true ratio clears it (so the bug would reject an adopt).
    """
    from agentdescent.meta import MetaOutcome, _mean_gain, meta_validate

    slot = _sampler_slot()
    seed_text = slot.render(slot.initial())

    def strong_train(value, seed):
        # evolved beats seed by ~0.8 (not 1.0) so ratio != mean_gain
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.9, 0.9], final=0.9)
        return MetaOutcome(curve=[0.1, 0.1], final=0.1)

    def weak_val(value, seed):
        # positive but small val gain (~0.05) — classic overfit row
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.15, 0.15], final=0.15)
        return MetaOutcome(curve=[0.1, 0.1], final=0.1)

    def matched_val(value, seed):
        # val gain ~0.3; paired with modest_train (~0.3) so ratio ~1
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.55, 0.55], final=0.55)
        return MetaOutcome(curve=[0.25, 0.25], final=0.25)

    def modest_train(value, seed):
        if getattr(value, "c", 0.2) > 0.2:
            return MetaOutcome(curve=[0.6, 0.6], final=0.6)
        return MetaOutcome(curve=[0.3, 0.3], final=0.3)

    # --- overfit: large train gain, small positive val gain ---------------
    overfit, _ = meta_run(
        {"a": strong_train, "b": strong_train}, {"v": weak_val},
        slot="task_sampler", spec=slot, propose=_deterministic_propose,
        seeds=(0, 1), val_seeds=(10, 11),
        min_gain=0.0, transfer_floor=0.3, rounds=1, n_workers=1)
    assert overfit.mean_gain > 0.0
    assert overfit.transfer is not None
    # recover the train gain the decision used
    train_rows = meta_validate(
        slot, seed_text, overfit.value, {"a": strong_train, "b": strong_train},
        seeds=(0, 1))
    train_gain = _mean_gain(train_rows)
    assert train_gain > overfit.mean_gain > 0.0
    assert overfit.transfer == pytest.approx(
        overfit.mean_gain / train_gain, rel=1e-6)
    assert overfit.transfer != overfit.mean_gain, (
        "transfer must be a ratio vs train gain, not a duplicate of mean_gain")
    assert overfit.transfer < 0.3
    assert overfit.verdict == "reject"
    assert "does not transfer" in overfit.reason or "transfer_floor" in overfit.reason

    # --- transfers: ratio clears a floor that mean_gain alone would miss -
    # train gain ~0.3, val gain ~0.3 → ratio ~1.0; floor=0.5.
    # mean_gain (~0.3) < 0.5 so transfer==mean_gain would reject; true ratio
    # adopts.
    ok, _ = meta_run(
        {"a": modest_train, "b": modest_train}, {"v": matched_val},
        slot="task_sampler", spec=slot, propose=_deterministic_propose,
        seeds=(0, 1), val_seeds=(10, 11),
        min_gain=0.0, transfer_floor=0.5, rounds=1, n_workers=1)
    assert ok.mean_gain > 0.0
    assert ok.mean_gain < 0.5, (
        "fixture must keep mean_gain below transfer_floor so a buggy "
        "transfer==mean_gain would reject")
    train_rows_ok = meta_validate(
        slot, seed_text, ok.value, {"a": modest_train, "b": modest_train},
        seeds=(0, 1))
    train_gain_ok = _mean_gain(train_rows_ok)
    assert ok.transfer == pytest.approx(ok.mean_gain / train_gain_ok, rel=1e-6)
    assert ok.transfer >= 0.5
    assert ok.verdict == "adopt"


def test_meta_run_rejects_overlapping_val_set():
    slot = _sampler_slot()
    train = {"a": _inner_problem(_tasks(prefix="a"))}
    with pytest.raises(ValueError, match="overlap training problems"):
        meta_run(train, {"a": _inner_problem(_tasks(prefix="z"))},
                 slot="task_sampler", spec=slot,
                 propose=_deterministic_propose, seeds=(0,), val_seeds=(10,))
    with pytest.raises(ValueError, match="overlap training seeds"):
        meta_run(train, {"v": _inner_problem(_tasks(prefix="z"))},
                 slot="task_sampler", spec=slot,
                 propose=_deterministic_propose, seeds=(0, 1), val_seeds=(1, 2))


def test_meta_run_accepts_distinct_sequence_problems(tmp_path):
    """#356: sequence train/val must not collide on synthetic p0/p1 labels.

    Distinct Problem callables with disjoint seeds must pass isolation and
    reach meta_evolve — the advertised Sequence[Problem] API.
    """
    train_a = _inner_problem(_tasks(prefix="a"))
    train_b = _inner_problem(_tasks(prefix="b"))
    unseen_c = _inner_problem(_tasks(prefix="c"))
    report, result = meta_run(
        [train_a, train_b], [unseen_c],
        slot="task_sampler", spec=_sampler_slot(),
        propose=_deterministic_propose,
        seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1,
        repo_path=str(tmp_path / "seq"),
    )
    assert isinstance(report, MetaRunReport)
    assert report.verdict in ("adopt", "reject")
    assert result.meta is not None
    # validation keys are the sequence-local synthetic names for val only
    assert set(report.validation.keys()) == {"p0"}


def test_meta_run_rejects_shared_sequence_problem_identity():
    """#356: genuine shared Problem objects still raise under the sequence API.

    Cosmetic name prefixes alone must not weaken the guard — the same
    callable in both train and val is an isolation failure even when the
    synthetic labels would otherwise differ after a naive rename.
    """
    slot = _sampler_slot()
    shared = _inner_problem(_tasks(prefix="shared"))
    other = _inner_problem(_tasks(prefix="other"))
    with pytest.raises(ValueError, match="overlap training problems"):
        meta_run(
            [shared, other], [shared],
            slot="task_sampler", spec=slot,
            propose=_deterministic_propose,
            seeds=(0, 1), val_seeds=(10, 11),
        )
    # mapping + sequence sharing the same object: names differ, identity must
    # still catch the leak
    with pytest.raises(ValueError, match="overlap training problems"):
        meta_run(
            {"a": shared}, [shared],
            slot="task_sampler", spec=slot,
            propose=_deterministic_propose,
            seeds=(0,), val_seeds=(10,),
        )


def test_meta_run_exports():
    import agentdescent
    import agentdescent.meta as meta
    assert "meta_run" in agentdescent.__all__
    assert "MetaRunReport" in agentdescent.__all__
    assert "meta_run" in meta.__all__
    assert "MetaRunReport" in meta.__all__


def test_meta_run_offline_and_deterministic(tmp_path):
    train = {"a": _inner_problem(_tasks(prefix="a")),
             "b": _inner_problem(_tasks(prefix="b"))}
    val = {"v": _inner_problem(_tasks(prefix="z"))}
    r1, _ = meta_run(train, val, slot="task_sampler", spec=_sampler_slot(),
                     propose=_deterministic_propose, seeds=(0, 1),
                     val_seeds=(10, 11), rounds=1, n_workers=1,
                     repo_path=str(tmp_path / "r1"))
    r2, _ = meta_run(train, val, slot="task_sampler", spec=_sampler_slot(),
                     propose=_deterministic_propose, seeds=(0, 1),
                     val_seeds=(10, 11), rounds=1, n_workers=1,
                     repo_path=str(tmp_path / "r2"))
    assert r1.value == r2.value
    assert r1.validation == r2.validation
