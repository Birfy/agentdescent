"""Joint multi-slot evolution has to be right where the decision plane is one artifact.

A joint candidate is a bundle ``{slot: value}``; each outer rollout installs all
the slots and runs the inner search, so an effect that only appears when two
rules change together is visible to the search. ``joint_meta_validate``
attributes the change per slot and isolates one that regresses. The fixtures
pin the surfaces that make this honest: the bundle parses/renders/targets slots,
a coupled landscape is only visible jointly, an untrusted slot is rolled back
(or the whole change rejected fail-closed), an adopted bundle archives one rule
per slot, and the isolation guards refuse overlap.
"""

import pytest

from agentdescent import Policies, SourceSlot, Task
from agentdescent.joint_meta import (
    JointMetaReport,
    PoliciesBundle,
    joint_meta_run,
    joint_meta_validate,
    joint_problem,
)
from agentdescent.meta import MetaOutcome, _archived_rules

CONFIG = {"dataset": "joint-demo", "scorer": "exact"}


def _spec(prefix):
    return SourceSlot(initial_value=f"{prefix}:0")


def _specs():
    return {"selection": _spec("s"), "task_sampler": _spec("t")}


def _bundle_problem(landscape):
    """A bundle-aware problem: ``(bundle, seed) -> MetaOutcome``."""
    def problem(bundle, seed):
        vals = {slot: str(value) for slot, value in bundle.items()}
        r = float(landscape(vals))
        return MetaOutcome(curve=[r], final=r, rollouts=1)
    return problem


# -- the bundle artifact -------------------------------------------------------


def test_bundle_parses_renders_and_targets_slots():
    b = PoliciesBundle(_specs())
    state = b.initial()
    assert set(state) == {"selection", "task_sampler"}
    rendered = b.render(state)
    assert "=== selection ===" in rendered and "=== task_sampler ===" in rendered
    # a header proposal targets its slot
    d = b.to_diff(state, "=== selection ===\ns:2", "w1", 0, "a")
    assert d.ops == {"selection": "s:2"}
    # a no-op is refused
    assert b.to_diff(state, "=== selection ===\ns:0", "w1", 0, "a") is None
    # several blocks are one coherent multi-slot change
    d = b.to_diff(state, "=== selection ===\ns:2\n=== task_sampler ===\nt:2",
                  "w1", 0, "a")
    assert d.ops == {"selection": "s:2", "task_sampler": "t:2"}
    # a bare proposal that compiles for BOTH slots is ambiguous -> refused
    assert b.to_diff(state, "garbage", "w1", 0, "a") is None


# -- the P7 proof: a coupled landscape is only visible jointly -----------------


def test_a_coupled_landscape_is_only_visible_jointly():
    b = PoliciesBundle(_specs())
    before = b.render(b.initial())

    def coupled(vals):
        return 1.0 if (vals["selection"] == "s:2"
                       and vals["task_sampler"] == "t:2") else 0.0

    val = {"v": _bundle_problem(coupled)}
    # either slot alone changes nothing ...
    sel_only = b.render({"selection": "s:2", "task_sampler": "t:0"})
    lone = joint_meta_validate(b, before, sel_only, val, seeds=(0, 1),
                               isolate=False)
    assert lone["joint_gain"] == pytest.approx(0.0)
    # ... but together they are decisive, and the leave-one-out attribution
    # credits each slot as load-bearing (reverting either collapses it)
    both = b.render({"selection": "s:2", "task_sampler": "t:2"})
    rep = joint_meta_validate(b, before, both, val, seeds=(0, 1), isolate=False)
    assert rep["joint_gain"] == pytest.approx(1.0)
    assert rep["slot_gains"]["selection"] == pytest.approx(1.0)
    assert rep["slot_gains"]["task_sampler"] == pytest.approx(1.0)
    assert rep["untrusted"] == ()
    assert rep["verdict"] == "adopt"


# -- isolating an untrusted rule ----------------------------------------------


def _hurt_landscape(vals):
    return (0.1 + 0.3 * (vals["selection"] == "s:2")
            - 0.2 * (vals["task_sampler"] == "t:bad"))


def test_an_untrusted_slot_is_isolated():
    b = PoliciesBundle(_specs())
    before = b.render(b.initial())
    val = {"v": _bundle_problem(_hurt_landscape)}
    after = b.render({"selection": "s:2", "task_sampler": "t:bad"})
    rep = joint_meta_validate(b, before, after, val, seeds=(0, 1), isolate=True)
    assert rep["verdict"] == "adopt"
    assert rep["untrusted"] == ("task_sampler",)
    assert rep["isolated"] == ("task_sampler",)
    # the corrected bundle keeps the helpful slot and rolls back the harmful one
    assert "s:2" in rep["value"] and "t:bad" not in rep["value"]
    assert rep["slot_gains"]["selection"] > 0
    assert rep["slot_gains"]["task_sampler"] < 0
    # correcting improves on the raw joint change
    assert rep["joint_gain"] > 0.1


def test_a_coupled_change_is_adopted_with_positive_min_gain():
    # leave-one-out attribution credits a load-bearing slot inside a synergy,
    # so a coupled change is not rejected just because no slot pays alone
    b = PoliciesBundle(_specs())
    before = b.render(b.initial())

    def coupled(vals):
        return 1.0 if (vals["selection"] == "s:2"
                       and vals["task_sampler"] == "t:2") else 0.0

    val = {"v": _bundle_problem(coupled)}
    after = b.render({"selection": "s:2", "task_sampler": "t:2"})
    rep = joint_meta_validate(b, before, after, val, seeds=(0, 1),
                              min_gain=0.05, isolate=True)
    assert rep["verdict"] == "adopt"
    assert rep["untrusted"] == ()
    assert rep["joint_gain"] == pytest.approx(1.0)
    # each slot's contribution is the whole synergy: reverting it collapses it
    assert rep["slot_gains"]["selection"] == pytest.approx(1.0)
    assert rep["slot_gains"]["task_sampler"] == pytest.approx(1.0)


def test_fail_closed_without_isolation():
    b = PoliciesBundle(_specs())
    before = b.render(b.initial())
    val = {"v": _bundle_problem(_hurt_landscape)}
    after = b.render({"selection": "s:2", "task_sampler": "t:bad"})
    rep = joint_meta_validate(b, before, after, val, seeds=(0, 1), isolate=False)
    assert rep["verdict"] == "reject"
    assert rep["isolated"] == ()
    assert rep["untrusted"] == ("task_sampler",)


def test_a_value_with_a_foreign_header_is_not_split():
    # a value containing "=== foo ===" (foo is not a slot) is content, not a
    # delimiter -- the parser only splits on declared slot names
    b = PoliciesBundle(_specs())
    state = b.initial()
    state["selection"] = "code\n=== foo ===\nmore code"
    rendered = b.render(state)
    assert b.compile(rendered)["selection"] == "code\n=== foo ===\nmore code"


def test_report_slot_values_ignore_a_foreign_header_in_a_value(tmp_path):
    # a value that itself contains "=== note ===" must not be split when the
    # report reads the adopted bundle back
    def problem(bundle, seed):
        v = str(bundle["selection"])
        r = 1.0 if "a" in v else 0.1
        return MetaOutcome(curve=[r], final=r, rollouts=1)

    def propose(rendered, task, output, reward):
        return "=== selection ===\na\n=== note ===\nb"

    report, _ = joint_meta_run(
        {"t": problem}, {"v": problem}, slots=["selection"],
        specs=_specs(), propose=propose, seeds=(0, 1, 2, 3),
        val_seeds=(4, 5, 6, 7), rounds=1, n_workers=1, repo_path=None)
    assert report.verdict == "adopt"
    assert report.slot_values["selection"] == "a\n=== note ===\nb"


# -- the loop -----------------------------------------------------------------


def _improving_landscape(vals):
    return (0.1 + 0.2 * (vals["selection"] == "s:2")
            + 0.2 * (vals["task_sampler"] == "t:2"))


def _joint_propose(rendered, task, output, reward):
    return "=== selection ===\ns:2\n=== task_sampler ===\nt:2"


def test_joint_meta_run_improves_both_slots():
    train = {f"t{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    val = {f"v{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    report, result = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=_joint_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        rounds=2, n_workers=1, repo_path=None)
    assert isinstance(report, JointMetaReport)
    assert report.verdict == "adopt"
    assert report.joint_gain > 0
    assert report.slot_values["selection"] == "s:2"
    assert report.slot_values["task_sampler"] == "t:2"
    assert result.meta["verdict"] == "adopt"


def test_joint_meta_run_isolates_a_bad_slot_end_to_end():
    train = {f"t{i}": _bundle_problem(_hurt_landscape) for i in range(2)}
    val = {f"v{i}": _bundle_problem(_hurt_landscape) for i in range(2)}

    def propose(rendered, task, output, reward):
        return "=== selection ===\ns:2\n=== task_sampler ===\nt:bad"

    report, result = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        isolate=True, rounds=2, n_workers=1, repo_path=None)
    assert report.verdict == "adopt"
    assert report.isolated == ("task_sampler",)
    assert report.slot_values["selection"] == "s:2"
    assert report.slot_values["task_sampler"] == "t:0"      # rolled back to seed


# -- archive + seeding ---------------------------------------------------------


def test_an_adopted_bundle_archives_per_slot_and_seeds(tmp_path):
    from agentdescent import ConsolidationStore
    store = ConsolidationStore(str(tmp_path / "archive"))
    train = {f"t{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    val = {f"v{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    report, _ = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=_joint_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        store=store, config=CONFIG, run_id="joint-1",
        rounds=2, n_workers=1, repo_path=None)
    assert report.verdict == "adopt"
    # one ordinary meta-rule record per slot, found by the normal ranking
    for slot in ("selection", "task_sampler"):
        rules = _archived_rules(store, slot=slot, spec=_specs()[slot],
                                config=CONFIG, artifact_id="meta-rule")
        assert rules, f"no archived rule for {slot}"
    # a second run seeds from those rules, not from the raw initial values
    report2, _ = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=_joint_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        store=store, config=CONFIG, run_id="joint-2",
        rounds=1, n_workers=1, repo_path=None)
    assert "s:2" in report2.seed_value and "t:2" in report2.seed_value


def test_an_isolated_slot_is_not_archived(tmp_path):
    from agentdescent import ConsolidationStore
    store = ConsolidationStore(str(tmp_path / "archive"))
    train = {f"t{i}": _bundle_problem(_hurt_landscape) for i in range(2)}
    val = {f"v{i}": _bundle_problem(_hurt_landscape) for i in range(2)}

    def propose(rendered, task, output, reward):
        return "=== selection ===\ns:2\n=== task_sampler ===\nt:bad"

    report, _ = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        store=store, config=CONFIG, run_id="j1", isolate=True,
        rounds=2, n_workers=1, repo_path=None)
    assert report.isolated == ("task_sampler",)
    assert store.find("j1:selection") is not None
    assert store.find("j1:task_sampler") is None   # rolled back, not adopted


def test_archive_records_the_joint_contribution_and_bundle(tmp_path):
    from agentdescent import ConsolidationStore
    store = ConsolidationStore(str(tmp_path / "archive"))
    train = {f"t{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    val = {f"v{i}": _bundle_problem(_improving_landscape) for i in range(2)}
    report, _ = joint_meta_run(
        train, val, slots=["selection", "task_sampler"], specs=_specs(),
        propose=_joint_propose, seeds=(0, 1, 2, 3), val_seeds=(4, 5, 6, 7),
        store=store, config=CONFIG, run_id="j2", isolate=True,
        rounds=2, n_workers=1, repo_path=None)
    assert report.verdict == "adopt"
    for slot in ("selection", "task_sampler"):
        rec = store.find(f"j2:{slot}")
        assert rec is not None
        assert rec.problem["joint_run"] == "j2"
        assert rec.problem["joint_slots"] == ["selection", "task_sampler"]
        assert rec.final_reward > 0          # the synergy contribution, not 0


# -- isolation guards ----------------------------------------------------------


def test_joint_meta_run_refuses_overlapping_names():
    problem = _bundle_problem(_improving_landscape)
    with pytest.raises(ValueError):
        joint_meta_run({"same": problem}, {"same": problem},
                       slots=["selection", "task_sampler"], specs=_specs(),
                       propose=_joint_propose, seeds=(0, 1, 2, 3),
                       val_seeds=(4, 5, 6, 7))


def test_joint_meta_run_refuses_seed_overlap():
    problem = _bundle_problem(_improving_landscape)
    with pytest.raises(ValueError):
        joint_meta_run({"t": problem}, {"v": problem},
                       slots=["selection", "task_sampler"], specs=_specs(),
                       propose=_joint_propose, seeds=(0, 1),
                       val_seeds=(1, 2))


# -- the inner problem installs the whole bundle ------------------------------


def test_joint_problem_installs_all_slots(monkeypatch):
    import agentdescent.joint_meta as jm
    from agentdescent import EvolutionResult
    seen = {}

    def fake_evolve(tasks, reward, *, seed, policies, **kwargs):
        seen["policies"] = policies
        return EvolutionResult(state={}, rendered="", final_reward=0.5,
                               history=[], ledger_log=[])

    monkeypatch.setattr(jm, "evolve", fake_evolve)
    problem = joint_problem([], lambda t, o: 0.0, base=Policies())
    outcome = problem({"selection": "S", "task_sampler": "T"}, 0)
    assert seen["policies"].selection == "S"
    assert seen["policies"].task_sampler == "T"
    assert outcome.final == pytest.approx(0.5)
