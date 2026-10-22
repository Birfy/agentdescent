"""The self-improvement loop has to be right where evidence drives the next move.

`evidence` reconciles the archive (per-slot return-per-cost, exam stamps, the
harness leaderboard) into one deterministic view; `rank_slots` turns it into the
next targets; `admit` composes an adoption's trust rings into one verdict; and
`self_improve` runs the loop -- read evidence, evolve the paying slots, archive,
trust-gate, move the budget -- for a bounded number of ticks. The fixtures pin
the surfaces that make this honest: the snapshot counts exams and ports, the
ranking prefers paying slots and falls back on a cold archive, the gate is
fail-closed, and a tick improves the slots the evidence routed it to and writes
them to the same archive everything else reads.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    SelfImproveConfig,
    SourceSlot,
    admit,
    admit_record,
    problem_descriptor,
    rank_slots,
    self_improve,
    snapshot,
)
from agentdescent.meta import MetaOutcome

CONFIG = {"dataset": "self-improve-demo", "scorer": "exact"}


def _archive(store, run_id, slot, reward, rounds, *, artifact_id="meta-rule"):
    from agentdescent.evolution import RoundInfo
    history = [RoundInfo(round=i, held_out_reward=reward, n_items=1,
                         committed=1, rejected=0) for i in range(rounds)]
    store.archive(
        EvolutionResult(state={"value": "x"}, rendered="x",
                        final_reward=reward, history=history, ledger_log=[]),
        run_id=run_id, strategy="SourceSlot", config=CONFIG,
        artifact_id=artifact_id,
        problem=problem_descriptor(slot, rounds=rounds, n_workers=1))


# -- evidence ------------------------------------------------------------------


def test_snapshot_composes_slots_ports_and_exam_counts(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    # slot A: three rules, mean reward 0.8 over 5 rounds -> rpc 0.16
    for i in range(3):
        _archive(store, f"a{i}", "selection", 0.8, 5)
    # slot B: two rules, low return -> below the floor
    _archive(store, "b0", "task_sampler", 0.1, 5)
    _archive(store, "b1", "task_sampler", 0.1, 5)
    store.mark_exam("b0", exam="flagged")
    store.mark_exam("b1", exam="demoted")
    # a harness port outcome
    _archive(store, "h0", "selection", 0.9, 5, artifact_id="harness:quick")

    snap = snapshot(store, config=CONFIG)
    assert set(snap.slots) == {"selection", "task_sampler"}
    a = snap.slots["selection"]
    assert a.samples == 3 and a.rpc == pytest.approx(0.16)
    assert a.trusted == 3 and a.flagged == 0 and a.demoted == 0
    b = snap.slots["task_sampler"]
    assert b.flagged == 1 and b.demoted == 1 and b.trusted == 0
    assert snap.total_rules == 5 and snap.demoted_rules == 1
    assert snap.flagged_rules == 1
    # harness leaderboard is composed in
    assert "quick" in snap.harnesses and snap.harnesses["quick"].best == 0.9


def test_rank_slots_prefers_paying_slots_and_skips_demoted(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    for i in range(3):
        _archive(store, f"a{i}", "selection", 0.8, 5)      # rpc 0.16
    for i in range(3):
        _archive(store, f"b{i}", "task_sampler", 0.1, 5)   # rpc 0.02
    # a high-rpc slot whose rules are demoted must not be targeted
    for i in range(3):
        _archive(store, f"c{i}", "acceptance", 0.9, 5)
    store.mark_exam("c0", exam="demoted")

    snap = snapshot(store, config=CONFIG)
    ranked = rank_slots(snap, k=3, floor=0.05,
                        allow=["selection", "task_sampler", "acceptance"])
    assert ranked == ["selection"]         # b below floor, c has a demotion


def test_rank_slots_falls_back_on_a_cold_archive(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    snap = snapshot(store)
    # nothing qualifies -> the declared order, so the loop still rotates
    assert rank_slots(snap, k=2, allow=["selection", "task_sampler"]) == \
        ["selection", "task_sampler"]
    assert rank_slots(snap, k=2) == []


# -- admit ---------------------------------------------------------------------


def test_admit_composes_rings_fail_closed():
    assert admit().verdict == "pass"
    assert admit(exam="passed", audit="verified").verdict == "pass"
    assert admit(exam="demoted").verdict == "demote"
    assert admit(audit="failed").verdict == "demote"
    assert admit(exam="passed", chain="unverified").verdict == "flag"
    # an unknown status never passes silently
    assert admit(exam="weird").verdict == "flag"


def test_admit_record_reads_the_exam_stamp(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive(store, "r0", "selection", 0.8, 5)
    record = store.find("r0")
    assert admit_record(record).verdict == "pass"        # provisional default
    store.mark_exam("r0", exam="demoted")
    assert admit_record(store.find("r0")).verdict == "demote"


# -- the loop ------------------------------------------------------------------


def _spec():
    return SourceSlot(initial_value="0")


def _specs():
    return {"selection": _spec(), "task_sampler": _spec()}


def _problem(value, seed):
    r = int(str(value)) / 10.0
    return MetaOutcome(curve=[r], final=r, rollouts=1)


def _improving_propose(rendered, task, output, reward):
    return "5"


def _val_problem(value, seed):
    r = int(str(value)) / 10.0
    return MetaOutcome(curve=[r], final=r, rollouts=1)


def _domains():
    # distinct Problem objects on each side, so meta_run's identity isolation
    # check sees them as disjoint
    return {"d": ({"t0": _problem, "t1": _problem},
                  {"v0": _val_problem, "v1": _val_problem})}


def test_self_improve_runs_a_tick_and_archives(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection", "task_sampler"], specs=_specs(),
        domains=_domains(), ticks=1, top_k=2, min_gain=0.0,
        config=CONFIG, run_id_prefix="s1")
    report = self_improve(store, cfg, propose=_improving_propose)
    assert report.improved
    assert report.total_gain > 0
    assert len(report.ticks) == 1
    tick = report.ticks[0]
    assert set(tick.targets) == {"selection", "task_sampler"}
    assert all(v == "adopt" for v in tick.verdicts.values())
    # two adopted rules (one per slot) in the same archive everything reads
    assert store.count() == 2
    assert set(tick.budget_after) == {"meta:selection", "meta:task_sampler"}


def test_self_improve_compounds_and_does_not_readopt(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection", "task_sampler"], specs=_specs(),
        domains=_domains(), ticks=2, top_k=2, min_gain=0.0,
        config=CONFIG, run_id_prefix="s2")
    report = self_improve(store, cfg, propose=_improving_propose)
    assert len(report.ticks) == 2
    # tick 2's budget_before reflects tick 1's archives
    assert report.ticks[1].budget_before["meta:selection"] > 0
    # the proposer always returns "5": tick 1 adopts, tick 2 finds nothing new
    # (seeded from tick 1's rule) and must not re-archive or add gain
    assert store.count() == 2
    assert set(report.ticks[1].verdicts.values()) == {"reject"}
    assert all(g == 0.0 for g in report.ticks[1].gains.values())
    assert report.total_gain == pytest.approx(1.0)   # tick 0's two rows only


def test_self_improve_rejects_a_different_but_equivalent_rule(tmp_path):
    # meta_run's guard is ``mean_gain <= min_gain``: a tick whose evolved text
    # differs from the prior but scores the same is not an improvement, so it
    # must not be archived, counted, or feed the budget.
    store = ConsolidationStore(str(tmp_path / "archive"))

    def equivalent(rendered, task, output, reward):
        text = str(rendered).strip()
        if text == "0":
            return "5"
        if text == "5":
            return "05"          # different text, same int() -> same score
        return None

    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(), ticks=2,
        top_k=1, min_gain=0.0, config=CONFIG, run_id_prefix="s6")
    report = self_improve(store, cfg, propose=equivalent)
    assert report.ticks[0].verdicts["selection:d"] == "adopt"
    assert report.ticks[1].verdicts["selection:d"] == "reject"
    assert report.ticks[1].gains["selection:d"] == pytest.approx(0.0)
    assert store.count() == 1                   # only tick 0's real improvement
    assert report.total_gain == pytest.approx(0.5)


def test_self_improve_honours_the_transfer_floor(tmp_path):
    # A rule that gains on the train problems but barely transfers to the
    # unseen ones is a fit, and meta_run's transfer_floor must refuse it.
    store = ConsolidationStore(str(tmp_path / "archive"))

    def tiny_val(value, seed):
        r = 0.01 if int(str(value)) >= 5 else 0.0
        return MetaOutcome(curve=[r], final=r, rollouts=1)

    domains = {"fit": ({"t0": _problem}, {"v0": tiny_val})}
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=domains, ticks=1,
        top_k=1, min_gain=0.0, transfer_floor=0.5, config=CONFIG,
        run_id_prefix="s7")
    report = self_improve(store, cfg, propose=_improving_propose)
    assert not report.improved
    assert report.ticks[0].verdicts["selection:fit"] == "reject"
    assert store.count() == 0
    assert report.ticks[0].gains["selection:fit"] == pytest.approx(0.01)


def test_self_improve_trust_gate_quarantines_the_record(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(), ticks=1,
        top_k=1, config=CONFIG, run_id_prefix="s3")
    report = self_improve(store, cfg, propose=_improving_propose,
                          admit_fn=lambda record: "demote")
    assert not report.improved
    assert set(report.ticks[0].verdicts.values()) == {"demote"}
    # the verdict is stamped on the record, so the archive's own demotion
    # path (and the next snapshot) sees it -- not just this report
    record = store.find("s3-t0-selection-d")
    assert record is not None and record.exam == "demoted"
    assert snapshot(store, config=CONFIG).demoted_rules == 1


def test_self_improve_explores_unseen_slots(tmp_path):
    # three slots, top_k=2: the first tick cannot reach all three, but the
    # exploration share + rotation must reach the third within a few ticks
    store = ConsolidationStore(str(tmp_path / "archive"))
    specs3 = {"selection": _spec(), "task_sampler": _spec(),
              "acceptance": _spec()}
    cfg = SelfImproveConfig(
        slots=list(specs3), specs=specs3, domains=_domains(), ticks=3,
        top_k=2, config=CONFIG, run_id_prefix="s4")
    report = self_improve(store, cfg, propose=_improving_propose)
    tried = {slot for tick in report.ticks for slot in tick.targets}
    assert tried == {"selection", "task_sampler", "acceptance"}


def test_self_improve_tags_records_with_their_domain(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(), ticks=1,
        top_k=1, config=CONFIG, run_id_prefix="s5")
    self_improve(store, cfg, propose=_improving_propose)
    record = store.find("s5-t0-selection-d")
    assert record is not None and record.problem["domain"] == "d"


def test_self_improve_refuses_overlapping_seeds(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(),
        seeds=(0, 1), val_seeds=(1, 2))
    with pytest.raises(ValueError):
        self_improve(store, cfg, propose=_improving_propose)
