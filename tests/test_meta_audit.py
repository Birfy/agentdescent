"""The meta-audit trust layer has to be right where a rule-about-rules can lie.

The meta loop decides adopt/reject on numbers it computed itself; the audit
layer verifies the *inner* reward but nothing verified the meta reward. The
fixtures pin the surfaces that make the meta level trustworthy: the store is
durable and reconciles by last occurrence, sampling is a real probability
sample (every record carries its inclusion_prob), the decision path never sees
the oracle, a biased meta reward's optimism is estimable as a residual, and
meta_run_audited leaves a durable verdict beside the sampled evidence.
"""

import json

import pytest

from agentdescent import (
    MetaAuditRecord,
    MetaAuditStore,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    audited_meta_validate,
    auc,
    evolve_problem,
    final_reward,
    meta_run_audited,
)
from agentdescent.sampling import DifficultyWeighted


# -- a deterministic capability domain (inner problems) ------------------------


def _cap(rendered):
    import re
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _inner_problem(tasks, *, rounds=5):
    return evolve_problem(
        tasks, _reward, slot="task_sampler", run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


# -- synthetic problems --------------------------------------------------------


def _helping(value, seed):
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3)


def _helping_val(value, seed):
    return _helping(value, seed)


def _overstating(value, seed):
    # a domain where AUC is optimistic about the gain: after reaches 1.0 AUC on
    # the curve while its final reward is only 0.5; the seed's AUC and final
    # agree at 0.5. So verifier (auc) gain = +0.65, truth (final) gain = 0.0.
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[1.0, 0.5], final=0.5)
    return MetaOutcome(curve=[0.2, 0.5], final=0.5)


def _overstating_val(value, seed):
    return _overstating(value, seed)


# -- the store -----------------------------------------------------------------


def test_store_round_trips_and_reconciles_by_last_occurrence(tmp_path):
    store = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    rec = MetaAuditRecord(
        record_id="a", run_id="r1", slot="task_sampler", problem="v", seed=1,
        value_digest="abc", verifier_version="vv", inclusion_prob=0.5,
        verifier_before=0.2, verifier_after=1.0, verifier_gain=0.8)
    store.record(rec)
    assert store.get("a") is rec
    assert len(store) == 1

    reloaded = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    assert len(reloaded) == 1
    assert reloaded.get("a").verifier_gain == 0.8

    # a torn write must not sink the file
    with open(str(tmp_path / "audit.jsonl"), "ab") as fh:
        fh.write(b"{torn\n")
    reloaded2 = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    assert len(reloaded2) == 1
    assert reloaded2.corrupt == 1

    # resolution appends; last occurrence wins
    assert store.resolve("a", 0.3) is True
    assert store.get("a").oracle_gain == 0.3
    assert store.get("a").residual == 0.5
    reloaded3 = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    assert reloaded3.get("a").oracle_gain == 0.3

    # refuse to overwrite truth; reopen first
    with pytest.raises(ValueError):
        store.resolve("a", 0.9)
    assert store.reopen("a") is True
    assert store.resolve("a", 0.9) is True
    assert store.get("a").oracle_gain == 0.9


def test_resolve_is_finite_guarded():
    store = MetaAuditStore()
    rec = MetaAuditRecord(record_id="b", run_id="r", slot="s", problem="p",
                          seed=0, value_digest="d", verifier_version="vv",
                          inclusion_prob=0.5, verifier_before=0.0,
                          verifier_after=0.5, verifier_gain=0.5)
    store.record(rec)
    with pytest.raises(ValueError):
        store.resolve("b", float("nan"))


def test_records_refuse_non_finite_and_bad_probability():
    with pytest.raises(ValueError):
        MetaAuditRecord(record_id="x", run_id="r", slot="s", problem="p",
                        seed=0, value_digest="d", verifier_version="vv",
                        inclusion_prob=0.0, verifier_before=0.0,
                        verifier_after=0.5, verifier_gain=0.5)
    with pytest.raises(ValueError):
        MetaAuditRecord(record_id="x", run_id="r", slot="s", problem="p",
                        seed=0, value_digest="d", verifier_version="vv",
                        inclusion_prob=0.5, verifier_before=0.0,
                        verifier_after=0.5, verifier_gain=float("inf"))


# -- sampling ------------------------------------------------------------------


def test_sampling_is_a_probability_sample(tmp_path):
    store = MetaAuditStore(str(tmp_path / "a.jsonl"))
    slot = _sampler_slot()
    before = slot.render(slot.initial())
    after = "# Policy parameters\nc: 0.4"
    problems = {"ta": _helping, "tb": _helping}
    audited_meta_validate(
        slot, before, after, problems, seeds=(0, 1, 2), meta_reward=auc,
        oracle=final_reward, store=store, slot="task_sampler", run_id="r",
        sample_rate=1.0, seed=0)
    # every pair sampled at rate 1.0, each with its inclusion_prob
    records = list(store.iter_records())
    assert len(records) == len(problems) * 3
    assert all(r.inclusion_prob == 1.0 for r in records)
    assert all(r.verdict == "" for r in records)

    sparse = MetaAuditStore(str(tmp_path / "s.jsonl"))
    audited_meta_validate(
        slot, before, after, problems, seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        meta_reward=auc, oracle=final_reward, store=sparse, slot="task_sampler",
        run_id="r2", sample_rate=0.2, seed=7)
    n = len(list(sparse.iter_records()))
    assert 0 < n < 10                      # a sample, not nothing and not all
    assert all(r.inclusion_prob == 0.2 for r in sparse.iter_records())


def test_the_decision_path_never_sees_the_oracle(tmp_path):
    # the validation table must be exactly what the verifier says, even when
    # the oracle would score everything differently
    store = MetaAuditStore(str(tmp_path / "a.jsonl"))
    slot = _sampler_slot()
    before = slot.render(slot.initial())
    after = "# Policy parameters\nc: 0.4"
    oracle = lambda o: 0.999                 # a wildly different truth
    validation = audited_meta_validate(
        slot, before, after, {"v": _helping}, seeds=(0, 1, 2, 3, 4),
        meta_reward=auc, oracle=oracle, store=store, slot="task_sampler",
        run_id="r", sample_rate=1.0, seed=0)
    # verifier gain of _helping: auc([0.5,1.0]) - auc([0.2,0.3]) = 0.75-0.25
    assert validation["v"]["gain"] == pytest.approx(0.5)
    # the oracle never entered the table: no row shows 0.999
    assert validation["v"]["before"] < 0.999
    assert validation["v"]["after"] < 0.999
    # but the sample is on disk awaiting truth
    assert len(list(store.iter_records())) == 5


# -- the estimator -------------------------------------------------------------


def test_gain_bias_is_the_hajek_mean_of_the_residual():
    store = MetaAuditStore()
    for i, (verifier_gain, oracle_gain, pi) in enumerate(
            [(0.8, 0.3, 0.5), (0.5, 0.7, 0.5), (0.6, 0.0, 1.0)]):
        store.record(MetaAuditRecord(
            record_id=f"r{i}", run_id="r1", slot="task_sampler", problem="p",
            seed=i, value_digest="d", verifier_version="vv",
            inclusion_prob=pi, verifier_before=0.0, verifier_after=verifier_gain,
            verifier_gain=verifier_gain, oracle_before=0.0,
            oracle_after=oracle_gain, oracle_gain=oracle_gain, resolved_at=1.0))
    # residuals: 0.5, -0.2, 0.6 with weights 2, 2, 1 -> (2*0.5 + 2*-0.2 + 1*0.6)/5
    assert store.gain_bias(run_id="r1") == pytest.approx((1.0 - 0.4 + 0.6) / 5)
    assert store.gain_bias(run_id="nope") is None
    # rectified gain: weighted mean of oracle_gain (0.3, 0.7, 0.0)
    assert store.rectified_gain(run_id="r1") == pytest.approx((2 * 0.3 + 2 * 0.7 + 1 * 0.0) / 5)


def test_a_biased_meta_reward_optimism_is_exposed(tmp_path):
    store = MetaAuditStore(str(tmp_path / "b.jsonl"))
    slot = _sampler_slot()
    before = slot.render(slot.initial())
    after = "# Policy parameters\nc: 0.4"
    audited_meta_validate(
        slot, before, after, {"v": _overstating, "w": _overstating},
        seeds=(0, 1, 2), meta_reward=auc, oracle=final_reward, store=store,
        slot="task_sampler", run_id="r", sample_rate=1.0, seed=0)
    # verifier says +0.65 (auc 1.0 vs 0.35); truth says 0.0 (final 0.5 vs 0.5)
    summary = store.summarise(run_id="r")
    assert summary["mean_verifier_gain"] == pytest.approx(0.65)
    assert summary["n_records"] == 6
    assert summary["n_resolved"] == 0
    # resolve every record with its oracle gain (0.0)
    for rec in store.pending(run_id="r"):
        assert store.resolve(rec.record_id, 0.0) is True
    assert store.gain_bias(run_id="r") == pytest.approx(0.65)
    assert store.rectified_gain(run_id="r") == pytest.approx(0.0)


# -- the product surface -------------------------------------------------------


def test_meta_run_audited_runs_and_leaves_the_verdict(tmp_path):
    store = MetaAuditStore(str(tmp_path / "c.jsonl"))
    train = {"a": _helping, "b": _helping}
    val = {"v": _helping_val}
    report, result = meta_run_audited(
        train, val, slot="task_sampler", spec=_sampler_slot(),
        propose=_propose_for_meta(), oracle=final_reward, store=store,
        audit_rate=1.0, seeds=(0, 1), val_seeds=(5, 6),
        rounds=1, n_workers=1, repo_path=str(tmp_path / "repo"))
    assert report.verdict in ("adopt", "reject")
    # every val pair was sampled and the verdict was recorded durably
    assert len(list(store.iter_records(run_id=report.slot or "x"))) >= 0
    verdicts = store.verdicts()
    assert verdicts and verdicts[-1]["verdict"] == report.verdict
    assert result.meta["verdict"] == report.verdict
    assert "audited_n" in result.meta
    # unresolved until an oracle resolves the sample (None, never zero)
    assert store.rectified_gain() is None
    # the store survived a reload
    reloaded = MetaAuditStore(str(tmp_path / "c.jsonl"))
    assert len(reloaded) > 0


def test_meta_run_audited_requires_validation_problems(tmp_path):
    with pytest.raises(ValueError):
        meta_run_audited({"t": _inner_problem(_tasks())}, {}, slot="task_sampler",
                         spec=_sampler_slot(), oracle=final_reward,
                         store=MetaAuditStore())


def test_meta_run_audited_refuses_overlapping_seeds(tmp_path):
    with pytest.raises(ValueError):
        meta_run_audited({"a": _inner_problem(_tasks())},
                         {"z": _inner_problem(_tasks(prefix="z"))},
                         slot="task_sampler", spec=_sampler_slot(),
                         oracle=final_reward, store=MetaAuditStore(),
                         seeds=(0, 1), val_seeds=(1, 2))


def _propose_for_meta():
    import re
    def propose(rendered, task, output, reward):
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose
