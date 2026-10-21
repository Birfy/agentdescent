"""Chain audit has to be right where the framework's evidence promise is made
inspectable.

One adoption is traced through its rings — governance, decision, meta-gain,
self-exam, inner-reward, harness — and a ring is verified when its evidence is
present and within tolerance, flagged when outside it, and unverified when the
evidence simply is not there. The fixtures pin the surfaces that make this
honest: a fully-verified adoption is certified, missing evidence is reported
ring by ring (never silently assumed), a flagged ring (a demoted rule, a biased
meta gain) fails the chain, and an unknown run is refused.
"""

import pytest

from agentdescent import (
    AuditRecord,
    ChainAuditReport,
    ConsolidationStore,
    EvolutionResult,
    MetaAuditRecord,
    MetaAuditStore,
    ParamSlot,
    Purpose,
    SingleSlot,
    Task,
    chain_audit,
    exam_archive,
    final_reward,
    problem_descriptor,
)
from agentdescent.sampling import DifficultyWeighted


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


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


def _archive_rule(store, run_id, slot="task_sampler", *, c=0.4):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=0.8, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        artifact_id="meta-rule",
        problem=problem_descriptor(slot, **INNER))


def _resolved_meta_audit(tmp_path, run_id, *, bias=0.0):
    store = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    store.record(MetaAuditRecord(
        record_id="a", run_id=run_id, slot="task_sampler", problem="v", seed=1,
        value_digest="d", verifier_version="vv", inclusion_prob=0.5,
        verifier_before=0.2, verifier_after=0.2 + 0.3 + bias,
        verifier_gain=0.3 + bias, oracle_before=0.2, oracle_after=0.5,
        oracle_gain=0.3, resolved_at=1.0))
    store.record_verdict(run_id=run_id, slot="task_sampler", verdict="adopt",
                         reason="", mean_gain=0.3, transfer=1.0, audited_n=1,
                         oracle_gain=0.3, value_digest="d")
    return store


def _resolved_inner_audit(tmp_path, *, residual=0.0):
    from agentdescent.audit import AuditStore
    store = AuditStore(str(tmp_path / "inner.jsonl"))
    store.append(AuditRecord(
        record_id="x", task_id="t", artifact_signature="s", output="o",
        verifier_version="vv", verifier_score=0.5 + residual,
        inclusion_prob=0.5, purpose=Purpose.CALIBRATION,
        oracle_score=0.5, resolved_at=1.0))
    return store


def _exam_pass(archive, tmp_path, run_id):
    exam_archive(
        archive, slot="task_sampler", spec=_sampler_slot(),
        oracle=final_reward, tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        min_gain=0.0)


# -- the chain -----------------------------------------------------------------


def test_a_fully_verified_adoption_is_certified(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1")
    _exam_pass(archive, tmp_path, "run-1")          # exam -> passed
    meta = _resolved_meta_audit(tmp_path, "run-1", bias=0.0)
    inner = _resolved_inner_audit(tmp_path, residual=0.0)
    report = chain_audit(
        archive, "run-1", meta_audit=meta, inner_audit=inner,
        harness_evidence=lambda config: ("deep", 0.9))
    assert isinstance(report, ChainAuditReport)
    assert report.verified is True
    assert report.reason == "all 6 rings verified"
    by_layer = {l.layer: l.status for l in report.links}
    assert set(by_layer) == {"governance", "decision", "meta-gain", "self-exam",
                             "inner-reward", "harness"}
    assert all(s == "verified" for s in by_layer.values())
    # the evidence travelled with the links
    governance = next(l for l in report.links if l.layer == "governance")
    assert governance.evidence["class"] == "decision-plane (L1 harness)"


def test_missing_evidence_is_reported_ring_by_ring(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1")
    report = chain_audit(archive, "run-1")          # no stores at all
    assert report.verified is False
    by_layer = {l.layer: l.status for l in report.links}
    assert by_layer["governance"] == "verified"
    assert by_layer["decision"] == "unverified"
    assert by_layer["meta-gain"] == "unverified"
    assert by_layer["self-exam"] == "unverified"
    assert by_layer["inner-reward"] == "unverified"
    assert by_layer["harness"] == "unverified"
    assert "not verified" in report.reason
    assert all(layer in report.reason for layer in
               ("decision", "meta-gain", "self-exam", "inner-reward", "harness"))


def test_a_flagged_ring_fails_the_chain(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1")
    archive.mark_exam("run-1", exam="demoted", flags=2,
                      note="failed truth twice")
    report = chain_audit(archive, "run-1")
    self_exam = next(l for l in report.links if l.layer == "self-exam")
    assert self_exam.status == "flagged"
    assert report.verified is False


def test_a_biased_meta_gain_flags_the_ring(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1")
    meta = _resolved_meta_audit(tmp_path, "run-1", bias=0.3)   # residual +0.3
    report = chain_audit(archive, "run-1", meta_audit=meta, bias_threshold=0.1)
    meta_gain = next(l for l in report.links if l.layer == "meta-gain")
    assert meta_gain.status == "flagged"
    assert meta_gain.evidence["gain_bias"] == pytest.approx(0.3)


def test_an_unknown_run_is_refused(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(KeyError):
        chain_audit(archive, "nope")


def test_machinery_slot_is_classified_as_such(tmp_path):
    archive = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(archive, "run-1", slot="verifier")
    report = chain_audit(archive, "run-1")
    governance = next(l for l in report.links if l.layer == "governance")
    assert "machinery" in governance.evidence["class"]
