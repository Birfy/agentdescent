"""Verified seeding has to be right where trust becomes a gate.

The self-exam stamps adoptions and `_archived_rules` retires demoted ones; this
is the rung that gates *seeding* on trust: `verified_rules` returns only the
rules that meet the caller's bar (exam at or above `min_exam`, gain bias within
`bias_threshold` when an audit store is supplied) and explains every exclusion;
`continuous_meta_run(min_exam=...)` seeds the next search only from certified
rules. The fixtures pin the surfaces that make this honest: an unexamined rule
does not seed, a flagged one does not seed, an audit-biased one does not seed
even when it passed the exam, and `min_exam=None` reproduces the old behaviour
exactly.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaAuditRecord,
    MetaAuditStore,
    ParamSlot,
    SingleSlot,
    Task,
    continuous_meta_run,
    meta_population_run,
    problem_descriptor,
    verified_rules,
)
from agentdescent.meta import _archived_rules
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


def _archive_rule(store, run_id, *, c=0.4, reward=0.8):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=reward, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        artifact_id="meta-rule",
        problem=problem_descriptor("task_sampler", **INNER))


def _helping(value, seed):
    from agentdescent import MetaOutcome
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3)


def _helping_val(value, seed):
    return _helping(value, seed)


def _nudge_up():
    import re
    def propose(rendered, task, output, reward):
        m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
        c = float(m.group(1)) if m else 0.2
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose


# -- the gate -------------------------------------------------------------------


def test_unexamined_rules_do_not_seed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1")                  # exam is None
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG)
    assert seed.rules == ()
    assert seed.excluded == {"run-1": "rule has not been examined yet"}


def test_flagged_rules_do_not_seed_and_passed_ones_do(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-good")
    _archive_rule(store, "run-flagged")
    store.mark_exam("run-flagged", exam="flagged", flags=1, note="failed once")
    store.mark_exam("run-good", exam="passed", flags=0, note="ok")
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG)
    assert [r.run_id for r in seed.rules] == ["run-good"]
    assert "run-flagged" in seed.excluded
    assert "exam is 'flagged'" in seed.excluded["run-flagged"]


def test_an_audit_biased_rule_does_not_seed_even_when_passed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1")
    store.mark_exam("run-1", exam="passed", flags=0, note="ok")
    audit = MetaAuditStore(str(tmp_path / "audit.jsonl"))
    audit.record(MetaAuditRecord(
        record_id="a", run_id="run-1", slot="task_sampler", problem="v", seed=1,
        value_digest="d", verifier_version="vv", inclusion_prob=0.5,
        verifier_before=0.2, verifier_after=0.9, verifier_gain=0.7,
        oracle_before=0.2, oracle_after=0.5, oracle_gain=0.3, resolved_at=1.0))
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG, meta_audit=audit, bias_threshold=0.1)
    assert seed.rules == ()
    assert "gain bias" in seed.excluded["run-1"]


def test_min_exam_none_reproduces_old_behaviour(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1")                  # unexamined, not demoted
    plain = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                            config=CONFIG, artifact_id="meta-rule")
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG, min_exam=None)
    assert [r.run_id for r in seed.rules] == [r.run_id for r in plain]
    assert seed.excluded == {}


def test_min_exam_is_a_floor_not_equality(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-flagged", reward=0.7)
    store.mark_exam("run-flagged", exam="flagged", flags=1, note="failed once")
    _archive_rule(store, "run-passed", reward=0.8)
    store.mark_exam("run-passed", exam="passed", flags=0, note="ok")
    # "flagged" is below "passed": the passed rule is at or above the bar
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG, min_exam="flagged")
    assert {r.run_id for r in seed.rules} == {"run-flagged", "run-passed"}
    assert seed.excluded == {}


def test_an_unknown_min_exam_raises(tmp_path):
    # a typo must not silently empty the seed set
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1")
    store.mark_exam("run-1", exam="passed", flags=0, note="ok")
    with pytest.raises(ValueError, match="min_exam"):
        verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                       config=CONFIG, min_exam="pased")


def test_demoted_rules_never_seed(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1")
    store.mark_exam("run-1", exam="demoted", flags=2, note="failed twice")
    seed = verified_rules(store, slot="task_sampler", spec=_sampler_slot(),
                          config=CONFIG)
    assert seed.rules == ()
    assert seed.excluded == {}                     # excluded upstream, no reason here


# -- the knob -------------------------------------------------------------------


def test_continuous_meta_run_honours_min_exam(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-passed")
    store.mark_exam("run-passed", exam="passed", flags=0, note="ok")
    _archive_rule(store, "run-flagged", reward=0.9)
    store.mark_exam("run-flagged", exam="flagged", flags=1, note="failed once")

    # min_exam="passed": only the passed rule can seed
    s1, _ = continuous_meta_run(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, inner_propose=_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        min_exam="passed", seeds=(0, 1), val_seeds=(10, 11), rounds=1,
        n_workers=1, repo_path=str(tmp_path / "repo1"),
        problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert s1.seeded_from == "run-passed"          # not run-flagged, not run-passed's rival

    # default: the best rule seeds regardless of exam (old behaviour)
    s2, _ = continuous_meta_run(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, inner_propose=_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        seeds=(0, 1), val_seeds=(10, 11), rounds=1, n_workers=1,
        repo_path=str(tmp_path / "repo2"),
        problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert s2.seeded_from == "run-flagged"         # best reward, exam not consulted


def test_population_honours_min_exam(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-passed", reward=0.8)
    store.mark_exam("run-passed", exam="passed", flags=0, note="ok")
    _archive_rule(store, "run-flagged", reward=0.9)
    store.mark_exam("run-flagged", exam="flagged", flags=1, note="failed once")
    report, _ = meta_population_run(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, inner_propose=_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        islands=1, min_exam="passed", seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1, repo_dir=str(tmp_path / "repos"),
        problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert report.islands[0].seeded_from == "run-passed"
