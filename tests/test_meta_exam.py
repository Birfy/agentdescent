"""The self-exam has to be right where the archive's adoptions are re-judged.

Adoptions are provisional until an oracle re-validates them: a rule that passes
is certified, one that fails truth is flagged, and one flagged often enough is
demoted and retired from future seeding. The fixtures pin the surfaces that
make this honest: the stamp is durable and backwards compatible, the pass/flag/
demote ladder is deterministic, a demoted rule disappears from `_archived_rules`
(so seeding and population selection skip it), and the whole thing round-trips
through `continuous_meta_run`.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaExamReport,
    ParamSlot,
    SingleSlot,
    Task,
    continuous_meta_run,
    exam_archive,
    final_reward,
    problem_descriptor,
)
from agentdescent.meta import _archived_rules
from agentdescent.sampling import DifficultyWeighted


# -- the deterministic capability domain ---------------------------------------


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


def _archive_rule(store, run_id, c, *, reward=0.8):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=reward, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")


def _exam(store, tmp_path, **kw):
    base = dict(store=store, slot="task_sampler", spec=_sampler_slot(),
                oracle=final_reward, tasks_for=lambda rec: _tasks(),
                run=_run, reward=_reward, propose=_propose,
                strategy=SingleSlot(initial_value="cap:0"), config=CONFIG)
    base.update(kw)
    return exam_archive(**base)


# -- the exam ladder -----------------------------------------------------------


def test_a_rule_that_clears_min_gain_passes_and_resets_flags(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.4)
    # the cap domain is c-insensitive, so oracle gain is exactly 0: at
    # min_gain=0.0 (must not regress under truth) the rule passes
    report = _exam(store, tmp_path, min_gain=0.0)
    assert isinstance(report, MetaExamReport)
    assert report.passed == 1 and report.flagged == 0 and report.demoted == 0
    assert report.examined[0].verdict == "passed"
    # the stamp is durable and the note is readable
    reloaded = ConsolidationStore(str(tmp_path / "archive"))
    rec = reloaded.find("run-1")
    assert rec.exam == "passed"
    assert rec.exam_flags == 0
    assert "oracle gain" in rec.exam_note
    # the rule is still seedable
    assert [r.run_id for r in _archived_rules(
        store, slot="task_sampler", spec=_sampler_slot(), config=CONFIG,
        artifact_id="meta-rule")] == ["run-1"]


def test_a_rule_that_fails_truth_twice_is_demoted_and_retired(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-bad", 0.2)
    # with a positive min_gain the oracle's zero gain is a failure of truth
    r1 = _exam(store, tmp_path, min_gain=0.05)
    assert r1.flagged == 1
    assert store.find("run-bad").exam == "flagged"
    assert store.find("run-bad").exam_flags == 1

    r2 = _exam(store, tmp_path, min_gain=0.05)
    assert r2.demoted == 1
    assert store.find("run-bad").exam == "demoted"
    assert store.find("run-bad").exam_flags == 2
    # a demoted rule disappears from the seeding candidates entirely
    assert _archived_rules(
        store, slot="task_sampler", spec=_sampler_slot(), config=CONFIG,
        artifact_id="meta-rule") == []


def test_an_exam_skips_rules_already_demoted(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-good", 0.4)
    _archive_rule(store, "run-bad", 0.2)
    store.mark_exam("run-bad", exam="demoted", flags=2,
                    note="pre-marked: failed truth twice")
    # the demoted rule is not re-examined; the seedable one is
    report = _exam(store, tmp_path, min_gain=0.05)
    assert [e.run_id for e in report.examined] == ["run-good"]
    assert store.find("run-bad").exam == "demoted"


def test_a_passing_exam_resets_the_flag_count(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.4)
    _exam(store, tmp_path, min_gain=0.05)          # flagged once
    assert store.find("run-1").exam_flags == 1
    _exam(store, tmp_path, min_gain=0.0)           # now passes: resets
    assert store.find("run-1").exam == "passed"
    assert store.find("run-1").exam_flags == 0


def test_max_flags_must_be_positive(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.4)
    with pytest.raises(ValueError):
        _exam(store, tmp_path, max_flags=0)


def test_mark_exam_refuses_unknown_run(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(KeyError):
        store.mark_exam("nope", exam="passed", flags=0, note="")


# -- the round trip ------------------------------------------------------------


def test_round_trips_with_continuous_meta_run(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    s1, _ = continuous_meta_run(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, inner_propose=_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        run_id="run-1", inner=dict(INNER), seeds=(0, 1), val_seeds=(10, 11),
        rounds=1, n_workers=1, repo_path=str(tmp_path / "repo"),
        problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert s1.archived
    # the adopted rule is examined; the stamp lands next to the evidence
    report = _exam(store, tmp_path, min_gain=0.0)
    assert report.examined[0].run_id == "run-1"
    assert store.find("run-1").exam == "passed"
    # a following continuous session still seeds from the examined rule
    s2, _ = continuous_meta_run(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
        reward=_reward, inner_propose=_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
        seeds=(0, 1), val_seeds=(10, 11), rounds=1, n_workers=1,
        repo_path=str(tmp_path / "repo2"),
        problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert s2.seeded_from == "run-1"


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
