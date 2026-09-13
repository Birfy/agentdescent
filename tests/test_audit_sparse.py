"""The sparse audit layer: a cheap verifier paired against ground truth.

The load-bearing claim is the first test in this file. Everything else in the
package is bookkeeping in service of it: **turning the audit on does not change
the run**. If that ever stops being true the layer is not an instrument any more,
it is a second optimiser with no gate in front of it.
"""

import json
from difflib import SequenceMatcher
from typing import Optional

import pytest

from agentdescent.audit import (AuditedReward, AuditRecord, AuditStore,
                                DeferredOracle, GoldAnswer, NullOracle, Purpose,
                                RenderTap, resolve_from_mapping, summarise,
                                verifier_fingerprint)
from agentdescent.evolution import Task, evolve


# -- fixtures ----------------------------------------------------------------

def _tasks(n=12):
    return [Task(id=f"t{i}", prompt=f"  MiXeD Case {i}!  ",
                 meta={"expected": f"mixed case {i}!".strip()}) for i in range(n)]


def _truth(task, output):
    """Ground truth: does the output match the expected answer exactly?"""
    return 1.0 if output == task.meta["expected"] else 0.0


def _generous_judge(task, output):
    """A cheap proxy that is systematically kind -- the failure mode this exists for.

    It gives partial credit for near-misses, so it scores *above* the truth on
    every output that is close but wrong. A loop optimising it will accept
    changes that ground truth says did nothing.
    """
    return SequenceMatcher(None, output, task.meta["expected"]).ratio()


class GoodAgent:
    def solve(self, skill_text: str, task: Task) -> str:
        out = task.prompt
        if "lowercase" in skill_text:
            out = out.lower()
        if "Strip" in skill_text:
            out = out.strip()
        return out

    def propose(self, skill_text, task, output, reward) -> Optional[str]:
        if "lowercase" not in skill_text:
            return "Convert the text to lowercase."
        if "Strip" not in skill_text:
            return "Strip surrounding whitespace."
        return None


def _record(**over):
    base = dict(record_id="r1", task_id="t0", artifact_signature="sig",
                output="out", verifier_version="v1", verifier_score=0.5,
                inclusion_prob=0.25, purpose=Purpose.CALIBRATION)
    base.update(over)
    return AuditRecord(**base)


# -- the claim ---------------------------------------------------------------

def test_auditing_a_run_does_not_change_the_run():
    """Constraint 1, end to end: same seed, same result, audit on or off.

    Not a unit test of the return value -- a whole `evolve()`, so that anything
    that leaked truth into a gate, a cache key, or the order of a concurrent
    evaluation would show up here.
    """
    kw = dict(agent=GoodAgent(), rounds=5, n_workers=2, seed=7)

    plain = evolve(_tasks(), _generous_judge, **kw)

    audited = AuditedReward(_generous_judge, oracle=GoldAnswer(_truth),
                            sample_rate=1.0)          # audit *everything*
    with_audit = evolve(_tasks(), audited, **kw)

    assert with_audit.final_reward == plain.final_reward
    assert [r.held_out_reward for r in with_audit.history] == \
           [r.held_out_reward for r in plain.history]
    # and the audit really did run, so the equality above is not vacuous
    assert audited.audited > 0
    assert len(audited.calibration_set()) > 0


def test_the_tap_returns_the_verifiers_score_untouched():
    audited = AuditedReward(_generous_judge,
                            oracle=GoldAnswer(lambda t, o: 0.0),
                            sample_rate=1.0)
    task = _tasks(1)[0]
    for out in ["mixed case 0!", "MiXeD Case 0!", "", "wrong"]:
        assert audited(task, out) == _generous_judge(task, out)


def test_a_generous_judge_shows_up_as_a_positive_residual():
    """The whole point: the pair (f, Y) is measured on the *same* output."""
    audited = AuditedReward(_generous_judge, oracle=GoldAnswer(_truth),
                            sample_rate=1.0)
    task = _tasks(1)[0]
    audited(task, "mixed case 0")            # one character short of correct
    rec = audited.store.all()[0]
    assert rec.verifier_score > 0.9          # the judge is nearly convinced
    assert rec.oracle_score == 0.0           # the truth is not
    assert rec.residual > 0.9                # and the bias has a sign


# -- sampling ----------------------------------------------------------------

def test_inclusion_is_deterministic_in_the_unit_not_in_call_order():
    """Concurrency must not decide who gets audited.

    Evaluation runs on up to `eval_concurrency` threads. Drawing from a shared
    RNG would make inclusion depend on which thread arrived first, so the same
    run replayed would audit a different sample and `sampler_seed` would document
    nothing.
    """
    tasks = _tasks(60)
    a = AuditedReward(_generous_judge, sample_rate=0.3, seed=3)
    b = AuditedReward(_generous_judge, sample_rate=0.3, seed=3)

    for t in tasks:
        a(t, t.prompt)
    for t in reversed(tasks):                 # same units, opposite order
        b(t, t.prompt)

    assert {r.task_id for r in a.store.all()} == {r.task_id for r in b.store.all()}
    assert 0 < a.audited < len(tasks)         # the rate actually did something


def test_a_different_seed_draws_a_different_sample():
    tasks = _tasks(60)
    a = AuditedReward(_generous_judge, sample_rate=0.3, seed=1)
    b = AuditedReward(_generous_judge, sample_rate=0.3, seed=2)
    for t in tasks:
        a(t, t.prompt)
        b(t, t.prompt)
    assert {r.task_id for r in a.store.all()} != {r.task_id for r in b.store.all()}


def test_the_recorded_inclusion_probability_is_the_one_that_drew_the_unit():
    """Constraint 4. Per-stratum rates make this non-trivial: a single global
    number on the record would be wrong for every stratum but one."""
    audited = AuditedReward(
        _generous_judge, sample_rate=0.0,
        stratify=lambda t, o, s: "high" if s > 0.5 else "low",
        rates={"high": 1.0, "low": 1.0})
    task = _tasks(1)[0]
    audited(task, task.meta["expected"])      # scores 1.0 -> "high"
    audited(task, "zzzzzzzz")                 # scores ~0  -> "low"
    by_stratum = {r.stratum: r for r in audited.store.all()}
    assert set(by_stratum) == {"high", "low"}
    assert all(r.inclusion_prob == 1.0 for r in by_stratum.values())


def test_a_zero_rate_stratum_is_never_audited():
    audited = AuditedReward(
        _generous_judge, sample_rate=1.0,
        stratify=lambda t, o, s: "skip" if s > 0.5 else "keep",
        rates={"skip": 0.0})
    task = _tasks(1)[0]
    audited(task, task.meta["expected"])
    audited(task, "zzzzzzzz")
    assert [r.stratum for r in audited.store.all()] == ["keep"]


def test_the_two_pools_are_disjoint_and_roughly_the_requested_split():
    tasks = _tasks(400)
    audited = AuditedReward(_generous_judge, sample_rate=1.0,
                            calibration_fraction=0.7, seed=11)
    for t in tasks:
        audited(t, t.prompt)
    recs = audited.store.all()
    cal = [r for r in recs if r.purpose is Purpose.CALIBRATION]
    imp = [r for r in recs if r.purpose is Purpose.IMPROVEMENT]
    assert len(cal) + len(imp) == len(recs)
    assert not ({r.record_id for r in cal} & {r.record_id for r in imp})
    assert 0.6 < len(cal) / len(recs) < 0.8


# -- the store ---------------------------------------------------------------

def test_append_resolve_round_trip_through_disk(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    rec = store.append(_record())
    assert store.resolve(rec.record_id, 0.25)

    reread = AuditStore(path)
    assert len(reread) == 1
    got = reread.get(rec.record_id)
    assert got.oracle_score == 0.25
    assert got.resolved_at is not None
    assert got.residual == pytest.approx(0.25)


def test_for_calibration_never_returns_an_improvement_record():
    """Constraint 2. The pool used to *edit* the verifier is biased towards
    agreeing with it, so a bias estimated on it reads as honest."""
    store = AuditStore()
    store.append(_record(record_id="c", purpose=Purpose.CALIBRATION))
    store.append(_record(record_id="i", purpose=Purpose.IMPROVEMENT))
    store.resolve("c", 0.1)
    store.resolve("i", 0.9)
    got = store.for_calibration("v1")
    assert [r.record_id for r in got] == ["c"]


def test_for_calibration_filters_by_verifier_version():
    """Constraint 5: a correction belongs to the instrument that earned it."""
    store = AuditStore()
    store.append(_record(record_id="a", verifier_version="v1"))
    store.append(_record(record_id="b", verifier_version="v2"))
    store.resolve("a", 0.1)
    store.resolve("b", 0.2)
    assert [r.record_id for r in store.for_calibration("v1")] == ["a"]
    assert [r.record_id for r in store.for_calibration("v2")] == ["b"]


def test_an_unresolved_record_does_not_enter_calibration():
    store = AuditStore()
    store.append(_record(record_id="pending"))
    assert store.for_calibration("v1") == []
    assert [r.record_id for r in store.pending()] == ["pending"]


def test_a_torn_final_line_is_skipped_rather_than_raised(tmp_path):
    """A process killed mid-write damages the last line and nothing before it."""
    path = str(tmp_path / "audit.jsonl")
    store = AuditStore(path)
    store.append(_record(record_id="a"))
    store.append(_record(record_id="b"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"record_id": "c", "task_id": "t0", "outp')   # killed here

    reread = AuditStore(path)
    assert {r.record_id for r in reread.all()} == {"a", "b"}
    assert reread.corrupt == 1


def test_a_foreign_schema_version_is_an_error_not_a_guess(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    payload = _record().to_dict()
    payload["schema_version"] = 99
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        AuditStore(path)


def test_resolving_twice_is_refused():
    """Replaying a half-failed batch file must not silently change the sample."""
    store = AuditStore()
    store.append(_record())
    store.resolve("r1", 0.3)
    with pytest.raises(ValueError, match="already resolved"):
        store.resolve("r1", 0.9)
    assert store.get("r1").oracle_score == 0.3
    assert store.reopen("r1")
    assert store.resolve("r1", 0.9)
    assert store.get("r1").oracle_score == 0.9


def test_resolving_an_unknown_record_reports_rather_than_raises():
    assert AuditStore().resolve("nope", 1.0) is False


def test_an_impossible_inclusion_probability_is_rejected():
    with pytest.raises(ValueError, match="inclusion_prob"):
        _record(inclusion_prob=0.0)
    with pytest.raises(ValueError, match="inclusion_prob"):
        _record(inclusion_prob=1.5)


def test_a_score_without_a_resolution_time_is_rejected():
    with pytest.raises(ValueError, match="together"):
        _record(oracle_score=0.5)


# -- oracle sources ----------------------------------------------------------

def test_a_deferred_oracle_records_the_question_and_returns_immediately():
    """Constraint 3: an experiment cannot be awaited on the evaluation path."""
    oracle = DeferredOracle()
    audited = AuditedReward(_generous_judge, oracle=oracle, sample_rate=1.0)
    task = _tasks(1)[0]
    assert audited(task, "mixed case 0") == pytest.approx(
        _generous_judge(task, "mixed case 0"))

    pending = audited.pending()
    assert len(pending) == 1
    assert pending[0].oracle_score is None
    assert oracle.queued() == [pending[0].record_id]
    # the output is kept, because that is what the experiment is run against
    assert pending[0].output == "mixed case 0"


def test_a_deferred_answer_lands_later_possibly_from_another_process(tmp_path):
    path = str(tmp_path / "audit.jsonl")
    audited = AuditedReward(_generous_judge, oracle=DeferredOracle(),
                            store=AuditStore(path), sample_rate=1.0)
    for t in _tasks(3):
        audited(t, t.prompt)

    # ... a week passes, the experiments run, a different process reports back
    elsewhere = AuditStore(path)
    answers = {r.record_id: 0.5 for r in elsewhere.pending()}
    assert resolve_from_mapping(elsewhere, answers) == 3
    assert elsewhere.pending() == []
    assert summarise(elsewhere.all())["n_resolved"] == 3


def test_a_null_oracle_still_accumulates_answerable_questions():
    audited = AuditedReward(_generous_judge, oracle=NullOracle(), sample_rate=1.0)
    task = _tasks(1)[0]
    audited(task, "whatever")
    rec = audited.store.all()[0]
    assert rec.oracle_score is None
    assert rec.inclusion_prob == 1.0 and rec.verifier_version


def test_an_oracle_that_raises_cannot_fail_the_rollout():
    """The audit is a side channel. It may lose a record; it may not lose a score."""
    def explodes(task, output):
        raise RuntimeError("the instrument is down")

    oracle = GoldAnswer(explodes)
    audited = AuditedReward(_generous_judge, oracle=oracle, sample_rate=1.0)
    task = _tasks(1)[0]
    assert audited(task, "mixed case 0") == pytest.approx(
        _generous_judge(task, "mixed case 0"))
    assert len(oracle.errors) == 1
    assert audited.store.all()[0].oracle_score is None   # recorded as unanswered


def test_a_store_that_raises_cannot_fail_the_rollout():
    class Broken(AuditStore):
        def append(self, record):
            raise OSError("disk full")

    audited = AuditedReward(_generous_judge, store=Broken(), sample_rate=1.0)
    task = _tasks(1)[0]
    assert audited(task, "x") == pytest.approx(_generous_judge(task, "x"))


# -- versioning --------------------------------------------------------------

def test_editing_the_verifier_changes_its_fingerprint():
    def judge_v1(task, output):
        return 1.0

    def judge_v2(task, output):
        return 0.5                       # a different body

    assert verifier_fingerprint(judge_v1) != verifier_fingerprint(judge_v2)
    assert verifier_fingerprint(judge_v1) == verifier_fingerprint(judge_v1)


def test_an_agent_verifier_is_versioned_by_what_it_is_told_not_by_its_source():
    """An LLM judge's behaviour lives in its prompt and model, neither of which
    appears in the source of the function that calls it."""
    def judge(task, output):
        return 1.0

    a = verifier_fingerprint(judge, extra={"model": "m1", "prompt": "p1"})
    b = verifier_fingerprint(judge, extra={"model": "m2", "prompt": "p1"})
    assert a != b


def test_records_carry_the_version_that_scored_them():
    audited = AuditedReward(_generous_judge, sample_rate=1.0,
                            version_extra={"prompt": "v-alpha"})
    task = _tasks(1)[0]
    audited(task, "x")
    assert audited.store.all()[0].verifier_version == audited.verifier_version
    other = AuditedReward(_generous_judge, sample_rate=1.0,
                          version_extra={"prompt": "v-beta"})
    assert other.verifier_version != audited.verifier_version


# -- tracing which artifact answered -----------------------------------------

def test_the_render_tap_records_which_artifact_produced_the_output():
    audited = AuditedReward(_generous_judge, sample_rate=1.0)
    tap = RenderTap(lambda rendered, task: rendered.upper())
    task = _tasks(1)[0]
    out = tap("some skill text", task)
    audited(task, out)
    assert audited.store.all()[0].artifact_signature != ""


def test_without_a_render_tap_the_signature_is_empty_rather_than_wrong():
    audited = AuditedReward(_generous_judge, sample_rate=1.0)
    audited(_tasks(1)[0], "out")
    assert audited.store.all()[0].artifact_signature == ""


# -- guardrails on the summary ------------------------------------------------

def test_summarise_counts_a_generator_correctly():
    store = AuditStore()
    store.append(_record(record_id="a"))
    store.append(_record(record_id="b"))
    store.resolve("a", 0.25)
    got = summarise(r for r in store.all())
    assert got["n_records"] == 2 and got["n_resolved"] == 1
    assert got["mean_residual"] == pytest.approx(0.25)
