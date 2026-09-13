"""The offline half of `scripts/audit_diagnose.py`.

The script is the reproducible source of every number the audit package's
docstrings quote, so the parts that do not need the network are tested like
library code. The dataset join is not: it needs `hf_rows`, and a test that
silently skips when a cache is cold would be a test of nothing.
"""

import json
import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.diagnose import evaluate_fix
from scripts.audit_modes import resolved_records
from scripts.audit_diagnose import (echoes_the_question,
                                    far_shorter_than_reference,
                                    make_fix, normalise)


def _rec(f, y, *, out="o", task="t"):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task, artifact_signature="s",
        output=out, verifier_version="v1", verifier_score=f, inclusion_prob=1.0,
        purpose=Purpose.CALIBRATION, oracle_score=y, resolved_at=1.0)


# -- normalisation -----------------------------------------------------------

def test_normalise_is_the_oracles_own_function_not_a_copy_of_it():
    """A FORMATTING disagreement is defined as one the oracle's normalisation
    already handles, so the diagnosis has to use the oracle's normalisation --
    the same object, not an equivalent one. A copy that replaced punctuation
    with a space instead of deleting it turned `cat's` into `cat s`."""
    from scripts import audit_phase0

    assert normalise is audit_phase0.normalize
    assert normalise("The Answer!") == "answer"
    assert normalise("  a   Cat's  paw ") == "cats paw"
    assert normalise("from 1986 to 2013") == "from 1986 to 2013"


# -- the rules ---------------------------------------------------------------

def test_the_echo_rule_fires_on_an_answer_lifted_from_the_question():
    q = "Which county is Fairfax City surrounded by?"
    assert echoes_the_question("Fairfax City", q)
    assert not echoes_the_question("Fairfax County", q)


def test_the_echo_rule_ignores_an_empty_answer():
    assert not echoes_the_question("", "anything at all")


def test_the_length_rule_needs_both_sides_and_a_real_difference():
    assert far_shorter_than_reference("Childers", "Robert Erskine Childers DSC", 0.6)
    assert not far_shorter_than_reference("Childers", "Childers", 0.6)
    assert not far_shorter_than_reference("", "Childers", 0.6)
    assert not far_shorter_than_reference("Childers", "", 0.6)


def test_the_length_rules_threshold_is_a_dial_not_a_constant():
    """The verdict on the pair of rules turns on it, which is the point of
    passing it in rather than hard-coding 0.6."""
    assert far_shorter_than_reference("ab", "abcdef", 0.5)
    assert not far_shorter_than_reference("abcd", "abcdef", 0.5)
    assert far_shorter_than_reference("abcd", "abcdef", 0.9)


# -- composition -------------------------------------------------------------

_CONTEXT = {"echo": ("where is Fairfax City", "Fairfax County"),
            "short": ("who was he", "Robert Erskine Childers DSC"),
            "fine": ("who was he", "Robert Erskine Childers DSC")}


def _sample():
    return [_rec(1.0, 0.0, task="echo", out="Fairfax City"),
            _rec(1.0, 0.0, task="short", out="Childers"),
            _rec(1.0, 1.0, task="fine", out="Robert Erskine Childers DSC")]


def test_each_rule_can_be_run_alone():
    recs = _sample()
    echo_only = [make_fix(echo=True, shorter=0.0)(r, _CONTEXT[r.task_id])
                 for r in recs]
    short_only = [make_fix(echo=False, shorter=0.6)(r, _CONTEXT[r.task_id])
                  for r in recs]
    assert echo_only == [0.0, 1.0, 1.0]
    assert short_only == [1.0, 0.0, 1.0]


def test_a_bundle_is_the_union_of_what_each_rule_rejects():
    recs = _sample()
    both = [make_fix(echo=True, shorter=0.6)(r, _CONTEXT[r.task_id]) for r in recs]
    assert both == [0.0, 0.0, 1.0]


def test_a_rule_never_promotes_an_answer_the_verifier_already_rejected():
    """It can only reject, so a score of 0 comes back untouched -- otherwise a
    "hard rule" could raise the false-negative rate and the score at once."""
    rejected = _rec(0.0, 1.0, task="echo", out="Fairfax City")
    assert make_fix()(rejected, _CONTEXT["echo"]) == 0.0


def test_a_record_with_no_joined_question_is_left_as_it_was():
    assert make_fix()(_rec(1.0, 0.0, out="anything"), None) == 1.0


def test_the_bundle_hides_a_rule_that_does_not_help():
    """The finding, in miniature.

    Rule A trades one error for one; rule B corrects two and breaks nothing.
    Alone, A does not help. Bundled, the pair does -- and every number the
    bundle reports is better than the verifier it replaces.
    """
    traded = [_rec(1.0, 0.0, task="a-fixes"), _rec(1.0, 1.0, task="a-breaks")]
    clean = [_rec(1.0, 0.0, task=f"b{i}") for i in range(2)]
    recs = [_rec(1.0, 1.0, task=f"ok{i}") for i in range(7)] + traded + clean
    a_targets = {traded[0].record_id, traded[1].record_id}
    b_targets = {r.record_id for r in clean}

    def rule_a(rec, ctx):
        return 0.0 if rec.record_id in a_targets else rec.verifier_score

    def rule_b(rec, ctx):
        return 0.0 if rec.record_id in b_targets else rec.verifier_score

    def both(rec, ctx):
        return min(rule_a(rec, ctx), rule_b(rec, ctx))

    assert not evaluate_fix(recs, rule_a).helps
    assert evaluate_fix(recs, rule_b).helps
    bundle = evaluate_fix(recs, both)
    assert bundle.helps and bundle.broken == 1, (
        "the bundle passes while still carrying the rule that breaks things")


# -- loading -----------------------------------------------------------------
#
# The scripts used to carry two hand-rolled JSONL readers, which were
# reimplementing `AuditStore`'s parser next to it. They are gone; these tests
# now pin the properties the scripts *depend on*, against `resolved_records`.

def _line(**kw):
    row = dict(record_id="r1", task_id="t", artifact_signature="s", output="o",
               verifier_version="v1", verifier_score=1.0, inclusion_prob=1.0,
               purpose="calibration", stratum="all", oracle_score=None,
               dispatched_at=1.0, resolved_at=None, sampler_seed=0,
               schema_version=1)
    row.update(kw)
    return json.dumps(row)


def _resolved(**kw):
    return _line(oracle_score=0.0, resolved_at=2.0, **kw)


def test_the_last_occurrence_of_a_record_wins(tmp_path):
    """Resolution is an append in the store's own format, so a naive reader
    would report every audited unit as still pending."""
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join([_line(), _resolved()]))
    got = resolved_records(path)
    assert len(got) == 1 and got[0].oracle_score == 0.0


def test_an_unresolved_record_is_not_returned(tmp_path):
    """Every caller filtered for this itself; the filter is the only thing any
    of them wanted on top of the store's own reader."""
    path = tmp_path / "a.jsonl"
    path.write_text(_line())
    assert resolved_records(path) == []


def test_moments_snapshots_are_not_records(tmp_path):
    """A store holds unlabelled-moment snapshots alongside records, and they
    must not come back as audited units.

    The snapshot here is the **real** shape. The version this replaces was a
    stub -- `kind` plus three fields, no `moments` -- which the hand-rolled
    reader accepted because it only ever looked at `kind`. `AuditStore` parses
    it, so a malformed snapshot now raises instead of being silently skipped,
    and the fixture had to become a real one to keep the test honest."""
    path = tmp_path / "a.jsonl"
    path.write_text("\n".join([
        _resolved(),
        json.dumps({"kind": "unlabelled_moments", "verifier_version": "v1",
                    "stratum": "all",
                    "moments": {"n": 10, "mean": 0.5, "m2": 2.0}}),
        "",
    ]))
    got = resolved_records(path)
    assert len(got) == 1 and got[0].record_id == "r1"


def test_the_purpose_comes_back_as_the_enum(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(_resolved(purpose="improvement"))
    assert resolved_records(path)[0].purpose is Purpose.IMPROVEMENT
