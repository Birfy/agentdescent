"""One error-mode definition per workload, in `scripts/audit_modes.py`.

The coverage number decides whether the improvement pool still gets budget, so
two definitions that drift would answer that question differently in two files
with no way to tell which one ran. These tests are mostly about that: the
identity of the shared functions, and the boundaries between GSM8K's modes.
"""

import uuid

import pytest

from agentdescent.audit import AuditRecord, Purpose
from scripts import audit_diagnose, audit_modes, audit_phase0


def _rec(out, *, task="t", f=1.0, y=0.0):
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task, artifact_signature="s",
        output=out, verifier_version="v1", verifier_score=f, inclusion_prob=1.0,
        purpose=Purpose.IMPROVEMENT, oracle_score=y, resolved_at=1.0)


# -- one definition, not two --------------------------------------------------

def test_the_diagnosis_uses_the_shared_mode_function_rather_than_its_own():
    assert audit_diagnose.error_mode is audit_modes.ERROR_MODES["hotpot"]


def test_the_normaliser_is_still_the_oracles_own_object():
    """A copy of this replaced punctuation with a space where the original
    deletes it, so `cat's` normalised to `cat s` here and `cats` there."""
    assert audit_modes.normalise is audit_phase0.normalize
    assert audit_diagnose.normalise is audit_phase0.normalize


def test_every_workload_has_a_mode_function():
    assert set(audit_modes.ERROR_MODES) == set(audit_phase0.WORKLOADS)
    assert set(audit_modes.ERROR_MODES) == set(audit_phase0.ORACLES)


# -- the GSM8K modes ----------------------------------------------------------

CTX = ("Janet sells eggs. How much?", "18", "16 - 3 - 4 = 9\n9 * 2 = 18")


@pytest.mark.parametrize("out,want", [
    # The mode this workload was added for: correct-looking arithmetic, wrong
    # final number. Neither string workload can produce it -- their outputs
    # carry no derivation to be seduced by.
    ("16 - 3 - 4 = 9, and 9 * 2 = 17 a day.", "wrong-number-with-working"),
    ("17", "wrong-number-bare"),
    ("I could not work this out.", "no-number"),
    # A criticism of the *oracle*, not the judge: the reference number is in the
    # output and the convention of reading the last number did not find it.
    # AMBIGUOUS, and must not be "fixed".
    ("the answer is 18 (over 7 days)", "gold-not-final"),
    ("18", "judge-rejected-right-number"),
])
def test_gsm8k_modes_separate_the_ways_of_being_wrong(out, want):
    assert audit_modes.gsm8k_error_mode(_rec(out), CTX) == want


def test_a_long_wrong_sentence_is_not_counted_as_working():
    """Working means arithmetic, not length. A length threshold would put the
    bare case in the bucket this workload was added to isolate."""
    wordy = "I considered the problem at some length and concluded it is 17."
    assert audit_modes.gsm8k_error_mode(_rec(wordy), CTX) == "wrong-number-bare"


def test_an_unjoined_record_has_no_mode_rather_than_a_wrong_one():
    """An unclassified error is not evidence that a key is exhausted, and
    `coverage_of` does not count `None` as a mode."""
    assert audit_modes.gsm8k_error_mode(_rec("17"), None) is None
    assert audit_modes.text_error_mode(_rec("x"), None) is None


# -- the text modes are unchanged --------------------------------------------

def test_text_modes_still_classify_what_they_did_before_the_move():
    ctx = ("Who wrote the book about the sea?", "Robert Erskine Childers", None)
    mode = audit_modes.text_error_mode
    assert mode(_rec("Who wrote the book about the sea"), ctx) == "echoes-question"
    assert mode(_rec("Robert Erskine"), ctx) == "substring-of-reference"
    assert mode(_rec(""), ctx) == "empty"
    assert mode(_rec("Childers wrote it in 1903 while living in London indeed"),
                ctx) == "far-longer"


def test_predicates_kept_their_meaning_through_the_move():
    assert audit_modes.echoes_the_question("the answer", "what is the answer?")
    assert not audit_modes.echoes_the_question("", "anything")
    assert audit_modes.far_shorter_than_reference(
        "Childers", "Robert Erskine Childers DSC", 0.6)
    assert not audit_modes.far_shorter_than_reference("same", "same", 0.6)
    # `normalise` drops articles, so a bare "a" normalises to nothing and the
    # predicate declines to call an empty string far shorter than anything.
    assert not audit_modes.far_shorter_than_reference("a", "a long one", 0.6)
