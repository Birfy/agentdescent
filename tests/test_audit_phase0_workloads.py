"""The workload registry in `scripts/audit_phase0.py`, once there were three.

Everything here was a single-workload assumption that a second or third
workload made wrong. The dataset loaders are not tested: they need `hf_rows`,
and a test that silently skipped on a cold cache would be a test of nothing.
"""

import pytest

from scripts import audit_phase0 as P


def _task(gold, *, meta=None):
    return P.Task(id="t", prompt="q",
                  meta={"gold": gold, "expected": gold, **(meta or {})})


def test_every_workload_has_an_oracle_a_label_a_near_miss_and_an_indexer():
    """Six tables keyed by workload, and a workload missing from any one of them
    fails at a different point -- the oracle at the first score, the label only
    in the written report, and `_INDEXERS` not until something asks for that
    workload's gold answers, which is *after* its run has finished and paid for
    itself. `gsm_hard` shipped with a loader, four table entries and no indexer
    branch; it would have failed at `context_for` with the records already
    written, which is why `task_index` dispatches through a table now."""
    for table in (P.ORACLES, P._ORACLE_LABELS, P._WORKLOAD_LABELS,
                  P._NEAR_MISS, P._WRONG, P._INDEXERS):
        assert set(table) == set(P.WORKLOADS)


def test_the_free_text_oracle_is_shared_and_the_numeric_one_is_not():
    assert P.ORACLES["hotpot"] is P.ORACLES["bbh"] is P.exact_match
    assert P.ORACLES["gsm8k"] is P.number_match


def test_exact_match_would_have_scored_gsm8ks_own_answers_wrong():
    """Why the oracle had to become per-workload: a workload whose oracle
    refuses its own correct answers measures the oracle, not the judge."""
    task = _task("18")
    # `normalize` deletes punctuation rather than replacing it, so `18.00`
    # becomes `1800`. `$18` survives it and is *not* the example to use here --
    # the first draft of this test used it and passed for the wrong reason.
    assert P.exact_match(task, "18.00") == 0.0
    assert P.number_match(task, "18.00") == 1.0


def test_each_workloads_near_miss_is_one_its_own_oracle_refuses():
    """`The answer is 18.` is a near-miss for exact match and simply *correct*
    for `number_match`, so the shared string gave the GSM8K dry run no
    disagreements -- a rehearsal that exercises none of the paths it exists to
    rehearse."""
    for workload, oracle in P.ORACLES.items():
        # Picked by the oracle, not by the workload's name: a numeric oracle
        # scores "Ottawa" wrong against itself, which would pass the near-miss
        # assertion below for the wrong reason and fail the sanity one.
        gold = "18" if oracle is P.number_match else "Ottawa"
        near = P._NEAR_MISS[workload](gold)
        assert oracle(_task(gold), near) == 0.0, workload
        assert oracle(_task(gold), gold) == 1.0, workload


def test_the_offline_judge_takes_the_workloads_oracle_not_a_default():
    """With `exact_match` hard-coded the GSM8K dry run's known `Delta` -- the one
    thing the stand-in exists to make known -- would be about the wrong
    oracle."""
    judge = P.offline_judge(generosity=0.0, seed=0, oracle=P.number_match)
    assert judge(_task("18"), "18.00") == 1.0      # never marked down
    lenient = P.offline_judge(generosity=0.0, seed=0, oracle=P.exact_match)
    assert lenient(_task("18"), "18.00") == 0.0


def test_the_offline_judge_only_ever_scores_up():
    """Symmetric noise would make a broken estimator look fine -- it is unbiased
    on balanced binary outcomes."""
    judge = P.offline_judge(generosity=1.0, seed=0, oracle=P.number_match)
    assert judge(_task("18"), "18") == 1.0
    assert judge(_task("18"), "17") == 1.0        # forgiven, never marked down


def test_gsm8k_wrong_answers_span_more_than_one_error_mode():
    """A single `"unknown"` for every wrong answer gives the whole dry run one
    error mode, so `P(new)` comes back 0.0 and anything gated on coverage
    refuses to run -- which reads as a finding about the workload and is a
    property of the stand-in solver."""
    import random

    from scripts.audit_modes import gsm8k_error_mode

    class _Rec:
        task_id = "t"

        def __init__(self, output):
            self.output = output

    rng = random.Random(0)
    modes = {gsm8k_error_mode(_Rec(shape("18", rng)), ("q", "18", None))
             for shape in P._WRONG["gsm8k"] for _ in range(20)}
    assert len(modes) >= 3


def test_a_gsm8k_row_without_a_marked_answer_is_skipped_rather_than_guessed():
    assert P._gsm8k_task({"question": "q", "answer": "no marker here"}) is None
    assert P._gsm8k_task({"question": "", "answer": "x\n#### 18"}) is None


def test_the_gsm8k_task_id_is_hashed_from_the_question_not_its_position():
    """A positional id is only meaningful together with the `limit` and `seed`
    that produced the shuffle, so a records file written today could not be
    re-joined to its gold answers tomorrow."""
    row = {"question": "Janet sells eggs.", "answer": "16 - 7 = 9\n#### 18"}
    first = P._gsm8k_task(row)
    assert first.id == P._gsm8k_task(dict(row)).id
    assert first.id.startswith("gsm8k:")
    assert first.meta["gold"] == "18"
    # The derivation is kept for the diagnosis and never scored.
    assert first.meta["worked"] == "16 - 7 = 9"
    other = P._gsm8k_task({"question": "Something else.", "answer": "#### 18"})
    assert other.id != first.id
