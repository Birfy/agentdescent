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
        # A gold its own oracle can actually score. Picked by the oracle rather
        # than by the workload's name -- a numeric oracle scores "Ottawa" wrong
        # against itself, which would pass the near-miss assertion below for the
        # wrong reason and fail the sanity one.
        if oracle is P.tests_pass:
            gold = "def add(a, b):\n    return a + b"
            task = _task(gold, meta={"tests": ["assert add(1, 2) == 3"]})
        else:
            gold = "18" if oracle is P.number_match else "Ottawa"
            task = _task(gold)
        near = P._NEAR_MISS[workload](gold)
        assert oracle(task, near) == 0.0, f"{workload}: near-miss was accepted"
        assert oracle(task, gold) == 1.0, f"{workload}: gold was refused"


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


# -- MBPP: the oracle executes ------------------------------------------------

def test_the_mbpp_oracle_runs_the_tasks_own_asserts():
    task = P.Task(id="t", prompt="q", meta={
        "gold": "def add(a, b):\n    return a + b",
        "expected": "", "tests": ["assert add(1, 2) == 3"]})
    assert P.tests_pass(task, "def add(a, b):\n    return a + b") == 1.0
    assert P.tests_pass(task, "def add(a, b):\n    return a * b") == 0.0


@pytest.mark.parametrize("code,ok", [
    ("def f(a, b): return a + b", True),
    ("def f(a, b): return a * b", False),
    ("def f(a, b): raise ValueError()", False),      # a crash is wrong
    ("def f(:\n  bad", False),                       # so is a syntax error
])
def test_a_crash_a_hang_and_a_failed_assert_are_all_one_answer(code, ok):
    """The distinction matters to whoever is fixing the candidate and not to an
    oracle, whose whole job is a bit."""
    assert P.run_tests(code, ["assert f(1, 2) == 3"]) is ok


def test_a_hanging_candidate_is_wrong_rather_than_hanging_the_run():
    assert P.run_tests("while True:\n    pass", ["assert True"], timeout=3) is False


def test_the_oracle_reads_code_out_of_a_fenced_block():
    """A model asked for a function answers with prose around a fence about half
    the time. Executing the prose is a syntax error, which the oracle would
    score as a wrong answer -- so the oracle would be measuring markdown."""
    fenced = "Here you go:\n```python\ndef f(a, b): return a + b\n```\nHope that helps."
    assert P.extract_code(fenced) == "def f(a, b): return a + b"
    assert P.run_tests(P.extract_code(fenced), ["assert f(1, 2) == 3"]) is True


def test_an_mbpp_task_carries_the_first_assert_in_its_prompt():
    """MBPP's description does not name the function, so without the assert
    every candidate fails on the name and the oracle measures naming rather
    than correctness."""
    task = P._mbpp_task({"text": "Write a function to add two numbers.",
                         "code": "def add(a, b): return a + b",
                         "test_list": ["assert add(1, 2) == 3",
                                       "assert add(2, 2) == 4"]})
    assert "assert add(1, 2) == 3" in task.prompt
    assert task.meta["gold"] == "def add(a, b): return a + b"
    assert task.meta["tests"] == ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]


def test_the_judge_is_not_shown_the_tests():
    """`_JUDGE_TMPL` is formatted with `meta["gold"]`. A judge shown the asserts
    could evaluate them in its head and would be doing the oracle's job -- which
    is the one thing that would make this workload measure nothing."""
    task = P._mbpp_task({"text": "Add two numbers.",
                         "code": "def add(a, b): return a + b",
                         "test_list": ["assert add(1, 2) == 3"]})
    shown = P._JUDGE_TMPL.format(question=task.prompt, gold=task.meta["gold"],
                                 candidate="def add(a, b): return a + b")
    assert "assert add(2, 2)" not in shown          # only the signature hint
    assert task.meta["tests"][0] not in shown.split("Reference answer:")[1]


def test_an_mbpp_row_without_tests_or_a_reference_is_skipped():
    assert P._mbpp_task({"text": "x", "code": "def f(): pass", "test_list": []}) is None
    assert P._mbpp_task({"text": "", "code": "c", "test_list": ["assert 1"]}) is None
