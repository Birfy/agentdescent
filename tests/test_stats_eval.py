"""Statistical evidence must be reproducible and contamination must be visible."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from agentdescent import (bootstrap_ci, contamination_overlap, is_significant,
                          paired_gain)


def test_paired_gain_uses_only_shared_keys():
    diffs = paired_gain({"a": 0.2, "b": 0.5},
                        {"a": 0.6, "b": 0.1, "c": 1.0})
    assert sorted(diffs) == pytest.approx([-0.4, 0.4])


def test_a_clear_positive_gain_is_significant_and_reproducible():
    before = {str(i): 0.2 for i in range(50)}
    after = {str(i): 0.8 for i in range(50)}
    one = bootstrap_ci(before, after, seed=7)
    two = bootstrap_ci(before, after, seed=7)
    assert one == two and one.mean == pytest.approx(0.6)
    assert one.lo > 0 and one.prob_positive == 1.0
    assert is_significant(one)


def test_noise_does_not_pass_the_effect_bar():
    before = {str(i): 0.5 for i in range(20)}
    after = {str(i): 0.5 for i in range(20)}
    result = bootstrap_ci(before, after, seed=1)
    assert result.mean == 0.0 and not is_significant(result, min_effect=0.01)


def test_one_pair_is_not_significant():
    result = bootstrap_ci({"a": 0.1}, {"a": 0.9}, seed=0)
    assert result.n == 1 and not is_significant(result)


def test_contamination_finds_an_exact_duplicate():
    train = ["the quick brown fox jumps over the lazy dog"]
    val = ["the quick brown fox jumps over the lazy dog"]
    report = contamination_overlap(train, val, n=5)
    assert report.overlap == 1.0 and report.shared
    assert any(gram.startswith("the quick brown") for gram in report.shared)


def test_contamination_separates_disjoint_text():
    assert contamination_overlap(["alpha beta gamma delta epsilon"],
                                 ["one two three four five"], n=3).overlap == 0.0


def test_parameters_are_validated():
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci({}, {}, alpha=0)
    with pytest.raises(ValueError, match="n"):
        contamination_overlap([], [], n=0)


def test_exact_duplicates_are_visible_even_without_enough_words():
    for text in ("What is 2+2?", "把这句话翻译成英文并解释原因"):
        report = contamination_overlap([text], [text], n=5)
        assert report.overlap == 1.0
        assert report.exact_duplicates == 1


def test_no_grams_is_none_not_zero():
    report = contamination_overlap(["different"], ["ok"], n=5)
    assert report.overlap is None and report.held_out_grams == 0


def test_a_short_unique_validation_item_is_visible_as_no_coverage():
    report = contamination_overlap(["alpha beta"], ["ok"], n=5)
    assert report.overlap is None and report.held_out_grams == 0


def test_alpha_mismatch_is_refused_instead_of_false():
    result = bootstrap_ci({"a": 0.1, "b": 0.1}, {"a": 0.9, "b": 0.9},
                          alpha=0.1, seed=0)
    assert is_significant(result)
    with pytest.raises(ValueError, match="alpha"):
        is_significant(result, alpha=0.05)


def test_non_finite_scores_are_refused():
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_ci({"a": 0.0, "b": 0.0}, {"a": float("nan"), "b": 1.0})


def test_one_pair_reports_no_evidence_not_certainty():
    result = bootstrap_ci({"a": 0.1}, {"a": 0.9}, seed=0)
    assert result.prob_positive == 0.5 and result.n == 1


def test_bootstrap_is_reproducible_across_hash_seeds():
    code = (
        "from agentdescent import bootstrap_ci\n"
        "before={f't{i}': (i%3)/10 for i in range(12)}\n"
        "after={f't{i}': ((i*7)%10)/10 for i in range(12)}\n"
        "r=bootstrap_ci(before, after, seed=0)\n"
        "print(repr((r.lo, r.hi, r.prob_positive)))\n"
    )
    outputs = []
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True).strip())
    assert len(set(outputs)) == 1


def test_short_validation_text_inside_a_longer_training_prompt_is_found():
    report = contamination_overlap(
        ["Q: What is 2+2? Answer with only the number please."],
        ["What is 2+2?"])
    assert report.overlap is not None and report.overlap > 0
    assert report.exact_duplicates == 0
