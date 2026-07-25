"""Offline tests for the EvoSkill example.

Exercise the faithful numeric scorer (unit-aware tolerance + multi-tolerance
weighting), the bounded top-K frontier, doc retrieval, and the loop. No network.
"""

from examples.evoskill_skill_discovery import (
    Frontier,
    PASS_THRESHOLD,
    extract_numbers,
    fuzzy_match_answer,
    retrieve_context,
    run_evoskill,
    score_multi_tolerance,
)


def test_extract_numbers_scales_units():
    assert 2602.0 in extract_numbers("2,602")
    scaled = extract_numbers("2,602 million")
    assert 2602e6 in scaled and 2602.0 in scaled     # scaled AND bare kept


def test_fuzzy_match_tolerance():
    assert fuzzy_match_answer("2602", "2,602", 0.0)
    assert fuzzy_match_answer("2602", "2650", 0.05)          # within 5%
    assert not fuzzy_match_answer("2602", "3000", 0.05)      # outside 5%
    assert fuzzy_match_answer("507", "about 507 million dollars", 0.0)


def test_fuzzy_match_multi_number_requires_all():
    assert fuzzy_match_answer("100 and 200", "we found 200 and 100", 0.0)
    assert not fuzzy_match_answer("100 and 200", "only 100", 0.0)


def test_multi_tolerance_weighting():
    assert abs(score_multi_tolerance("2,602", "2,602") - 1.0) < 1e-9   # exact -> 1
    assert score_multi_tolerance("", "2602") == 0.0                    # empty -> 0
    partial = score_multi_tolerance("2650", "2602")   # matches loose tols only
    assert 0.0 < partial < 1.0


def test_pass_threshold_is_0_8():
    assert PASS_THRESHOLD == 0.8


def test_frontier_bounded_topk():
    f = Frontier(max_size=2)
    assert f.update({"a": "1"}, 0.5)          # room
    assert f.update({"b": "2"}, 0.7)          # room
    assert not f.update({"c": "3"}, 0.4)      # full, worse than worst(0.5) -> reject
    assert f.update({"d": "4"}, 0.9)          # replaces the worst member
    assert abs(f.select_parent()[1] - 0.9) < 1e-9   # parent = best


def test_retrieve_context_picks_relevant_lines():
    doc = "\n".join(["defense spending 1940 was 2602",
                     "an unrelated line about weather", "veterans 507"])
    ctx = retrieve_context(doc, "total national defense expenditures 1940", n_lines=1)
    assert "defense" in ctx


def test_loop_discovers_a_helpful_skill():
    class Stub:
        def __call__(self, prompt):
            if "Skill Proposer" in prompt:
                return "create defense-lookup\nLook up defense totals."
            if "Skill Generator" in prompt:
                return "Defense lookup\n- Find the national defense row and report millions."
            return "Answer: 2602" if "skill:" in prompt else "Answer: 0"

    train = [{"question": "defense 1940?", "answer": "2602",
              "source_files": "", "difficulty": "hard"} for _ in range(4)]
    val = [{"question": "defense?", "answer": "2602",
            "source_files": "", "difficulty": "hard"} for _ in range(3)]
    res = run_evoskill(Stub(), {}, train, val, iterations=3, seed=0)
    assert res.seed_score == 0.0
    assert res.best_score > res.seed_score
    assert len(res.skills) >= 1
