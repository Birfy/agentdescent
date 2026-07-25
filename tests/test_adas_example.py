"""Offline tests for the ADAS Meta Agent Search example.

Exercise the safe DSL validation + interpreter, MGSM scoring, bootstrap fitness,
the DGM parent-selection formula, and the search loop. No network or LLM calls.
"""

from examples.adas_meta_agent_search import (
    Interpreter,
    _extract_int,
    _majority,
    _parse_agent,
    bootstrap_ci,
    dgm_parent_weights,
    run_meta_agent_search,
    score_mgsm,
    seed_archive,
    validate_program,
)


def test_dsl_validation_rejects_unknown_blocks():
    assert validate_program({"block": "cot"})
    assert validate_program({"block": "ensemble",
                             "children": [{"block": "cot"}, {"block": "reflexion", "n": 1}]})
    assert not validate_program({"block": "exec", "code": "import os"})  # arbitrary code
    assert not validate_program({"block": "ensemble", "children": []})   # empty ensemble
    assert not validate_program("cot")                                   # not a dict


def test_all_seeds_are_valid_programs():
    assert all(validate_program(a["program"]) for a in seed_archive())
    assert len(seed_archive()) == 7   # ADAS's seven MGSM seeds


def test_mgsm_scoring_matches_adas():
    assert score_mgsm("18", "18")
    assert score_mgsm("18", "18.0")        # trailing-zero strip
    assert score_mgsm("1024", "1,024")     # comma strip
    assert not score_mgsm("18", "19")
    assert not score_mgsm("18", None)


def test_answer_extraction_and_majority():
    assert _extract_int("so the Answer: 1,024 dollars") == "1024"
    assert _extract_int("... = 70000") == "70000"
    assert _extract_int("no number here") is None
    assert _majority(["3", "3", "4", None]) == "3"


def test_bootstrap_ci_mean():
    mean, lo, hi = bootstrap_ci([1, 1, 1, 0, 0], seed=0)
    assert abs(mean - 0.6) < 1e-9
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_dgm_parent_weights_favor_performance_and_novelty():
    # higher score -> higher weight
    w = dgm_parent_weights([0.9, 0.5, 0.1], [0, 0, 0])
    assert w[0] > w[1] > w[2]
    assert abs(sum(w) - 1.0) < 1e-9
    # same score, more children -> lower weight (novelty discount)
    w2 = dgm_parent_weights([0.9, 0.9], [0, 5])
    assert w2[0] > w2[1]


def test_interpreter_runs_blocks_with_stub():
    stub = lambda prompt: "reasoning... Answer: 42"
    interp = Interpreter(stub)
    assert interp.run({"block": "cot"}, "q") == "42"
    assert interp.run({"block": "cot_sc", "k": 3}, "q") == "42"
    assert interp.run({"block": "reflexion", "n": 1}, "q") == "42"
    assert interp.run({"block": "ensemble",
                       "children": [{"block": "cot"}, {"block": "step_back"}]}, "q") == "42"


def test_meta_agent_parsing_validates_program():
    good = '{"thought":"t","name":"N","program":{"block":"cot_sc","k":3}}'
    assert _parse_agent(good)["program"]["block"] == "cot_sc"
    assert _parse_agent('{"program":{"block":"nope"}}') is None   # invalid block
    assert _parse_agent("no json here") is None


def test_search_evaluates_seed_archive():
    stub = lambda prompt: "Answer: 42"
    val = [("q1", "42"), ("q2", "7"), ("q3", "42")]
    result = run_meta_agent_search(stub, val, generations=1, seed=0)
    # seeds all score 2/3 on this val; the archive keeps every agent (keep-all).
    assert abs(result.seed_fitness - 2 / 3) < 1e-9
    assert len(result.archive) >= len(seed_archive())
