"""Offline tests for the ADAS Meta Agent Search example.

Exercise the safe DSL validation + interpreter, MGSM scoring, bootstrap fitness,
the DGM parent-selection formula, and the search loop. No network or LLM calls.
"""

import json

from examples.adas.adas_meta_agent_search import (
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
    # runs Meta Agent Search THROUGH evolve() (needs >= 4 tasks to split).
    stub = lambda prompt: "Answer: 42"
    val = [("q1", "42"), ("q2", "7"), ("q3", "42"), ("q4", "42"), ("q5", "1"), ("q6", "42")]
    result = run_meta_agent_search(stub, val, generations=1, seed=0)
    # keep-all archive holds at least every seed; fitness is a valid rate.
    assert len(result.archive) >= len(seed_archive())
    assert 0.0 <= result.seed_fitness <= 1.0
    assert result.best_fitness >= result.seed_fitness


# --- concurrency: ADAS is the most expensive example, and it was serial --------

def test_evaluate_agent_scores_examples_concurrently():
    """This loop *is* the run: every candidate is scored on every example, and
    each score is a multi-step program. Serially, ADAS's effective concurrency was
    ~1 no matter how many workers the engine was given -- measured, a 6-example
    generation had not finished after 25 minutes.
    """
    import threading
    import time

    from examples.adas.adas_meta_agent_search import evaluate_agent

    live, peak, lock = [0], [0], threading.Lock()

    def slow(prompt):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return "Answer: 42"

    examples = [(f"q{i}", "42") for i in range(8)]
    scores = evaluate_agent(Interpreter(slow), {"block": "cot"}, examples)
    assert scores == [1.0] * 8
    assert peak[0] > 1, "evaluate_agent scored the examples serially"


def test_evaluate_agent_handles_an_empty_split():
    from examples.adas.adas_meta_agent_search import evaluate_agent
    assert evaluate_agent(Interpreter(lambda p: "Answer: 1"), {"block": "cot"}, []) == []


def test_evaluate_agent_preserves_order():
    """Concurrency must not scramble the per-instance outcomes -- the bootstrap
    CI is computed over them."""
    from examples.adas.adas_meta_agent_search import evaluate_agent

    def echo(prompt):
        return f"Answer: {prompt.count('#')}"

    examples = [("#" * i, str(i)) for i in range(1, 7)]
    assert evaluate_agent(Interpreter(echo), {"block": "cot"}, examples) == [1.0] * 6


def test_hard_mode_keeps_items_a_single_call_gets_wrong():
    """`--hard` exists because MGSM is ~95% solved by one call, so the search has
    no gradient. The filter is `select_hard` over a structure-free baseline."""
    from agentdescent.dataloader import select_hard

    pool = [(f"q{i}", str(i)) for i in range(40)]
    # a "baseline" that only gets the even ones right
    kept = select_hard(pool, lambda ex: 1.0 if int(ex[1]) % 2 == 0 else 0.0)
    assert kept == [ex for ex in pool if int(ex[1]) % 2]


# --- measurability: the fixes that turned "no lift demonstrated" into a number --

def test_parse_agent_survives_prose_around_the_json():
    """The meta-agent wraps its JSON in a fence and a closing remark. A single
    greedy `\\{.*\\}` spans from the first brace to the last one anywhere in the
    reply, so one stray brace in the prose discarded the whole proposal -- and a
    generation produces one proposal."""
    reply = ('Here is my design:\n```json\n'
             '{"thought":"t","name":"N","program":{"block":"cot_sc","k":3}}\n'
             '```\nNote: {this} composes two ideas.')
    assert _parse_agent(reply)["name"] == "N"


def test_parse_agent_rejects_a_program_it_cannot_afford():
    """Cost is multiplicative in this DSL and evaluation is candidates x items x
    cost, so one unbounded proposal outweighs the rest of the search."""
    from examples.adas.adas_meta_agent_search import MAX_PROGRAM_CALLS, program_cost

    greedy = {"block": "ensemble", "children": [
        {"block": "debate", "roles": ["a", "b", "c"], "rounds": 2},
        {"block": "debate", "roles": ["a", "b", "c"], "rounds": 2}]}
    assert program_cost(greedy) > MAX_PROGRAM_CALLS
    assert _parse_agent(json.dumps({"program": greedy})) is None
    # ... and every shipped seed stays comfortably inside the cap
    assert all(program_cost(s["program"]) <= MAX_PROGRAM_CALLS for s in seed_archive())


def test_program_cost_counts_the_calls_the_interpreter_makes():
    """The budget line and the cap both read this, so it has to match the
    interpreter rather than approximate it."""
    from examples.adas.adas_meta_agent_search import program_cost

    for program in ([s["program"] for s in seed_archive()] +
                    [{"block": "debate", "roles": ["x", "y"], "rounds": 2}]):
        calls = [0]

        def count(prompt):
            calls[0] += 1
            return "Answer: 42"

        Interpreter(count).run(program, "q")
        assert calls[0] == program_cost(program), program


def test_propose_keeps_a_valid_draft_when_a_later_round_degrades():
    """Two Reflexion rounds refine the draft; returning whatever the *last* round
    emitted throws away a good agent whenever the last one is malformed."""
    from examples.adas.adas_meta_agent_search import propose_agent

    good = '{"thought":"t","name":"Good","program":{"block":"step_back"}}'
    replies = iter([good] + ["not json"] * 6)

    agent = propose_agent(lambda prompt: next(replies), seed_archive())
    assert agent is not None and agent["name"] == "Good"


def test_seed_archive_is_scored_before_the_first_proposal():
    """The meta-agent's only conditioning signal is the archive. Scoring the seeds
    inside the first `step()` means generation 0 designs against seven entries all
    reading `unevaluated`."""
    seen = []

    def stub(prompt):
        if "archive of agents discovered so far" in prompt:
            seen.append(prompt)
        return "Answer: 42"

    val = [(f"q{i}", "42" if i % 2 else "7") for i in range(8)]
    run_meta_agent_search(stub, val, generations=1, seed=0)
    assert seen, "the meta-agent was never asked for a design"
    assert "unevaluated" not in seen[0], (
        "the first proposal was conditioned on an unscored archive")


def test_search_reports_the_best_seed_for_comparison():
    """On a `--hard` subset the structure-free baseline is 0.000 by construction,
    so searched-vs-seed is the only comparison that carries information."""
    val = [(f"q{i}", "42" if i % 2 else "7") for i in range(8)]
    result = run_meta_agent_search(lambda p: "Answer: 42", val, generations=1, seed=0)
    assert validate_program(result.best_seed.get("program", {}))
    assert any(a.get("seed") for a in result.archive)
    assert result.best_fitness >= result.seed_fitness


def test_merge_reports_name_the_outcome():
    """`committed_version is None` is the driver's whole tally, so an archive whose
    best design simply did not change was reported identically to a rejection."""
    from agentdescent.evolution import Task, evolve

    from examples.adas.adas_meta_agent_search import (AdasContext, AgentDesignStrategy,
                                                 MetaSearchAggregator, make_propose)

    ctx = AdasContext()
    interp = Interpreter(lambda p: "Answer: 42")
    tasks = [Task(id=f"t{i}", prompt="q", meta={"answer": "42"}) for i in range(8)]

    def factory(ledger, verifier, audit, config, policy):
        agg = MetaSearchAggregator(ledger, verifier, ctx, artifact_id="agentic_system")
        agg.seed()
        return agg

    out = evolve(tasks, lambda t, o: 1.0 if o == "42" else 0.0,
                 run=lambda r, t: interp.run(json.loads(r), t.prompt) or "",
                 propose=make_propose(ctx, lambda p: "Answer: 42"),
                 strategy=AgentDesignStrategy(), artifact_id="agentic_system",
                 blast_radius=0.6, rounds=1, n_workers=1, self_verify=False,
                 held_out_frac=0.5, aggregator_factory=factory)
    assert set(out.outcomes()) <= {"committed", "already-best", "no-candidates",
                                   "cas-conflict"}
    assert "unknown" not in out.outcomes()


def test_search_surfaces_why_each_generation_went_as_it_did():
    """`already-best` and `no-candidates` both print as `+0/-1` on the driver's
    line, and they need opposite fixes. The run that concluded "the archive
    rejected every candidate" had no way to tell which one it had seen."""
    # a stub that answers every task correctly -> no trigger rollout ever fails,
    # so evolve() never asks for a proposal and nothing reaches the archive.
    val = [(f"q{i}", "42") for i in range(12)]
    r = run_meta_agent_search(lambda p: "Answer: 42", val, generations=2, seed=0)
    assert r.outcomes == {"no-candidates": 2}, r.outcomes
    assert r.stop_reason == "rounds" and r.error is None


def test_empty_completions_are_counted_not_silently_scored_wrong():
    """The most dangerous failure this example has is silent: a reasoning model
    that spends its whole token budget on hidden reasoning returns EMPTY visible
    content, `_extract_int("")` is None, and `score_mgsm` scores that as a wrong
    answer. Nothing raises, so a starved run reports a low accuracy that looks
    exactly like a model which cannot do the problems. Measured on
    deepseek-v4-flash at the library default of 4096: 0 of 4 meta-agent calls
    returned anything at all."""
    from examples.adas.adas_meta_agent_search import EmptyCompletionGuard

    replies = iter(["", "Answer: 42", "   ", "Answer: 7"])
    guard = EmptyCompletionGuard(lambda p: next(replies))
    assert [guard(f"p{i}") for i in range(4)] == ["", "Answer: 42", "   ", "Answer: 7"]
    assert guard.blank == 2 and guard.total == 4
    msg = guard.report()
    assert msg and "EMPTY" in msg and "50.0%" in msg

    quiet = EmptyCompletionGuard(lambda p: "Answer: 1")
    quiet("p")
    assert quiet.report() is None      # silent when there is nothing to report


def test_an_empty_completion_scores_as_a_wrong_answer():
    """Why the guard has to exist: this is indistinguishable from being wrong."""
    from examples.adas.adas_meta_agent_search import evaluate_agent

    scores = evaluate_agent(Interpreter(lambda p: ""), {"block": "cot"},
                            [("q", "42"), ("q2", "7")])
    assert scores == [0.0, 0.0]        # no error, no warning -- just "wrong"


def test_design_identity_ignores_key_order():
    """Every dedup in the search is string equality on the design's JSON, and a
    proposal's key order is whatever the model emitted. Unsorted, a semantically
    identical design counted as new: it passed `to_diff`, was scored on every val
    item, tied the design it duplicated, failed the strict `>` test, and appeared
    as one more `+0/-1` -- having spent a full evaluation sweep to learn nothing."""
    from examples.adas.adas_meta_agent_search import AgentDesignStrategy, canonical

    a = {"block": "cot_sc", "k": 3}
    b = {"k": 3, "block": "cot_sc"}
    assert canonical(a) == canonical(b)
    assert json.dumps(a) != json.dumps(b)      # why this function has to exist

    # the head is stored canonically, so a re-proposal of it is dropped, not scored
    strat = AgentDesignStrategy()
    state = strat.initial()
    same = json.dumps({"name": "N", "thought": "t",
                       "program": {"k": 3, "block": "cot"}})   # CoT, reordered+extra
    head_prog = json.loads(state["design"])
    assert head_prog == {"block": "cot"}
    resend = json.dumps({"name": "N", "thought": "t", "program": {"block": "cot"}})
    assert strat.to_diff(state, resend, "w0", 1, "agentic_system") is None


def test_a_missing_score_prints_as_missing():
    """The test split is scored *after* the search is finished and paid for. When
    that call fails -- an account ran out of credit inside it, 79 minutes in --
    the validation numbers are still in hand, so they must still print. Formatting
    `None` with `:.3f` raises, which would discard them along with the failure."""
    from examples.adas.adas_meta_agent_search import fmt_score

    assert fmt_score(0.4375).strip() == "0.438"
    assert fmt_score(None).strip() == "n/a"
    assert len(fmt_score(None)) == len(fmt_score(0.5))    # column stays aligned
