"""Dynamic Cheatsheet (DC-Cu) against ``suzgunmirac/dynamic-cheatsheet@5cfe3c3``.

The distinctive claims, each pinned here without a model or a network:

* Gen always reads the current cheatsheet;
* Cur sees ``(memory, question, the model's own answer)`` -- never a gold
  answer and never the grader's verdict;
* the default is online (every rewrite kept) and ``--gated`` restores the
  framework's held-out gate, which can reject a rewrite that drops content.
"""
from __future__ import annotations

import itertools
import re
from fractions import Fraction

import pytest

from agentdescent.aggregator import AggregatorConfig
from agentdescent.evolution import Task, evolve
from agentdescent.policies import AcceptDecision

from examples._measure import PhasedLLM, Recorder
from examples.dynamic_cheatsheet import dynamic_cheatsheet_cumulative as dc
from examples.dynamic_cheatsheet import game24
from agentdescent.agents import Usage, metered


TIP_SHEET = ("Version: 1\nGENERAL META-REASONING STRATEGIES\n"
             "Use brute force search over every expression tree.")


# ---------------------------------------------------------------------------
# The domain: independently re-derived, deterministic, safe to grade.
# ---------------------------------------------------------------------------

def _solvable(numbers):
    """A second, independent solver: every permutation x operators x bracketing."""
    ops = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b,
           lambda a, b: a / b if b else None]
    for a, b, c, d in set(itertools.permutations(numbers)):
        for f, g, h in itertools.product(ops, repeat=3):
            trees = (
                lambda: _apply(h, _apply(g, _apply(f, a, b), c), d),
                lambda: _apply(h, _apply(f, a, b), _apply(g, c, d)),
                lambda: _apply(f, a, _apply(g, b, _apply(h, c, d))),
            )
            if any(t() == 24 for t in trees):
                return True
    return False


def _apply(f, x, y):
    if x is None or y is None:
        return None
    return f(Fraction(x), Fraction(y))


def test_every_puzzle_is_solvable_and_the_reference_scores_one():
    assert len(game24.PUZZLES) == len(set(game24.PUZZLES)) == 36
    for numbers in game24.PUZZLES:
        assert _solvable(numbers), numbers
        reference = game24.solve_exact(numbers)
        answer = f"FINAL ANSWER:\n<answer>\n{reference}\n</answer>"
        assert game24.score_answer(numbers, answer) == 1.0


@pytest.mark.parametrize("reply, expected", [
    ("<answer>(7 - (8 / 8)) * 4</answer>", 1.0),
    ("<answer>(7 - (8 / 8)) * 4 = 24</answer>", 1.0),
    ("<answer>(7 - (8 \u00f7 8)) \u00d7 4</answer>", 1.0),
    ("<answer>8 * 3 + 7 - 8 / 8 * 0</answer>", 0.0),   # wrong numbers
    ("<answer>(7 - 8 / 8) * 4 * 1</answer>", 0.0),     # a number used twice
    ("<answer>7 + 8 + 8 + 4</answer>", 0.0),           # wrong value
    ("(7 - (8 / 8)) * 4", 0.0),                        # no <answer> block
    ("<answer>__import__('os').system('true')</answer>", 0.0),
    ("<answer>(7 - (8 / 0)) * 4</answer>", 0.0),
    ("<answer>-(7) + 31 + 0 * 8 * 8 * 4</answer>", 0.0),
    ("<answer>2 ** 3 * 3 + 4 - 4</answer>", 0.0),
])
def test_grader_checks_numbers_and_value_and_never_evals(reply, expected):
    assert game24.score_answer([4, 7, 8, 8], reply) == expected


def test_splits_are_disjoint_frozen_per_seed_and_wide_enough():
    for seed in (0, 1, 7):
        train, held_out, test = game24.game24_splits(seed)
        assert (len(train), len(held_out), len(test)) == (12, 12, 12)
        prompts = [t.prompt for t in train + held_out + test]
        assert len(set(prompts)) == 36
        ids = [t.id for t in train + held_out + test]
        assert len(set(ids)) == 36
    again = game24.game24_splits(0)
    assert [t.id + t.prompt for t in again[2]] == [
        t.id + t.prompt for t in game24.game24_splits(0)[2]]


# ---------------------------------------------------------------------------
# Gen: the current cheatsheet is always in the generator's prompt.
# ---------------------------------------------------------------------------

class _Recording:
    """A completion that records every prompt and answers from a function."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply(prompt)


def _llm(completion):
    usage = Usage()
    recorder = Recorder(metered(completion, usage), usage)
    return PhasedLLM(recorder, "run:dynamic_cheatsheet")


@pytest.mark.parametrize("cheatsheet", [
    dc.INITIAL_CHEATSHEET,
    TIP_SHEET,
    "line one\n\n<memory_item>x</memory_item>\n** Count: 3",
    "mentions [[QUESTION]] and [[MODEL_ANSWER]] literally",
])
def test_the_generator_prompt_always_contains_the_current_cheatsheet(cheatsheet):
    policy = dc.build(0)
    recording = _Recording(lambda prompt: "ok")
    for task in policy.train_tasks[:3]:
        policy.solve(_llm(recording), cheatsheet, task)
    assert len(recording.prompts) == 3
    for task, prompt in zip(policy.train_tasks, recording.prompts):
        assert cheatsheet in prompt
        assert f"Input: {task.prompt}" in prompt
        # The cheatsheet is substituted once and the question is the task's own.
        assert prompt.count(task.prompt) >= 1


def test_the_initial_cheatsheet_is_upstreams_empty_marker():
    strategy = dc.build(0).strategy
    assert strategy.render(strategy.initial()) == "(empty)"


def test_placeholder_substitution_is_one_pass():
    """Upstream chains `str.replace`, so a model answer that mentions a
    placeholder is itself substituted. One pass substitutes only the template."""
    task = dc.build(0).train_tasks[0]
    prompt = dc.curator_prompt("MEMORY", task, "I wrote [[PREVIOUS_CHEATSHEET]]")
    assert "I wrote [[PREVIOUS_CHEATSHEET]]" in prompt
    assert prompt.count("MEMORY") == 1


# ---------------------------------------------------------------------------
# Cur: no gold, no reward, anywhere in the curator path.
# ---------------------------------------------------------------------------

def test_the_curator_template_has_no_gold_or_reward_slot():
    slots = set(re.findall(r"\[\[([A-Z_]+)\]\]", dc.CURATOR_TEMPLATE))
    assert slots == {"PREVIOUS_CHEATSHEET", "QUESTION", "MODEL_ANSWER"}
    generator_slots = set(re.findall(r"\[\[([A-Z_]+)\]\]", dc.GENERATOR_TEMPLATE))
    assert generator_slots == {"CHEATSHEET", "QUESTION"}
    for template in (dc.CURATOR_TEMPLATE, dc.GENERATOR_TEMPLATE):
        lowered = template.lower()
        for word in ("reward", "gold", "reference solution", "ground truth",
                     "correct answer is"):
            assert word not in lowered


def test_curator_prompts_never_carry_a_gold_answer_or_the_reward():
    policy = dc.build(0)
    every = (policy.train_tasks + policy.held_out_tasks + policy.test_tasks)
    assert len(every) == 36
    for task in every:
        reference = task.meta["reference"]
        assert reference  # the gold exists on the task...
        seen = []
        for reward in (0.0, 1.0, 0.123456):
            recording = _Recording(lambda prompt: "<cheatsheet>x</cheatsheet>")
            policy.propose(_llm(recording), "(empty)", task,
                           "Working.\nFINAL ANSWER:\n<answer>\n1 + 1\n</answer>",
                           reward)
            (prompt,) = recording.prompts
            seen.append(prompt)
            # ...and never reaches the curator, in any form.
            assert reference not in prompt
            assert reference.replace("(", "").replace(")", "") not in prompt
            assert "0.123456" not in prompt
        # The verdict is not an input: the prompt is identical for every reward.
        assert seen[0] == seen[1] == seen[2]


def test_the_curator_curates_after_success_and_failure_alike():
    """Upstream curates after every query. A port that skipped successes would
    change what the cheatsheet is made of."""
    policy = dc.build(0)
    task = policy.train_tasks[0]
    for reward in (0.0, 1.0):
        recording = _Recording(lambda prompt: "<cheatsheet>x</cheatsheet>")
        out = policy.propose(_llm(recording), "(empty)", task, "answer", reward)
        assert out == "<cheatsheet>x</cheatsheet>"
        assert len(recording.prompts) == 1


# ---------------------------------------------------------------------------
# Validation lives once, in to_diff: strict block, no fallback substitution.
# ---------------------------------------------------------------------------

def test_a_curator_reply_without_a_cheatsheet_block_is_refused_not_replaced():
    strategy = dc.build(0).strategy
    state = strategy.initial()
    assert strategy.to_diff(state, "I think you should use Python.", "w", 1,
                            "a") is None
    assert strategy.invalid_proposals == 1
    # Upstream would have returned the OLD cheatsheet here; nothing is substituted.
    assert strategy.to_diff(state, "", "w", 1, "a") is None
    assert strategy.to_diff(state, None, "w", 1, "a") is None
    assert strategy.invalid_proposals == 3


def test_an_empty_cheatsheet_block_is_refused():
    strategy = dc.build(0).strategy
    assert strategy.to_diff(strategy.initial(), "<cheatsheet>\n \n</cheatsheet>",
                            "w", 1, "a") is None
    assert strategy.invalid_proposals == 1


def test_a_well_formed_reply_becomes_one_whole_value_replacement():
    strategy = dc.build(0).strategy
    reply = "Sure!\n<cheatsheet>\nVersion: 2\nTip A\n</cheatsheet>\ntrailing"
    diff = strategy.to_diff(strategy.initial(), reply, "w", 1, "a")
    assert diff is not None
    assert diff.ops == {"cheatsheet": "Version: 2\nTip A"}
    assert strategy.render({"cheatsheet": "Version: 2\nTip A"}) == (
        "Version: 2\nTip A")


def test_the_last_cheatsheet_block_wins():
    reply = ("<cheatsheet>draft</cheatsheet> hmm, better: "
             "<cheatsheet>final</cheatsheet>")
    assert dc.extract_cheatsheet(reply) == "final"


def test_merge_validator_is_the_tag_free_shape_check():
    """The runner hands `strategy.validator` to the reflective merge, whose
    output is plain text with no <cheatsheet> wrapper."""
    strategy = dc.build(0).strategy
    assert strategy.validator("plain merged text") == "plain merged text"
    assert strategy.validator("<cheatsheet>wrapped</cheatsheet>") == "wrapped"
    with pytest.raises(ValueError):
        strategy.validator("   ")
    with pytest.raises(ValueError):
        strategy.validator("x" * (dc.MAX_CHEATSHEET_CHARS + 1))


# ---------------------------------------------------------------------------
# Online vs gated: the ledger round trip, and the gate's veto.
# ---------------------------------------------------------------------------

def _fake(curator_sheet):
    """Gen solves only with the brute-force tip in memory; Cur returns
    ``curator_sheet`` regardless of what it was shown."""
    def reply(prompt):
        lower = prompt.lower()
        if "cheatsheet reference curator" in lower:
            return f"<cheatsheet>\n{curator_sheet}\n</cheatsheet>"
        numbers = re.findall(r"Input: ([\d ]+)", prompt)[-1].strip()
        sheet = prompt.split("CHEATSHEET:", 1)[1].split("Now it is time", 1)[0]
        expression = (game24.reference_answers()[numbers]
                      if "brute force" in sheet.lower() else "1 + 1")
        return f"FINAL ANSWER:\n<answer>\n{expression}\n</answer>"
    return reply


def _run(gated, curator_sheet, initial_sheet):
    """One real `evolve()` round with the port's own pieces. Returns the result
    and the strategy so the ledger's state can be inspected."""
    policy = dc.build(0, gated=gated)
    usage = Usage()
    recorder = Recorder(metered(_fake(curator_sheet), usage), usage)
    run_llm = PhasedLLM(recorder, "run:dynamic_cheatsheet")
    prop_llm = PhasedLLM(recorder, "proposal:dynamic_cheatsheet")
    result = evolve(
        policy.engine_tasks, policy.reward,
        run=lambda rendered, task: policy.solve(run_llm, rendered, task),
        propose=lambda rendered, task, output, reward: policy.propose(
            prop_llm, rendered, task, output, reward),
        strategy=policy.strategy,
        artifact_id="candidate-dynamic_cheatsheet",
        initial_state={"cheatsheet": initial_sheet},
        n_workers=2, rounds=1, max_concurrency=1, blast_radius=policy.blast_radius,
        held_out_frac=policy.held_out_frac, solved_threshold=1.1,
        agg_config=AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        shuffle=False, seed=0, verbose=False, policies=policy.engine,
    )
    return result, policy


USELESS = "Version: 2\nnothing worth keeping"


def test_online_mode_keeps_every_rewrite_and_it_round_trips_the_ledger():
    result, policy = _run(False, USELESS, TIP_SHEET)
    assert result.error is None
    # The rewrite dropped the tip, held-out collapsed, and it was kept anyway:
    # online DC-Cu has no acceptance test, exactly as upstream.
    assert result.state["cheatsheet"] == USELESS
    assert result.rendered == USELESS
    assert result.outcomes().get("committed", 0) >= 1
    assert result.ledger_log  # the commit is in the ledger's own log
    # The memory is what the next generator call reads back out of the ledger.
    recording = _Recording(lambda prompt: "ok")
    policy.solve(_llm(recording), result.rendered, policy.train_tasks[0])
    assert USELESS in recording.prompts[0]


def test_gated_mode_rejects_a_rewrite_that_empties_useful_content():
    result, _ = _run(True, USELESS, TIP_SHEET)
    assert result.error is None
    assert result.state["cheatsheet"] == TIP_SHEET
    assert result.outcomes().get("committed", 0) == 0


def test_gated_mode_still_commits_a_rewrite_that_helps():
    result, _ = _run(True, TIP_SHEET, "(empty)")
    assert result.state["cheatsheet"] == TIP_SHEET
    assert result.final_reward > 0.5


def _through_the_shared_runner(gated, mode="sync_parallel"):
    """The same scenario through ``run_port`` itself, so the runner's own
    blast-radius wiring is what is under test, not the test's."""
    from examples._method_runner import run_port

    policy = dc.build(0, gated=gated)
    usage = Usage()
    recorder = Recorder(metered(_fake(USELESS), usage), usage)
    return run_port(policy, recorder, mode=mode, seed=0, workers=2,
                    candidate_budget=2, max_seconds=30.0, shutdown_grace=5.0)


@pytest.mark.parametrize("mode", ["serial", "sync_parallel", "async_pipeline"])
def test_the_shared_runner_really_is_ungated_online_and_gated_with_the_flag(mode):
    """Held-out is 0.0 before and after (the curator's text never helps), so
    nothing here *improves* -- yet online mode commits. That is only true if no
    L1 commit audit sits in front of the acceptance policy."""
    online = _through_the_shared_runner(False, mode)
    assert online.accepted >= 1
    assert online.framework["acceptance"] == "CommitEveryUpdate"
    gated = _through_the_shared_runner(True, mode)
    assert gated.accepted == 0
    assert gated.framework["acceptance"] == "default"


def test_the_runner_passes_the_policys_blast_radius_to_evolve(monkeypatch):
    import examples._method_runner as runner
    from examples._method_policy import MethodPolicy

    seen = {}
    real = runner.evolve

    def spy(*args, **kwargs):
        seen["blast_radius"] = kwargs["blast_radius"]
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "evolve", spy)
    _through_the_shared_runner(False, "sync_parallel")
    assert seen["blast_radius"] == dc.build(0).blast_radius == 0.2
    # Every other port keeps the L1 default it always had.
    fields = MethodPolicy.__dataclass_fields__
    assert fields["blast_radius"].default == 0.6


def test_the_acceptance_seam_is_the_only_difference_between_the_modes():
    assert dc.build(0).engine.acceptance.__class__ is dc.CommitEveryUpdate
    assert dc.build(0, gated=True).engine.acceptance is None
    decision = dc.CommitEveryUpdate().accept(object())
    assert isinstance(decision, AcceptDecision) and decision.accept


# ---------------------------------------------------------------------------
# The command line is offline.
# ---------------------------------------------------------------------------

def test_dry_run_is_offline_and_names_the_acceptance_rule(capsys, monkeypatch):
    import socket

    def refuse(*args, **kwargs):  # pragma: no cover - only runs on a bug
        raise AssertionError("dry-run touched the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert dc.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dynamic_cheatsheet [mechanism_microport]" in out
    assert "acceptance=CommitEveryUpdate" in out
    assert "no model API was accessed" in out
    assert dc.main(["--dry-run", "--gated"]) == 0
    assert "acceptance=" not in capsys.readouterr().out


def test_no_embedding_or_retrieval_dependency_is_imported():
    banned = ("sentence_transformers", "faiss", "openai", "voyageai", "cohere")
    source = open(dc.__file__).read() + open(game24.__file__).read()
    for name in banned:
        assert f"import {name}" not in source
        assert f"from {name}" not in source
