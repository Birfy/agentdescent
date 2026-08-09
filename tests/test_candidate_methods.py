"""Offline tests for the candidate-method ports: every method, every mode,
no API key. The fake completion below plays every actor; the engine, the
strategies, the selection policies, and the reflective merge path are real.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agentdescent.agents import Usage, metered
from agentdescent.evolution import Task
from agentdescent.fusion import KeepContradictions, ReflectiveFusion

from bench.candidate_methods import (ALGORITHMS,
                                     PROPOSAL_CALLS_PER_CANDIDATE, main)
from examples._measure import MODES, Recorder, parse_json_object
from examples._method_policy import FieldSlots, WindowedMemory, read_fields
from examples._method_runner import ProposalLimiter, run_port
from examples._money_domain import TASKS, parse_integer_answer, money_splits
from examples._selfplay_domain import selfplay_splits, trajectory_reward
from examples.agent0.agent0_tool_curriculum import calculator
from examples.aflow.aflow_workflow_search import SoftMixed
from examples.promptbreeder.promptbreeder_genetic_prompts import BinaryTournament
from examples.sica.sica_self_edit import compile_policy
from examples.voyager.voyager_skill_library import simulate as voyager_simulate
import examples.sica.sica_self_edit as sica_module
import examples.godel_agent.godel_agent_self_modify as godel_module


ANSWERS = {task.question: task.answer_cents for task in TASKS}

FULL_RECIPE = [
    "sanitize:vessel", "collect:water", "collect:{ingredient}",
    "heat:water", "combine:water+{ingredient}", "serve:drink",
]


def _money_answer(prompt):
    for question, answer in ANSWERS.items():
        if question not in prompt:
            continue
        if "integer cents" in prompt.lower():
            return answer
        return f"${int(answer) / 100:.2f}"
    return None


def fake_completion(prompt):
    lower = prompt.lower()
    # PromptBreeder's nine operators and its population seeding all end with one
    # of three JSON contracts, so the fake answers the contract rather than each
    # operator's wording. Keying on the wording is how this stub silently stopped
    # covering the port: the operators were rewritten, every match fell through
    # to the prose fallback, `parse_json_object` rejected it, and the offline run
    # produced no valid proposal at all while still passing a
    # `final >= baseline` assertion.
    if 'with "task_prompt" and "mutation_prompt" strings' in lower:
        return json.dumps(
            {
                "task_prompt": "Compute carefully and return integer cents only.",
                "mutation_prompt": "Generalize evaluator feedback into domain rules.",
            }
        )
    if 'with a "mutation_prompt" string' in lower:
        return json.dumps(
            {"mutation_prompt": "Generalize evaluator feedback into strict domain output rules."})
    if 'with a "task_prompt" string' in lower:
        # Varied per prompt so the population is a population: identical units
        # deduplicate to one entry and no tournament ever has two to sample.
        tag = abs(hash(prompt)) % 1000
        return json.dumps(
            {"task_prompt": f"Compute carefully and return integer cents only. [{tag}]"})
    if "aflow's graph optimizer" in lower:
        return json.dumps(
            {
                "modification": "learn integer-cents formatting",
                "solve_instruction": "Compute carefully and return integer cents only.",
                "review_instruction": "Verify arithmetic and return integer cents only.",
            }
        )
    if "reflexion's verbal reflection module" in lower:
        return "Compute carefully and return monetary answers as integer cents only."
    if "self-refine's feedback module" in lower:
        return "The answer used dollars; the evaluator expects integer cents."
    if "self-refine's refine module" in lower:
        return "Compute carefully and return monetary answers as integer cents only."
    if "sica's meta-improvement tool" in lower:
        return """```python
def agent_prompt(question):
    return "Compute carefully and return integer cents only.\\n\\n" + question
```"""
    if "recursively improve either or both functions" in lower:
        return """```python
def solve_prompt(question):
    return "Compute carefully and return integer cents only.\\n\\n" + question

def self_improvement_prompt(source, feedback):
    return "Improve this source using feedback:\\n" + feedback + "\\n" + source
```"""
    if "voyager repairs executable programs" in lower:
        return json.dumps({"steps": FULL_RECIPE})
    if "voyager's action agent" in lower:
        ingredient_match = re.search(r"Visible ingredient: (\w+)", prompt)
        ingredient = ingredient_match.group(1) if ingredient_match else "mint"
        if "sanitize:vessel" in lower:
            actions = [step.replace("{ingredient}", ingredient)
                       for step in FULL_RECIPE]
        else:
            actions = ["collect:water", f"collect:{ingredient}", "serve:drink"]
        return json.dumps({"actions": actions})
    if "skillweaver hone" in lower:
        return json.dumps(
            {
                "calls": [
                    "open:{page}", "wait:hydration-complete",
                    "fill:{field}={value}", "click:save", "assert:saved-toast",
                ]
            }
        )
    if "operate a settings website" in lower:
        task_match = re.search(r"Task: set (\w+) to (\w+) on (/\S+)", prompt)
        field, value, page = (
            task_match.groups()
            if task_match
            else ("timezone", "UTC", "/settings/profile")
        )
        calls = [f"open:{page}", f"fill:{field}={value}", "click:save"]
        if "hydration-complete" in lower:
            calls = [
                f"open:{page}", "wait:hydration-complete",
                f"fill:{field}={value}", "click:save", "assert:saved-toast",
            ]
        return json.dumps({"calls": calls})
    if "zero-data self-play loop" in lower:
        return '{"item_cents":[125,240],"quantities":[1,2]}'
    if "agent0's executor" in lower:
        return '{"tool":"calculator","expression":"125+240*2"}'
    if "continue the same agent0 trajectory" in lower:
        return "605"
    if "solve the problem. follow policy memory" in lower:
        return "605"
    if any(
        marker in lower
        for marker in (
            "absolute zero updates the same model",
            "update only r-zero's challenger",
            "update only r-zero's solver",
            "agent0 co-evolution update",
        )
    ):
        return "Represent monetary totals as integer cents and output only that integer."
    if "produce their union" in lower:
        return ("Merged union: compute carefully and return monetary answers "
                "as integer cents only, keeping every proposed rule.")
    answer = _money_answer(prompt)
    if answer is not None:
        return answer
    return "Represent monetary totals as integer cents and output only that integer."


def _offline_run(algorithm, mode):
    usage = Usage()
    recorder = Recorder(metered(fake_completion, usage), usage)
    fidelity, builder = ALGORITHMS[algorithm]
    policy = builder(0)
    assert policy.fidelity == fidelity
    return run_port(
        policy,
        recorder,
        mode=mode,
        seed=0,
        workers=2,
        candidate_budget=2,
        max_seconds=30.0,
        shutdown_grace=5.0,
    ).compact()


# ---------------------------------------------------------------------------
# The matrix: every method through the real engine in every mode, offline.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("mode", MODES)
def test_every_candidate_method_uses_the_framework_in_every_mode(algorithm, mode):
    payload = _offline_run(algorithm, mode)
    assert 0.0 <= payload["baseline_quality"] <= 1.0
    assert 0.0 <= payload["final_quality"] <= 1.0
    assert payload["final_quality"] >= payload["baseline_quality"]
    assert payload["candidates"] == 2
    assert payload["budget"]["observed_candidates"] == 2
    assert payload["budget"]["matched"] is True
    assert payload["budget"]["observed_proposal_calls"] <= (
        2 * PROPOSAL_CALLS_PER_CANDIDATE[algorithm]
    )
    assert payload["usage"]["calls"] == len(payload["events"])
    assert payload["framework"]["runtime"] == (
        "async_evolve" if mode == "async_pipeline" else "evolve"
    )
    assert payload["framework"]["rollouts"] >= 2
    assert not ({"prompt", "response", "source", "artifact"} & payload.keys())


# ---------------------------------------------------------------------------
# No fallback substitution anywhere: invalid proposals are counted, not fixed.
# ---------------------------------------------------------------------------

def test_invalid_proposals_are_counted_not_replaced():
    _, builder = ALGORITHMS["sica"]
    policy = builder(0)
    strategy = policy.strategy
    state = strategy.initial()
    assert strategy.to_diff(state, "this is not python", "w", 1, "a") is None
    assert strategy.invalid_proposals == 1
    # The old hardcoded improved-source fallbacks are gone.
    assert not hasattr(sica_module, "SICA_IMPROVED_SOURCE")
    assert not hasattr(godel_module, "GODEL_IMPROVED_SOURCE")


def test_reflective_and_verify_flags_by_artifact_kind():
    flags = {name: ALGORITHMS[name][1](0) for name in ALGORITHMS}
    for name in ("promptbreeder", "aflow", "reflexion", "self_refine",
                 "absolute_zero", "r_zero", "agent0"):
        assert flags[name].reflective, name
    for name in ("voyager", "skillweaver", "sica", "godel_agent"):
        assert not flags[name].reflective, name
    assert flags["voyager"].self_verify and flags["skillweaver"].self_verify


def test_reflective_fusion_synthesises_with_one_model_call():
    calls = []

    def complete(prompt):
        calls.append(prompt)
        assert "Produce their UNION" in prompt
        return "merged value keeping both improvements"

    fusion = ReflectiveFusion(complete)
    fusion._fallback = SimpleNamespace(select=lambda a, d: (d[0], a, False))
    diffs = [
        SimpleNamespace(diff_id="a", target="x", ops={"value": "rule one"},
                        contract_breaking=False),
        SimpleNamespace(diff_id="b", target="x", ops={"value": "rule two"},
                        contract_breaking=False),
    ]
    artifact = SimpleNamespace(id="x", state={"value": "seed"},
                               apply=lambda diff: artifact)
    union, _, fused = fusion.select(artifact, list(diffs))
    assert fused and len(calls) == 1
    assert union.ops["value"] == "merged value keeping both improvements"
    assert isinstance(KeepContradictions().resolve(artifact, diffs)[0], list)


# ---------------------------------------------------------------------------
# Shared strategies.
# ---------------------------------------------------------------------------

def test_field_slots_partial_updates_and_render_roundtrip():
    slots = FieldSlots(fields={"a": "seed a", "b": "seed b"},
                       parse=parse_json_object)
    state = slots.initial()
    diff = slots.to_diff(state, '{"a": "new a"}', "w", 1, "t")
    assert diff.ops == {"a": "new a"}
    assert slots.to_diff(state, "no json here", "w", 1, "t") is None
    assert slots.invalid_proposals == 1
    assert read_fields(slots.render({"a": "x", "b": "y"})) == {"a": "x", "b": "y"}


def test_windowed_memory_is_append_only_and_bounded():
    memory = WindowedMemory(seed_text="seed", window=3)
    state = {}
    for version, text in enumerate(["one", "two", "three", "four"]):
        diff = memory.to_diff(state, text, "w", version, "t")
        assert diff is not None
        state.update(diff.ops)
    rendered = memory.render(state)
    assert "one" not in rendered
    assert all(word in rendered for word in ("two", "three", "four"))
    assert memory.to_diff(state, "four", "w", 9, "t") is None  # duplicate


def test_skill_library_accepts_known_keys_and_rejects_unknown():
    _, builder = ALGORITHMS["voyager"]
    strategy = builder(0).strategy
    state = strategy.initial()
    good = strategy.to_diff(
        state, 'skill generic: {"steps": ["collect:water"]}', "w", 1, "t")
    assert good is not None and "skill generic" in good.ops
    assert strategy.to_diff(state, 'skill unknown-key: {"steps": ["x"]}',
                            "w", 1, "t") is None
    assert strategy.invalid_proposals == 1


# ---------------------------------------------------------------------------
# Selection policies.
# ---------------------------------------------------------------------------

def _candidates(scores):
    return [SimpleNamespace(score=score) for score in scores]


def test_binary_tournament_prefers_the_pair_winner():
    policy = BinaryTournament(seed=0)
    ctx = SimpleNamespace(candidates=_candidates([0.1, 0.9, 0.5]))
    winners = policy.select(ctx, 50)
    assert all(w.score in (0.1, 0.5, 0.9) for w in winners)
    # The best candidate wins every tournament it is drawn into, so it must
    # appear strictly more often than the worst.
    tally = {0.1: 0, 0.5: 0, 0.9: 0}
    for w in winners:
        tally[w.score] += 1
    assert tally[0.9] > tally[0.1]


def test_soft_mixed_keeps_the_seed_and_favours_scores():
    policy = SoftMixed(alpha=5.0, lam=0.2, top_k=2, seed=0)
    seed_candidate = SimpleNamespace(score=0.0)
    ctx = SimpleNamespace(
        candidates=[seed_candidate] + _candidates([0.2, 0.9, 0.8]))
    picks = policy.select(ctx, 200)
    scores = [p.score for p in picks]
    assert scores.count(0.9) > scores.count(0.8) >= 0
    # λ-uniform keeps the seed reachable even at score 0.
    assert 0.0 in scores


# ---------------------------------------------------------------------------
# Domains.
# ---------------------------------------------------------------------------

def test_strict_grader_is_the_one_the_ports_use():
    assert parse_integer_answer("Answer: 415") == "415"
    assert parse_integer_answer("415") == "415"
    assert parse_integer_answer("$4.15") is None
    assert parse_integer_answer("about 415 cents") is None
    # The convention is still something a method has to discover -- these are the
    # forms it must NOT accept, on the line it reads.
    assert parse_integer_answer("2.75 + 1.40 = 4.15") is None
    assert parse_integer_answer("") is None


def test_the_grader_reads_the_final_line_so_working_and_format_are_not_exclusive():
    """It used to `fullmatch` the whole reply, which made the output convention
    and the reasoning mutually exclusive: a prompt that satisfied the format had
    to forbid the working, and without the working the arithmetic collapsed.

    Measured on `deepseek-v4-flash` across all 48 items -- same model, same
    convention: no working scored 0.562, *with* working scored **0.000** because
    the working itself failed the grader, and with working under a final-line
    grader scored 0.979. The middle number is the point: a correct answer scored
    zero for having shown its arithmetic.
    """
    worked = ("Notebook 2.75 and pen 1.40.\n"
              "2.75 + 1.40 = 4.15 dollars\n"
              "4.15 x 100 = 415 cents\n"
              "Answer: 415")
    assert parse_integer_answer(worked) == "415"
    # A reply that stops mid-working still scores zero: the last line is not an
    # answer, and reading further back would start grading intermediate values.
    assert parse_integer_answer("2.75 + 1.40 = 4.15\n4.15 x 100 = 415 cents") is None
    assert parse_integer_answer("Answer: 415\nbut I am not sure") is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_money_splits_are_disjoint_and_complete(seed):
    train, held_out, test = money_splits(seed)
    ids = [{task.id for task in rows} for rows in (train, held_out, test)]
    assert [len(rows) for rows in ids] == [16, 16, 16]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_the_train_split_is_wide_enough_for_the_workers_the_matrix_runs():
    """`run_port` refuses a run whose train split is smaller than the worker
    count, so the domain's size *is* the ceiling on this family's parallelism.
    At twelve items the splits were 4/4/4 and every one of these eleven ports
    was capped at four workers with a four-item acceptance gate."""
    train, held_out, _ = money_splits(0)
    assert len(train) >= 16 and len(held_out) >= 16


def test_every_money_answer_survives_an_independent_rederivation():
    """A wrong answer in this table does not raise. It silently becomes a task no
    method can solve, and the run then reads as a method that failed to learn --
    so the arithmetic is checked rather than trusted.

    The table is built in integer cents; this recomputes each row with `Decimal`
    from the dollar figures the question itself prints, which is a different
    numeric type applied to a different source of truth (the prose, not the
    generator).
    """
    import re
    from decimal import Decimal

    checked = 0
    for task in TASKS:
        q = task.question
        dollars = [Decimal(m) for m in re.findall(r"\$(\d+\.\d\d)", q)]
        counts = [int(m) for m in re.findall(r"(?<![.\d$])\b(\d+)\b(?!\.\d)", q)]
        pct = re.search(r"(\d+) percent", q)
        counts = [c for c in counts if not (pct and str(c) == pct.group(1))]
        # The original twelve spell their multipliers as words ("Three labels",
        # "Four friends"); the generated rows use digits. Both are read here, so
        # this check covers the hand-written half of the table too.
        words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}
        counts += [words[w] for w in re.findall(r"\b([A-Za-z]+)\b", q.lower())
                   if w in words]

        if "coupon removes" in q or "change is due" in q:
            want = dollars[0] - dollars[1]
        elif "split" in q:
            want = dollars[0] / counts[0]
        elif "tip to" in q or "percent tax" in q:
            want = dollars[0] * (100 + Decimal(pct.group(1))) / 100
        elif "discounted by" in q:
            want = dollars[0] * (100 - Decimal(pct.group(1))) / 100
        elif "each and" in q:                      # n of one, plus one other
            want = counts[0] * dollars[0] + dollars[1]
        elif " are $" in q:                        # one item, plus n of another
            want = dollars[0] + counts[0] * dollars[1]
        elif len(dollars) == 2:
            want = dollars[0] + dollars[1]
        elif counts:                               # n at one price
            want = counts[0] * dollars[0]
        else:
            raise AssertionError(f"no rule matched, so nothing was checked: {q}")

        assert int(want * 100) == int(task.answer_cents), (
            f"{task.id}: prose says {want}, table says {task.answer_cents}")
        assert f"${want}" not in q, f"{task.id} prints its own answer"
        checked += 1
    assert checked == len(TASKS) == 48


def test_selfplay_evaluation_splits_are_frozen_and_seeded():
    train, held_out, test = selfplay_splits(7, "absolute_zero")
    again = selfplay_splits(7, "absolute_zero")
    other = selfplay_splits(8, "absolute_zero")
    assert [t.meta for t in held_out] == [t.meta for t in again[1]]
    assert [t.meta for t in held_out] != [t.meta for t in other[1]]
    assert all(t.meta["kind"] == "frozen" and t.meta["answer"] for t in held_out + test)
    assert all(t.meta["kind"] == "selfplay" for t in train)
    kinds = {("abduction" if "hidden" in t.prompt else "deduction") for t in test}
    assert kinds == {"deduction", "abduction"}


def test_trajectory_reward_uses_the_tolerant_parse():
    record = {"item_cents": [125, 240], "quantities": [1, 2], "kind": "deduction",
              "valid": True}
    task = Task("t", "p")
    assert trajectory_reward(task, json.dumps({**record, "final": "605"})) == 1.0
    assert trajectory_reward(task, json.dumps({**record, "final": "Answer: 605"})) == 1.0
    assert trajectory_reward(task, json.dumps({**record, "final": "$6.05"})) == 0.0
    assert trajectory_reward(task, json.dumps({**record, "valid": False,
                                               "final": "605"})) == 0.0


def test_voyager_environment_reports_failures_not_gold_traces():
    ok, message = voyager_simulate(
        ["sanitize:vessel", "collect:water", "collect:mint", "heat:water",
         "combine:water+mint", "serve:drink"], "mint")
    assert ok
    ok, message = voyager_simulate(["collect:water", "serve:drink"], "mint")
    assert not ok
    # The feedback names the failed step's verb, never the full required list.
    assert "sanitize" in message
    assert "combine:water+mint" not in message


def test_generated_policy_gate_rejects_calls_and_imports():
    with pytest.raises(ValueError):
        compile_policy(
            "import os\ndef agent_prompt(question):\n return question\n",
            {"agent_prompt": 1},
        )
    with pytest.raises(ValueError):
        compile_policy(
            "def agent_prompt(question):\n return str(question)\n",
            {"agent_prompt": 1},
        )


def test_calculator_is_bounded():
    assert calculator("125+240*2") == 605
    with pytest.raises(ValueError):
        calculator("__import__('os').system('id')")


def test_proposal_limiter_caps_concurrent_async_overshoot():
    calls = []

    def propose(rendered, task, output, reward):
        calls.append(task.id)
        return "candidate"

    limiter = ProposalLimiter(propose, 2)
    task = Task("lane", "task")
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(
            pool.map(
                lambda _: limiter("artifact", task, "output", 0.0),
                range(8),
            )
        )
    assert rows.count("candidate") == 2
    assert calls == ["lane", "lane"]
    assert limiter.claimed == 2


# ---------------------------------------------------------------------------
# The bench entry point stays offline for --dry-run.
# ---------------------------------------------------------------------------

def test_benchmark_dry_run_is_offline_and_names_framework_runtime(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--dry-run", "--algorithms", "reflexion", "sica"]) == 0
    output = capsys.readouterr().out
    assert "reserved proposal calls=12" in output
    assert "serial/sync runtime=evolve" in output
    assert "async runtime=async_evolve" in output


def test_dry_run_counts_two_call_proposals(capsys):
    assert main(["--dry-run", "--algorithms", "self_refine", "r_zero"]) == 0
    assert "reserved proposal calls=24" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The population aggregator: a declared selection policy really sees a
# multi-candidate archive on the single-head ledger.
# ---------------------------------------------------------------------------

def test_population_aggregator_feeds_selection_a_real_archive():
    """AFlow rather than PromptBreeder.

    PromptBreeder used to be the vehicle here, and stopped being one when its
    tournament moved out of the selection seam: Algorithm 1 has to *evaluate*
    both sampled units on a training batch and *replace* the loser, and a
    `SelectionPolicy` can do neither -- it receives candidates with cached scores
    and returns one. AFlow's `SoftMixed` is a selection rule in the sense the
    seam means, so it is the honest test of the shared aggregator.
    """
    from dataclasses import replace as _replace

    from agentdescent.policies import Policies as _Policies
    from examples.aflow.aflow_workflow_search import SoftMixed

    class RecordingSelection(SoftMixed):
        def __init__(self):
            super().__init__(seed=0)
            self.sizes = []

        def select(self, ctx, n):
            self.sizes.append(len(ctx.candidates))
            return super().select(ctx, n)

    spy = RecordingSelection()
    _, builder = ALGORITHMS["aflow"]
    policy = _replace(builder(0), engine=_Policies(selection=spy))
    usage = Usage()
    recorder = Recorder(metered(fake_completion, usage), usage)
    payload = run_port(
        policy, recorder, mode="sync_parallel", seed=0, workers=2,
        candidate_budget=4, max_seconds=30.0, shutdown_grace=5.0,
    ).compact()
    # The archive held the seed plus at least one committed candidate, so the
    # selection rule ran over a genuine population at least once.
    assert spy.sizes, "selection policy was never consulted"
    assert max(spy.sizes) >= 2, f"archive never grew: {spy.sizes}"
    assert payload["final_quality"] >= payload["baseline_quality"]
