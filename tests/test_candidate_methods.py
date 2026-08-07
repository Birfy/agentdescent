import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from agentdescent.agents import Usage, metered
from agentdescent.evolution import Task
from examples.candidate_methods.benchmark import (
    ALGORITHMS,
    PROPOSAL_CALLS_PER_CANDIDATE,
    main,
)
from examples.candidate_methods.domain import TASKS, parse_integer_answer, split_tasks
from examples.candidate_methods.framework import ProposalLimiter, run_port
from examples.candidate_methods.runtime import MODES, Recorder, parse_json_object
from examples.candidate_methods.self_edit_methods import _compile_policy
from examples.candidate_methods.self_play_methods import _calculator, _generated_task


ANSWERS = {task.question: task.answer_cents for task in TASKS}


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
    if "promptbreeder's mutation operator" in lower:
        return json.dumps(
            {
                "task_prompt": "Compute carefully and return integer cents only.",
                "mutation_prompt": "Generalize evaluator feedback into domain rules.",
            }
        )
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
        return "Compute carefully and return monetary answers as integer cents only."
    if "complete revised python source" in lower and "agent_prompt" in lower:
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
    if "voyager's automatic curriculum" in lower:
        return '{"task_id":"assigned"}'
    if "voyager repairs executable programs" in lower:
        return json.dumps(
            {
                "steps": [
                    "sanitize:vessel",
                    "collect:water",
                    "collect:{ingredient}",
                    "heat:water",
                    "combine:water+{ingredient}",
                    "serve:drink",
                ]
            }
        )
    if "voyager's critic" in lower:
        return '{"success":true,"critique":"verified from events"}'
    if "voyager's action agent" in lower:
        ingredient_match = re.search(r"Visible ingredient: (\w+)", prompt)
        ingredient = ingredient_match.group(1) if ingredient_match else "mint"
        actions = ["collect:water", f"collect:{ingredient}", "serve:drink"]
        if "sanitize:vessel" in lower:
            actions = [
                "sanitize:vessel",
                "collect:water",
                f"collect:{ingredient}",
                "heat:water",
                f"combine:water+{ingredient}",
                "serve:drink",
            ]
        return json.dumps({"actions": actions})
    if "skillweaver propose" in lower:
        return json.dumps(
            {"calls": ["open:{page}", "fill:{field}={value}", "click:save"]}
        )
    if "skillweaver practice" in lower:
        return '{"page":"/settings/profile","field":"timezone","value":"UTC"}'
    if "skillweaver hone" in lower:
        return json.dumps(
            {
                "calls": [
                    "open:{page}",
                    "wait:hydration-complete",
                    "fill:{field}={value}",
                    "click:save",
                    "assert:saved-toast",
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
                f"open:{page}",
                "wait:hydration-complete",
                f"fill:{field}={value}",
                "click:save",
                "assert:saved-toast",
            ]
        return json.dumps({"calls": calls})
    if "zero-data self-play loop" in lower:
        return '{"item_cents":[125,240],"quantities":[1,2]}'
    if "agent0's executor" in lower:
        return '{"tool":"calculator","expression":"125+240+240"}'
    if "continue the same agent0 trajectory" in lower:
        return "605" if "integer cents" in lower else "$6.05"
    if "solve the self-generated problem" in lower:
        return "605" if "integer cents" in lower else "$6.05"
    if any(
        marker in lower
        for marker in (
            "absolute zero updates",
            "update only r-zero's challenger",
            "update only r-zero's solver",
            "agent0 co-evolution update",
        )
    ):
        return "Represent monetary totals as integer cents and output only that integer."
    answer = _money_answer(prompt)
    if answer is not None:
        return answer
    return "Represent monetary totals as integer cents and output only that integer."


def _offline_run(algorithm, mode):
    usage = Usage()
    recorder = Recorder(metered(fake_completion, usage), usage)
    fidelity, factory = ALGORITHMS[algorithm]
    spec = factory(recorder, 0)
    assert spec.fidelity == fidelity
    return run_port(
        spec,
        recorder,
        mode=mode,
        seed=0,
        workers=2,
        candidate_budget=2,
        max_seconds=30.0,
        shutdown_grace=5.0,
    ).compact()


def test_structured_parsers_reject_ambiguous_values():
    assert parse_json_object("prose ```json\n{\"ok\": true}\n```") == {"ok": True}
    with pytest.raises(ValueError):
        parse_json_object("there is no object here")
    assert parse_integer_answer("Answer: 415") == "415"
    assert parse_integer_answer("$4.15") is None


def test_money_splits_are_disjoint_and_complete():
    train, held_out, test = split_tasks(0)
    ids = [{task.id for task in rows} for rows in (train, held_out, test)]
    assert [len(rows) for rows in ids] == [4, 4, 4]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_generated_policy_gate_rejects_calls_and_imports():
    with pytest.raises(ValueError):
        _compile_policy(
            "import os\ndef agent_prompt(question):\n return question\n",
            {"agent_prompt": 1},
        )
    with pytest.raises(ValueError):
        _compile_policy(
            "def agent_prompt(question):\n return str(question)\n",
            {"agent_prompt": 1},
        )


def test_zero_data_curriculum_and_calculator_are_bounded():
    task = _generated_task(
        {
            "item_cents": [437, 892, 1543, 267, 3189, 76, 1105],
            "quantities": [14, 9, 5, 23, 3, 41, 7],
        },
        slot=0,
    )
    assert task.answer == "48420"
    assert "type-7" in task.question
    assert _calculator("125+240*2") == 605
    with pytest.raises(ValueError):
        _calculator("__import__('os').system('id')")


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
    assert payload["budget"]["observed_proposal_calls"] == (
        2 * PROPOSAL_CALLS_PER_CANDIDATE[algorithm]
    )
    assert payload["usage"]["calls"] == len(payload["events"])
    assert payload["framework"]["runtime"] == (
        "async_evolve" if mode == "async_pipeline" else "evolve"
    )
    assert payload["framework"]["rollouts"] >= 2
    assert not ({"prompt", "response", "source", "artifact"} & payload.keys())


def test_benchmark_dry_run_is_offline_and_names_framework_runtime(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--dry-run", "--algorithms", "reflexion", "sica"]) == 0
    output = capsys.readouterr().out
    assert "reserved proposal calls=12" in output
    assert "serial/sync runtime=evolve" in output
    assert "async runtime=async_evolve" in output


def test_dry_run_reports_extra_environment_proposal_budget(capsys):
    assert main(["--dry-run", "--algorithms", "voyager", "skillweaver"]) == 0
    assert "reserved proposal calls=36" in capsys.readouterr().out
