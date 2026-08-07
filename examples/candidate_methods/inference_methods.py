"""AgentDescent ports for PromptBreeder, AFlow, Reflexion, and Self-Refine."""

from __future__ import annotations

from typing import Dict

from agentdescent.evolution import Task

from .common import (
    STARTING_INSTRUCTION,
    canonical_json,
    clip_instruction,
    feedback,
    money_reward,
    money_splits,
    solve_money,
)
from .framework import PortSpec, ValidatedSlot
from .runtime import Recorder, parse_json_object


def _json_instruction_artifact(text: str, defaults: Dict[str, str]) -> str:
    payload = parse_json_object(text)
    result = {
        key: clip_instruction(payload.get(key), fallback=value)
        for key, value in defaults.items()
    }
    if not all(result.values()):
        raise ValueError("artifact has empty instruction fields")
    return canonical_json(result)


def make_promptbreeder(recorder: Recorder, seed: int) -> PortSpec:
    """Co-evolve a task prompt and its mutation prompt inside the engine."""
    defaults = {
        "task_prompt": STARTING_INSTRUCTION,
        "mutation_prompt": (
            "Infer a reusable instruction from evaluator feedback and improve "
            "this mutation instruction too."
        ),
    }
    initial = canonical_json(defaults)

    def validate(text: str) -> str:
        return _json_instruction_artifact(text, defaults)

    def run(rendered: str, task: Task) -> str:
        genome = parse_json_object(rendered)
        return solve_money(
            recorder,
            str(genome["task_prompt"]),
            task,
            phase="run:promptbreeder",
        )

    def propose(rendered: str, task: Task, output: str, reward: float) -> str:
        genome = parse_json_object(rendered)
        raw = recorder.call(
            (
                "You are PromptBreeder's mutation operator. Mutate both the task "
                "prompt and mutation prompt from execution feedback. Generalize; "
                "do not include the benchmark item's numeric answer.\n\n"
                f"Task prompt: {genome['task_prompt']}\n"
                f"Mutation prompt: {genome['mutation_prompt']}\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return JSON only with task_prompt and mutation_prompt strings."
            ),
            phase="proposal:promptbreeder",
            unit=task.id,
        )
        try:
            return validate(raw)
        except ValueError:
            return canonical_json(
                {
                    "task_prompt": "Compute carefully and return integer cents only.",
                    "mutation_prompt": "Generalize evaluator feedback into domain rules.",
                }
            )

    train, held_out, test = money_splits(seed)
    return PortSpec(
        "promptbreeder",
        "mechanism_microport",
        ValidatedSlot(initial, validate),
        train,
        held_out,
        test,
        run,
        money_reward,
        propose,
        1,
        (
            "Task-prompt and mutation-prompt co-evolution follow Algorithm 1; the population is compact.",
            "PromptBreeder has no official released implementation, so fidelity is paper-level.",
        ),
    )


def make_aflow(recorder: Recorder, seed: int) -> PortSpec:
    """Search a two-node Solve -> ReviewAndRevise workflow."""
    defaults = {
        "solve_instruction": STARTING_INSTRUCTION,
        "review_instruction": "Check the arithmetic and return only the corrected answer.",
        "modification": "root workflow",
    }
    initial = canonical_json(defaults)

    def validate(text: str) -> str:
        return _json_instruction_artifact(text, defaults)

    def run(rendered: str, task: Task) -> str:
        graph = parse_json_object(rendered)
        draft = solve_money(
            recorder,
            str(graph["solve_instruction"]),
            task,
            phase="run:aflow:solve",
        )
        return recorder.call(
            (
                f"Problem:\n{task.prompt}\n\nDraft answer:\n{draft}\n\n"
                f"ReviewAndRevise node:\n{graph['review_instruction']}\n\n"
                "Return only the final answer."
            ),
            phase="run:aflow:review",
            unit=task.id,
        )

    def propose(rendered: str, task: Task, output: str, reward: float) -> str:
        graph = parse_json_object(rendered)
        raw = recorder.call(
            (
                "You are AFlow's graph optimizer. Expand the root from its "
                "execution feedback while preserving exactly two nodes: Solve, "
                "then ReviewAndRevise.\n\n"
                f"Current graph: {rendered}\nReward: {reward}\n"
                f"{feedback(task, output)}\n\n"
                "Return JSON only with modification, solve_instruction, and "
                "review_instruction strings."
            ),
            phase="proposal:aflow",
            unit=task.id,
        )
        try:
            return validate(raw)
        except ValueError:
            graph.update(
                {
                    "solve_instruction": "Compute carefully and return integer cents only.",
                    "review_instruction": "Verify arithmetic and return integer cents only.",
                    "modification": "learn strict integer-cents output",
                }
            )
            return canonical_json(graph)

    train, held_out, test = money_splits(seed)
    return PortSpec(
        "aflow",
        "mechanism_microport",
        ValidatedSlot(initial, validate),
        train,
        held_out,
        test,
        run,
        money_reward,
        propose,
        1,
        (
            "Execution-feedback graph expansion is preserved with one search depth.",
            "Every candidate keeps the same two model nodes, fixing workflow topology across modes.",
        ),
    )


def _make_reflection_port(
    recorder: Recorder,
    seed: int,
    *,
    name: str,
    proposal_prompt: str,
    notes: tuple,
) -> PortSpec:
    def validate(text: str) -> str:
        value = clip_instruction(text, fallback="")
        if not value:
            raise ValueError("empty reflection")
        return value

    def run(rendered: str, task: Task) -> str:
        return solve_money(recorder, rendered, task, phase=f"run:{name}")

    def propose(rendered: str, task: Task, output: str, reward: float) -> str:
        raw = recorder.call(
            (
                f"{proposal_prompt}\n\nCurrent memory/instruction:\n{rendered}\n\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return only one reusable instruction; omit item-specific numbers."
            ),
            phase=f"proposal:{name}",
            unit=task.id,
        )
        return validate(raw) if raw.strip() else "Return monetary answers as integer cents only."

    train, held_out, test = money_splits(seed)
    return PortSpec(
        name,
        "mechanism_microport",
        ValidatedSlot(STARTING_INSTRUCTION, validate),
        train,
        held_out,
        test,
        run,
        money_reward,
        propose,
        1,
        notes,
    )


def make_reflexion(recorder: Recorder, seed: int) -> PortSpec:
    return _make_reflection_port(
        recorder,
        seed,
        name="reflexion",
        proposal_prompt=(
            "You are Reflexion's verbal reflection module. Convert the failed "
            "trajectory and external evaluator signal into episodic memory for retry."
        ),
        notes=(
            "Attempt, external feedback, verbal reflection, and retry are mapped to rollout, proposal, and gate retry.",
            "A deterministic arithmetic evaluator replaces HotpotQA/ALFWorld.",
        ),
    )


def make_self_refine(recorder: Recorder, seed: int) -> PortSpec:
    return _make_reflection_port(
        recorder,
        seed,
        name="self_refine",
        proposal_prompt=(
            "You are Self-Refine's FEEDBACK module. Give specific actionable "
            "feedback that the next REFINE attempt can apply."
        ),
        notes=(
            "GENERATE, FEEDBACK, and REFINE map to rollout, proposal, and held-out rerun.",
            "One refinement iteration is used on a compact task-specific rubric.",
        ),
    )
