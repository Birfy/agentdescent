"""AgentDescent zero-data analogues for Absolute Zero, R-Zero, and Agent0."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from agentdescent.evolution import Task

from .common import STARTING_INSTRUCTION, canonical_json, clip_instruction
from .framework import PortSpec, ValidatedSlot
from .runtime import Recorder, parse_json_object


@dataclass(frozen=True)
class GeneratedMoneyTask:
    item_cents: Tuple[int, ...]
    quantities: Tuple[int, ...]

    @property
    def question(self) -> str:
        terms = []
        for index, (price, quantity) in enumerate(zip(self.item_cents, self.quantities), 1):
            noun = "item" if quantity == 1 else "items"
            terms.append(f"{quantity} type-{index} {noun} at ${price / 100:.2f} each")
        return "A cart contains " + ", ".join(terms) + ". What is the total amount?"

    @property
    def answer(self) -> str:
        return str(sum(price * quantity for price, quantity in zip(self.item_cents, self.quantities)))


def _generated_task(payload: Dict[str, Any], *, slot: int) -> GeneratedMoneyTask:
    item_cents = payload.get("item_cents")
    quantities = payload.get("quantities")
    if not isinstance(item_cents, list) or not isinstance(quantities, list):
        raise ValueError("item_cents and quantities must be lists")
    if not 2 <= len(item_cents) == len(quantities) <= 8:
        raise ValueError("generated task must have two to eight terms")
    if not all(isinstance(value, int) and 1 <= value <= 20_000 for value in item_cents):
        raise ValueError("item_cents values are invalid")
    if not all(isinstance(value, int) and 1 <= value <= 100 for value in quantities):
        raise ValueError("quantity values are invalid")
    return GeneratedMoneyTask(tuple(item_cents), tuple(quantities))


def _fallback_task(slot: int) -> GeneratedMoneyTask:
    return GeneratedMoneyTask((125 + 37 * slot, 240 + 29 * slot), (1, 2))


def _selfplay_tasks(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    rows = [
        Task(
            id=f"zero:{seed + index}",
            prompt="Generate and solve one verifier-grounded latent curriculum item.",
            meta={"slot": seed + index},
        )
        for index in range(12)
    ]
    return rows[:4], rows[4:8], rows[8:12]


def _proposer_prompt(role: str, slot: int, memory: str) -> str:
    return (
        f"You are the {role} in a zero-data self-play loop. Propose a novel "
        "money calculation near the solver's capability frontier. A trusted local "
        "renderer creates the word problem and verifies sum(item_cents[i] * "
        "quantities[i]). Use 2 to 8 terms, item_cents 1..20000, quantities "
        "1..100. No benchmark label is available.\n\n"
        f"Curriculum memory: {memory}\nSlot: {slot}\n"
        'Return JSON only with integer-list "item_cents" and "quantities".'
    )


def _solver_prompt(task: GeneratedMoneyTask, memory: str) -> str:
    return (
        "Solve the self-generated problem. Follow policy memory exactly and "
        "return only the final answer.\n\n"
        f"Policy memory:\n{memory}\n\nProblem:\n{task.question}"
    )


def _trajectory(task: GeneratedMoneyTask, final: str, *, valid: bool) -> str:
    return canonical_json(
        {
            "item_cents": list(task.item_cents),
            "quantities": list(task.quantities),
            "final": final.strip()[:120],
            "valid": valid,
        }
    )


def _trajectory_reward(_task: Task, output: str) -> float:
    try:
        payload = parse_json_object(output)
        if payload.get("valid") is not True:
            return 0.0
        generated = _generated_task(payload, slot=0)
        return float(str(payload.get("final", "")).strip() == generated.answer)
    except ValueError:
        return 0.0


def _plain_memory(text: str) -> str:
    value = clip_instruction(text, fallback="")
    if not value:
        raise ValueError("empty policy memory")
    return value


def make_absolute_zero(recorder: Recorder, seed: int) -> PortSpec:
    def run(rendered: str, task: Task) -> str:
        slot = int(task.meta["slot"])
        raw = recorder.call(
            _proposer_prompt("Absolute Zero proposer", slot, rendered),
            phase="run:absolute_zero:proposer",
            unit=task.id,
        )
        valid = True
        try:
            generated = _generated_task(parse_json_object(raw), slot=slot)
        except ValueError:
            generated = _fallback_task(slot)
            valid = False
        final = recorder.call(
            _solver_prompt(generated, rendered),
            phase="run:absolute_zero:solver",
            unit=task.id,
        )
        return _trajectory(generated, final, valid=valid)

    def propose(rendered: str, task: Task, output: str, score: float) -> str:
        payload = parse_json_object(output)
        generated = _generated_task(payload, slot=int(task.meta["slot"]))
        raw = recorder.call(
            (
                "Absolute Zero updates the same model from grounded verifier "
                "reward. Distill one reusable policy lesson, including the output "
                "representation; omit item-specific numbers.\n\n"
                f"Problem: {generated.question}\nAttempt: {payload.get('final')}\n"
                f"Verifier answer: {generated.answer}\nReward: {score}"
            ),
            phase="proposal:absolute_zero:update",
            unit=task.id,
        )
        return _plain_memory(raw) if raw.strip() else "Return monetary totals as integer cents only."

    train, held_out, test = _selfplay_tasks(seed)
    return PortSpec(
        "absolute_zero",
        "inference_analogue",
        ValidatedSlot(STARTING_INSTRUCTION, _plain_memory),
        train,
        held_out,
        test,
        run,
        _trajectory_reward,
        propose,
        1,
        (
            "Proposer, solver, grounded verifier, and zero-data self-play curriculum are preserved.",
            "Verbal policy memory replaces PPO weight updates on the available 8 GB GPU.",
        ),
    )


def _rzero_artifact(text: str) -> str:
    payload = parse_json_object(text)
    challenger = clip_instruction(payload.get("challenger_memory"), fallback="")
    solver = clip_instruction(payload.get("solver_memory"), fallback="")
    if not challenger or not solver:
        raise ValueError("R-Zero artifact requires both role memories")
    return canonical_json(
        {"challenger_memory": challenger, "solver_memory": solver}
    )


def make_r_zero(recorder: Recorder, seed: int) -> PortSpec:
    initial = canonical_json(
        {
            "challenger_memory": "Increase curriculum difficulty gradually.",
            "solver_memory": STARTING_INSTRUCTION,
        }
    )

    def run(rendered: str, task: Task) -> str:
        memory = parse_json_object(rendered)
        slot = int(task.meta["slot"])
        raw = recorder.call(
            _proposer_prompt(
                "R-Zero Challenger",
                slot,
                str(memory["challenger_memory"]),
            ),
            phase="run:r_zero:challenger",
            unit=task.id,
        )
        valid = True
        try:
            generated = _generated_task(parse_json_object(raw), slot=slot)
        except ValueError:
            generated = _fallback_task(slot)
            valid = False
        final = recorder.call(
            _solver_prompt(generated, str(memory["solver_memory"])),
            phase="run:r_zero:solver",
            unit=task.id,
        )
        return _trajectory(generated, final, valid=valid)

    def propose(rendered: str, task: Task, output: str, score: float) -> str:
        payload = parse_json_object(output)
        generated = _generated_task(payload, slot=int(task.meta["slot"]))
        challenger = recorder.call(
            (
                "Update only R-Zero's Challenger policy toward a valid task at "
                "the Solver frontier. Return one general lesson.\n\n"
                f"Generated problem: {generated.question}\nSolver reward: {score}"
            ),
            phase="proposal:r_zero:challenger_update",
            unit=task.id,
        )
        solver = recorder.call(
            (
                "Update only R-Zero's Solver policy from verifier feedback. State "
                "the answer representation and omit item-specific numbers.\n\n"
                f"Problem: {generated.question}\nAttempt: {payload.get('final')}\n"
                f"Verifier answer: {generated.answer}\nReward: {score}"
            ),
            phase="proposal:r_zero:solver_update",
            unit=task.id,
        )
        return canonical_json(
            {
                "challenger_memory": clip_instruction(
                    challenger, fallback="Increase curriculum difficulty gradually."
                ),
                "solver_memory": clip_instruction(
                    solver, fallback="Return monetary totals as integer cents only."
                ),
            }
        )

    train, held_out, test = _selfplay_tasks(seed)
    return PortSpec(
        "r_zero",
        "inference_analogue",
        ValidatedSlot(initial, _rzero_artifact),
        train,
        held_out,
        test,
        run,
        _trajectory_reward,
        propose,
        2,
        (
            "Separate Challenger and Solver roles, frontier tasks, rewards, and role updates are preserved.",
            "Two-model GRPO is replaced by separate verbal role memories.",
        ),
    )


def _calculator(expression: str) -> int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("calculator expression is invalid") from error
    allowed = (ast.Expression, ast.BinOp, ast.Add, ast.Mult, ast.Constant)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("calculator accepts only integer addition and multiplication")
    if any(
        isinstance(node, ast.Constant)
        and (not isinstance(node.value, int) or not 0 <= node.value <= 100_000)
        for node in ast.walk(tree)
    ):
        raise ValueError("calculator operands exceed bounds")
    value = eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
    if not isinstance(value, int) or not 0 <= value <= 20_000_000:
        raise ValueError("calculator result exceeds bounds")
    return value


def make_agent0(recorder: Recorder, seed: int) -> PortSpec:
    def run(rendered: str, task: Task) -> str:
        slot = int(task.meta["slot"])
        raw = recorder.call(
            _proposer_prompt("Agent0 curriculum agent", slot, rendered),
            phase="run:agent0:curriculum",
            unit=task.id,
        )
        valid = True
        try:
            generated = _generated_task(parse_json_object(raw), slot=slot)
        except ValueError:
            generated = _fallback_task(slot)
            valid = False
        request = recorder.call(
            (
                "You are Agent0's Executor. Use the calculator tool before the "
                "final answer. Prices are integer cents.\n\n"
                f"Problem: {generated.question}\nItem cents: {generated.item_cents}\n"
                f"Quantities: {generated.quantities}\n"
                'Return JSON only as {"tool":"calculator","expression":"..."}.'
            ),
            phase="run:agent0:tool_request",
            unit=task.id,
        )
        try:
            tool = parse_json_object(request)
            expression = tool.get("expression")
            if tool.get("tool") != "calculator" or not isinstance(expression, str):
                raise ValueError("invalid tool request")
            result = _calculator(expression)
        except ValueError:
            result = -1
            valid = False
        final = recorder.call(
            (
                "Continue the same Agent0 trajectory after the external tool call. "
                "Use policy memory and return only the final answer.\n\n"
                f"Policy memory: {rendered}\nProblem: {generated.question}\n"
                f"Calculator result: {result}"
            ),
            phase="run:agent0:tool_result",
            unit=task.id,
        )
        return _trajectory(generated, final, valid=valid)

    def propose(rendered: str, task: Task, output: str, score: float) -> str:
        payload = parse_json_object(output)
        generated = _generated_task(payload, slot=int(task.meta["slot"]))
        raw = recorder.call(
            (
                "Agent0 co-evolution update: distill a reusable Executor policy "
                "lesson from the tool trajectory and grounded reward. Include the "
                "answer representation; omit item-specific numbers.\n\n"
                f"Problem: {generated.question}\nFinal: {payload.get('final')}\n"
                f"Verifier answer: {generated.answer}\nReward: {score}"
            ),
            phase="proposal:agent0:update",
            unit=task.id,
        )
        return _plain_memory(raw) if raw.strip() else "Use calculator totals and return integer cents only."

    train, held_out, test = _selfplay_tasks(seed)
    return PortSpec(
        "agent0",
        "inference_analogue",
        ValidatedSlot(STARTING_INSTRUCTION, _plain_memory),
        train,
        held_out,
        test,
        run,
        _trajectory_reward,
        propose,
        1,
        (
            "Curriculum/Executor co-evolution and multi-turn calculator use are preserved.",
            "Verbal policy memory replaces RL post-training.",
        ),
    )
