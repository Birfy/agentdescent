"""Inference analogue: **Agent0** (aiming-lab/Agent0@f775b510).

Preserved: Curriculum and Executor roles from the same base; **multi-turn
tool-integrated rollouts** through a sandboxed calculator (stop-and-go: request
tool, execute locally, continue with the result); the curriculum reward's
components surfaced in the update prompt -- uncertainty ``1−2|p̂−0.5|`` and the
tool-use count, upstream's ``R_unc``/``R_tool``; and frontier targeting via
:class:`~agentdescent.sampling.DifficultyWeighted`, whose ``4p(1−p)`` weight is
the same curve as ``1−2|p̂−0.5|``.

Boundaries: verbal policy memory replaces ADPO; one calculator tool replaces a
Python interpreter; the BLEU repetition penalty is omitted; evaluation carts
are frozen from the seed.
"""

from __future__ import annotations

import ast
from typing import Optional

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import parse_json_object
from examples._method_policy import MethodPolicy, ValidatedSlot, clip_text
from examples._method_runner import standard_main
from examples._money_domain import STARTING_INSTRUCTION
from examples._selfplay_domain import (CartTask, mixed_reward, proposer_prompt,
                                       selfplay_splits, trajectory,
                                       validate_generated)


FIDELITY = "inference_analogue"

_FALLBACK = CartTask((125, 240), (1, 2))


def calculator(expression: str) -> int:
    """AST-gated integer arithmetic: the sandboxed external tool."""
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


def _memory(text: str) -> str:
    value = clip_text(text)
    if not value:
        raise ValueError("empty policy memory")
    return value


def _tool_turns(llm, memory: str, question: str, unit: str) -> str:
    """The Executor's stop-and-go trajectory: request tool, run it, continue."""
    request = llm(
        (
            "You are Agent0's Executor. Use the calculator tool before the "
            "final answer. Prices are integer cents.\n\n"
            f"Policy memory: {memory}\nProblem: {question}\n"
            'Return JSON only as {"tool":"calculator","expression":"..."}.'
        ),
        subphase="tool_request",
        unit=unit,
    )
    try:
        tool = parse_json_object(request)
        expression = tool.get("expression")
        if tool.get("tool") != "calculator" or not isinstance(expression, str):
            raise ValueError("invalid tool request")
        result = calculator(expression)
    except ValueError:
        result = -1
    return llm(
        (
            "Continue the same Agent0 trajectory after the external tool call. "
            "Use policy memory and return only the final answer.\n\n"
            f"Policy memory: {memory}\nProblem: {question}\n"
            f"Calculator result: {result}"
        ),
        subphase="tool_result",
        unit=unit,
    )


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        if task.meta.get("kind") == "frozen":
            return _tool_turns(llm, rendered, task.prompt, task.id)
        slot = int(task.meta["slot"])
        raw = llm(proposer_prompt("Agent0 curriculum agent", slot, rendered),
                  subphase="curriculum", unit=task.id)
        valid = True
        try:
            cart = validate_generated(parse_json_object(raw))
        except ValueError:
            cart, valid = _FALLBACK, False
        final = _tool_turns(llm, rendered, cart.question, task.id)
        return trajectory(cart, final, valid=valid)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        uncertainty = 1.0 - 2.0 * abs(score - 0.5)
        return llm(
            (
                "Agent0 co-evolution update: distill one reusable Executor "
                "policy lesson from the tool trajectory and grounded reward. "
                "The curriculum reward combines uncertainty 1-2|p-0.5| "
                f"(currently {uncertainty:.2f}) with a tool-use bonus, so favor "
                "lessons that keep the calculator in the loop on frontier "
                "tasks. Include the answer representation; omit item-specific "
                "numbers.\n\n"
                f"Trajectory: {output[:400]}\nReward: {score}\n"
                f"Current policy memory:\n{rendered}"
            ),
            subphase="update",
            unit=task.id,
        )

    train, held_out, test = selfplay_splits(seed, "agent0")
    return MethodPolicy(
        name="agent0",
        fidelity=FIDELITY,
        notes=(
            "Curriculum/Executor co-evolution and multi-turn calculator use are preserved; frozen evaluation carts also run the tool loop.",
            "The curriculum update surfaces upstream's uncertainty and tool-use reward components.",
            "DifficultyWeighted's 4p(1-p) sampling weight is the same curve as Agent0's 1-2|p-0.5| uncertainty reward.",
            "Verbal policy memory replaces ADPO post-training; one calculator replaces the Python interpreter.",
        ),
        strategy=ValidatedSlot(initial_value=STARTING_INSTRUCTION,
                               validator=_memory),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=mixed_reward,
        proposal_calls_per_candidate=1,
        engine=Policies(task_sampler=DifficultyWeighted()),
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
