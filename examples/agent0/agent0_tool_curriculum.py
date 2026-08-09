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
import threading
from typing import Dict, Optional, Tuple

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import parse_json_object
from examples._method_policy import MethodPolicy, ValidatedSlot, clip_text
from examples._method_runner import standard_main
from examples._money_domain import STARTING_INSTRUCTION
from examples._selfplay_domain import (CartTask, majority_share, mixed_reward,
                                       proposer_prompt, selfplay_splits,
                                       trajectory, validate_generated)


FIDELITY = "inference_analogue"

_FALLBACK = CartTask((125, 240), (1, 2))

#: Executor samples per generated task. `generate_results` is the same
#: computation R-Zero uses, and `score = max_count / len(results)` needs more
#: than one sample to be anything but 0 or 1.
EXECUTOR_SAMPLES = 4

#: `calculate_tool_reward(predict, weight=0.05, cap=4)`: `min(calls, 4) * 0.05`,
#: counting ```` ```output ```` markers in the trajectory.
TOOL_WEIGHT = 0.05
TOOL_CAP = 4


def tool_reward(tool_calls: int) -> float:
    """`min(tool_call_count, cap) * weight`, upstream's `R_tool`."""
    return min(max(tool_calls, 0), TOOL_CAP) * TOOL_WEIGHT


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


def _tool_turns(llm, memory: str, question: str, unit: str) -> Tuple[str, int]:
    """The Executor's stop-and-go trajectory: request tool, run it, continue.

    Returns the final answer and how many tool calls actually executed --
    upstream counts ```` ```output ```` markers in the trajectory for `R_tool`,
    and a tool that was requested but rejected produced no output block.
    """
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
    calls = 0
    try:
        tool = parse_json_object(request)
        expression = tool.get("expression")
        if tool.get("tool") != "calculator" or not isinstance(expression, str):
            raise ValueError("invalid tool request")
        result = calculator(expression)
        calls = 1
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
    ), calls


def build(seed: int) -> MethodPolicy:
    signals: Dict[str, Tuple[float, int]] = {}
    lock = threading.Lock()

    def solve(llm, rendered: str, task: Task) -> str:
        if task.meta.get("kind") == "frozen":
            return _tool_turns(llm, rendered, task.prompt, task.id)[0]
        slot = int(task.meta["slot"])
        raw = llm(proposer_prompt("Agent0 curriculum agent", slot, rendered),
                  subphase="curriculum", unit=task.id)
        valid = True
        try:
            cart = validate_generated(parse_json_object(raw))
        except ValueError:
            cart, valid = _FALLBACK, False
        # `generate_results` samples the executor repeatedly and takes the
        # majority share. One sample makes it 0 or 1, and the uncertainty term
        # is zero at both -- the curriculum signal then never exists.
        runs = [_tool_turns(llm, rendered, cart.question, task.id)
                for _ in range(EXECUTOR_SAMPLES)]
        finals = [answer for answer, _ in runs]
        with lock:
            signals[task.id] = (majority_share(finals),
                                sum(calls for _, calls in runs))
        return trajectory(cart, finals[0], valid=valid)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        with lock:
            p_hat, calls = signals.get(task.id, (1.0, 0))
        # `min(score, 1 - score)`, on the executor's *self-consistency*. The
        # port used the grounded reward, which is 0 or 1 on a single rollout, so
        # the term was zero on every item of every run.
        uncertainty = min(p_hat, 1.0 - p_hat)
        bonus = tool_reward(calls)
        return llm(
            (
                "Agent0 co-evolution update: distill one reusable Executor "
                "policy lesson from the tool trajectory and grounded reward. "
                "The curriculum reward is min(p,1-p) + R_tool. Uncertainty is "
                f"{uncertainty:.2f} on this item (the executor agreed with "
                f"itself {p_hat:.0%} of the time) and R_tool is {bonus:.2f} "
                f"from {calls} tool call(s), capped at {TOOL_CAP}. Favor lessons "
                "that keep the calculator in the loop on frontier tasks. "
                "Include the answer representation; omit item-specific "
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
            "The curriculum update surfaces both reward components as numbers: min(p,1-p) on the executor's self-consistency over repeated samples, and R_tool = min(calls, 4) * 0.05.",
            "DifficultyWeighted's 4p(1-p) sampling weight shares its peak and zeros with min(p,1-p); upstream's term is min(p,1-p) rather than the 1-2|p-0.5| this port used, which is exactly twice it.",
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
