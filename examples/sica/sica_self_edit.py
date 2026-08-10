"""Self-edit analogue: **SICA** (MaximeRobeyns/self_improving_coding_agent@ed8275dc).

Preserved: the meta loop edits real Python policy source through an AST gate,
utility is the measured benchmark score, and the next base is the **best agent
in the archive of prior versions** -- `get_best_agent_iteration` takes
``idxmax()`` of the mean benchmark score, and `runner.py` then runs
``archive.agent_{best_iter}.agent_code``. That is `Archive`'s ``best`` mode, not
its ``performance`` one: a softmax over scores in ``[0, 1]`` at temperature 1
leaves only ``exp(1)/exp(0) = 2.7`` between the best and worst entry, so a run
configured as "performance" starts from the worst agent in the archive about a
quarter of the time. (The paper's cost/time composite is unimplemented upstream
too.)

An unparseable self-edit costs its candidate and produces no diff -- there is
no substitute source. ``reflective=False``: synthesised merges of Python source
would bypass the AST gate.
"""

from __future__ import annotations

import ast
from types import FunctionType
from typing import Callable, Dict, Mapping, Optional

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.selection import Archive

from examples._measure import extract_python
from examples._method_policy import MethodPolicy, ValidatedSlot
from examples._method_runner import standard_main
from examples._gsm8k_domain import feedback, gsm8k_reward, gsm8k_splits


FIDELITY = "self_edit_analogue"

SICA_INITIAL_SOURCE = '''def agent_prompt(question):
    return "Solve the arithmetic problem carefully. Return only the final answer.\\n\\n" + question
'''

_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.BinOp, ast.Add, ast.Constant, ast.Name, ast.Load,
)


def compile_policy(source: str, required: Mapping[str, int]) -> Dict[str, Callable[..., str]]:
    """AST-gated compilation shared by the self-edit analogues."""
    if not source or len(source) > 4_000:
        raise ValueError("policy source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("policy source is not valid Python") from error
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != len(required) or {n.name for n in functions} != set(required):
        raise ValueError("policy source has the wrong function surface")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"policy source contains forbidden AST node {type(node).__name__}")
    for function in functions:
        if len(function.args.args) != required[function.name]:
            raise ValueError(f"{function.name} has the wrong arity")
        if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
            raise ValueError(f"{function.name} must contain exactly one return")
    namespace: Dict[str, object] = {"__builtins__": {}}
    exec(compile(tree, "<candidate-policy>", "exec"), namespace, namespace)
    compiled: Dict[str, Callable[..., str]] = {}
    for name in required:
        function = namespace.get(name)
        if not isinstance(function, FunctionType):
            raise ValueError(f"{name} did not compile to a function")
        compiled[name] = function
    return compiled


def policy_prompt(function: Callable[[str], str], task: Task) -> str:
    value = function(task.prompt)
    if not isinstance(value, str) or not value or len(value) > 4_000:
        raise ValueError("policy returned an invalid prompt")
    return value


def _sica_source(text: str) -> str:
    source = extract_python(text)
    compile_policy(source, {"agent_prompt": 1})
    return source.strip() + "\n"


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        functions = compile_policy(rendered, {"agent_prompt": 1})
        return llm(policy_prompt(functions["agent_prompt"], task), unit=task.id)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        return llm(
            (
                "You are SICA's meta-improvement tool. Edit the coding agent's "
                "own Python source to improve measured accuracy. Preserve "
                "exactly agent_prompt(question), with one return expression "
                "made only from string constants, +, and question. No imports, "
                "calls, assignments, or comments.\n\n"
                f"Current source:\n```python\n{rendered}```\n\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return only complete revised Python source in one code fence."
            ),
            unit=task.id,
        )

    train, held_out, test = gsm8k_splits(seed)
    return MethodPolicy(
        name="sica",
        fidelity=FIDELITY,
        notes=(
            "The meta loop edits real Python policy source and the framework gate measures utility.",
            "Archive selection in `best` mode -- `get_best_agent_iteration` takes idxmax() of the mean benchmark score, so the next meta-improvement starts from exactly that agent and never from a sampled one; the run finalises on the archive's best scorer.",
            "An unparseable self-edit costs its candidate and produces no diff; there is no substitute source.",
            "The editable surface is one AST-gated function rather than SWE-bench Docker.",
        ),
        strategy=ValidatedSlot(initial_value=SICA_INITIAL_SOURCE,
                               validator=_sica_source),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=gsm8k_reward,
        proposal_calls_per_candidate=1,
        engine=Policies(selection=Archive(sampling="best", seed=seed)),
        reflective=False,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
