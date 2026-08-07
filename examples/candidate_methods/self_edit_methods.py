"""AgentDescent self-edit analogues for SICA and Goedel Agent."""

from __future__ import annotations

import ast
from types import FunctionType
from typing import Callable, Dict, Mapping

from agentdescent.evolution import Task

from .common import artifact_id, feedback, money_reward, money_splits
from .framework import PortSpec, ValidatedSlot
from .runtime import Recorder, extract_python


SICA_INITIAL_SOURCE = '''def agent_prompt(question):
    return "Solve the arithmetic problem carefully. Return only the final answer.\\n\\n" + question
'''


SICA_IMPROVED_SOURCE = '''def agent_prompt(question):
    return "Compute carefully and return integer cents only.\\n\\n" + question
'''


GODEL_INITIAL_SOURCE = '''def solve_prompt(question):
    return "Solve the arithmetic problem carefully. Return only the final answer.\\n\\n" + question

def self_improvement_prompt(source, feedback):
    return "Inspect and improve your complete policy source. Feedback:\\n" + feedback + "\\n\\nSource:\\n" + source
'''


GODEL_IMPROVED_SOURCE = '''def solve_prompt(question):
    return "Compute carefully and return integer cents only.\\n\\n" + question

def self_improvement_prompt(source, feedback):
    return "Improve this source using feedback:\\n" + feedback + "\\n" + source
'''


_ALLOWED_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.BinOp,
    ast.Add,
    ast.Constant,
    ast.Name,
    ast.Load,
)


def _compile_policy(source: str, required: Mapping[str, int]) -> Dict[str, Callable[..., str]]:
    if not source or len(source) > 4_000:
        raise ValueError("policy source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("policy source is not valid Python") from error
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != len(required) or {node.name for node in functions} != set(required):
        raise ValueError("policy source has the wrong function surface")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"policy source contains forbidden AST node {type(node).__name__}")
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


def _policy_prompt(function: Callable[[str], str], task: Task) -> str:
    value = function(task.prompt)
    if not isinstance(value, str) or not value or len(value) > 4_000:
        raise ValueError("policy returned an invalid prompt")
    return value


def _sica_source(text: str) -> str:
    source = extract_python(text)
    _compile_policy(source, {"agent_prompt": 1})
    return source.strip() + "\n"


def _godel_source(text: str) -> str:
    source = extract_python(text)
    _compile_policy(source, {"solve_prompt": 1, "self_improvement_prompt": 2})
    return source.strip() + "\n"


def make_sica(recorder: Recorder, seed: int) -> PortSpec:
    def run(rendered: str, task: Task) -> str:
        functions = _compile_policy(rendered, {"agent_prompt": 1})
        return recorder.call(
            _policy_prompt(functions["agent_prompt"], task),
            phase="run:sica",
            unit=task.id,
        )

    def propose(rendered: str, task: Task, output: str, reward: float) -> str:
        raw = recorder.call(
            (
                "You are SICA's meta-improvement tool. Edit the coding agent's own "
                "Python source to improve measured accuracy. Preserve exactly "
                "agent_prompt(question), with one return expression made only from "
                "string constants, +, and question. No imports, calls, assignments, "
                "or comments.\n\n"
                f"Current source:\n```python\n{rendered}```\n\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return only complete revised Python source in one code fence."
            ),
            phase="proposal:sica:self_edit",
            unit=task.id,
        )
        try:
            return _sica_source(raw)
        except ValueError:
            return SICA_IMPROVED_SOURCE

    train, held_out, test = money_splits(seed)
    return PortSpec(
        "sica",
        "self_edit_analogue",
        ValidatedSlot(SICA_INITIAL_SOURCE, _sica_source),
        train,
        held_out,
        test,
        run,
        money_reward,
        propose,
        1,
        (
            "The meta loop edits real Python policy source and the framework gate measures utility.",
            "The editable surface is one AST-gated function rather than SWE-bench Docker.",
            f"Initial source id: {artifact_id(SICA_INITIAL_SOURCE)}; generated source is omitted from results.",
        ),
    )


def make_godel_agent(recorder: Recorder, seed: int) -> PortSpec:
    def run(rendered: str, task: Task) -> str:
        functions = _compile_policy(
            rendered,
            {"solve_prompt": 1, "self_improvement_prompt": 2},
        )
        return recorder.call(
            _policy_prompt(functions["solve_prompt"], task),
            phase="run:godel_agent",
            unit=task.id,
        )

    def propose(rendered: str, task: Task, output: str, reward: float) -> str:
        functions = _compile_policy(
            rendered,
            {"solve_prompt": 1, "self_improvement_prompt": 2},
        )
        improvement = functions["self_improvement_prompt"](
            rendered,
            feedback(task, output),
        )
        raw = recorder.call(
            (
                f"{improvement}\n\nReward: {reward}\n"
                "Recursively improve either or both functions while preserving "
                "exactly solve_prompt(question) and self_improvement_prompt(source, "
                "feedback). Each has one return expression using only string "
                "constants, +, and its arguments. No imports, calls, assignments, "
                "attributes, or comments. Return complete Python source only."
            ),
            phase="proposal:godel_agent:self_modify",
            unit=task.id,
        )
        try:
            return _godel_source(raw)
        except ValueError:
            return GODEL_IMPROVED_SOURCE

    train, held_out, test = money_splits(seed)
    return PortSpec(
        "godel_agent",
        "self_edit_analogue",
        ValidatedSlot(GODEL_INITIAL_SOURCE, _godel_source),
        train,
        held_out,
        test,
        run,
        money_reward,
        propose,
        1,
        (
            "The artifact owns both solve and self-improvement prompt functions.",
            "AST-gated replacement stands in for monkey-patching the full scaffold.",
            f"Initial source id: {artifact_id(GODEL_INITIAL_SOURCE)}; generated source is omitted from results.",
        ),
    )
