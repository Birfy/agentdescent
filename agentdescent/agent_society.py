"""Agent society: the framework evolves a cooperating topology, not just one artifact.

Everything else in this project evolves one artifact against one task set. This
module adds the topology dimension (Anthropic's multi-agent research system;
AgentEvolve): a **society** is a graph of agent nodes -- each node an
instruction and a budget -- with delegation edges, and the whole graph -- which
nodes exist, who hands to whom, what each node's instruction is -- is one
evolvable artifact. ``society_evolve`` runs the ordinary ``evolve()`` over
society-run tasks: a candidate is a gated source defining the graph, a task is
one multi-step problem routed through the graph, and the reward is correctness
minus cost, so the search finds the topology that solves the task with the
least delegation. A single-node society is just the old one-agent loop; a
society that delegates to a specialist that the single node lacked is the
dimension up.
"""

from __future__ import annotations

import ast
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task, evolve
from .meta import MetaOutcome, slot_reflector

__all__ = ["AgentNode", "SocietyGraph", "society_spec", "society_problem",
           "SocietyReport", "society_evolve"]


@dataclass(frozen=True)
class AgentNode:
    """One node of a society: a name, an instruction (a callable), and a
    budget (how many uses the node may spend)."""

    name: str
    fn: Callable[[Any], Any]
    budget: int = 1


@dataclass
class SocietyGraph:
    """A directed graph of :class:`AgentNode` with delegation edges.

    ``solve(input)`` walks a **single delegation chain** from ``root``: each
    node's output is handed to its first successor edge, a node runs at most
    ``budget`` times, a cycle stops the walk, and a node with no outgoing edges
    is the sink whose output is the society's answer. (Edges beyond the first
    per node are accepted and recorded but a chain, not a fan-out, is walked;
    ``used`` is the number of node executions.)
    """

    nodes: Dict[str, AgentNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    root: str = ""

    def solve(self, value: Any) -> Tuple[Any, int]:
        used = 0
        current = value
        current_node = self.root
        seen: set = set()
        node_uses: Dict[str, int] = {}
        while current_node in self.nodes:
            node = self.nodes[current_node]
            if node_uses.get(current_node, 0) >= node.budget:
                break                      # this node's budget is spent
            if current_node in seen:
                break                      # a cycle is not walked twice
            seen.add(current_node)
            node_uses[current_node] = node_uses.get(current_node, 0) + 1
            current = node.fn(current)
            used += 1
            outs = [b for a, b in self.edges if a == current_node]
            if not outs:
                break
            current_node = outs[0]
        return current, used


SOCIETY_SEED = """def society():
    # one node that doubles the input -- the single-agent baseline
    return {
        "nodes": {"double": {"budget": 1, "code": "lambda x: x * 2"}},
        "edges": [],
        "root": "double",
    }
"""


def _gate_society(source: str) -> ast.Module:
    """The society's gate: exactly one ``society()`` function whose body is a
    single ``return <dict literal>``, built only from safe syntax (no imports,
    no dunder, no interpreter calls, no frame/attribute access)."""
    if not source or len(source) > 12_000:
        raise ValueError("society source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("society source is not valid Python") from error
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "society":
        raise ValueError("society source must define exactly one function, society")
    body = functions[0].body
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        raise ValueError("society() must be exactly 'return <dict literal>'")
    if not _is_literal(body[0].value):
        raise ValueError("society() must return a literal dict/list/str/number, "
                         "not computed code")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            raise ValueError("imports are not allowed in a society")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute):
            # frame attributes and dunder attributes alike -- the body is data
            raise ValueError("attribute access is forbidden in a society")
        if isinstance(node, ast.Call):
            raise ValueError("calls are forbidden in a society: the body must "
                             "be a literal")
    return tree


def _is_literal(node: ast.AST) -> bool:
    """A structure ``ast.literal_eval`` can evaluate: constants and
    dict/list/tuple/set literals of those, with string keys."""
    if isinstance(node, (ast.Constant, ast.NameConstant)):
        return True
    if isinstance(node, ast.Dict):
        return all(isinstance(k, ast.Constant) and _is_literal(v)
                   for k, v in zip(node.keys, node.values))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_literal(el) for el in node.elts)
    return False


def _gate_lambda(code: str) -> ast.Expression:
    """A node's ``code`` must be exactly one lambda expression, and nothing the
    outer gate forbids: no imports, no dunder names, no attribute access, no
    interpreter/reflector calls. The string constant is invisible to the outer
    AST walk, so it gets its own gate before it is ever evaluated."""
    if not code or len(code) > 2_000:
        raise ValueError("node code is empty or too long")
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as error:
        raise ValueError("node code is not valid Python") from error
    if not isinstance(tree.body, ast.Lambda):
        raise ValueError("node code must be a lambda")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden in node code")
        if isinstance(node, ast.Attribute):
            # e.g. ().__class__ / __init__.__globals__ -- no attribute access
            raise ValueError("attribute access is forbidden in node code")
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and
                    func.id in ("exec", "eval", "open", "__import__",
                                "getattr", "setattr", "globals", "vars")):
                raise ValueError(f"forbidden call: {func.id}")
            if isinstance(func, ast.Attribute):
                raise ValueError("method calls are forbidden in node code")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            raise ValueError("attribute indexing is forbidden in node code")
    return tree


def _compile_society(source: str) -> SocietyGraph:
    source = (source or "").strip()
    _gate_society(source)
    tree = ast.parse(source)
    fn = tree.body[0]
    spec = ast.literal_eval(fn.body[0].value)   # pure data: no exec at all
    if not isinstance(spec, dict) or "nodes" not in spec:
        raise ValueError("society() must return a dict with 'nodes'")
    safe_builtins = {
        k: (__builtins__[k] if isinstance(__builtins__, dict)
            else getattr(__builtins__, k))
        for k in ("abs", "all", "any", "bool", "dict", "float", "int", "len",
                  "list", "map", "max", "min", "range", "set", "str", "tuple",
                  "zip", "True", "False", "None",
                  "Exception", "ValueError", "TypeError", "KeyError")
        if k in (__builtins__ if isinstance(__builtins__, dict)
                 else dir(__builtins__))}
    nodes = {}
    for name, info in spec.get("nodes", {}).items():
        code = str(info["code"])
        _gate_lambda(code)
        nodes[name] = AgentNode(
            name=str(name),
            fn=eval(code, {"__builtins__": safe_builtins}),
            budget=int(info.get("budget", 1)))
    edges = [tuple(e) for e in spec.get("edges", [])]
    root = str(spec.get("root", next(iter(nodes), "")))
    return SocietyGraph(nodes=nodes, edges=edges, root=root)


def society_spec(seed: str = SOCIETY_SEED) -> Any:
    """The evolvable spec: the value is a gated ``society()`` source compiled
    into a :class:`SocietyGraph`. ``SOCIETY_SEED`` is the single-agent baseline
    -- the topology the search is asked to beat."""
    from .meta import SourceSlot
    return SourceSlot(
        initial_value=seed.strip() + "\n",
        validate=lambda text: (text or "").strip() + "\n",
        build=_compile_society,
        description=(
            "The value is one function `society()` whose body is exactly "
            "`return <dict literal>` returning "
            "{'nodes': {name: {'budget': int, 'code': 'lambda x: ...'}}, "
            "'edges': [(a, b), ...], 'root': name}. The body is data, not "
            "code: no imports, no attribute access, no calls, no dunder. "
            "Reply with the complete function in one ```python fence."))


def society_problem(
    tasks: Mapping[str, Any],
    *,
    gold: Callable[[Any], Any],
    lam: float = 0.05,
) -> Callable:
    """A :class:`~agentdescent.meta.Problem` over multi-step tasks.

    ``(graph, seed) -> MetaOutcome``: ``tasks[seed % len(tasks)]`` is routed
    through the graph, and the reward is ``1.0 - lam * nodes_used`` when the
    final output matches ``gold`` (0 otherwise) -- correctness minus
    delegation cost, so a topology that solves the task with the least
    hand-off wins."""

    keys = list(tasks)

    def problem(graph: Any, seed: int) -> MetaOutcome:
        task = tasks[keys[seed % len(keys)]]
        output, used = graph.solve(task)
        reward = (1.0 - lam * used) if output == gold(task) else 0.0
        return MetaOutcome(curve=[reward, reward], final=reward, rollouts=used,
                           detail={"used": used, "correct": output == gold(task)})

    return problem


@dataclass(frozen=True)
class SocietyReport:
    """What :func:`society_evolve` found: the evolved graph, its cost-adjusted
    correctness on validation tasks, and the same for the seed."""

    seed_value: str
    value: str
    seed_score: float
    evolved_score: float
    gain: float


def society_evolve(
    problems: Union[Sequence[Callable], Mapping[str, Callable]],
    val_problems: Union[Sequence[Callable], Mapping[str, Callable]],
    *,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    val_seeds: Sequence[int] = (100, 101, 102, 103),
    seed_source: str = SOCIETY_SEED,
    **evolve_kwargs: Any,
) -> Tuple[SocietyReport, EvolutionResult]:
    """Evolve the society topology on correctness-minus-cost.

    A bespoke ``evolve()`` over ``society_problem`` tasks: the artifact is the
    graph source, ``propose`` rewrites it, the reward is correctness minus
    delegation cost. The report compares the evolved graph with the seed on
    ``val_problems`` at ``val_seeds`` (disjoint from the training ``seeds``),
    so "the delegation won" is measured, not assumed.
    """
    if propose is None and model is None:
        raise ValueError("society_evolve() needs propose= or model=")
    spec = society_spec(seed=seed_source)
    named = dict(problems)
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for seed in seeds for name in named]
    reflector = propose if propose is not None else slot_reflector(model, spec)
    spec.compile(spec.render(spec.initial()))

    def run(rendered: str, task: Task) -> str:
        try:
            graph = spec.compile(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": f"compile: {error}"}).to_json()
        outcome = named[task.meta["problem"]](graph, int(task.meta["seed"]))
        return outcome.to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(MetaOutcome.from_json(output).final)
        except (TypeError, ValueError, KeyError):
            return 0.0

    result = evolve(
        tasks, reward, run=run, propose=reflector, strategy=spec,
        initial_state=spec.initial(), self_verify=False, solved_threshold=1.1,
        **evolve_kwargs)
    value = result.rendered or spec.render(spec.initial())
    seed_value = spec.render(spec.initial())

    seed_score = _mean_score(spec.compile(seed_value), val_problems, val_seeds)
    evolved_score = _mean_score(spec.compile(value), val_problems, val_seeds)
    report = SocietyReport(
        seed_value=seed_value, value=value,
        seed_score=seed_score, evolved_score=evolved_score,
        gain=evolved_score - seed_score)
    try:
        result.meta = {"seed_score": seed_score,
                       "evolved_score": evolved_score,
                       "gain": evolved_score - seed_score}
    except AttributeError:
        pass
    return report, result


def _mean_score(graph: Any, problems: Mapping[str, Callable],
                seeds: Sequence[int]) -> float:
    total = 0.0
    n = 0
    for problem in problems.values():
        for seed in seeds:
            total += float(problem(graph, seed).final)
            n += 1
    return total / n if n else 0.0
