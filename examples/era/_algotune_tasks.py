"""Loading an AlgoTune task, and turning its reference into a seed program.

Standard library only, and loadable by file path, because both sides of the
sandbox need it: the host reads a task's source to derive the root node's
program, and :mod:`examples.era._era_algotune_runner` -- which runs inside
Bubblewrap with ``python -I`` and no package on ``sys.path`` -- loads this file
the same way ``_era_integration_runner.py`` loads the integrand catalogue.

What an AlgoTune task is
------------------------
Upstream (`oripress/AlgoTune`) ships one class per task under
``AlgoTuneTasks/<name>/<name>.py``::

    @register_task("svd")
    class SVD(Task):
        def generate_problem(self, n, random_seed=1) -> dict: ...
        def solve(self, problem) -> Any: ...
        def is_solution(self, problem, solution) -> bool: ...

``generate_problem`` is the dataset, ``solve`` is the reference implementation
the candidate has to beat, and ``is_solution`` is the correctness oracle. Those
three are the whole benchmark: the score is *how much faster than ``solve``* a
program is, on problems ``is_solution`` accepts.

Two things have to happen before that class can be used here.

**The import has to resolve.** Every task file begins ``from AlgoTuneTasks.base
import register_task, Task``, and ``AlgoTuneTasks.base`` drags in the whole
upstream harness -- litellm, orjson, a dataset manager, a multiprocessing pool.
:func:`install_shim` supplies the four names a task file actually touches, so a
task loads from its own source with nothing installed.

**The reference has to become a program.** ERA searches over *self-contained
programs*: the root node has to be runnable source, and the mutation prompt
shows it to the model as the code to improve. A bound method on a class the
sandbox does not have is neither, so :func:`derive_seed_program` rewrites the
class's ``solve`` into a module-level ``solve(problem)`` -- inlining the
``self.`` attributes it reads and the helper methods it calls, and dropping
everything else. The result is checked, not trusted: `prepare_suite` scores it
through the sandbox before a search starts, and a task whose derivation does not
reproduce the reference is refused rather than silently searched.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple


#: The upstream revision every URL, task source and published problem size in
#: this port is read from. Pinned for the same reason `_era_support` pins its
#: Kaggle CSV: a benchmark that follows a moving branch is not a benchmark.
UPSTREAM_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"


# --------------------------------------------------------------------------
# The import shim
# --------------------------------------------------------------------------


def install_shim() -> Dict[str, type]:
    """Register a minimal ``AlgoTuneTasks.base`` and return its task registry.

    Upstream's real ``base.py`` is 1100 lines of dataset management, JSON
    round-tripping and a multiprocessing validation pool, and importing it needs
    most of AlgoTune's dependency list. A task *file* uses four names from it:
    the ``register_task`` decorator, the ``Task`` base class, and -- through
    ``Task.__init__`` -- ``self.task_name`` and ``self.oracle``. Those are what
    this installs.

    Idempotent, and it hands back the *live* registry dict rather than a copy,
    so a second load of a second task file appends to the same mapping.
    """
    existing = sys.modules.get("AlgoTuneTasks.base")
    if existing is not None:
        return existing.TASK_REGISTRY  # type: ignore[attr-defined]

    registry: Dict[str, type] = {}

    def register_task(name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            registry.setdefault(name, cls)
            return cls
        return decorator

    class Task:
        """The attributes an AlgoTune task file reads off its own base class."""

        def __init__(self, n: Optional[int] = None, dataset_size: Optional[int] = None,
                     target_time_ms: Optional[int] = None,
                     data_dir: Optional[str] = None, **_kwargs: Any) -> None:
            self.task_name = type(self).__name__
            self.k: Optional[int] = None
            self.n = n
            self.dataset_size = dataset_size
            self.target_time_ms = target_time_ms
            self.data_dir = data_dir
            self.oracle = self.solve

        def generate_problem(self, n: int, random_seed: int = 1) -> Any:
            raise NotImplementedError

        def solve(self, problem: Any) -> Any:
            raise NotImplementedError

        def is_solution(self, problem: Any, solution: Any) -> bool:
            raise NotImplementedError

    package = types.ModuleType("AlgoTuneTasks")
    package.__path__ = []  # type: ignore[attr-defined]
    base = types.ModuleType("AlgoTuneTasks.base")
    base.register_task = register_task  # type: ignore[attr-defined]
    base.Task = Task  # type: ignore[attr-defined]
    base.TASK_REGISTRY = registry  # type: ignore[attr-defined]
    package.base = base  # type: ignore[attr-defined]
    sys.modules["AlgoTuneTasks"] = package
    sys.modules["AlgoTuneTasks.base"] = base
    return registry


def load_module(path: str, name: str) -> types.ModuleType:
    """Import a file by path, registering it before it executes.

    Registered first, not after, for the reason ``_era_integration_runner``
    gives: ``@dataclass`` resolves its own module through
    ``sys.modules[cls.__module__]``, and a module that is not there yet fails
    with an error that reads like a broken candidate.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_task(path: str, task_name: str) -> Any:
    """Instantiate the task class registered by the file at ``path``."""
    registry = install_shim()
    load_module(path, f"algotune_task_{task_name}")
    cls = registry.get(task_name)
    if cls is None:
        raise RuntimeError(
            f"{path} registered {sorted(registry)} rather than {task_name!r}")
    return cls()


# --------------------------------------------------------------------------
# The reference, as a program
# --------------------------------------------------------------------------


class _SelfRewriter(ast.NodeTransformer):
    """Rewrite ``self.x`` into the module-level name ``x`` was lifted to."""

    def __init__(self, names: Dict[str, str]) -> None:
        self.names = names

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            target = self.names.get(node.attr)
            if target is None:
                raise DerivationError(f"self.{node.attr} cannot be lifted")
            return ast.copy_location(ast.Name(id=target, ctx=node.ctx), node)
        return node


class DerivationError(RuntimeError):
    """The task's reference could not be expressed as a standalone program."""


def _task_class(tree: ast.Module) -> ast.ClassDef:
    """The class ``register_task`` decorates, or the only class in the file.

    Two of the 154 task files define a helper class before the task, so "the
    first ClassDef" is wrong. The decorator is the upstream marker and is what
    :func:`install_shim` keys its registry on, so it is what this looks for.
    """
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    decorated = [
        node for node in classes
        if any(isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
               and dec.func.id == "register_task" for dec in node.decorator_list)
    ]
    if decorated:
        return decorated[-1]
    if len(classes) == 1:
        return classes[0]
    raise DerivationError("no @register_task class in the task source")


def _init_constants(cls: ast.ClassDef) -> Dict[str, ast.expr]:
    """``self.x = <expr>`` assignments in ``__init__``, by attribute name."""
    constants: Dict[str, ast.expr] = {}
    init = next((node for node in cls.body
                 if isinstance(node, ast.FunctionDef) and node.name == "__init__"), None)
    if init is None:
        return constants
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                constants[target.attr] = node.value
    return constants


def _module_header(tree: ast.Module) -> List[ast.stmt]:
    """Imports, module constants and free functions, minus the AlgoTune import.

    ``from AlgoTuneTasks.base import register_task, Task`` is the one import a
    derived program must not keep: the shim it would reach is not in the
    sandbox, and the gate would refuse it in any case.
    """
    header: List[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias for alias in node.names
                     if not alias.name.split(".")[0] == "AlgoTuneTasks"]
            if names:
                header.append(ast.Import(names=names))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "AlgoTuneTasks":
                continue
            header.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign)):
            header.append(node)
        elif isinstance(node, ast.Try):
            # A module-level `try: from x import y / except: y = None` is how an
            # optional dependency gets bound, and dropping it does not drop a
            # feature -- it leaves the *name* unbound, so the derived reference
            # raises NameError from whatever used it. `polynomial_real` binds
            # `threadpool_limits` this way and died on exactly that, which reads
            # as a broken task rather than a derivation that threw a statement
            # away. Kept whole: it is upstream's code, and rewriting the fallback
            # would be guessing at what upstream meant.
            header.append(node)
    return header


def _lifted_name(attr: str, *, method: bool) -> str:
    """The module-level name a lifted attribute takes.

    Prefixed so a lifted ``self.mode`` cannot collide with a local named
    ``mode`` in the method that reads it, and cased by kind so a reader of the
    seed program can see at a glance which names were class state.
    """
    stem = attr.lstrip("_")
    return f"_ref_{stem}" if method else f"_REF_{stem.upper()}"


def derive_seed_program(source: str, *, entry: str = "solve") -> str:
    """Rewrite a task class's ``solve`` into a standalone ``solve(problem)``.

    The transform is deliberately small, and it fails loudly rather than
    guessing:

    * the module's imports, constants and free functions are kept as they are;
    * ``solve`` becomes a module-level function with ``self`` dropped;
    * every ``self.x`` it reads is lifted -- a *method* becomes another
      module-level function (processed the same way, recursively), an attribute
      assigned in ``__init__`` becomes a module-level constant;
    * anything else -- state built somewhere other than ``__init__``, a
      reference to ``self`` that is not an attribute access -- raises
      :class:`DerivationError`, and the task is refused.

    Comments do not survive: the output is ``ast.unparse`` of the rewritten
    tree, so the model sees the reference's code and docstrings but not its
    ``#`` annotations. That is the price of a mechanical derivation over a
    hand-copied one, and it is worth paying -- a hand-copied reference is a
    second thing to keep in step with upstream, and 154 of them is a benchmark
    that rots.
    """
    tree = ast.parse(source)
    cls = _task_class(tree)
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    constants = _init_constants(cls)
    if entry not in methods:
        raise DerivationError(f"the task class defines no {entry}()")

    body: List[ast.stmt] = []
    emitted: Dict[str, str] = {}
    lifted_constants: Dict[str, str] = {}
    pending: List[Tuple[str, str]] = [(entry, entry)]
    while pending:
        method, out_name = pending.pop(0)
        if method in emitted:
            continue
        emitted[method] = out_name
        function = methods[method]
        mapping: Dict[str, str] = {}
        needed = sorted({
            node.attr for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        })
        for attr in needed:
            if attr in methods:
                target = _lifted_name(attr, method=True)
                mapping[attr] = target
                pending.append((attr, target))
            elif attr in constants:
                target = _lifted_name(attr, method=False)
                mapping[attr] = target
                if attr not in lifted_constants:
                    lifted_constants[attr] = target
                    body.append(ast.Assign(
                        targets=[ast.Name(id=target, ctx=ast.Store())],
                        value=constants[attr]))
            else:
                raise DerivationError(
                    f"self.{attr} is set outside __init__, so {method}() cannot "
                    f"be lifted out of the class")
        # Re-parsed rather than mutated in place: the class body's nodes carry
        # the original file's column offsets, and `ast.unparse` of a rewritten
        # method that still holds them is a syntax error on the first `if`.
        lifted = ast.parse(ast.unparse(function)).body[0]
        assert isinstance(lifted, ast.FunctionDef)
        lifted = _SelfRewriter(mapping).visit(lifted)
        lifted.name = out_name
        lifted.decorator_list = []
        if lifted.args.args and lifted.args.args[0].arg == "self":
            lifted.args.args = lifted.args.args[1:]
        elif lifted.args.posonlyargs and lifted.args.posonlyargs[0].arg == "self":
            lifted.args.posonlyargs = lifted.args.posonlyargs[1:]
        else:
            raise DerivationError(f"{method}() is not an instance method")
        body.append(lifted)

    module = ast.Module(body=_module_header(tree) + body, type_ignores=[])
    ast.fix_missing_locations(module)
    rendered = ast.unparse(module)
    if "self" in {node.id for node in ast.walk(ast.parse(rendered))
                  if isinstance(node, ast.Name)}:
        raise DerivationError("the derived program still refers to self")
    return rendered.rstrip() + "\n"


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def best_seconds(
    call: Callable[[Any], Any],
    problem: Any,
    *,
    repeats: int,
    deadline: Optional[float] = None,
) -> Tuple[float, Any, int]:
    """The fastest of ``repeats`` runs, and the output of the last one.

    Minimum rather than mean, which is what AlgoTune reports: its benchmark
    keeps ``min_time_ms`` and ``calculate_input_speedup`` divides those. On a
    shared machine the minimum is the estimate of the program's own cost that
    interference can only bias one way, and the same rule is applied to the
    reference in the same process moments earlier, so the ratio is what moves.

    Each run is handed a **deep copy** of the problem, made outside the timed
    region. That is not politeness about mutation: identical arguments across
    repeats would let a candidate memoise on the first run and report the
    dictionary lookups of the second as its runtime. Copying costs the same on
    both sides of the ratio.

    Returns ``(seconds, last_output, runs)``. ``runs`` is short of ``repeats``
    when the deadline cut the loop off -- a program 400x slower than the
    reference is measured once and reported, not run five times.
    """
    best = float("inf")
    output: Any = None
    runs = 0
    for _ in range(max(1, repeats)):
        payload = copy.deepcopy(problem)
        started = time.perf_counter()
        output = call(payload)
        elapsed = time.perf_counter() - started
        runs += 1
        best = min(best, elapsed)
        if deadline is not None and time.perf_counter() > deadline:
            break
    return best, output, runs
