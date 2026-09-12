"""The compact formation domain: a language toolchain grown from an empty repo.

Upstream's formation run starts from a repository with **no implementation in
it** and ends with a 248,989-line C compiler after 123.4 hours and 1,019
episodes. That is not a thing a test suite can run, so this is the same *shape*
at a size that finishes offline in seconds: a human supplies the architecture,
the constraints and a frozen validation suite; the agents supply every line of
implementation, growing the tree as they go.

What is faithful here
---------------------
* The repository starts implementation-empty -- ``CONTEXT.md`` records and a
  frozen spec, and nothing else. Every source file in the result was created by
  an episode.
* Validation is **staged**, as a real compiler suite is: a case exercises the
  lexer, the parser, or the evaluator. So the suite has a gradient before the
  toolchain is complete, which is what lets any gate -- the engine's or the
  parent's -- see progress at all. Upstream gets this from c-testsuite's own
  spread; without it, formation has literally zero signal until the last file
  lands, and no acceptance rule can help.
* The frozen files are the human's, and stay the human's: ``spec/**`` is refused
  to every proposal and restored pristine before scoring.

What is a surrogate, and says so
--------------------------------
``--offline`` proposes with rule-based actors that reveal pre-written module
implementations one step at a time. It exercises *this port's mechanism* --
delegation, the spatial contract, the octopus merge, the parent's verdict -- and
it does not exercise a model's ability to write a compiler. It is the same
device as DGM's surrogate objective and the molecule search's offline operators,
and it is why the offline numbers are a property of the harness rather than a
result about models. Pass ``--model`` for actors that are asked to write the code.

One human-supplied constraint is worth naming because the domain does not work
without it: ``src/__init__.py`` imports each stage **lazily**, so a missing
parser does not stop the lexer's cases from scoring. That is the architectural
kind of constraint the paper says the human provides.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agentdescent.evolution import Task
from agentdescent.filetree import materialize, parse_tree

from ._delegation import Brief, Delegation, Edit
from ._spatial import SITUATED_EDIT_PROTOCOL, parse_situated_edits
from ._world import normalise

__all__ = ["FROZEN", "build_tasks", "initial_files", "llm_executor", "llm_manager",
           "make_runner", "offline_executor", "offline_manager", "reward"]

#: Human-supplied and never the agents'. L0 in this repository's sense, and the
#: role c-testsuite / LLVM / Csmith play upstream.
FROZEN = ("spec/**",)

ENTRY = "src/__init__.py"
LEXER = "src/frontend/lexer.py"
PARSER = "src/frontend/parser.py"
EVALUATOR = "src/backend/evaluator.py"


# ---------------------------------------------------------------------------
# The human's repository: context, constraints, validation. No implementation.
# ---------------------------------------------------------------------------

_SPEC = '''# minilang -- the language this project implements

An expression language over integers.

    expr   := term (("+" | "-") term)*
    term   := atom (("*" | "/") atom)*
    atom   := NUMBER | "-" atom | "(" expr ")"

`/` is floor division. Whitespace is insignificant.

## The three validated stages

| stage | entry point | what a case checks |
|---|---|---|
| `tok`  | `src.tokenize(source)` | the token list, as `repr` |
| `ast`  | `src.parse(source)`    | the parse tree, as `repr` |
| `val`  | `src.evaluate(source)` | the integer value, as `repr` |

A stage is scored independently of the ones after it, so `src/__init__.py` MUST
import each stage lazily, inside the function that needs it. A module-level
import of a stage that does not exist yet would fail every case in the suite,
including the ones that stage has nothing to do with.

Tokens are `("num", <int>)` and `("op", <str>)`. The parse tree is nested tuples:
`("num", n)`, `("neg", x)`, and `(op, left, right)` for the four binary operators.
'''

_ROOT_CONTEXT = '''# minilang -- root

## Intent
Implement the language in `spec/CONTEXT.md`. The specification and its staged
validation are given; no implementation exists yet.

## Routing table
- `./src/` -> the implementation, and the three stage entry points

## Constraints
- `spec/` is read-only. It is the validation contract, not a work item.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = '''# src -- implementation root

## Intent
Own `src/__init__.py`, the public surface named by the spec: `tokenize`, `parse`
and `evaluate`. Each MUST import its stage lazily.

## Routing table
- `./src/frontend/` -> tokenizer and parser
- `./src/backend/`  -> evaluation of the parse tree
'''

_FRONTEND_CONTEXT = '''# src/frontend -- source text to parse tree

## Intent
`lexer.py` turns source text into the token list; `parser.py` turns tokens into
the nested-tuple parse tree, honouring precedence and parentheses.
'''

_BACKEND_CONTEXT = '''# src/backend -- parse tree to value

## Intent
`evaluator.py` walks the parse tree. One function per node kind, so that two
agents fixing two different node kinds are editing two different parts of the
file rather than the same one.
'''


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from."""
    return {
        "CONTEXT.md": _ROOT_CONTEXT,
        "spec/CONTEXT.md": _SPEC,
        "src/CONTEXT.md": _SRC_CONTEXT,
        "src/frontend/CONTEXT.md": _FRONTEND_CONTEXT,
        "src/backend/CONTEXT.md": _BACKEND_CONTEXT,
    }


# ---------------------------------------------------------------------------
# The reference implementation: what the offline actors reveal, and the oracle
# the frozen suite's expectations are computed from.
# ---------------------------------------------------------------------------

_REF_ENTRY = '''"""Public surface. Each stage is imported lazily -- see spec/CONTEXT.md."""


def tokenize(source):
    from .frontend.lexer import tokenize as _tokenize
    return _tokenize(source)


def parse(source):
    from .frontend.parser import parse as _parse
    return _parse(source)


def evaluate(source):
    from .backend.evaluator import evaluate as _evaluate
    return _evaluate(parse(source))
'''

_REF_LEXER = '''"""Source text -> a list of ("num", int) / ("op", str) tokens."""

import re

_TOKEN = re.compile(r"\\s*(?:(\\d+)|(.))")


def tokenize(source):
    tokens, pos = [], 0
    while pos < len(source):
        match = _TOKEN.match(source, pos)
        if match is None:
            break
        pos = match.end()
        number, char = match.group(1), match.group(2)
        if number is not None:
            tokens.append(("num", int(number)))
        elif char is not None and not char.isspace():
            tokens.append(("op", char))
    return tokens
'''

_REF_PARSER = '''"""Tokens -> a nested-tuple parse tree, by recursive descent."""

from .lexer import tokenize


def parse(source):
    node, _pos = _expr(tokenize(source), 0)
    return node


def _expr(tokens, pos):
    node, pos = _term(tokens, pos)
    while pos < len(tokens) and tokens[pos][0] == "op" and tokens[pos][1] in "+-":
        op = tokens[pos][1]
        right, pos = _term(tokens, pos + 1)
        node = (op, node, right)
    return node, pos


def _term(tokens, pos):
    node, pos = _atom(tokens, pos)
    while pos < len(tokens) and tokens[pos][0] == "op" and tokens[pos][1] in "*/":
        op = tokens[pos][1]
        right, pos = _atom(tokens, pos + 1)
        node = (op, node, right)
    return node, pos


def _atom(tokens, pos):
    kind, value = tokens[pos]
    if kind == "num":
        return ("num", value), pos + 1
    if value == "-":
        node, pos = _atom(tokens, pos + 1)
        return ("neg", node), pos
    if value == "(":
        node, pos = _expr(tokens, pos + 1)
        return node, pos + 1
    raise SyntaxError(value)
'''

#: The shape the backend CONTEXT.md asks for: one function per node kind, so two
#: agents filling two different kinds edit two different hunks. That is what
#: makes `OctopusConflict` do something a keyed union cannot.
_EVALUATOR_SKELETON = '''"""Parse tree -> value. One function per node kind."""


def evaluate(node):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "neg":
        return eval_neg(node)
    if kind in ("+", "-"):
        return eval_addsub(node)
    if kind in ("*", "/"):
        return eval_muldiv(node)
    raise ValueError(kind)


def eval_neg(node):
    raise NotImplementedError("neg")


def eval_addsub(node):
    raise NotImplementedError("addsub")


def eval_muldiv(node):
    raise NotImplementedError("muldiv")
'''

_STUBS = {
    "neg": ('def eval_neg(node):\n    raise NotImplementedError("neg")\n',
            'def eval_neg(node):\n    return -evaluate(node[1])\n'),
    "addsub": ('def eval_addsub(node):\n    raise NotImplementedError("addsub")\n',
               'def eval_addsub(node):\n    left, right = evaluate(node[1]), evaluate(node[2])\n'
               '    return left + right if node[0] == "+" else left - right\n'),
    "muldiv": ('def eval_muldiv(node):\n    raise NotImplementedError("muldiv")\n',
               'def eval_muldiv(node):\n    left, right = evaluate(node[1]), evaluate(node[2])\n'
               '    return left * right if node[0] == "*" else left // right\n'),
}


_PKG_INIT = '"""Package marker."""\n'


def _reference_tree() -> Dict[str, str]:
    """The finished repository -- the oracle the frozen suite is computed from."""
    evaluator = _EVALUATOR_SKELETON
    for stub, filled in _STUBS.values():
        evaluator = evaluator.replace(stub, filled)
    tree = dict(initial_files())
    tree.update({
        ENTRY: _REF_ENTRY,
        "src/frontend/__init__.py": _PKG_INIT,
        "src/backend/__init__.py": _PKG_INIT,
        LEXER: _REF_LEXER,
        PARSER: _REF_PARSER,
        EVALUATOR: evaluator,
    })
    return tree


# ---------------------------------------------------------------------------
# Running a candidate repository
# ---------------------------------------------------------------------------

#: Executed inside the materialised workspace, in a child process. A candidate
#: is agent-written code: it can loop, raise or exit, and none of those may take
#: the run with it.
_HARNESS = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
cases = json.loads(sys.argv[2])
import src
out = []
for kind, source in cases:
    try:
        if kind == "tok":
            value = src.tokenize(source)
        elif kind == "ast":
            value = src.parse(source)
        else:
            value = src.evaluate(source)
        out.append(repr(value))
    except Exception as exc:            # a stage that is not built yet
        out.append("ERROR:" + type(exc).__name__)
print(json.dumps(out), end="")
"""

#: What a case scores when the workspace could not be run at all -- a syntax
#: error in a generated module, a timeout, a missing package marker. Distinct
#: from a wrong answer only in the transcript; both score zero, and conflating
#: them in the *reward* would be the bug, not here.
CRASHED = "ERROR:workspace"


def _run_cases(state: Mapping[str, str], cases: Sequence[Sequence[str]],
               *, timeout: float = 30.0) -> List[str]:
    """Materialise ``state`` and evaluate every case in one child process."""
    if not cases:
        return []
    workspace = tempfile.mkdtemp(prefix="genesis-run-")
    try:
        materialize(state, workspace)
        proc = subprocess.run(
            [sys.executable, "-c", _HARNESS, workspace, json.dumps(list(cases))],
            capture_output=True, text=True, timeout=timeout, cwd=workspace)
        if proc.returncode != 0:
            return [CRASHED] * len(cases)
        return list(json.loads(proc.stdout))
    except (OSError, ValueError, subprocess.SubprocessError):
        return [CRASHED] * len(cases)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


#: The frozen validation suite: (stage, source). Ordered so that the held-out
#: tail `evolve()` cuts is a mix of all three stages rather than one of them.
_CASES = (
    ("tok", "1 + 2"), ("ast", "1 + 2"), ("val", "1 + 2"),
    ("tok", "7"), ("ast", "7"), ("val", "7"),
    ("tok", "2 * 3"), ("ast", "2 * 3"), ("val", "2 * 3"),
    ("tok", "-5"), ("ast", "-5"), ("val", "-5"),
    ("tok", "8 / 2"), ("ast", "8 / 2"), ("val", "8 / 2"),
    ("tok", "(1 + 2) * 3"), ("ast", "(1 + 2) * 3"), ("val", "(1 + 2) * 3"),
    ("tok", "10 - 4 - 3"), ("ast", "10 - 4 - 3"), ("val", "10 - 4 - 3"),
    ("tok", "2 * 3 + 4"), ("ast", "2 * 3 + 4"), ("val", "2 * 3 + 4"),
    ("tok", "-(2 + 3)"), ("ast", "-(2 + 3)"), ("val", "-(2 + 3)"),
    ("tok", "100 / 7"), ("ast", "100 / 7"), ("val", "100 / 7"),
)


def build_tasks(limit: Optional[int] = None) -> List[Task]:
    """The suite, with every expected answer computed from the reference tree.

    Computed rather than typed out: a hand-written expectation drifts from the
    implementation it is supposed to pin, and a suite that disagrees with its own
    oracle scores a correct candidate wrong. This is the loader, so it is also
    the boundary ``--dry-run`` must not cross.
    """
    cases = list(_CASES)[:limit] if limit else list(_CASES)
    gold = _run_cases(_reference_tree(), cases)
    tasks: List[Task] = []
    for i, ((kind, source), expected) in enumerate(zip(cases, gold)):
        if expected.startswith("ERROR:"):
            raise RuntimeError(
                f"the reference implementation failed case {kind} {source!r} "
                f"({expected}); the suite would score a correct candidate wrong")
        tasks.append(Task(id=f"{kind}{i:02d}", prompt=source,
                          meta={"kind": kind, "gold": expected}))
    return tasks


def make_runner() -> Callable[[str, Task], str]:
    """``run(rendered, task)`` -- one case against one candidate repository."""

    def run(rendered: str, task: Task) -> str:
        try:
            state = dict(parse_tree(rendered))
        except Exception:  # noqa: BLE001 - an empty artifact, before anything exists
            return CRASHED
        # The frozen files are restored from the human's copy rather than taken
        # from the candidate: a proposal can never write them, and a resumed
        # ledger or a hand-built aggregator could still put state in front of
        # this. Scoring against the agents' copy of the suite is the one failure
        # mode that cannot be detected from a completed run.
        for path, content in initial_files().items():
            if path.startswith("spec/"):
                state[path] = content
        return _run_cases(state, [(task.meta["kind"], task.prompt)])[0]

    return run


def reward(task: Task, output: str) -> float:
    """Exact match against the frozen suite's expectation."""
    return 1.0 if output == task.meta.get("gold") else 0.0


# ---------------------------------------------------------------------------
# The offline actors: rule-based, deterministic, and a surrogate (see the header)
# ---------------------------------------------------------------------------

def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    """What each node still owes, in the order its CONTEXT.md implies."""
    owed: Dict[str, List[str]] = {"src": [], "src/frontend": [], "src/backend": []}
    if ENTRY not in state:
        owed["src"].append(ENTRY)
    for node, files in (("src/frontend", ("src/frontend/__init__.py", LEXER, PARSER)),
                        ("src/backend", ("src/backend/__init__.py", EVALUATOR))):
        owed[node] = [f for f in files if f not in state]
    if EVALUATOR in state:
        body = state[EVALUATOR]
        owed["src/backend"] += [f"{EVALUATOR}#{name}"
                                for name, (stub, _) in _STUBS.items() if stub in body]
    return owed


def offline_manager(brief: Brief) -> Sequence[Delegation]:
    """Decompose along the node's routing table, or become its executor.

    The decomposition is **read from ``CONTEXT.md``**, not hardcoded here:
    ``LocalWorld.routing`` parses the routing table of the node the agent is
    standing on, which is what that table is for upstream. What stays a surrogate
    is only the choice of *which* routed child to work on, and that is decided by
    what the node still owes rather than by a fixed list.
    """
    path = normalise(brief.world.path)
    if path == "src" and ENTRY not in brief.state:
        return ()                           # this node's own file: do it here
    return [Delegation(node, f"clear the outstanding work under {node}/")
            for node in brief.world.routing(brief.state)
            if _owes(brief.state, node)]


def _owes(state: Mapping[str, str], node: str) -> bool:
    """Does ``node`` or anything beneath it still have outstanding work?

    Asked of the subtree rather than the node, so the root keeps delegating into
    ``src/`` while the work is two levels further down.
    """
    node = normalise(node)
    return any(items for key, items in _outstanding(state).items()
               if key == node or key.startswith(node + "/"))


def offline_executor(brief: Brief) -> Sequence[Edit]:
    """Write exactly one file, the way a bounded episode does."""
    path, state = normalise(brief.world.path), brief.state
    owed = _outstanding(state)

    if path == "src" and ENTRY not in state:
        return [Edit(owner=path, path=ENTRY, content=_REF_ENTRY)]
    if path == "src/frontend":
        for target, content in ((("src/frontend/__init__.py"), _PKG_INIT),
                                (LEXER, _REF_LEXER), (PARSER, _REF_PARSER)):
            if target not in state:
                return [Edit(owner=path, path=target, content=content)]
        return ()
    if path == "src/backend":
        if "src/backend/__init__.py" not in state:
            return [Edit(owner=path, path="src/backend/__init__.py", content=_PKG_INIT)]
        if EVALUATOR not in state:
            return [Edit(owner=path, path=EVALUATOR, content=_EVALUATOR_SKELETON)]
        name = _stub_for(brief, state)
        if name is None:
            return ()
        stub, filled = _STUBS[name]
        return [Edit(owner=path, path=EVALUATOR,
                     content=state[EVALUATOR].replace(stub, filled))]
    return ()


def _stub_for(brief: Brief, state: Mapping[str, str]) -> Optional[str]:
    """Which node kind this episode's failing case points at.

    The *point* of routing by the failure: two workers holding two different
    failing cases fill two different functions of one file, in the same round.
    Those are two values for one key, so a keyed union drops one of them and a
    three-way merge keeps both -- which is the whole comparison this port exists
    to make runnable.
    """
    body = state.get(EVALUATOR, "")
    open_stubs = [name for name, (stub, _) in _STUBS.items() if stub in body]
    if not open_stubs:
        return None
    source = str(getattr(brief.task, "prompt", ""))
    wants: List[str] = []
    if _has_unary_minus(source):
        wants.append("neg")
    if "*" in source or "/" in source:
        wants.append("muldiv")
    if "+" in source or "-" in source:
        wants.append("addsub")
    for name in wants:
        if name in open_stubs:
            return name
    # Nothing this case needs is still open. Upstream a manager would close the
    # task; here the run would then stall with a stage unbuilt, so the episode
    # takes the next outstanding item instead -- a surrogate's convenience, and
    # the reason the offline arm always completes the toolchain.
    return open_stubs[0]


def _has_unary_minus(source: str) -> bool:
    previous = ""
    for char in source:
        if char == "-" and previous in ("", "(", "+", "-", "*", "/"):
            return True
        if not char.isspace():
            previous = char
    return False


# ---------------------------------------------------------------------------
# The LLM actors: the same two seams, asked rather than computed
# ---------------------------------------------------------------------------

_MANAGER_PROMPT = """You are a manager agent in a recursive software world. You are \
situated at the repository path `{path}` and you own everything under it.

{context}

This node's routing table says its children are: {routes}

OBJECTIVE
{objective}

You do not write code. Decide whether to delegate to more specific paths inside \
your own subtree, or to handle this yourself.

Reply with ONE JSON object and nothing else:
{{"delegations": [{{"path": "<a node inside {path}>", "objective": "<one sentence>"}}]}}

Rules:
- A node is a **directory**, never a file. `src/frontend` is a node; \
`src/frontend/lexer.py` is a file that belongs to the agent situated at \
`src/frontend`, and delegating to it is refused.
- Prefer a child this node already routes to. Naming a new one is allowed and \
adds it to this node's routing table -- do that only when the work genuinely \
belongs to a new part of the tree.
- An empty list means you will handle it at your own path, writing the files \
that belong to `{path}` itself.
- Never name a path outside your subtree -- it belongs to another agent."""

_EXECUTOR_PROMPT = """You are an executor agent in a recursive software world. You are \
situated at the repository path `{path}` and you may write ONLY files under it.

{context}

OBJECTIVE
{objective}

A validation case failed:
  input    {prompt}
  produced {output}
  score    {reward:.2f}

{protocol}"""


def llm_manager(complete) -> Callable[[Brief], Sequence[Delegation]]:
    """Ask a model where to situate children. A bad reply means "handle it here"."""

    def manager(brief: Brief) -> Sequence[Delegation]:
        routes = brief.world.routing(brief.state)
        reply = _ask(complete, _MANAGER_PROMPT.format(
            path=brief.world.path or "./", context=brief.context,
            routes=", ".join(f"`{r}/`" for r in routes) or "(none yet)",
            objective=brief.objective))
        try:
            data = json.loads(_first_object(reply) or "{}")
            items = data.get("delegations") or []
        except Exception:  # noqa: BLE001 - malformed model output, not a bug
            return ()
        out = []
        for item in items:
            if isinstance(item, dict) and item.get("path"):
                out.append(Delegation(normalise(str(item["path"])),
                                      str(item.get("objective", brief.objective))))
        return out

    return manager


def llm_executor(complete, *, editable: Sequence[str] = ("**",),
                 frozen: Sequence[str] = FROZEN) -> Callable[[Brief], Sequence[Edit]]:
    """Ask a model for situated edits. Unparseable replies cost their episode."""
    protocol = SITUATED_EDIT_PROTOCOL.format(
        editable=", ".join(editable) or "(none)", frozen=", ".join(frozen) or "(none)")

    def executor(brief: Brief) -> Sequence[Edit]:
        reply = _ask(complete, _EXECUTOR_PROMPT.format(
            path=brief.world.path or "./", context=brief.context,
            objective=brief.objective,
            prompt=getattr(brief.task, "prompt", ""),
            output=(brief.output or "")[:400], reward=brief.reward,
            protocol=protocol))
        return [Edit(owner=brief.world.path or "", path=edit["path"],
                     content=edit["content"])
                for edit in parse_situated_edits(reply)]

    return executor


def _ask(complete, prompt: str) -> str:
    try:
        return complete(prompt) or ""
    except Exception:  # noqa: BLE001 - a dead backend costs this episode, not the run
        return ""


def _first_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
