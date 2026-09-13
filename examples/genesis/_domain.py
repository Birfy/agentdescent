"""minilang: an integer expression language, grown from an empty repository.

The first of two formation domains -- :mod:`examples.genesis._stackvm` is the
other, four nodes deep where this is two -- and the shallower one, so the
reasoning they share is written out here and referred to from there. What both
sit on is :mod:`examples.genesis._suite`: the harness, the loader, the frozen-file
restore, the parent's integration check and the LLM actors. A domain is data.

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
The default (no ``--model``) proposes with rule-based actors that reveal
pre-written module implementations one step at a time, driven by ``_PLAN``. It exercises *this port's mechanism* --
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

from typing import Dict, List, Mapping, Optional, Sequence

from ._delegation import Brief, Delegation, Edit
from ._suite import PYTHON_MODULE_SKILL, Suite, plan_delegations, reward
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR, normalise

__all__ = ["CASE_NOUN", "CONTRACTS", "FROZEN", "GROUP_NOUN", "HELD_OUT_FRAC",
           "SCORING", "MINILANG", "build_tasks", "initial_files", "llm_executor",
           "llm_manager", "make_runner", "offline_executor", "offline_manager",
           "reference_tree", "reward", "suite_review"]

#: Human-supplied and never the agents'. L0 in this repository's sense, and the
#: role c-testsuite / LLVM / Csmith play upstream.
FROZEN = ("spec/**",)

#: What is pushed into every brief.
CONTRACTS = FROZEN

#: How the run is scored, and what the header line says about it.
SCORING = "reference oracle, exact match"
CASE_NOUN = "validation cases"
GROUP_NOUN = "stages"
#: `evolve()`'s default. The cases are sampled expressions over one grammar, so the
#: held-out tail is a genuine generalisation estimate -- unlike md, where every task
#: is a requirement and the tail is an audit set instead.
HELD_OUT_FRAC = 0.4

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

## API Surface
Nothing at the root. Every stage is reached through `src/__init__.py`.

## Routing Table
- `./src/` -> the implementation, and the three stage entry points

## Constraints
- `spec/` is read-only. It is the validation contract, not a work item.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = '''# src -- implementation root

## Intent
Own `src/__init__.py`, the public surface named by the spec: `tokenize`, `parse`
and `evaluate`. Each MUST import its stage lazily.

## API Surface
- `src/__init__.py`: `tokenize(source)`, `parse(source)`, `evaluate(source)` -- each
  takes the **source text**, not the previous stage's output.

## Routing Table
- `./src/frontend/` -> tokenizer and parser
- `./src/backend/`  -> evaluation of the parse tree
'''

_FRONTEND_CONTEXT = '''# src/frontend -- source text to parse tree

## Intent
`lexer.py` turns source text into the token list; `parser.py` turns tokens into
the nested-tuple parse tree, honouring precedence and parentheses.

## API Surface
- `lexer.py`: `tokenize(source)`
- `parser.py`: `parse(source)` -- the source text, and it tokenizes internally
'''

_BACKEND_CONTEXT = '''# src/backend -- parse tree to value

## Intent
`evaluator.py` walks the parse tree. One function per node kind, so that two
agents fixing two different node kinds are editing two different parts of the
file rather than the same one.

## API Surface
- `evaluator.py`: `evaluate(node)`, plus one `eval_*(node)` per node kind
'''


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from.

    Context, constraints, one reusable skill -- the things the paper says an
    accepted version carries besides source (3.1) -- and no implementation.
    """
    return {
        "CONTEXT.md": _ROOT_CONTEXT,
        "spec/CONTEXT.md": _SPEC,
        "src/CONTEXT.md": _SRC_CONTEXT,
        "src/frontend/CONTEXT.md": _FRONTEND_CONTEXT,
        "src/backend/CONTEXT.md": _BACKEND_CONTEXT,
        f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL,
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


def _filled_evaluator() -> str:
    """The evaluator with every stub filled -- the oracle's copy."""
    body = _EVALUATOR_SKELETON
    for stub, filled in _STUBS.values():
        body = body.replace(stub, filled)
    return body


def reference_tree() -> Dict[str, str]:
    """The finished repository -- the oracle the frozen suite is computed from."""
    return MINILANG.reference_tree()


# ---------------------------------------------------------------------------
# Running a candidate repository
# ---------------------------------------------------------------------------

#: Executed inside the materialised workspace, in a child process. A candidate
#: is agent-written code: it can loop, raise or exit, and none of those may take
#: the run with it.
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



#: minilang as a :class:`~examples.genesis._suite.Suite`. Everything above is data;
#: the harness, the loader, the runner and the parent's integration check are
#: shared with every other formation domain.
MINILANG = Suite(
    name="minilang",
    given=initial_files(),
    frozen=FROZEN,
    stages={"tok": "tokenize", "ast": "parse", "val": "evaluate"},
    cases=_CASES,
    reference={
        ENTRY: _REF_ENTRY,
        "src/frontend/__init__.py": _PKG_INIT,
        "src/backend/__init__.py": _PKG_INIT,
        LEXER: _REF_LEXER,
        PARSER: _REF_PARSER,
        EVALUATOR: _filled_evaluator(),
    },
)

build_tasks = MINILANG.build_tasks
make_runner = MINILANG.make_runner
suite_review = MINILANG.review


def llm_manager(complete):
    """The shared manager actor. Re-exported so a caller needs one import."""
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    """The shared executor actor, with this domain's frozen paths."""
    return _llm_executor(complete, editable=editable, frozen=frozen)


# ---------------------------------------------------------------------------
# The offline actors: rule-based, deterministic, and a surrogate (see the header)
# ---------------------------------------------------------------------------

#: What each node owes, in the order its CONTEXT.md implies. The plan is the whole
#: surrogate: a real executor decides what to write, this one is told. Same shape
#: as :mod:`examples.genesis._stackvm`'s, so the two domains read alike.
_PLAN = {
    "src": [(ENTRY, _REF_ENTRY)],
    "src/frontend": [("src/frontend/__init__.py", _PKG_INIT),
                     (LEXER, _REF_LEXER), (PARSER, _REF_PARSER)],
    "src/backend": [("src/backend/__init__.py", _PKG_INIT),
                    (EVALUATOR, _EVALUATOR_SKELETON)],
}


def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    owed = {node: [path for path, _ in steps if path not in state]
            for node, steps in _PLAN.items()}
    if EVALUATOR in state:
        body = state[EVALUATOR]
        owed["src/backend"] += [f"{EVALUATOR}#{name}"
                               for name, (stub, _) in _STUBS.items() if stub in body]
    return owed


def _owes(state: Mapping[str, str], node: str) -> bool:
    """Does ``node`` or anything beneath it still have outstanding work?

    Asked of the subtree rather than the node, so the root keeps delegating into
    ``src/`` while the work is two levels further down.
    """
    node = normalise(node)
    return any(items for key, items in _outstanding(state).items()
               if key == node or key.startswith(node + "/"))


def offline_manager(brief: Brief) -> Sequence[Delegation]:
    """Decompose along the node's routing table; accountability writes its own.

    The decomposition is **read from ``CONTEXT.md``**, not hardcoded here:
    ``LocalWorld.routing`` parses the routing table of the node the agent is
    standing on, which is what that table is for upstream. What stays a surrogate
    is only the choice of *which* routed child to work on, and that is decided by
    what the node still owes rather than by a fixed list.

    Cold-started (`--cold-start`) there is no table to read, so it opens the nodes
    its plan needs and the run records them -- see :func:`plan_delegations`.
    """
    return plan_delegations(brief, _owes, list(_PLAN))


def offline_executor(brief: Brief) -> Sequence[Edit]:
    """Write exactly one file, the way a bounded episode does."""
    path, state = normalise(brief.world.path), brief.state
    for target, content in _PLAN.get(path, ()):
        if target not in state:
            return [Edit(owner=path, path=target, content=content)]
    if path == "src/backend" and EVALUATOR in state:
        name = _stub_for(brief, state)
        if name is not None:
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
