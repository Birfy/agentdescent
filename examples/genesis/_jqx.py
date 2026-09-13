"""jqx: a JSON query tool, grown from an empty repository.

The third formation domain, and the one meant to come out as **software someone
can use**: a filter language with a lexer, a parser, an evaluator, a builtin
table and an output layer, behind a frozen command-line entry point the agents
never touch. Run it when it is done::

    python jqx.py '.users | map(.age) | add' people.json

Why a third domain. minilang showed the mechanism, stackvm gave the recursion
somewhere to go, and neither produces anything you would keep. This one does, and
it is also the harder test: four validated stages rather than three, nine files
across four nodes, and a builtin table whose five functions are five independent
edits to one file -- the case a keyed union cannot fuse.

What is faithful and what is a surrogate is as it is for the other two, and the
reasoning is written out on :mod:`examples.genesis._domain`. One thing is new:
``jqx.py`` is **frozen** alongside the specification. The agents grow a library;
the human supplies the shell around it, so a finished run is a program rather than
a package nobody can invoke.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from ._delegation import Brief, Delegation, Edit
from ._suite import PYTHON_MODULE_SKILL, Suite, reward
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR, normalise

__all__ = ["FROZEN", "JQX", "build_tasks", "initial_files", "llm_executor",
           "llm_manager", "make_runner", "offline_executor", "offline_manager",
           "reward", "suite_review"]

#: The specification and the command-line shell. Human-supplied, refused to every
#: proposal, and restored pristine before scoring.
FROZEN = ("spec/**", "jqx.py")

ENTRY = "src/__init__.py"
LANG_INIT = "src/lang/__init__.py"
LEXER = "src/lang/lexer.py"
PARSER = "src/lang/parser.py"
EVAL_INIT = "src/eval/__init__.py"
BUILTINS = "src/eval/builtins.py"
ENGINE = "src/eval/engine.py"
UI_INIT = "src/ui/__init__.py"
FORMAT = "src/ui/format.py"


# ---------------------------------------------------------------------------
# The human's repository: the language, the decomposition, the entry point
# ---------------------------------------------------------------------------

_SPEC = r'''# jqx -- the filter language this project implements

A filter maps one JSON value to a **list** of JSON values.

## Grammar

    filter   := pipeline
    pipeline := term ("|" term)*            left-associative
    term     := path | NAME | NAME "(" pipeline ")"
    path     := "." step*
    step     := NAME | "[" INT "]" | "[" "]"

`.` on its own is the identity filter. `.a.b` is sugar for `.a | .b`, and the
parser MUST desugar it: the AST never contains a multi-step path.

## AST

| node | means |
|---|---|
| `("identity",)` | the input, unchanged |
| `("field", name)` | the value at `name`, or `null` when the input is not an object or has no such key |
| `("index", n)` | element `n` of an array (negative counts from the end), or `null` when out of range or not an array |
| `("iterate",)` | every element of an array, or every value of an object **in sorted-key order**; an empty list for anything else |
| `("pipe", left, right)` | every result of `right` applied to every result of `left` |
| `("call", name)` | a builtin with no argument |
| `("call1", name, filter)` | a builtin with one filter argument; only `map` exists |

## Builtins

| builtin | on | result |
|---|---|---|
| `length` | array, object, string | its length; `null` -> `0`; anything else raises |
| `keys` | object, array | sorted keys; for an array, `[0, 1, ...]` |
| `values` | object, array | values in sorted-key order; for an array, itself |
| `add` | array | the elements combined left to right with `+`; `[]` -> `null` |
| `type` | anything | `"null"` `"boolean"` `"number"` `"string"` `"array"` `"object"` |
| `map(f)` | array | one result: the concatenation of `f` applied to each element |

## The four validated stages

| stage | entry point | the argument | what a case checks |
|---|---|---|---|
| `lex` | `src.tokenize(filter)` | the filter text | the token list, as `repr` |
| `ast` | `src.parse(filter)` | the filter text | the AST, as `repr` |
| `val` | `src.query(payload)` | `{"filter": ..., "input": ...}` as JSON text | the result list, as `repr` |
| `out` | `src.render(payload)` | the same payload | the text: one result per line, `json.dumps` with `sort_keys=True` and `(",", ":")` separators |

Tokens are `("dot",)` `("name", str)` `("int", int)` `("lbracket",)`
`("rbracket",)` `("lparen",)` `("rparen",)` `("pipe",)`.

A stage is scored independently of the ones after it, so `src/__init__.py` MUST
import each stage lazily, inside the function that needs it.
'''

_ROOT_CONTEXT = r'''# jqx -- root

## Intent
Implement the filter language in `spec/CONTEXT.md` as a library under `src/`.
`jqx.py` is the command-line entry point and is already written.

## Routing Table
- `./src/` -> the library, and the four stage entry points

## Constraints
- `spec/` and `jqx.py` are read-only. They are the contract, not work items.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = r'''# src -- library root

## Intent
Own `src/__init__.py`, the public surface named by the spec: `tokenize`, `parse`,
`query` and `render`. Each MUST import its stage lazily.

## Routing Table
- `./src/lang/` -> filter text to an AST
- `./src/eval/` -> an AST applied to a value
- `./src/ui/`   -> results to text
'''

_LANG_CONTEXT = r'''# src/lang -- filter text to an AST

## Intent
`lexer.py` turns filter text into the token list. `parser.py` turns tokens into
the AST, desugaring `.a.b` into a pipe so the evaluator never sees a multi-step
path, and rejecting a filter with tokens left over.
'''

_EVAL_CONTEXT = r'''# src/eval -- an AST applied to a value

## Intent
`engine.py` walks the AST; every node returns a **list** of results, so a pipe is
a flat-map and nothing else needs to know about streams. `builtins.py` holds the
builtins, **one function per builtin**, plus the table that maps a name to one --
so that two agents fixing two different builtins are editing two different parts
of the file rather than the same one.
'''

_UI_CONTEXT = r'''# src/ui -- results to text

## Intent
`format.py` renders a result list the way the spec's `out` stage describes: one
JSON value per line, keys sorted, compact separators, a trailing newline on every
line including the last.
'''

_ENTRY_SCRIPT = r'''#!/usr/bin/env python3
"""jqx -- query JSON from the command line.

Human-supplied, frozen, and never written by an agent: the library under ``src/``
is what the run grows, and this is the shell around it so the result is a program
rather than a package nobody can invoke.

    python jqx.py '.users[].name' people.json
    cat people.json | python jqx.py '.users | map(.age) | add'
    python jqx.py --help
"""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="jqx", description="Query JSON with a small filter language.",
        epilog="See spec/CONTEXT.md for the filter language.")
    parser.add_argument("filter", help="the filter, e.g. '.users[].name'")
    parser.add_argument("file", nargs="?",
                        help="JSON input file (default: standard input)")
    parser.add_argument("-i", "--indent", type=int, default=None, metavar="N",
                        help="pretty-print each result with N spaces of indent")
    args = parser.parse_args(argv)

    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    try:
        document = json.loads(text)
    except ValueError as exc:
        print(f"jqx: input is not JSON: {exc}", file=sys.stderr)
        return 2

    from src import query
    payload = json.dumps({"filter": args.filter, "input": document})
    try:
        results = query(payload)
    except Exception as exc:                # a filter error is the user's, not a crash
        print(f"jqx: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for value in results:
        print(json.dumps(value, sort_keys=True, indent=args.indent,
                         separators=None if args.indent else (",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from."""
    return {
        "CONTEXT.md": _ROOT_CONTEXT,
        "spec/CONTEXT.md": _SPEC,
        "jqx.py": _ENTRY_SCRIPT,
        "src/CONTEXT.md": _SRC_CONTEXT,
        "src/lang/CONTEXT.md": _LANG_CONTEXT,
        "src/eval/CONTEXT.md": _EVAL_CONTEXT,
        "src/ui/CONTEXT.md": _UI_CONTEXT,
        f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL,
    }


# ---------------------------------------------------------------------------
# The reference implementation: the oracle, and what the offline actors reveal
# ---------------------------------------------------------------------------

_PKG_INIT = '"""Package marker."""\n'

_REF_ENTRY = r'''"""Public surface. Each stage is imported lazily -- see spec/CONTEXT.md."""


def tokenize(source):
    from .lang.lexer import tokenize as _tokenize
    return _tokenize(source)


def parse(source):
    from .lang.parser import parse as _parse
    return _parse(source)


def query(payload):
    from .eval.engine import query as _query
    return _query(payload)


def render(payload):
    from .ui.format import render as _render
    return _render(payload)
'''

_REF_LEXER = r'''"""Filter text -> tokens."""

_PUNCT = {".": "dot", "[": "lbracket", "]": "rbracket",
          "(": "lparen", ")": "rparen", "|": "pipe"}


def tokenize(source):
    tokens, i = [], 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch in _PUNCT:
            tokens.append((_PUNCT[ch],))
            i += 1
            continue
        if ch == "-" or ch.isdigit():
            j = i + 1
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(("int", int(source[i:j])))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == "_"):
                j += 1
            tokens.append(("name", source[i:j]))
            i = j
            continue
        raise ValueError("unexpected character: " + ch)
    return tokens
'''

_REF_PARSER = r'''"""Tokens -> an AST. `.a.b` desugars to a pipe, so the evaluator stays small."""

from .lexer import tokenize


def parse(source):
    tokens = tokenize(source)
    node, pos = _pipeline(tokens, 0)
    if pos != len(tokens):
        raise ValueError("trailing tokens")
    return node


def _pipeline(tokens, pos):
    node, pos = _term(tokens, pos)
    while pos < len(tokens) and tokens[pos][0] == "pipe":
        right, pos = _term(tokens, pos + 1)
        node = ("pipe", node, right)
    return node, pos


def _term(tokens, pos):
    if pos >= len(tokens):
        raise ValueError("unexpected end of filter")
    kind = tokens[pos][0]
    if kind == "name":
        name = tokens[pos][1]
        pos += 1
        if pos < len(tokens) and tokens[pos][0] == "lparen":
            inner, pos = _pipeline(tokens, pos + 1)
            if pos >= len(tokens) or tokens[pos][0] != "rparen":
                raise ValueError("expected )")
            return ("call1", name, inner), pos + 1
        return ("call", name), pos
    if kind != "dot":
        raise ValueError("expected a path or a builtin")
    steps, pos = [], pos + 1
    while pos < len(tokens):
        kind = tokens[pos][0]
        if kind == "name":
            steps.append(("field", tokens[pos][1]))
            pos += 1
        elif kind == "lbracket":
            if pos + 1 < len(tokens) and tokens[pos + 1][0] == "rbracket":
                steps.append(("iterate",))
                pos += 2
            elif (pos + 2 < len(tokens) and tokens[pos + 1][0] == "int"
                  and tokens[pos + 2][0] == "rbracket"):
                steps.append(("index", tokens[pos + 1][1]))
                pos += 3
            else:
                raise ValueError("expected [] or [<int>]")
        elif kind == "dot":
            pos += 1
        else:
            break
    if not steps:
        return ("identity",), pos
    node = steps[0]
    for step in steps[1:]:
        node = ("pipe", node, step)
    return node, pos
'''

_REF_ENGINE = r'''"""AST -> a stream of results. A filter maps one value to a list of values."""

import json

from .builtins import TABLE


def query(payload):
    spec = json.loads(payload)
    from ..lang.parser import parse
    return apply_filter(parse(spec["filter"]), spec["input"])


def apply_filter(node, value):
    kind = node[0]
    if kind == "identity":
        return [value]
    if kind == "field":
        return [value.get(node[1]) if isinstance(value, dict) else None]
    if kind == "index":
        if isinstance(value, list) and -len(value) <= node[1] < len(value):
            return [value[node[1]]]
        return [None]
    if kind == "iterate":
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return [value[k] for k in sorted(value)]
        return []
    if kind == "pipe":
        out = []
        for item in apply_filter(node[1], value):
            out.extend(apply_filter(node[2], item))
        return out
    if kind == "call":
        return [TABLE[node[1]](value)]
    if kind == "call1":
        if node[1] != "map":
            raise ValueError("unknown builtin: " + node[1])
        if not isinstance(value, list):
            raise ValueError("map: not an array")
        mapped = []
        for item in value:
            mapped.extend(apply_filter(node[2], item))
        return [mapped]
    raise ValueError("unknown node: " + kind)
'''

_REF_FORMAT = r'''"""Results -> text. One JSON value per line, keys sorted, compact separators."""

import json


def render(payload):
    from ..eval.engine import query
    lines = [json.dumps(value, sort_keys=True, separators=(",", ":"))
             for value in query(payload)]
    return "".join(line + "\n" for line in lines)
'''

#: The shape `src/eval/CONTEXT.md` asks for: one function per builtin, five of
#: them, each independently fillable. Five edits to one file is what makes the
#: three-way merge do something a keyed union cannot.
_BUILTINS_SKELETON = r'''"""One function per builtin. Each takes a value and returns a value."""


def length(value):
    raise NotImplementedError("length")


def keys(value):
    raise NotImplementedError("keys")


def values(value):
    raise NotImplementedError("values")


def add(value):
    raise NotImplementedError("add")


def type_of(value):
    raise NotImplementedError("type_of")


TABLE = {"length": length, "keys": keys, "values": values, "add": add,
         "type": type_of}
'''

_STUBS = {'length': ('def length(value):\n    raise NotImplementedError("length")\n', 'def length(value):\n    if value is None:\n        return 0\n    if isinstance(value, (list, dict, str)):\n        return len(value)\n    raise ValueError("length: not a container")\n'), 'keys': ('def keys(value):\n    raise NotImplementedError("keys")\n', 'def keys(value):\n    if isinstance(value, dict):\n        return sorted(value)\n    if isinstance(value, list):\n        return list(range(len(value)))\n    raise ValueError("keys: not an object or array")\n'), 'values': ('def values(value):\n    raise NotImplementedError("values")\n', 'def values(value):\n    if isinstance(value, dict):\n        return [value[k] for k in sorted(value)]\n    if isinstance(value, list):\n        return list(value)\n    raise ValueError("values: not an object or array")\n'), 'add': ('def add(value):\n    raise NotImplementedError("add")\n', 'def add(value):\n    if not isinstance(value, list):\n        raise ValueError("add: not an array")\n    if not value:\n        return None\n    total = value[0]\n    for item in value[1:]:\n        total = total + item\n    return total\n'), 'type_of': ('def type_of(value):\n    raise NotImplementedError("type_of")\n', 'def type_of(value):\n    if value is None:\n        return "null"\n    if isinstance(value, bool):\n        return "boolean"\n    if isinstance(value, (int, float)):\n        return "number"\n    if isinstance(value, str):\n        return "string"\n    if isinstance(value, list):\n        return "array"\n    return "object"\n')}


def _filled_builtins() -> str:
    body = _BUILTINS_SKELETON
    for stub, filled in _STUBS.values():
        body = body.replace(stub, filled)
    return body


#: The frozen validation suite: (stage, source). `lex` and `ast` take the filter
#: text; `val` and `out` take the payload. Interleaved by filter so the held-out
#: tail `evolve()` cuts is a mix of stages rather than one stage's worth.
_CASES = (
    ("lex", 'keys'), ("ast", 'keys'),
    ("val", '{"filter": "keys", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": "keys", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[0] | keys'), ("ast", '.users[0] | keys'),
    ("val", '{"filter": ".users[0] | keys", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[0] | keys", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[0] | values'), ("ast", '.users[0] | values'),
    ("val", '{"filter": ".users[0] | values", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[0] | values", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users | length'), ("ast", '.users | length'),
    ("val", '{"filter": ".users | length", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users | length", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.n | type'), ("ast", '.n | type'),
    ("val", '{"filter": ".n | type", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".n | type", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", 'add'), ("ast", 'add'),
    ("val", '{"filter": "add", "input": [3, 1, 2]}'),
    ("out", '{"filter": "add", "input": [3, 1, 2]}'),
    ("lex", '.users | map(.age)'), ("ast", '.users | map(.age)'),
    ("val", '{"filter": ".users | map(.age)", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users | map(.age)", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users | map(.age) | add'), ("ast", '.users | map(.age) | add'),
    ("val", '{"filter": ".users | map(.age) | add", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users | map(.age) | add", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", 'length'), ("ast", 'length'),
    ("val", '{"filter": "length", "input": [3, 1, 2]}'),
    ("out", '{"filter": "length", "input": [3, 1, 2]}'),
    ("lex", '.'), ("ast", '.'),
    ("val", '{"filter": ".", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.n'), ("ast", '.n'),
    ("val", '{"filter": ".n", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".n", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[0].name'), ("ast", '.users[0].name'),
    ("val", '{"filter": ".users[0].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[0].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[].name'), ("ast", '.users[].name'),
    ("val", '{"filter": ".users[].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[0].tags | length'), ("ast", '.users[0].tags | length'),
    ("val", '{"filter": ".users[0].tags | length", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[0].tags | length", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.missing'), ("ast", '.missing'),
    ("val", '{"filter": ".missing", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".missing", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.users[-1].name'), ("ast", '.users[-1].name'),
    ("val", '{"filter": ".users[-1].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[-1].name", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("lex", '.[]'), ("ast", '.[]'),
    ("val", '{"filter": ".[]", "input": [3, 1, 2]}'),
    ("out", '{"filter": ".[]", "input": [3, 1, 2]}'),
    ("lex", '.users[0].tags[0]'), ("ast", '.users[0].tags[0]'),
    ("val", '{"filter": ".users[0].tags[0]", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
    ("out", '{"filter": ".users[0].tags[0]", "input": {"n": 2, "users": [{"age": 36, "name": "ada", "tags": ["x", "y"]}, {"age": 41, "name": "bob", "tags": []}]}}'),
)

JQX = Suite(
    name="jqx",
    given=initial_files(),
    frozen=FROZEN,
    stages={"lex": "tokenize", "ast": "parse", "val": "query", "out": "render"},
    cases=_CASES,
    reference={
        ENTRY: _REF_ENTRY,
        LANG_INIT: _PKG_INIT, LEXER: _REF_LEXER, PARSER: _REF_PARSER,
        EVAL_INIT: _PKG_INIT, BUILTINS: _filled_builtins(), ENGINE: _REF_ENGINE,
        UI_INIT: _PKG_INIT, FORMAT: _REF_FORMAT,
    },
)

build_tasks = JQX.build_tasks
make_runner = JQX.make_runner
suite_review = JQX.review


def llm_manager(complete):
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen)


def _reference_tree() -> Dict[str, str]:
    return JQX.reference_tree()


# ---------------------------------------------------------------------------
# The offline actors: rule-based, deterministic, and a surrogate
# ---------------------------------------------------------------------------

_PLAN = {
    "src": [(ENTRY, _REF_ENTRY)],
    "src/lang": [(LANG_INIT, _PKG_INIT), (LEXER, _REF_LEXER), (PARSER, _REF_PARSER)],
    "src/eval": [(EVAL_INIT, _PKG_INIT), (BUILTINS, _BUILTINS_SKELETON),
                 (ENGINE, _REF_ENGINE)],
    "src/ui": [(UI_INIT, _PKG_INIT), (FORMAT, _REF_FORMAT)],
}


def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    owed = {node: [path for path, _ in steps if path not in state]
             for node, steps in _PLAN.items()}
    if BUILTINS in state:
        body = state[BUILTINS]
        owed["src/eval"] += [f"{BUILTINS}#{name}"
                             for name, (stub, _) in _STUBS.items() if stub in body]
    return owed


def _owes(state: Mapping[str, str], node: str) -> bool:
    node = normalise(node)
    return any(items for key, items in _outstanding(state).items()
               if key == node or key.startswith(node + "/"))


def offline_manager(brief: Brief) -> Sequence[Delegation]:
    """Decompose along the node's routing table; accountability writes its own."""
    return [Delegation(node, f"clear the outstanding work under {node}/")
            for node in brief.world.routing(brief.state)
            if _owes(brief.state, node)]


def offline_executor(brief: Brief) -> Sequence[Edit]:
    """Write exactly one file, the way a bounded episode does."""
    path, state = normalise(brief.world.path), brief.state
    for target, content in _PLAN.get(path, ()):
        if target not in state:
            return [Edit(owner=path, path=target, content=content)]
    if path == "src/eval" and BUILTINS in state:
        name = _stub_for(brief, state)
        if name is not None:
            stub, filled = _STUBS[name]
            return [Edit(owner=path, path=BUILTINS,
                         content=state[BUILTINS].replace(stub, filled))]
    return ()


def _stub_for(brief: Brief, state: Mapping[str, str]) -> Optional[str]:
    """Which builtin this episode's failing filter points at.

    Two workers holding two different failing filters fill two different functions
    of one file -- two values for one key, which a keyed union drops one of and a
    three-way merge keeps both.
    """
    body = state.get(BUILTINS, "")
    open_stubs = [n for n, (stub, _) in _STUBS.items() if stub in body]
    if not open_stubs:
        return None
    source = str(getattr(brief.task, "prompt", ""))
    for name in open_stubs:
        if (name.replace("_of", "") in source):
            return name
    return open_stubs[0]
