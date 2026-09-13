"""The second formation domain: a stack machine, grown four levels deep.

minilang (:mod:`examples.genesis._domain`) is two nodes deep and four files. This
one is **four** nodes deep and ten, with a real instruction set: labels the
assembler has to resolve to indices, a dispatch table that depends on three
sibling modules, and a loop in the suite. The point is not that it is harder code
-- it is that the *recursion* has somewhere to go. ``src/vm/ops`` is a node whose
parent is itself a child, so an episode reaches depth 3 on the way to a leaf, and
``arith.py`` holds one independently-fillable function per opcode so two agents
working from two different failing programs edit two different parts of one file.

What is faithful and what is a surrogate is exactly as it is for minilang, and the
reasoning is written out on :mod:`examples.genesis._domain`: the repository starts implementation-empty, the
validation is staged so the run has a gradient before the toolchain is complete,
the spec is frozen and restored pristine before scoring, and ``--offline`` reveals
pre-written modules one step at a time and says so.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from ._delegation import Brief, Delegation, Edit
from ._suite import PYTHON_MODULE_SKILL, Suite, reward
from ._suite import llm_executor as _llm_executor
from ._suite import llm_manager as _llm_manager
from ._world import SKILLS_DIR, normalise

__all__ = ["CASE_NOUN", "FROZEN", "GROUP_NOUN", "HELD_OUT_FRAC",
           "SCORING", "STACKVM", "build_tasks", "initial_files", "llm_executor",
           "llm_manager", "make_runner", "offline_executor", "offline_manager",
           "reference_tree", "reward", "suite_review"]

FROZEN = ("spec/**",)

SCORING = "reference oracle, exact match"
CASE_NOUN = "validation cases"
GROUP_NOUN = "stages"
HELD_OUT_FRAC = 0.4

ENTRY = "src/__init__.py"
ASM_INIT = "src/asm/__init__.py"
LEXER = "src/asm/lexer.py"
PARSER = "src/asm/parser.py"
VM_INIT = "src/vm/__init__.py"
MACHINE = "src/vm/machine.py"
OPS_INIT = "src/vm/ops/__init__.py"
ARITH = "src/vm/ops/arith.py"
STACK = "src/vm/ops/stack.py"
CONTROL = "src/vm/ops/control.py"


# ---------------------------------------------------------------------------
# The human's repository: the instruction set, the decomposition, the contract
# ---------------------------------------------------------------------------

_SPEC = '''# stackvm -- the machine this project implements

A stack machine and its assembler.

## Instruction set

| instruction | effect |
|---|---|
| `push N` | push the integer N |
| `add` `sub` `mul` `div` | pop b, pop a, push `a OP b`; `div` is floor division |
| `dup` | push a copy of the top |
| `swap` | exchange the top two |
| `drop` | discard the top |
| `emit` | append the top to the output, without popping it |
| `jmp L` | continue at label L |
| `jz L` | pop; if it is 0, continue at label L |
| `halt` | stop |
| `L:` | declare label L at this position |

`#` starts a comment that runs to the end of the line. Blank lines are ignored.

## The three validated stages

| stage | entry point | what a case checks |
|---|---|---|
| `lex` | `src.tokenize(source)` | the token list, as `repr` |
| `asm` | `src.assemble(source)` | the program, as `repr` |
| `run` | `src.run(source)` | the list of emitted values, as `repr` |

A stage is scored independently of the ones after it, so `src/__init__.py` MUST
import each stage lazily, inside the function that needs it.

Tokens are `("label", name)`, `("op", name)` and `("arg", value)`, where an
integer argument is an `int` and a label reference is a `str`. A program is a list
of tuples `(op, *args)` in which **every label reference has been replaced by the
index of the instruction it labels**. Execution stops on `halt` or when the
program counter leaves the program.
'''

_ROOT_CONTEXT = '''# stackvm -- root

## Intent
Implement the machine in `spec/CONTEXT.md`. The specification and its staged
validation are given; no implementation exists yet.

## Routing Table
- `./src/` -> the implementation, and the three stage entry points

## Constraints
- `spec/` is read-only. It is the validation contract, not a work item.
- Every agent edits only files under its own path.
'''

_SRC_CONTEXT = '''# src -- implementation root

## Intent
Own `src/__init__.py`, the public surface named by the spec: `tokenize`,
`assemble` and `run`. Each MUST import its stage lazily.

## Routing Table
- `./src/asm/` -> assembly text to a program
- `./src/vm/`  -> executing a program
'''

_ASM_CONTEXT = '''# src/asm -- assembly text to a program

## Intent
`lexer.py` turns text into the token list. `parser.py` turns tokens into the
program, resolving every label reference to the index of the instruction it
labels -- so the VM never sees a label.
'''

_VM_CONTEXT = '''# src/vm -- executing a program

## Intent
Own `machine.py`: the machine state, the opcode table, and the interpreter loop.
`push` carries its operand so the loop handles it directly; every other opcode is
a function looked up in the table. Bound the loop so a program that never halts
cannot hang the suite.

## Routing Table
- `./src/vm/ops/` -> one module per family of opcodes
'''

_OPS_CONTEXT = '''# src/vm/ops -- the opcodes themselves

## Intent
One module per family, and **one function per opcode**, so that two agents fixing
two different opcodes are editing two different parts of the file rather than the
same one.

- `arith.py`   -> `add` `sub` `mul` `div`
- `stack.py`   -> `dup` `swap` `drop`
- `control.py` -> `jmp` `jz` `emit` `halt`

Every opcode function takes the machine as its first argument and mutates it;
`jmp` and `jz` take the target index as a second argument.
'''


def initial_files() -> Dict[str, str]:
    """The implementation-empty repository the run starts from."""
    return {
        "CONTEXT.md": _ROOT_CONTEXT,
        "spec/CONTEXT.md": _SPEC,
        "src/CONTEXT.md": _SRC_CONTEXT,
        "src/asm/CONTEXT.md": _ASM_CONTEXT,
        "src/vm/CONTEXT.md": _VM_CONTEXT,
        "src/vm/ops/CONTEXT.md": _OPS_CONTEXT,
        f"src/{SKILLS_DIR}/python-modules.md": PYTHON_MODULE_SKILL,
    }


# ---------------------------------------------------------------------------
# The reference implementation: the oracle, and what the offline actors reveal
# ---------------------------------------------------------------------------

_PKG_INIT = '"""Package marker."""\n'

_REF_ENTRY = '''"""Public surface. Each stage is imported lazily -- see spec/CONTEXT.md."""


def tokenize(source):
    from .asm.lexer import tokenize as _tokenize
    return _tokenize(source)


def assemble(source):
    from .asm.parser import assemble as _assemble
    return _assemble(source)


def run(source):
    from .vm.machine import run_program
    return run_program(assemble(source))
'''

_REF_LEXER = '''"""Assembly text -> ("label", name) / ("op", name) / ("arg", value) tokens."""


def tokenize(source):
    tokens = []
    for raw in source.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            tokens.append(("label", line[:-1].strip()))
            continue
        parts = line.split()
        tokens.append(("op", parts[0]))
        for arg in parts[1:]:
            tokens.append(("arg", int(arg) if _integer(arg) else arg))
    return tokens


def _integer(text):
    return text.lstrip("-").isdigit()
'''

_REF_PARSER = '''"""Tokens -> a program: (op, *args) per instruction, labels resolved to indices."""

from .lexer import tokenize


def assemble(source):
    labels, draft = {}, []
    for kind, value in tokenize(source):
        if kind == "label":
            labels[value] = len(draft)
        elif kind == "op":
            draft.append([value])
        else:
            draft[-1].append(value)
    program = []
    for instruction in draft:
        op, args = instruction[0], instruction[1:]
        program.append(tuple([op] + [labels.get(a, a) if isinstance(a, str) else a
                                     for a in args]))
    return program
'''

_REF_MACHINE = '''"""The interpreter loop and the opcode table."""

from .ops import arith, control, stack

MAX_STEPS = 10_000


class Machine:
    def __init__(self):
        self.stack = []
        self.pc = 0
        self.out = []
        self.halted = False


TABLE = {
    "add": arith.add, "sub": arith.sub, "mul": arith.mul, "div": arith.div,
    "dup": stack.dup, "swap": stack.swap, "drop": stack.drop,
    "jmp": control.jmp, "jz": control.jz, "emit": control.emit,
    "halt": control.halt,
}


def run_program(program):
    vm, steps = Machine(), 0
    while not vm.halted and 0 <= vm.pc < len(program):
        op, args = program[vm.pc][0], program[vm.pc][1:]
        vm.pc += 1
        if op == "push":
            vm.stack.append(args[0])
        else:
            TABLE[op](vm, *args)
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError("step limit exceeded")
    return vm.out
'''

_REF_STACK = '''"""Stack-shuffling opcodes."""


def dup(vm):
    vm.stack.append(vm.stack[-1])


def swap(vm):
    vm.stack[-1], vm.stack[-2] = vm.stack[-2], vm.stack[-1]


def drop(vm):
    vm.stack.pop()
'''

_REF_CONTROL = '''"""Control-flow and output opcodes."""


def jmp(vm, target):
    vm.pc = target


def jz(vm, target):
    if vm.stack.pop() == 0:
        vm.pc = target


def emit(vm):
    vm.out.append(vm.stack[-1])


def halt(vm):
    vm.halted = True
'''

#: The shape `src/vm/ops/CONTEXT.md` asks for: one function per opcode, each
#: independently fillable, so `OctopusConflict` has something a keyed union cannot
#: do -- two agents filling two opcodes in one file, in one round.
_ARITH_SKELETON = '''"""Arithmetic opcodes. Each pops two operands and pushes one result."""


def add(vm):
    raise NotImplementedError("add")


def sub(vm):
    raise NotImplementedError("sub")


def mul(vm):
    raise NotImplementedError("mul")


def div(vm):
    raise NotImplementedError("div")
'''

_STUBS = {'add': ('def add(vm):\n    raise NotImplementedError("add")\n', 'def add(vm):\n    b, a = vm.stack.pop(), vm.stack.pop()\n    vm.stack.append(a + b)\n'), 'sub': ('def sub(vm):\n    raise NotImplementedError("sub")\n', 'def sub(vm):\n    b, a = vm.stack.pop(), vm.stack.pop()\n    vm.stack.append(a - b)\n'), 'mul': ('def mul(vm):\n    raise NotImplementedError("mul")\n', 'def mul(vm):\n    b, a = vm.stack.pop(), vm.stack.pop()\n    vm.stack.append(a * b)\n'), 'div': ('def div(vm):\n    raise NotImplementedError("div")\n', 'def div(vm):\n    b, a = vm.stack.pop(), vm.stack.pop()\n    vm.stack.append(a // b)\n')}


def _filled_arith() -> str:
    body = _ARITH_SKELETON
    for stub, filled in _STUBS.values():
        body = body.replace(stub, filled)
    return body


#: The frozen validation suite: (stage, source). Interleaved by stage so the
#: held-out tail `evolve()` cuts is a mix rather than one stage's worth.
_CASES = (
    ("lex", 'push 2\npush 3\nadd\nemit\nhalt'), ("asm", 'push 2\npush 3\nadd\nemit\nhalt'), ("run", 'push 2\npush 3\nadd\nemit\nhalt'),
    ("lex", 'push 10\npush 4\nsub\nemit\nhalt'), ("asm", 'push 10\npush 4\nsub\nemit\nhalt'), ("run", 'push 10\npush 4\nsub\nemit\nhalt'),
    ("lex", 'push 6\npush 7\nmul\nemit\nhalt'), ("asm", 'push 6\npush 7\nmul\nemit\nhalt'), ("run", 'push 6\npush 7\nmul\nemit\nhalt'),
    ("lex", 'push 100\npush 7\ndiv\nemit\nhalt'), ("asm", 'push 100\npush 7\ndiv\nemit\nhalt'), ("run", 'push 100\npush 7\ndiv\nemit\nhalt'),
    ("lex", 'push 5\ndup\nadd\nemit\nhalt'), ("asm", 'push 5\ndup\nadd\nemit\nhalt'), ("run", 'push 5\ndup\nadd\nemit\nhalt'),
    ("lex", 'push 1\npush 2\nswap\nemit\nhalt'), ("asm", 'push 1\npush 2\nswap\nemit\nhalt'), ("run", 'push 1\npush 2\nswap\nemit\nhalt'),
    ("lex", 'push 1\npush 2\ndrop\nemit\nhalt'), ("asm", 'push 1\npush 2\ndrop\nemit\nhalt'), ("run", 'push 1\npush 2\ndrop\nemit\nhalt'),
    ("lex", 'push 0\njz skip\npush 99\nemit\nskip:\npush 7\nemit\nhalt'), ("asm", 'push 0\njz skip\npush 99\nemit\nskip:\npush 7\nemit\nhalt'), ("run", 'push 0\njz skip\npush 99\nemit\nskip:\npush 7\nemit\nhalt'),
    ("lex", 'push 3\nloop:\ndup\nemit\npush 1\nsub\ndup\njz done\njmp loop\ndone:\nhalt'), ("asm", 'push 3\nloop:\ndup\nemit\npush 1\nsub\ndup\njz done\njmp loop\ndone:\nhalt'), ("run", 'push 3\nloop:\ndup\nemit\npush 1\nsub\ndup\njz done\njmp loop\ndone:\nhalt'),
    ("lex", '# add two numbers\npush 8\n\npush 9   # second\nadd\nemit\nhalt'), ("asm", '# add two numbers\npush 8\n\npush 9   # second\nadd\nemit\nhalt'), ("run", '# add two numbers\npush 8\n\npush 9   # second\nadd\nemit\nhalt'),
)

STACKVM = Suite(
    name="stackvm",
    given=initial_files(),
    frozen=FROZEN,
    stages={"lex": "tokenize", "asm": "assemble", "run": "run"},
    cases=_CASES,
    reference={
        ENTRY: _REF_ENTRY,
        ASM_INIT: _PKG_INIT, LEXER: _REF_LEXER, PARSER: _REF_PARSER,
        VM_INIT: _PKG_INIT, MACHINE: _REF_MACHINE,
        OPS_INIT: _PKG_INIT, ARITH: _filled_arith(),
        STACK: _REF_STACK, CONTROL: _REF_CONTROL,
    },
)

build_tasks = STACKVM.build_tasks
make_runner = STACKVM.make_runner
suite_review = STACKVM.review


def llm_manager(complete):
    return _llm_manager(complete)


def llm_executor(complete, *, editable=("**",), frozen=FROZEN):
    return _llm_executor(complete, editable=editable, frozen=frozen)


def reference_tree() -> Dict[str, str]:
    return STACKVM.reference_tree()


# ---------------------------------------------------------------------------
# The offline actors: rule-based, deterministic, and a surrogate
# ---------------------------------------------------------------------------

#: What each node owes, in the order its CONTEXT.md implies. The plan is the whole
#: surrogate: a real executor decides what to write, this one is told.
_PLAN = {
    "src": [(ENTRY, _REF_ENTRY)],
    "src/asm": [(ASM_INIT, _PKG_INIT), (LEXER, _REF_LEXER), (PARSER, _REF_PARSER)],
    "src/vm": [(VM_INIT, _PKG_INIT), (MACHINE, _REF_MACHINE)],
    "src/vm/ops": [(OPS_INIT, _PKG_INIT), (ARITH, _ARITH_SKELETON),
                   (STACK, _REF_STACK), (CONTROL, _REF_CONTROL)],
}


def _outstanding(state: Mapping[str, str]) -> Dict[str, List[str]]:
    owed = {node: [path for path, _ in steps if path not in state]
             for node, steps in _PLAN.items()}
    if ARITH in state:
        body = state[ARITH]
        owed["src/vm/ops"] += [f"{ARITH}#{name}"
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
    if path == "src/vm/ops" and ARITH in state:
        name = _stub_for(brief, state)
        if name is not None:
            stub, filled = _STUBS[name]
            return [Edit(owner=path, path=ARITH,
                         content=state[ARITH].replace(stub, filled))]
    return ()


def _stub_for(brief: Brief, state: Mapping[str, str]) -> Optional[str]:
    """Which opcode this episode's failing program points at.

    The point of routing by the failure: two workers holding two different failing
    programs fill two different functions of one file, which is two values for one
    key -- a keyed union drops one and a three-way merge keeps both.
    """
    body = state.get(ARITH, "")
    open_stubs = [n for n, (stub, _) in _STUBS.items() if stub in body]
    if not open_stubs:
        return None
    source = str(getattr(brief.task, "prompt", ""))
    for name in open_stubs:
        if name in source.split():
            return name
    return open_stubs[0]
