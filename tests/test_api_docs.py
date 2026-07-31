"""Every public knob must be documented.

`evolve()` grew to 25 parameters with only 9 mentioned in its docstring, and
`async_evolve()` documented 5 of 23 -- including the ones that silently change
cost (`self_verify`) or bound the run (`max_seconds`, `max_iters`). This keeps
the docstrings honest as the signatures change.
"""

import inspect
import re

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import evolve


def _documented_parameters(fn):
    """Names with a real entry under ``Parameters``, not merely a mention.

    This used to be ``p not in doc`` -- a substring match against the whole
    docstring. Delete `async_evolve`'s entire Parameters section and **20 of its
    27 parameters still passed**, because its opening paragraph lists names in
    prose ("`tasks`, `reward`, `agent` ... mean exactly what they do in evolve").
    `evolve` kept 11 of 30 the same way. Substrings made it worse still: `run`
    matches "running", `agent` matches "agents", `parallel` matches "parallelism".

    Both docstrings are numpydoc, so an entry is a line at a fixed indent ending
    in a colon, and one line may name several (``run, propose:``).
    """
    body = _parameters_section(fn)
    documented = set()
    for line in body.splitlines():
        # `getdoc` dedents against the summary line, so the section body keeps its
        # own indent. An entry is a name (or comma-separated names) then a colon.
        m = re.match(r"^\s*(\w[\w, ]*):\s*$", line)
        if m:
            documented.update(n.strip() for n in m.group(1).split(","))
    return documented


def _parameters_section(fn) -> str:
    """Everything after the numpydoc ``Parameters`` underline, or ``""``."""
    doc = inspect.getdoc(fn) or ""
    m = re.search(r"^\s*Parameters\s*\n\s*-{3,}\s*\n", doc, re.M)
    return doc[m.end():] if m else ""


def _undocumented(fn):
    documented = _documented_parameters(fn)
    return [p for p in inspect.signature(fn).parameters if p not in documented]


def test_evolve_documents_every_parameter():
    assert _undocumented(evolve) == []


def test_async_evolve_documents_every_parameter():
    assert _undocumented(async_evolve) == []


def test_the_guard_would_notice_an_undocumented_parameter():
    """Guard the guard: the old substring version could not fail for most names.

    Stripping the Parameters section must make (almost) everything undocumented.
    If it does not, the check is reading prose somewhere else in the docstring.
    """
    for fn in (evolve, async_evolve):
        params = list(inspect.signature(fn).parameters)
        body = _parameters_section(fn)
        stripped = (inspect.getdoc(fn) or "").replace(body, "")
        survivors = [p for p in params if re.search(rf"^\s*{p}:\s*$", stripped, re.M)]
        assert not survivors, (
            f"{fn.__name__}: {survivors} would still look documented with the "
            "Parameters section removed")


def test_public_entry_points_have_docstrings():
    for fn in (evolve, async_evolve):
        assert (fn.__doc__ or "").strip(), f"{fn.__name__} has no docstring"


def test_result_documents_the_error_contract():
    """`error` is the field that distinguishes a died run from a converged one."""
    from agentdescent.evolution import EvolutionResult

    src = inspect.getsource(EvolutionResult)
    assert "error" in src and "clean run" in src


def test_docstring_constructor_examples_use_real_arguments():
    """A copy-pasteable example that raises TypeError is worse than none.

    `SingleSlot`'s docstring advertised `SingleSlot(initial=...)` when the field is
    `initial_value`, and described a `keep_longest` parameter that never existed.
    """
    import dataclasses
    import re

    import agentdescent.evolution as ev

    bad = []
    for name in dir(ev):
        obj = getattr(ev, name)
        if not (dataclasses.is_dataclass(obj) and isinstance(obj, type)):
            continue
        fields = {f.name for f in dataclasses.fields(obj)}
        doc = inspect.getdoc(obj) or ""
        for kwarg in re.findall(rf"\b{name}\(\s*([a-z_][a-z_0-9]*)\s*=", doc):
            if kwarg not in fields:
                bad.append(f"{name}(...) docstring passes '{kwarg}'; fields are "
                           f"{sorted(fields)}")
    assert not bad, "\n".join(bad)
