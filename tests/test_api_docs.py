"""Every public knob must be documented.

`evolve()` grew to 25 parameters with only 9 mentioned in its docstring, and
`async_evolve()` documented 5 of 23 -- including the ones that silently change
cost (`self_verify`) or bound the run (`max_seconds`, `max_iters`). This keeps
the docstrings honest as the signatures change.
"""

import inspect

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import evolve


def _undocumented(fn):
    doc = fn.__doc__ or ""
    return [p for p in inspect.signature(fn).parameters if p not in doc]


def test_evolve_documents_every_parameter():
    assert _undocumented(evolve) == []


def test_async_evolve_documents_every_parameter():
    assert _undocumented(async_evolve) == []


def test_public_entry_points_have_docstrings():
    for fn in (evolve, async_evolve):
        assert (fn.__doc__ or "").strip(), f"{fn.__name__} has no docstring"


def test_result_documents_the_error_contract():
    """`error` is the field that distinguishes a died run from a converged one."""
    from agentdescent.evolution import EvolutionResult

    src = inspect.getsource(EvolutionResult)
    assert "error" in src and "clean run" in src
