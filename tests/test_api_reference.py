"""`docs/api.md` must agree with the code it documents.

A hand-maintained API page is wrong the moment a signature changes, and the
failure is silent: a reader copies a call that no longer exists. The page is
therefore generated from `agentdescent.__all__`, and this test runs the
generator's `--check` mode, so a signature change that was not regenerated fails
here instead of shipping.

    python -m tools.gen_api_docs        # the fix, when this fails
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import agentdescent

ROOT = Path(__file__).resolve().parent.parent
API = ROOT / "docs" / "api.md"


def test_the_api_reference_is_in_sync_with_the_package():
    proc = subprocess.run([sys.executable, "-m", "tools.gen_api_docs", "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, (
        "docs/api.md is out of date -- run `python -m tools.gen_api_docs`.\n"
        + proc.stderr)


def test_every_exported_name_appears():
    """The point of generating it: nothing exported can be missing."""
    text = API.read_text(encoding="utf-8")
    missing = [n for n in agentdescent.__all__ if f"`{n}" not in text]
    assert not missing, f"exported but undocumented: {missing}"


def test_the_generator_reads_the_same_class_on_every_interpreter():
    """A generated file committed on one Python must check clean on another.

    `_methods` tested `callable(member)` before unwrapping `staticmethod`, and a
    `staticmethod` object only became callable in 3.10 (bpo-43682). So the page
    grew a row when regenerated on 3.11 and lost it again on 3.9, and CI
    contradicted itself across the matrix on one unchanged tree -- 3.9 green,
    3.11 and 3.12 red. Six members were missing from the reference on the
    version that wrote it.

    `--check` cannot catch this: it only ever sees the interpreter running it.
    """
    from tools.gen_api_docs import _methods

    class _Probe:
        @staticmethod
        def stat(a: int) -> str:
            """A documented staticmethod."""
            return ""

        @classmethod
        def clsm(cls, a: int) -> str:
            """A documented classmethod."""
            return ""

        def inst(self, a: int) -> str:
            """A documented instance method."""
            return ""

    got = {sig for sig, _ in _methods(_Probe)}
    assert got == {"stat(a: int) -> str", "clsm(a: int) -> str", "inst(a: int) -> str"}, got


def _section(text: str, heading: str) -> str:
    """One `### ` section of the page, up to the next heading of any level."""
    start = text.index(heading)
    rest = text[start + len(heading):]
    end = re.search(r"\n#{2,3} ", rest)
    return rest[: end.start()] if end else rest


@pytest.mark.parametrize("name", ["evolve", "async_evolve", "evolve_skill"])
def test_every_parameter_of_an_entry_point_reaches_the_page(name):
    """A heading that reads `evolve(...)` documents no parameters at all.

    67 signatures were too long for their heading and collapsed to `(...)`,
    including every entry point this library has. The parameters were in the
    code and the prose describing them was in the docstrings; neither reached a
    reader of the reference.
    """
    import inspect

    section = _section(API.read_text(encoding="utf-8"), f"### `{name}(...)`")
    missing = [p for p in inspect.signature(getattr(agentdescent, name)).parameters
               if f"`{p}`" not in section and f"`**{p}`" not in section]
    assert not missing, f"{name} parameters absent from docs/api.md: {missing}"


def test_the_keyword_only_marker_is_emitted_once_per_signature():
    """`*` used to be re-emitted before every keyword-only parameter.

    The guard read `"*" not in parts[-1:]`, which is true again as soon as one
    keyword-only parameter has been appended, so `evolve` rendered thirty-odd
    of them. It was invisible while every long signature collapsed to `(...)`.
    """
    section = _section(API.read_text(encoding="utf-8"), "### `evolve(...)`")
    block = section.split("```python")[1].split("```")[0]
    assert block.count("\n    *,\n") == 1, block


def test_a_table_of_blank_prose_is_not_printed():
    """A dataclass of counters documents none of its fields; a table of empty
    cells repeats the signature above it and says nothing."""
    section = _section(API.read_text(encoding="utf-8"), "### `FusionStats(...)`")
    assert "| parameter |" not in section


def test_no_sphinx_roles_leak_into_the_page():
    """Docstrings use `:class:` cross-references; a Markdown page must not."""
    text = API.read_text(encoding="utf-8")
    for role in (":class:`", ":func:`", ":meth:`", ":mod:`", ":data:`"):
        assert role not in text, f"unrendered Sphinx role {role} in docs/api.md"


@pytest.mark.parametrize("page", [
    "evolution.md", "quickstart-skill.md", "directory-evolution.md", "agents.md",
    "backends.md", "data-model.md", "aggregator.md", "ledger.md", "verifier.md",
    "governance.md", "staleness.md", "parallelism.md", "sampling.md",
    "duration-scheduling.md", "dataloader.md", "rewards.md", "async.md",
    "orchestrator.md",
])
def test_every_module_section_links_to_a_real_guide(page):
    """Each API section points at the page explaining *why*; those must exist."""
    assert (ROOT / "docs" / page).exists(), f"api.md links to a missing page: {page}"
