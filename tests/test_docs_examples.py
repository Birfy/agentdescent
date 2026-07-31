"""The advertised examples must actually run.

The README Quickstart used undefined names (`tasks`, `reward`), so copy-pasting it
-- the first thing a new user does -- raised NameError immediately, and it also
required API credentials. Docs that do not run are worse than no docs, so extract
the runnable blocks and execute them.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _first_python_block(markdown: str, after: str) -> str:
    """The first ```python fence following the marker text."""
    tail = markdown.split(after, 1)
    assert len(tail) == 2, f"marker not found in the document: {after!r}"
    m = re.search(r"```python\n(.*?)```", tail[1], re.DOTALL)
    assert m, "no python block after the marker"
    return m.group(1)


def test_readme_quickstart_runs_as_written():
    code = _first_python_block(
        (ROOT / "README.md").read_text(),
        "Runnable as-is")
    ns: dict = {}
    exec(compile(code, "README-quickstart", "exec"), ns)   # noqa: S102 - that is the test
    assert ns["result"].final_reward == 1.0
    assert ns["result"].error is None


def test_readme_quickstart_needs_no_credentials():
    """It must not reach for a provider -- a new user has no key yet."""
    code = _first_python_block((ROOT / "README.md").read_text(), "Runnable as-is")
    for forbidden in ("claude(", "openai_compatible(", "OPENAI_API_KEY"):
        assert forbidden not in code, f"the quickstart should not require {forbidden}"


def test_evolution_guide_bring_your_own_agent_block_is_complete():
    """The `evolve` page's headline snippet must at least name every plug-in.

    It cannot be executed (it takes a real model), so this checks the thing that
    actually broke before: a snippet that omits an argument the prose says is
    required. The runnable no-credentials example is covered by the README tests.
    """
    code = _first_python_block(
        (ROOT / "docs" / "evolution.md").read_text(),
        "## Bring an agent you already have")
    for required in ("run=", "propose=", "strategy=", "evolve("):
        assert required in code, f"the quickstart snippet dropped {required}"


def test_quickstart_defines_every_name_it_uses():
    """Guards the original bug: `tasks` and `reward` were never defined."""
    code = _first_python_block((ROOT / "README.md").read_text(), "Runnable as-is")
    for name in ("tasks", "reward", "run", "propose"):
        assert re.search(rf"^\s*(def\s+{name}\b|{name}\s*=)", code, re.M), \
            f"{name} is used but never defined in the quickstart"
