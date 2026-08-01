"""`docs/api.md` must agree with the code it documents.

A hand-maintained API page is wrong the moment a signature changes, and the
failure is silent: a reader copies a call that no longer exists. The page is
therefore generated from `agentdescent.__all__`, and this test runs the
generator's `--check` mode, so a signature change that was not regenerated fails
here instead of shipping.

    python -m tools.gen_api_docs        # the fix, when this fails
"""

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
