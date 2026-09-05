"""What `import agentdescent` gives you, and whether the docs agree with it.

`__init__.py` curates a large `__all__`, and the project's own documentation
almost never used it -- 63 submodule imports against 3 top-level ones, with
`evolve` never once shown as `from agentdescent import evolve`. So the top-level
surface was untested by usage, and it showed: a documented helper missing from it
(`tasks_from`, which the quickstart tells you to import from
`agentdescent.evolution`), the whole `ContractError` hierarchy unreachable
although `evolve`'s docstring tells callers to distinguish it from a backend
failure, and two classes exported that cannot be constructed from the public API.
"""

import pathlib
import re

import pytest

import agentdescent

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_every_exported_name_resolves():
    """The cheapest possible guard, and it did not exist."""
    missing = [n for n in agentdescent.__all__ if not hasattr(agentdescent, n)]
    assert not missing, f"__all__ names nothing can import: {missing}"


def test_all_is_sorted_free_of_duplicates():
    dupes = {n for n in agentdescent.__all__ if agentdescent.__all__.count(n) > 1}
    assert not dupes, f"duplicated in __all__: {sorted(dupes)}"


@pytest.mark.parametrize("name", [
    "evolve", "async_evolve", "scorer", "Task", "tasks_from",
    "AppendRules", "KeyedRules", "SingleSlot", "EvolutionResult",
])
def test_the_things_a_first_run_needs_are_top_level(name):
    """`tasks_from` was documented as `from agentdescent.evolution import ...`.

    Everything else the quickstart uses was already top-level, so importing the
    one helper from a submodule made the module layout look like public API.
    """
    assert hasattr(agentdescent, name)


@pytest.mark.parametrize("name", [
    "ContractError", "RewardContractError", "ProposalContractError",
    "AggregatorContractError", "GovernanceError", "GitError", "CASConflict",
])
def test_the_error_hierarchy_is_importable(name):
    """`evolve()` tells callers to tell a *caller* bug from a *backend* failure.

    `ContractError` is the base of that hierarchy and was not reachable from the
    package that documents it, so `except agentdescent.ContractError:` did not work.
    """
    assert hasattr(agentdescent, name)


@pytest.mark.parametrize("name", ["diffs_contradict", "fuse_diffs", "stable_hash",
                                  "assign_key_sections", "MergeOutcome"])
def test_the_extension_primitives_are_importable(name):
    """`aggregator_factory=` is advertised as the main extension point, and
    `diffs_contradict` / `fuse_diffs` are documented as its primitives."""
    assert hasattr(agentdescent, name)


# -- the two Task classes -------------------------------------------------------


def test_the_reference_domain_no_longer_shadows_the_engine_s_task():
    """`Task(id, prompt, meta)` vs `Task(text, label, keyword)` -- disjoint
    fields, no relationship, same name, and `orchestrator`/`worker` imported the
    other one."""
    from agentdescent.domains.router import RouterTask

    assert agentdescent.Task is not RouterTask
    assert {f for f in RouterTask.__dataclass_fields__} == {"text", "label", "keyword"}
    assert "id" in agentdescent.Task.__dataclass_fields__


def test_the_old_router_task_name_still_works():
    """An alias, so nothing that imported it breaks."""
    from agentdescent.domains.router import RouterTask, Task as LegacyTask

    assert LegacyTask is RouterTask


# -- the merge-outcome vocabulary (#8) ------------------------------------------


def test_merge_outcomes_are_str_so_existing_lookups_keep_working():
    """`outcomes()["below-threshold"]` must not break."""
    assert agentdescent.MergeOutcome.BELOW_THRESHOLD == "below-threshold"
    assert f"{agentdescent.MergeOutcome.COMMITTED}" == "committed"
    assert {agentdescent.MergeOutcome.COMMITTED: 1}["committed"] == 1


def test_outcomes_print_the_way_every_doc_shows_them():
    """`outcomes()` is a dict, and printing a dict uses `repr` on its keys.

    The README, both quickstarts, `evolution.md` and `MergeOutcome`'s own
    docstring all show `{'committed': 1}`; the enum repr printed
    `{<MergeOutcome.COMMITTED: 'committed'>: 1}` instead — noise in the one
    diagnostic a user is told to read first.
    """
    printed = str({agentdescent.MergeOutcome.COMMITTED: 2,
                   agentdescent.MergeOutcome.BELOW_THRESHOLD: 7})
    assert printed == "{'committed': 2, 'below-threshold': 7}", printed


def test_every_category_the_aggregator_emits_is_in_the_vocabulary():
    """The keys of `result.outcomes()` were bare literals written at six
    different return sites; to learn them you had to read the file."""
    src = (ROOT / "agentdescent" / "aggregator.py").read_text()
    emitted = set(re.findall(r"MergeOutcome\.([A-Z_]+)", src))
    declared = {m.name for m in agentdescent.MergeOutcome}
    assert emitted <= declared, f"emitted but undeclared: {emitted - declared}"
    assert "OVERSIZED" in declared, \
        "trust-region rejections were folded into all-stale, which points at the opposite fix"


# -- the docs must import what exists (#20) -------------------------------------


def _doc_imports():
    """Every `from agentdescent... import a, b` across the docs and README."""
    for path in sorted(list((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]):
        for block in re.findall(r"```python\n(.*?)```", path.read_text(), re.DOTALL):
            for module, names in re.findall(
                    r"^from (agentdescent[\w.]*) import ([^\n(]+)$", block, re.M):
                for name in names.split("#")[0].split(","):
                    name = name.strip()
                    if name and not name.startswith("*"):
                        yield path.name, module, name


def test_every_name_the_docs_import_actually_exists():
    """68 of 70 code blocks were never executed by anything.

    A rename, a typo or an unexported name in any of them was invisible. This
    resolves the imports without needing credentials or a network.
    """
    import importlib

    broken = []
    for doc, module, name in _doc_imports():
        try:
            if not hasattr(importlib.import_module(module), name):
                broken.append(f"{doc}: {module}.{name}")
        except ImportError as e:
            broken.append(f"{doc}: cannot import {module} ({e})")
    assert not broken, "docs import names that do not exist:\n  " + "\n  ".join(broken)


def test_the_docs_import_something_at_all():
    """Guard the premise: the check above is only worth anything if it has input."""
    assert len(list(_doc_imports())) > 30


# -- the version is single-sourced (#10) ----------------------------------------


def test_the_packaged_version_cannot_drift_from_the_module():
    """`__version__` said 0.7.0 while `pyproject.toml` published 0.2.0, so every
    wheel, the PyPI page and the README badge were five minor versions behind."""
    cfg = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in cfg, "the version must not be duplicated"
    assert 'attr = "agentdescent.__version__"' in cfg
    assert not re.search(r'^version\s*=\s*"', cfg, re.M), \
        "a static version in pyproject.toml is what drifted"
