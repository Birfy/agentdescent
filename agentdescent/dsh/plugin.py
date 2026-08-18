"""Evolving a DeepSeek Harness plugin -- the artifact that can rewrite the rules.

A skill is text the agent reads. A *plugin* is code the harness executes, and it
sits at L1 for that reason: every merge goes through the oracle, and a human
approves before it lands.

**The frozen set is the design.** ``evolve_agent_code`` already freezes the test
suite, because otherwise the shortest path to a high score is to weaken the thing
measuring it. A dsh plugin has two more files with exactly that property, and
neither is obvious until you ask what the file can do:

* ``cordis.patch.yml`` decides **which rows get inserted** into the plugin tree.
  A patch can insert or replace *any* row, including the approval gate, the
  sandbox backend, and the evolution engine's own. An artifact permitted to edit
  its own patch is an artifact permitted to remove the thing that reviews it.
* ``package.json`` carries ``prepare`` / ``postinstall`` -- code that runs at
  install time, outside whatever sandbox the agent runs under -- and the
  ``dsh.bundle`` pointer that says which patch file is authoritative.

So both are frozen here, alongside the tests. This is `governance.FROZEN_IDS`'s
reasoning applied to paths rather than artifact ids: whether a file can rewrite
the rules is a structural fact about the file, not something a blast radius can
measure.

Freezing is enforced twice, and both are needed: the proposal filter stops the
*reflector* from editing those paths, and the runner's pristine overlay stops the
*candidate* from rewriting them at run time.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence, Union

from ..agents import Completion
from ..evolution import EvolutionResult
from ..skilldir import HARNESS_BLAST_RADIUS, evolve_agent_code
from .locate import Plugin, find_plugin

__all__ = ["FROZEN_PLUGIN_PATHS", "evolve_dsh_plugin", "resolve_evolvable_plugin"]

#: Paths a plugin may never edit while being evolved. See the module docstring:
#: each one can either rewrite the rules or move the goalposts.
FROZEN_PLUGIN_PATHS: Sequence[str] = (
    "cordis.patch.yml",       # decides which rows mount, including the reviewer's
    "cordis.patch.yaml",
    "package.json",           # install-time scripts, and the bundle pointer
    "test/**",                # the gate
    "tests/**",
    "**/*.test.ts",
    "**/*.spec.ts",
)

#: Never proposals, never rollouts: build output and dependencies are not source.
IGNORED_PLUGIN_PATHS: Sequence[str] = ("node_modules/**", "lib/**", "dist/**", ".git/**")


class NotEvolvable(RuntimeError):
    """The named plugin cannot be evolved in place, and why."""


def resolve_evolvable_plugin(name: str, profile: str = "web", *,
                             environ: Optional[dict] = None) -> Plugin:
    """Find ``name`` in ``profile`` and refuse the cases that cannot work.

    A registry-installed plugin lives under ``node_modules`` as *built output*
    from a tarball. Evolving that would edit files with no source behind them,
    improve nothing reviewable, and be erased by the next install -- silently, at
    a time unrelated to the run. Only a ``link:``ed checkout is a real artifact.
    """
    plugin = find_plugin(name, profile, environ=environ)
    if plugin is None:
        raise NotEvolvable(
            f"no plugin {name!r} in profile {profile!r}. `dsh plugin --profile "
            f"{profile} add <path>` installs one.")
    if not plugin.linked:
        raise NotEvolvable(
            f"{name!r} is installed from the registry, not linked to a checkout. "
            f"What is on disk is built output with no source behind it, and the "
            f"next install would erase any edit. Re-add it as a local checkout "
            f"(`dsh plugin --profile {profile} add ./{name}`) to evolve it.")
    return plugin


def evolve_dsh_plugin(
    plugin: Union[str, Plugin],
    data: Sequence[Any],
    *,
    entrypoint: Sequence[str],
    reflect_with: Completion,
    profile: str = "web",
    score: Union[str, Callable] = "contains",
    test_cmd: Optional[Sequence[str]] = ("npm", "test"),
    setup_cmd: Optional[Sequence[str]] = ("npm", "ci", "--ignore-scripts"),
    frozen: Sequence[str] = (),
    timeout: float = 600.0,
    environ: Optional[dict] = None,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve a dsh plugin's source, gated on its own test suite.

    ``plugin`` is a name in ``profile`` or an already-resolved
    :class:`~agentdescent.dsh.locate.Plugin`. ``entrypoint`` is the command a
    rollout runs to answer a task with the candidate mounted -- normally a
    headless `dsh` invocation pointed at the candidate workspace.

    ``frozen`` **adds to** :data:`FROZEN_PLUGIN_PATHS` rather than replacing it.
    Replacing would let a caller unfreeze the patch file and the manifest by
    accident, which is the one mistake here with no visible symptom: the run
    still passes, still commits, and has quietly been allowed to edit the rows
    that decide whether anything reviews it.

    ``setup_cmd`` installs with ``--ignore-scripts`` on purpose. A candidate that
    could add a ``postinstall`` would otherwise run code on this machine before
    a single test did -- and the gate is supposed to be what runs first.
    """
    resolved = (plugin if isinstance(plugin, Plugin)
                else resolve_evolvable_plugin(plugin, profile, environ=environ))
    blocked = tuple(FROZEN_PLUGIN_PATHS) + tuple(frozen)
    return evolve_agent_code(
        resolved.path, data,
        entrypoint=entrypoint,
        reflect_with=reflect_with,
        score=score,
        name=os.path.basename(resolved.path),
        frozen=blocked,
        editable=("**",),
        setup_cmd=setup_cmd,
        test_cmd=test_cmd,
        timeout=timeout,
        # L1: a plugin is harness, so every merge is forced through the oracle.
        blast_radius=HARNESS_BLAST_RADIUS,
        **evolve_kwargs,
    )
