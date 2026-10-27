"""Scaffold for one named Algorithm card (#539).

Copy this file to ``agentdescent/algorithms/<name>.py`` (or keep the card in
``builtin.py`` when it is a few lines), replace every ``YOUR_*`` placeholder,
call ``register(...)``, then import the module from ``builtin.py`` the same way
``ace`` / ``gepa`` are imported.

This module is **not** imported by the package. Importing it does not register
anything until you call ``register(build_card())``.

For the step-by-step contributor path see ``docs/algorithm-authoring.md``.
Full research ports still use ``examples/_TEMPLATE.py`` and
``docs/porting-checklist.md``.
"""

from __future__ import annotations

from typing import Any, Dict

from . import Algorithm, BuildContext, Overrides, Param

__all__ = ["build_card"]


def _build(ctx: BuildContext, p: Dict[str, Any]) -> Overrides:
    """Turn validated parameters into policy / knob / agg_config defaults.

    ``ctx.spec`` is the EvolveSpec; ``ctx.reflect`` is the reflection model
    (may be ``None``). Raise ``AlgorithmError`` for impossible combinations.
    """
    # Example: only swap parent selection. Replace with your mechanism.
    # from ..selection import Archive
    # archive = Archive(sampling=p["sampling"], seed=p["seed"])
    return Overrides(
        # policies={"selection": archive},
        notes=[f"algorithm {YOUR_NAME}: describe what this card changes."],
    )


YOUR_NAME = "your_method"


def build_card() -> Algorithm:
    """Return the card; callers register it (in-package or via entry points)."""
    return Algorithm(
        name=YOUR_NAME,
        summary="One line: what a reader gets when they name this algorithm.",
        kinds=("text",),  # and/or skill_dir, agent_dir, agent_code, plugin
        fidelity="preset",  # or mechanism_microport / benchmark_faithful
        params=(
            Param("seed", "int", 0, "seed of any stochastic draw"),
            # Param("rate", "float", 0.5, "…", minimum=0.0, maximum=1.0),
            # Param("mode", "str", "a", "…", choices=("a", "b")),
        ),
        build=_build,
        source="Paper or engine component this card stands for",
        extra_calls="none",
        # doc="algo-your_method.md",
    )


# Wire-up (in-package):
#   from . import register
#   register(build_card())
#   # and in builtin.py: from . import <name>  # noqa: E402,F401
# Third-party package: expose build_card (or the Algorithm) under
#   [project.entry-points."agentdescent.algorithms"]
#   your_method = "your_pkg:build_card"
