"""The #539 card scaffold stays inert until a contributor wires register()."""

from __future__ import annotations

import importlib

from agentdescent import algorithms


def test_card_template_is_not_auto_registered():
    before = set(algorithms.names())
    mod = importlib.import_module("agentdescent.algorithms._TEMPLATE")
    assert set(algorithms.names()) == before
    card = mod.build_card()
    assert card.name == mod.YOUR_NAME
    assert card.fidelity in algorithms.FIDELITY
    assert callable(card.build)
    # Still not in the registry — register() is commented out on purpose.
    assert card.name not in algorithms.names()
