"""Moved to :mod:`agentdescent.population`; re-exported here for the ports.

The population layer was written here, in the examples, and installed by the
method runner -- which meant `Policies(selection=...)` meant three different
things depending on who read it: `evolve()` refused it, `async_evolve()` dropped
it, and this runner honoured it. It is the engine's now, so a caller gets the
same search from either driver without a runner in between.

Ports that subclass :class:`~agentdescent.population.PopulationAggregator` to
put their own algorithm inside the archive -- PromptBreeder's tournament, which
evaluates and replaces and so cannot be a `SelectionPolicy` -- still import it,
and importing it from here still works.
"""

from __future__ import annotations

from agentdescent.population import PopulationAggregator, population_factory

__all__ = ["PopulationAggregator", "population_factory"]
