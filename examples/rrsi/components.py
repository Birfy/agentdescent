"""Alias of :mod:`agentdescent.algorithms.rrsi.components` (moved into the package, #384)."""
import sys

from agentdescent.algorithms.rrsi import components as _moved

sys.modules[__name__] = _moved
