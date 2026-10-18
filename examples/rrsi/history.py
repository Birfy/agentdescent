"""Alias of :mod:`agentdescent.algorithms.rrsi.history` (moved into the package, #384)."""
import sys

from agentdescent.algorithms.rrsi import history as _moved

sys.modules[__name__] = _moved
