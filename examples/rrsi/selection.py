"""Alias of :mod:`agentdescent.algorithms.rrsi.selection` (moved into the package, #384)."""
import sys

from agentdescent.algorithms.rrsi import selection as _moved

sys.modules[__name__] = _moved
