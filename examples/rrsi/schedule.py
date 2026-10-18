"""Alias of :mod:`agentdescent.algorithms.rrsi.schedule` (moved into the package, #384)."""
import sys

from agentdescent.algorithms.rrsi import schedule as _moved

sys.modules[__name__] = _moved
