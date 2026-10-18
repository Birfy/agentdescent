"""Alias of :mod:`agentdescent.algorithms.rrsi.rrsi_policies` (moved into the package, #384)."""
import sys

from agentdescent.algorithms.rrsi import rrsi_policies as _moved

sys.modules[__name__] = _moved
