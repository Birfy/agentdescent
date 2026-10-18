"""Offline, mechanism-level building blocks for the Google RRSI port.

Part of the wheel since #384 (it moved from ``examples/rrsi/``, whose modules are
now aliases of these), and reachable from a spec as ``"algorithm": "rrsi"`` --
see :mod:`agentdescent.algorithms`. ``components``, ``history``, ``schedule`` and
``selection`` are adapted from google-research/rrsi and stay Apache-2.0 (each
keeps its header; see ``NOTICE``); ``rrsi_policies`` is original.

Currently ships:

* ``selection`` — Algorithm 2 round-selection rule
* ``schedule`` — cosine-annealed L0 edit budget
* ``components`` — the component vocabulary K / K_str and evidence-checked tags
* ``history`` — the per-edit history L_t: yield, prune set, stall exploration
* ``rrsi_policies`` — the same mechanisms wired into ``evolve()`` through the
  engine's ``Policies`` seams (screen / proposal / acceptance)
"""
