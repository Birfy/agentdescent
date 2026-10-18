"""Offline, mechanism-level building blocks for the Google RRSI port.

Currently ships:

* ``selection`` — Algorithm 2 round-selection rule
* ``schedule`` — cosine-annealed L0 edit budget
* ``components`` — the component vocabulary K / K_str and evidence-checked tags
* ``history`` — the per-edit history L_t: yield, prune set, stall exploration
* ``rrsi_policies`` — the same mechanisms wired into ``evolve()`` through the
  engine's ``Policies`` seams (screen / proposal / acceptance)
"""
