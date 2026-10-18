"""The RRSI mechanisms now live in :mod:`agentdescent.algorithms.rrsi`.

They moved into the package so the algorithm registry (``agentdescent algorithms
list``) can ship them in the wheel, which does not carry ``examples/`` (#384).
The modules here are aliases of the moved ones, so existing
``examples.rrsi.<module>`` imports keep working and resolve to the same objects.
"""
