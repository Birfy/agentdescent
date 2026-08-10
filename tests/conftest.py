"""The offline suite stays offline.

Six MethodPolicy ports read GSM8K now, and `build(seed)` loads it -- so importing
their test modules would reach the network, and CI has neither `pyarrow` nor a
`datasets-server` that answers reliably (it returned 502 and 429 on the same
run). This repository's suite has never needed the network, and moving a domain
to a real dataset is no reason to change that.

The committed 96-row slice is used **only** because this file asks for it. It is
never a fallback when a fetch fails: a sample that could stand in silently would
let a run report a number it measured on 96 rows while its output said GSM8K,
which is the direction of error nobody can see -- and is exactly what the
truncated 944-row download was.
"""
import os

os.environ.setdefault("AGENTDESCENT_GSM8K_SAMPLE", "1")
