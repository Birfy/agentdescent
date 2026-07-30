"""Seeded runs must be reproducible ACROSS PROCESSES.

Python randomises `hash()` of str/bytes per process unless PYTHONHASHSEED is
pinned, so using it to seed an RNG or assign a partition silently makes `seed=`
meaningless -- identical seeds gave different results on every run, including for
the deliberately deterministic synchronous orchestrator. Everything that must be
reproducible hashes through `stable_hash` instead.
"""

import subprocess
import sys

from agentdescent.evolvable import stable_hash


def _in_fresh_process(snippet: str) -> str:
    """Run a snippet in a NEW interpreter (fresh, randomised PYTHONHASHSEED)."""
    out = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def test_stable_hash_is_deterministic_across_processes():
    snippet = ("from agentdescent.evolvable import stable_hash;"
               "print(stable_hash('alpha'), stable_hash(('w1', 7)))")
    results = {_in_fresh_process(snippet) for _ in range(3)}
    assert len(results) == 1, f"stable_hash varied across processes: {results}"


def test_builtin_hash_would_have_varied():
    """Guard the premise: builtin hash() really is process-dependent here."""
    results = {_in_fresh_process("print(hash('alpha'))") for _ in range(5)}
    assert len(results) > 1, "expected builtin hash() to vary (PYTHONHASHSEED unset)"


def test_section_assignment_is_stable_across_processes():
    snippet = ("from agentdescent.parallel import section_of;"
               "print([section_of(k, 4) for k in ('alpha','beta','gamma','delta')])")
    results = {_in_fresh_process(snippet) for _ in range(3)}
    assert len(results) == 1, f"TP section assignment varied: {results}"


def test_stable_hash_is_a_nonnegative_int():
    for key in ("a", ("w0", 3), 42):
        h = stable_hash(key)
        assert isinstance(h, int) and h >= 0
