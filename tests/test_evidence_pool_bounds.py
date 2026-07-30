"""The settled-evidence pool must not grow without bound.

`settle()` is called on every discarded card -- stale, oversized, CAS-conflicted
-- and nothing in the library reads the pool back. Unbounded, it retained exactly
the payloads the trust region exists to reject.
"""
from agentdescent.aggregator import EvidenceBuffer
from agentdescent.evolvable import Diff, EvidenceCard


def _card(i, value="v"):
    return EvidenceCard(diff=Diff(diff_id=str(i), target="a", ops={"rule": value}),
                        base_version={}, touched=["rule"])


def test_card_count_is_capped():
    buf = EvidenceBuffer()
    for i in range(EvidenceBuffer.SETTLED_MAX_CARDS * 10):
        buf.settle([_card(i)])
    assert len(buf.settled) == EvidenceBuffer.SETTLED_MAX_CARDS


def test_payload_size_is_capped_even_when_the_count_is_not():
    """A runaway reflector's diffs are huge, so the count cap alone is no bound."""
    buf = EvidenceBuffer()
    for i in range(500):
        buf.settle([_card(i, "x" * 500_000)])
    assert len(buf.settled) < EvidenceBuffer.SETTLED_MAX_CARDS   # size, not count, bit
    assert buf.settled_chars <= EvidenceBuffer.SETTLED_MAX_CHARS


def test_eviction_keeps_the_newest():
    """A diagnostic ring is only useful if it holds the most recent failures."""
    buf = EvidenceBuffer()
    for i in range(EvidenceBuffer.SETTLED_MAX_CARDS + 50):
        buf.settle([_card(i)])
    ids = [c.diff.diff_id for c in buf.settled]
    assert ids[-1] == str(EvidenceBuffer.SETTLED_MAX_CARDS + 49)
    assert "0" not in ids


def test_accounting_stays_consistent_across_evictions():
    buf = EvidenceBuffer()
    for i in range(400):
        buf.settle([_card(i, "x" * 20_000)])
    assert buf.settled_chars == sum(
        len(str(v)) for c in buf.settled for v in c.diff.ops.values())


def test_a_real_run_leaves_a_bounded_pool():
    """End-to-end: oversized proposals go through the aggregator, not a stub."""
    from agentdescent.evolution import SingleSlot, Task, evolve

    tasks = [Task(id=str(i), prompt="p") for i in range(6)]
    res = evolve(tasks, lambda t, o: 0.5, run=lambda a, t: "x",
                 # a reflector echoing a huge blob -- rejected by the trust region
                 propose=lambda *a: "y" * 200_000,
                 strategy=SingleSlot(initial_value="v"),
                 rounds=8, n_workers=3, max_concurrency=3, held_out_frac=0.5)
    assert res.error is None
