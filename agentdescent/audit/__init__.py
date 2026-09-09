"""Sparse audit: pairing a cheap verifier against ground truth, without paying for truth.

The loop optimises whatever ``reward`` returns. When that reward is an agent
judging an output rather than a fact about it, the loop is optimising a proxy,
and a proxy can be wrong in a *direction* -- systematically generous about a kind
of answer it likes. Nothing in the loop can notice: every gate reads the same
proxy, so a change that games it looks exactly like a change that improves.

This package adds the only thing that can notice -- occasionally asking someone
who knows -- under three constraints that decide its whole shape:

1. **The answer must not reach the loop.** :class:`AuditedReward` returns the
   verifier's score, always. Enabling the audit cannot change which diffs
   commit, so it can be turned on mid-run and on a production run.
2. **Truth may take days.** A wet-lab experiment or a human reviewer cannot be
   awaited on the evaluation path, so submitting is decoupled from resolving and
   the record is durable enough to outlive the process that made it.
3. **The sample must be a probability sample.** Every record carries the
   ``inclusion_prob`` it was drawn with and the ``verifier_version`` that scored
   it. Neither can be reconstructed afterwards, and without both the labels are
   an anecdote.

Typical use, gold answers::

    from agentdescent.audit import AuditedReward, GoldAnswer, AuditStore

    audited = AuditedReward(
        llm_judge,                              # cheap, runs on every rollout
        oracle=GoldAnswer(exact_match),         # truth, runs on ~10% of them
        store=AuditStore("runs/audit.jsonl"),
        sample_rate=0.1,
    )
    result = evolve(tasks, reward=audited, agent=agent)

    print(summarise(audited.calibration_set()))

Typical use, an experiment that returns next week::

    audited = AuditedReward(simulator_score, oracle=DeferredOracle(),
                            store=AuditStore("runs/audit.jsonl"))
    ...
    # later, in another process
    store = AuditStore("runs/audit.jsonl")
    for rec in store.pending():
        print(rec.record_id, rec.output)        # go and measure these
    resolve_from_mapping(store, {"a1b2...": 0.83, ...})

What is deliberately **not** here: the estimator. Turning a set of paired
observations into a bias estimate with an interval is
prediction-powered inference, it is easy to get subtly wrong in ways coverage
tests do not catch, and it belongs in a module with its own golden vectors and
mutation tests. This package produces the input to that estimator and stops
there. :func:`~agentdescent.audit.store.summarise` reports a raw mean residual
for eyeballing and is named so that nobody mistakes it for the estimator.
"""

from .records import (SCHEMA_VERSION, AuditRecord, Purpose, new_record_id,
                      output_digest, verifier_fingerprint)
from .sources import (DeferredOracle, GoldAnswer, NullOracle, OracleSource,
                      resolve_from_mapping)
from .store import AuditStore, summarise
from .tap import AuditedReward, RenderTap

__all__ = [
    "SCHEMA_VERSION",
    "AuditRecord",
    "AuditStore",
    "AuditedReward",
    "DeferredOracle",
    "GoldAnswer",
    "NullOracle",
    "OracleSource",
    "Purpose",
    "RenderTap",
    "new_record_id",
    "output_digest",
    "resolve_from_mapping",
    "summarise",
    "verifier_fingerprint",
]
