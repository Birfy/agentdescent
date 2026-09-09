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

What is here, and what is not. :mod:`~agentdescent.audit.estimate` ships the
**design-based baseline** -- a Hajek (inclusion-probability-weighted) mean of the
residual with a bootstrap interval, which is what Phase 0 of the plan needs to
decide whether a bias exists and whether it is large next to the noise the
acceptance gate already carries. What is deliberately absent is
**prediction-powered inference**: borrowing strength from the unlabelled verifier
scores via a cross-fitted coefficient, per-stratum weights and a `t` quantile is
easy to get subtly wrong in ways a coverage test does not catch, and it belongs
in a module with its own golden vectors and mutation tests. The two are a
baseline and a refinement, not alternatives -- same point estimate in
expectation, wider interval -- so any PPI estimator added later is measured
against this one rather than trusted over it.

:func:`~agentdescent.audit.store.summarise` is neither: it reports a raw
*unweighted* mean for eyeballing a run, and is named so that nobody mistakes it
for an estimator.
"""

from .calibrator import STALE_INFLATION, Calibrator, Rectification
from .diagnose import (Direction, Disagreement, DisagreementReport,
                       FixReport, Kind, classify_disagreements,
                       evaluate_fix, reference_classifier, residual_stats)
from .estimate import bootstrap_ci, hajek_mean, residual_bias, standard_error
from .ppi import (MIN_N_DOMINANT, PPIError, PPIResult, Stratum,
                  ppi_mean_stratified, t_ppf)
from .records import (SCHEMA_VERSION, AuditRecord, Purpose, new_record_id,
                      output_digest, verifier_fingerprint)
from .sources import (DeferredOracle, GoldAnswer, NullOracle, OracleSource,
                      resolve_from_mapping)
from .sampler import (AuditPolicy, SamplePlan, boundary_stratifier,
                      observed_weights, resid_sd_from)
from .sampler import plan as plan_audit
from .store import AuditStore, summarise
from .tap import AuditedReward, RenderTap

__all__ = [
    "SCHEMA_VERSION",
    "AuditRecord",
    "MIN_N_DOMINANT",
    "AuditPolicy",
    "AuditStore",
    "AuditedReward",
    "Calibrator",
    "boundary_stratifier",
    "bootstrap_ci",
    "classify_disagreements",
    "DeferredOracle",
    "Direction",
    "Disagreement",
    "DisagreementReport",
    "FixReport",
    "Kind",
    "GoldAnswer",
    "NullOracle",
    "OracleSource",
    "PPIError",
    "PPIResult",
    "Purpose",
    "SamplePlan",
    "Rectification",
    "RenderTap",
    "STALE_INFLATION",
    "Stratum",
    "new_record_id",
    "evaluate_fix",
    "hajek_mean",
    "observed_weights",
    "output_digest",
    "plan_audit",
    "ppi_mean_stratified",
    "reference_classifier",
    "resid_sd_from",
    "residual_stats",
    "residual_bias",
    "resolve_from_mapping",
    "standard_error",
    "summarise",
    "t_ppf",
    "verifier_fingerprint",
]
