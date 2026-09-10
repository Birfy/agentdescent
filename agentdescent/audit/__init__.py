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

What measures what. :mod:`~agentdescent.audit.estimate` is the **design-based
baseline** -- a Hajek (inclusion-probability-weighted) mean of the residual with
a bootstrap interval, which is what deciding *whether there is a bias at all*
needs. :mod:`~agentdescent.audit.ppi` is the refinement that borrows strength
from the unlabelled verifier scores; the two are not alternatives -- same point
estimate in expectation, wider interval -- so the estimator is measured against
the baseline rather than trusted over it, and carries golden vectors and
mutation tests of its own. :func:`~agentdescent.audit.store.summarise` is
neither: a raw *unweighted* mean for eyeballing a run, named so nobody mistakes
it for an estimator.

And then three modules that do something with the answer.
:mod:`~agentdescent.audit.sampler` decides where the next labels should go,
:mod:`~agentdescent.audit.diagnose` sorts the verifier's errors by what fixing
them would cost, and :mod:`~agentdescent.audit.gate` is the only place any of it
changes an outcome -- by discounting the held-out evidence in proportion to how
much the verifier disagrees with ground truth. On the run that motivated this
package, that discount was **0.60**: thirty-two tasks judged by an LLM carried
the information of nineteen judged by exact match.
"""

from .calibrator import (STALE_INFLATION, Calibrator, Rectification,
                         population_resid_sd)
from .diagnose import (Direction, Disagreement, DisagreementReport,
                       FixReport, Kind, classify_disagreements,
                       evaluate_fix, reference_classifier, residual_stats)
from .estimate import bootstrap_ci, hajek_mean, residual_bias, standard_error
from .gate import (Adjustment, RectifiedAcceptance, VerifierWatch,
                   discount_for, rectified_counts)
from .ppi import (MIN_N_DOMINANT, PPIError, PPIResult, Stratum,
                  ppi_mean_stratified, t_ppf)
from .records import (SCHEMA_VERSION, AuditRecord, Purpose, new_record_id,
                      output_digest, verifier_fingerprint)
from .sources import (DeferredOracle, GoldAnswer, NullOracle, OracleSource,
                      resolve_from_mapping)
from .sampler import (AuditPolicy, SamplePlan, boundary_stratifier,
                      observed_weights, resid_sd_from)
from .sampler import plan as plan_audit
from .scorecard import (FLIP_ALARM, Cost, Goal, Metric, RescanReport,
                        Scorecard, rescan)
# Aliased for the same reason ``plan`` is: exporting it under its own name would
# shadow the module it lives in, so ``from agentdescent.audit import scorecard``
# would hand back a function and ``import agentdescent.audit.scorecard`` a
# module.
from .scorecard import scorecard as verifier_scorecard
from .store import AuditStore, summarise
from .tap import AuditedReward, RenderTap

__all__ = [
    "SCHEMA_VERSION",
    "AuditRecord",
    "MIN_N_DOMINANT",
    "AuditPolicy",
    "AuditStore",
    "AuditedReward",
    "Adjustment",
    "Cost",
    "Calibrator",
    "boundary_stratifier",
    "bootstrap_ci",
    "classify_disagreements",
    "DeferredOracle",
    "Direction",
    "Disagreement",
    "DisagreementReport",
    "FixReport",
    "FLIP_ALARM",
    "Goal",
    "Kind",
    "Metric",
    "GoldAnswer",
    "NullOracle",
    "OracleSource",
    "PPIError",
    "PPIResult",
    "Purpose",
    "RectifiedAcceptance",
    "VerifierWatch",
    "SamplePlan",
    "Rectification",
    "RescanReport",
    "Scorecard",
    "RenderTap",
    "STALE_INFLATION",
    "Stratum",
    "new_record_id",
    "discount_for",
    "evaluate_fix",
    "hajek_mean",
    "observed_weights",
    "output_digest",
    "plan_audit",
    "population_resid_sd",
    "ppi_mean_stratified",
    "rectified_counts",
    "rescan",
    "reference_classifier",
    "resid_sd_from",
    "residual_stats",
    "residual_bias",
    "resolve_from_mapping",
    "standard_error",
    "summarise",
    "t_ppf",
    "verifier_fingerprint",
    "verifier_scorecard",
]
