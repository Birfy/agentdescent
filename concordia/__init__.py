"""Concordia: a parallel, self-evolving framework for accelerating RSI.

A systematic port of the parallel-training playbook (data/tensor/pipeline
parallelism, parameter servers, decoupled/async RL, partial rollout) onto
recursive self-improvement, where the "parameters" are a library of evolvable
artifacts (skills, prompts, harness modules, verifiers) and the "gradients" are
diffs carrying evidence cards.

See ``concordia_design.md`` for the full design; each module cites the section
it implements.
"""

from .evolvable import (
    Contract,
    Diff,
    EvidenceCard,
    Evolvable,
    VersionVector,
    vv_dominates,
    vv_staleness,
)
from .ledger import Ledger, Snapshot, CASConflict, ContractRejected
from .verifier import ThreeLayerVerifier, VerifierBudget
from .scheduler import AuditScheduler, ResumeQueue, TaskCluster, TaskScheduler
from .governance import Layer, classify, assert_mutable, GovernanceError, L1SerialGate
from .aggregator import Aggregator, AggregatorConfig, MergeReport, EvidenceBuffer
from .staleness import (
    StaleAction,
    StalenessPolicy,
    FullStaleness,
    GuardedStaleness,
    ReflectiveStaleness,
    get_policy,
)
from .worker import Worker
from .orchestrator import Concordia, RoundStat, run_fork_baseline
from .async_runtime import AsyncConcordia, AsyncConfig, AsyncStats
from .parallel import (
    ParallelMode,
    TensorParallelMerge,
    PipelineChain,
    SectionViolation,
    assign_sections,
    section_of,
    shard_round_robin,
)

__version__ = "0.3.0"

__all__ = [
    "Contract",
    "Diff",
    "EvidenceCard",
    "Evolvable",
    "VersionVector",
    "vv_dominates",
    "vv_staleness",
    "Ledger",
    "Snapshot",
    "CASConflict",
    "ContractRejected",
    "ThreeLayerVerifier",
    "VerifierBudget",
    "AuditScheduler",
    "ResumeQueue",
    "TaskCluster",
    "TaskScheduler",
    "Layer",
    "classify",
    "assert_mutable",
    "GovernanceError",
    "L1SerialGate",
    "Aggregator",
    "AggregatorConfig",
    "MergeReport",
    "EvidenceBuffer",
    "StaleAction",
    "StalenessPolicy",
    "FullStaleness",
    "GuardedStaleness",
    "ReflectiveStaleness",
    "get_policy",
    "Worker",
    "Concordia",
    "RoundStat",
    "run_fork_baseline",
    "AsyncConcordia",
    "AsyncConfig",
    "AsyncStats",
    "ParallelMode",
    "TensorParallelMerge",
    "PipelineChain",
    "SectionViolation",
    "assign_sections",
    "section_of",
    "shard_round_robin",
]
