"""AgentDescent: a parallel, self-evolving framework for accelerating RSI.

A systematic port of the parallel-training playbook (data/tensor/pipeline
parallelism, parameter servers, decoupled/async RL, partial rollout) onto
recursive self-improvement, where the "parameters" are a library of evolvable
artifacts (skills, prompts, harness modules, verifiers) and the "gradients" are
diffs carrying evidence cards.

See ``agentdescent_design.md`` for the full design; each module cites the section
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
from . import backends, dataloader          # submodules: agentdescent.dataloader.hf_rows(...)
from .dataloader import Dataset, split_dataset
from .sampling import DifficultyWeighted, RoundRobin, TaskSampler
from .scheduler import (
    AuditScheduler,
    DurationEstimator,
    ResumeQueue,
    TaskCluster,
    TaskScheduler,
    fifo_makespan,
    lpt_schedule,
)
from .governance import Layer, classify, assert_mutable, GovernanceError, L1SerialGate
from .aggregator import (
    Aggregator,
    AggregatorConfig,
    AggregatorProtocol,
    AggregatorFactory,
    MergeReport,
    EvidenceBuffer,
)
from .staleness import (
    StaleAction,
    StalenessPolicy,
    FullStaleness,
    GuardedStaleness,
    ReflectiveStaleness,
    get_policy,
)
from .worker import Worker
from .orchestrator import AgentDescent, RoundStat, run_fork_baseline
from .agents import (
    Completion,
    claude,
    echo,
    from_callable,
    openai_compatible,
    with_retries,
)
from .evolution import (
    Agent,
    LLMAgent,
    Task,
    EvolvingArtifact,
    Strategy,
    AppendRules,
    KeyedRules,
    EvolutionResult,
    RoundInfo,
    evolve,
    claude_agent,
    rule_id,
)
from .async_evolve import async_evolve
from .async_runtime import AsyncAgentDescent, AsyncConfig, AsyncStats
from .parallel import (
    ParallelMode,
    ParallelStrategy,
    WorkUnit,
    DataParallel,
    TensorParallel,
    PipelineParallel,
    TensorParallelMerge,
    PipelineChain,
    SectionViolation,
    assign_sections,
    section_of,
    shard_round_robin,
)

__version__ = "0.7.0"

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
    "Dataset",
    "split_dataset",
    "backends",
    "dataloader",
    "DifficultyWeighted",
    "RoundRobin",
    "TaskSampler",
    "ResumeQueue",
    "DurationEstimator",
    "lpt_schedule",
    "fifo_makespan",
    "TaskCluster",
    "TaskScheduler",
    "Layer",
    "classify",
    "assert_mutable",
    "GovernanceError",
    "L1SerialGate",
    "Aggregator",
    "AggregatorConfig",
    "AggregatorProtocol",
    "AggregatorFactory",
    "MergeReport",
    "EvidenceBuffer",
    "StaleAction",
    "StalenessPolicy",
    "FullStaleness",
    "GuardedStaleness",
    "ReflectiveStaleness",
    "get_policy",
    "Worker",
    "AgentDescent",
    "RoundStat",
    "run_fork_baseline",
    "AsyncAgentDescent",
    "AsyncConfig",
    "AsyncStats",
    "ParallelMode",
    "ParallelStrategy",
    "WorkUnit",
    "DataParallel",
    "TensorParallel",
    "PipelineParallel",
    "TensorParallelMerge",
    "PipelineChain",
    "SectionViolation",
    "assign_sections",
    "section_of",
    "shard_round_robin",
    "Completion",
    "claude",
    "echo",
    "from_callable",
    "openai_compatible",
    "with_retries",
    "Agent",
    "LLMAgent",
    "Task",
    "EvolvingArtifact",
    "Strategy",
    "AppendRules",
    "KeyedRules",
    "EvolutionResult",
    "RoundInfo",
    "evolve",
    "async_evolve",
    "claude_agent",
    "rule_id",
]
