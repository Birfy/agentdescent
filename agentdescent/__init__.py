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
    ContractError,
    stable_hash,
    Diff,
    EvidenceCard,
    Evolvable,
    VersionVector,
    vv_dominates,
    vv_staleness,
)
from .ledger import (
    Ledger, Snapshot, CASConflict, ContractRejected, GitError, LedgerFailure,
)
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
from .governance import (
    FAST_MAX, FROZEN_IDS, Layer, classify, assert_mutable, GovernanceError,
    L1SerialGate,
)
from .aggregator import (
    Aggregator,
    AggregatorContractError,
    MergeOutcome,
    diffs_conflict,
    diffs_contradict,
    fuse_diffs,
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
from . import rewards                      # agentdescent.rewards.last_number(...)
from .skill import evolve_skill
from .worker import Worker
from .orchestrator import AgentDescent, RoundStat, run_fork_baseline
from .agents import (
    AgentError,
    Completion,
    Usage,
    claude,
    echo,
    from_callable,
    claude_code,
    cli_agent,
    codex,
    metered,
    openai_compatible,
    with_retries,
)
from .agents import WorkspaceAgent
from .evolution import (
    Agent,
    LLMAgent,
    reflector,
    Task,
    EvolvingArtifact,
    Strategy,
    AppendRules,
    KeyedRules,
    SingleSlot,
    EvolutionResult,
    ProposalContractError,
    RewardContractError,
    SOLVED,
    tasks_from,
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
    assign_key_sections,
    assign_sections,
    section_of,
    shard_round_robin,
)

__version__ = "0.7.0"

__all__ = [
    "Contract",
    "ContractError",
    "stable_hash",
    "Diff",
    "EvidenceCard",
    "Evolvable",
    "VersionVector",
    "vv_dominates",
    "vv_staleness",
    "Ledger",
    "Snapshot",
    "CASConflict",
    "GitError",
    "LedgerFailure",
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
    "FAST_MAX",
    "FROZEN_IDS",
    "L1SerialGate",
    "Aggregator",
    "AggregatorConfig",
    "AggregatorProtocol",
    "AggregatorFactory",
    "MergeReport",
    "MergeOutcome",
    "AggregatorContractError",
    "diffs_conflict",
    "diffs_contradict",
    "fuse_diffs",
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
    "assign_key_sections",
    "section_of",
    "shard_round_robin",
    "Completion",
    "codex",
    "claude_code",
    "cli_agent",
    "WorkspaceAgent",
    "AgentError",
    "Usage",
    "metered",
    "claude",
    "echo",
    "from_callable",
    "openai_compatible",
    "with_retries",
    "Agent",
    "LLMAgent",
    "reflector",
    "Task",
    "EvolvingArtifact",
    "Strategy",
    "AppendRules",
    "KeyedRules",
    "SingleSlot",
    "EvolutionResult",
    "RewardContractError",
    "ProposalContractError",
    "SOLVED",
    "tasks_from",
    "RoundInfo",
    "evolve",
    "async_evolve",
    "claude_agent",
    "rule_id",
]
