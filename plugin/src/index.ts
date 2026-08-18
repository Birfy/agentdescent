/**
 * A training loop for DeepSeek Harness.
 *
 * dsh makes every layer of the agent a swappable plugin, which is the first
 * time an agent's *parameters* -- its skills, prompt sections, tool set,
 * subagent presets -- have been addressable rather than buried in a fixed
 * stack. This bundle adds the third term: `model + harness + descent`.
 *
 * The design is `docs/design-dsh-plugin.md` in the AgentDescent repository.
 *
 * @module dsh-plugin-agentdescent
 */

export {
  EvolutionRegistry,
  default as evolution,
} from './evolution.js'

export { Peer, PeerGone, PROTOCOL_VERSION, RpcError } from './bridge.js'
export { startEngine } from './engine.js'
export { ENTRYPOINT, EVOLVED_SKILL_RANK, parseSkill, skillArtifact } from './artifacts/skill.js'
export { parseLibrary, skillLibraryArtifact } from './artifacts/library.js'
export {
  EVOLVED_SECTION_ORDER, promptArtifact, SECTION_FILE, whenIdle,
} from './artifacts/prompt.js'
export {
  parseToolPolicy, PERSONA_FILE, presetArtifact, TOOLS_FILE,
} from './artifacts/preset.js'

export type { PromptArtifactOptions } from './artifacts/prompt.js'
export type { PresetArtifactOptions, ToolPolicy } from './artifacts/preset.js'

export type { LibrarySkill, SkillLibraryOptions } from './artifacts/library.js'
export { describeChoices, parseEvolveInput, registerCommand, registerTools } from './surface.js'
export { inProcessRollout } from './rollout.js'
export { apply, inject, name, registerReflectionArtifact, REFLECTION_ARTIFACT } from './plugin.js'
export { CommitQueue, renderPending } from './approval.js'
export { idleTrigger } from './idle.js'
export { startAsJob } from './jobs.js'

export type { JobHost, JobsBridgeOptions } from './jobs.js'
export { transcriptTaskSource } from './transcripts.js'
export { buildScorer, buildTaskSource, ConfigError, readJsonl } from './declarative.js'

export type { DatasetSpec, ObjectiveDeps, ObjectiveSpec } from './declarative.js'

export type { TranscriptOptions, TranscriptSource } from './transcripts.js'

export type { IdleTriggerOptions } from './idle.js'

export type { Config } from './plugin.js'
export {
  all, commandScorer, efficiency, goldScorer, judgeScorer, replayPairwiseScorer,
} from './scorers.js'

export type {
  CommandOptions, EfficiencyOptions, GoldMatch, GoldOptions, Judge, JudgeOptions, ReplayOptions,
} from './scorers.js'

export type { CommitQueueOptions, PendingCommit, PublishOutcome } from './approval.js'
export {
  FROZEN_PLUGIN_PATHS, isFrozen, PLUGIN_BLAST_RADIUS, pluginArtifact,
} from './artifacts/plugin.js'

export type { PluginArtifactOptions } from './artifacts/plugin.js'

export type {
  AgentStatus, InProcessRolloutOptions, RolloutAgent, RolloutAgentHandle, RolloutHost,
} from './rollout.js'

export type { Handler } from './bridge.js'
export type { Completer, EngineOptions, RolloutRunner, Sidecar } from './engine.js'
export type { SkillArtifactOptions } from './artifacts/skill.js'

export type {
  ArtifactAdapter,
  ArtifactHead,
  ArtifactId,
  ArtifactState,
  CommitRecord,
  EvolutionEngine,
  EvolutionSpec,
  EvolutionTask,
  RolloutResult,
  RunId,
  RunPhase,
  RunStatus,
  Scorer,
  TaskSource,
} from './evolution.js'
