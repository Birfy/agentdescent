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
