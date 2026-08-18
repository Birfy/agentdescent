/**
 * The evolution capability seam.
 *
 * This package owns the Service Definition role: what it means for a harness to
 * improve one of its own artifacts, with no opinion on how. The optimiser is a
 * Service Provider registered through {@link EvolutionRegistry.registerEngine}
 * (`dsh-evolution-agentdescent` is one), and the tool, command and UI packages
 * are Consumers.
 *
 * Splitting it this way is not ceremony. dsh's own rule is that one role alone
 * is not a seam, and the practical payoff is that the thing being optimised and
 * the thing doing the optimising can be replaced independently: a different
 * optimiser keeps the `/evolve` command, and a new artifact kind works with an
 * optimiser that has never heard of it.
 *
 * @module dsh-evolution
 */

import { Context, Service } from '@deepseek-ai/cordis'

/** Addresses one evolvable artifact, e.g. `skill:sql` or `prompt:house-style`. */
export type ArtifactId = string

/** Identifies one evolution run for its lifetime. */
export type RunId = string

/**
 * An artifact's whole state as a flat `{relative path -> content}` map.
 *
 * The keys being paths is what gives aggregation file-level semantics for free:
 * two proposals touching different files are complementary and fuse, two
 * touching the same file contradict and are resolved on held-out score. The
 * engine never interprets a key.
 */
export type ArtifactState = Readonly<Record<string, string>>

/** One unit of work a candidate is measured on. */
export interface EvolutionTask {
  /** Stable identity, so a held-out split means the same thing across rounds. */
  readonly id: string
  readonly prompt: string
  /** Free-form; `gold` is where the shipped gold-answer scorers look. */
  readonly meta?: Readonly<Record<string, unknown>>
}

/** What one rollout produced, as far as scoring is concerned. */
export interface RolloutResult {
  /** The committed assistant text for the interval this rollout owned. */
  readonly output: string
  /** The `kind` of the last `turn/end`, e.g. `completed` / `max-tokens` / `error`. */
  readonly finishReason?: string
  readonly steps?: number
  readonly toolErrors?: number
  readonly tokens?: number
}

/**
 * Turns a rollout into a reward in `[0, 1]`.
 *
 * `reference` is the trajectory of what actually happened when this task was
 * first seen, when there is one. It is optional because the strongest case --
 * the user brought gold answers or an objective -- needs no reference at all,
 * and only the fallback rung compares against history.
 */
export interface Scorer {
  readonly name: string
  score(task: EvolutionTask, run: RolloutResult, reference?: RolloutResult): Promise<number>
}

/** Where tasks come from: a dataset, past transcripts, or synthesis. */
export interface TaskSource {
  readonly name: string
  fetch(count: number, signal?: AbortSignal): Promise<readonly EvolutionTask[]>
}

/**
 * Maps one artifact between the ledger and the live harness.
 *
 * `bind` and `publish` are the same registration at two scopes, which is the
 * whole reason rollouts can share a process: `bind` registers a candidate
 * against one child agent's context so N workers each see their own, and
 * `publish` registers the accepted state globally.
 */
export interface ArtifactAdapter {
  readonly id: ArtifactId
  /** Picks the governance layer: < 0.5 merges on score, >= 0.5 is oracle-gated. */
  readonly blastRadius: number
  load(): Promise<ArtifactState>
  /** Scope a candidate to one agent. Pass that agent's own `ctx`. */
  bind(scope: Context, state: ArtifactState): () => void
  publish(state: ArtifactState): Promise<() => void>
}

/** What to evolve, measured how, against what. */
export interface EvolutionSpec {
  readonly artifact: ArtifactId
  readonly taskSource: string
  readonly scorer: string
  readonly rounds?: number
  readonly workers?: number
  /** A hard ceiling, not advice: the engine stops starting rollouts at it. */
  readonly tokenBudget?: number
}

/** Where a run is. `budget-exhausted` is a normal ending, not a failure. */
export type RunPhase =
  | 'starting' | 'running' | 'stopping'
  | 'done' | 'failed' | 'cancelled' | 'budget-exhausted'
  /** Finished, and its commit is staged for a human -- see `/evolve pending`. */
  | 'awaiting-review'

export interface RunStatus {
  readonly runId: RunId
  readonly artifact: ArtifactId
  readonly phase: RunPhase
  readonly round: number
  readonly accepted: number
  readonly rejected: number
  readonly tokensSpent: number
  readonly error?: string
}

/** One accepted change to an artifact. */
export interface CommitRecord {
  readonly commitId: string
  readonly runId: RunId
  readonly artifact: ArtifactId
  readonly blastRadius: number
  readonly version: number
  readonly heldOutBefore: number
  readonly heldOutAfter: number
}

/** The current accepted state of an artifact. */
export interface ArtifactHead {
  readonly artifact: ArtifactId
  readonly version: number
  readonly heldOut: number
  readonly state: ArtifactState
}

/** The optimiser. One at a time -- see {@link EvolutionRegistry.registerEngine}. */
export interface EvolutionEngine {
  readonly name: string
  start(spec: EvolutionSpec): Promise<RunId>
  status(runId: RunId): RunStatus | undefined
  cancel(runId: RunId, reason?: string): Promise<void>
  head(artifact: ArtifactId): ArtifactHead | undefined
  history(artifact: ArtifactId, limit?: number): Promise<readonly CommitRecord[]>
}

const NO_ENGINE =
  'no evolution engine is mounted. The seam only defines the capability; ' +
  'mount a provider such as `dsh-evolution-agentdescent` to supply one.'

/**
 * `ctx.evolution` -- the registries, plus delegation to the mounted engine.
 */
export class EvolutionRegistry extends Service {
  private engine: EvolutionEngine | undefined
  private readonly adapters = new Map<ArtifactId, ArtifactAdapter>()
  private readonly scorersByName = new Map<string, Scorer>()
  private readonly sourcesByName = new Map<string, TaskSource>()
  private readonly commitListeners = new Set<(record: CommitRecord) => void>()

  constructor(ctx: Context) {
    super(ctx, 'evolution')
  }

  /**
   * Register the sole optimiser.
   *
   * Sole on purpose: two engines against one ledger is two optimisers writing
   * the same artifact with neither one's acceptance test aware of the other's
   * commits, which is not a configuration anyone wants to debug. A duplicate
   * throws at mount rather than at the first run.
   */
  registerEngine(engine: EvolutionEngine): () => void {
    if (this.engine !== undefined) {
      throw new Error(
        `evolution engine "${engine.name}" cannot mount beside "${this.engine.name}": ` +
        'one ledger takes one optimiser. Remove the other engine row from your profile.')
    }
    this.engine = engine
    return this.ctx.effect(() => () => {
      if (this.engine === engine) this.engine = undefined
    }, 'evolution.registerEngine()')
  }

  /** Register what an artifact id means. Duplicate ids throw. */
  registerArtifact(adapter: ArtifactAdapter): () => void {
    return this.insert(this.adapters, adapter.id, adapter, 'artifact')
  }

  registerScorer(scorer: Scorer): () => void {
    return this.insert(this.scorersByName, scorer.name, scorer, 'scorer')
  }

  registerTaskSource(source: TaskSource): () => void {
    return this.insert(this.sourcesByName, source.name, source, 'task source')
  }

  private insert<T>(into: Map<string, T>, key: string, value: T, kind: string): () => void {
    if (into.has(key)) {
      throw new Error(`${kind} "${key}" is already registered`)
    }
    into.set(key, value)
    return this.ctx.effect(() => () => {
      if (into.get(key) === value) into.delete(key)
    }, `evolution.register${kind}`)
  }

  artifact(id: ArtifactId): ArtifactAdapter | undefined { return this.adapters.get(id) }
  scorer(name: string): Scorer | undefined { return this.scorersByName.get(name) }
  taskSource(name: string): TaskSource | undefined { return this.sourcesByName.get(name) }

  /** Registered names, sorted, for discovery surfaces that list choices. */
  listArtifacts(): ArtifactId[] { return [...this.adapters.keys()].sort() }
  listScorers(): string[] { return [...this.scorersByName.keys()].sort() }
  listTaskSources(): string[] { return [...this.sourcesByName.keys()].sort() }

  /**
   * Start a run, after checking the spec names things that exist.
   *
   * Resolving here rather than in the engine keeps one error message for a
   * typo'd scorer regardless of which optimiser is mounted, and means an engine
   * never has to decide what a name it does not recognise should do.
   */
  async start(spec: EvolutionSpec): Promise<RunId> {
    const engine = this.requireEngine()
    if (this.artifact(spec.artifact) === undefined) {
      throw new Error(this.unknown('artifact', spec.artifact, this.listArtifacts()))
    }
    if (this.scorer(spec.scorer) === undefined) {
      throw new Error(this.unknown('scorer', spec.scorer, this.listScorers()))
    }
    if (this.taskSource(spec.taskSource) === undefined) {
      throw new Error(this.unknown('task source', spec.taskSource, this.listTaskSources()))
    }
    return engine.start(spec)
  }

  status(runId: RunId): RunStatus | undefined { return this.requireEngine().status(runId) }

  async cancel(runId: RunId, reason?: string): Promise<void> {
    await this.requireEngine().cancel(runId, reason)
  }

  head(artifact: ArtifactId): ArtifactHead | undefined {
    return this.requireEngine().head(artifact)
  }

  async history(artifact: ArtifactId, limit?: number): Promise<readonly CommitRecord[]> {
    return this.requireEngine().history(artifact, limit)
  }

  /**
   * Announce an accepted change. The engine calls this; consumers listen.
   *
   * Listener failures are contained: a UI that throws while rendering a commit
   * must not unmake the commit, which has already happened.
   */
  commit(record: CommitRecord): void {
    for (const listener of [...this.commitListeners]) {
      try {
        listener(record)
      } catch (error) {
        this.ctx.logger.warn(`evolution commit listener threw: ${String(error)}`)
      }
    }
  }

  onCommit(listener: (record: CommitRecord) => void): () => void {
    this.commitListeners.add(listener)
    return this.ctx.effect(() => () => { this.commitListeners.delete(listener) },
                           'evolution.onCommit()')
  }

  private requireEngine(): EvolutionEngine {
    if (this.engine === undefined) throw new Error(NO_ENGINE)
    return this.engine
  }

  private unknown(kind: string, name: string, known: readonly string[]): string {
    const list = known.length > 0 ? known.join(', ') : '(none registered)'
    return `unknown ${kind} "${name}". Registered: ${list}`
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    evolution: EvolutionRegistry
  }
}

export default EvolutionRegistry
