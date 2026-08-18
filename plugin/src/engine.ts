/**
 * The AgentDescent engine, as a provider for the `ctx.evolution` seam.
 *
 * This is the Service Provider role: it supplies `start` / `status` / `cancel` /
 * `head` / `history` by driving the Python optimiser over the bridge, and serves
 * the half the optimiser calls back into -- loading the artifact, fetching
 * tasks, executing rollouts, scoring them, and running the model.
 *
 * Everything model-shaped stays on this side. The engine has no API key, no
 * endpoint and no sandbox of its own, so the budget, the credentials and the
 * approval policy have exactly one home.
 *
 * @module dsh-plugin-agentdescent/engine
 */

import { spawn } from 'node:child_process'

import type { Context } from '@deepseek-ai/cordis'

import { CommitQueue, type PublishOutcome } from './approval.js'
import { Peer, PROTOCOL_VERSION } from './bridge.js'
import type {
  ArtifactHead,
  ArtifactId,
  ArtifactState,
  CommitRecord,
  EvolutionEngine,
  EvolutionRegistry,
  EvolutionSpec,
  EvolutionTask,
  RolloutResult,
  RunId,
  RunStatus,
} from './evolution.js'

/** Executes one task under one candidate state, and reports what happened. */
export type RolloutRunner = (
  artifact: ArtifactId, state: ArtifactState, task: EvolutionTask,
) => Promise<RolloutResult>

/** Runs one completion for the optimiser's reflection step. */
export type Completer = (prompt: string) => Promise<string | { text: string; tokens?: number }>

export interface EngineOptions {
  readonly evolution: EvolutionRegistry
  readonly rollout: RolloutRunner
  readonly complete: Completer
  /** How the sidecar is launched. Replaceable so tests can drive a fake. */
  readonly spawnSidecar?: () => Sidecar
  readonly python?: string
  /**
   * Where the sidecar runs.
   *
   * Not cosmetic: inheriting whatever directory the harness happened to be
   * launched from decides whether `python -m agentdescent...` resolves at all,
   * and the failure is a process that exits instantly with the bridge reporting
   * only that the peer went away.
   */
  readonly cwd?: string
  readonly handshakeTimeoutMs?: number
  /**
   * Holds commits above the auto-merge line for review. Supplied by the caller
   * so the command surface can list and approve the same queue.
   */
  readonly queue?: CommitQueue
}

const DEFAULT_HANDSHAKE_MS = 30_000

/** The two streams and the off switch -- all this needs of a child process. */
export interface Sidecar {
  readonly stdin: NodeJS.WritableStream
  readonly stdout: NodeJS.ReadableStream
  kill: () => void
}

/**
 * Wire a sidecar to the harness and register it as the evolution engine.
 *
 * The handshake happens before anything else and **refuses** a protocol
 * mismatch. Two versions that half-understand each other produce measurements
 * that look fine and are not, which is a worse failure than a process that will
 * not start.
 */
export async function startEngine(ctx: Context, options: EngineOptions): Promise<EvolutionEngine> {
  const child = (options.spawnSidecar ?? defaultSpawn(options.python, options.cwd))()
  const peer = new Peer(child.stdout as never, child.stdin as never,
                        (message) => { ctx.logger.warn(message) })

  serveHarnessHalf(ctx, peer, options)

  const shake = await peer.call<{ protocol: number; engine: string; version: string }>(
    'handshake', {}, options.handshakeTimeoutMs ?? DEFAULT_HANDSHAKE_MS)
  if (shake.protocol !== PROTOCOL_VERSION) {
    child.kill()
    throw new Error(
      `evolution bridge protocol mismatch: this plugin speaks ${PROTOCOL_VERSION}, ` +
      `${shake.engine} ${shake.version} speaks ${shake.protocol}. ` +
      'Upgrade whichever is older; the two are shipped together.')
  }

  const statuses = new Map<RunId, RunStatus>()
  const heads = new Map<ArtifactId, ArtifactHead>()
  const queue = options.queue ?? new CommitQueue()

  const engine: EvolutionEngine = {
    name: `${shake.engine}@${shake.version}`,
    start: async (spec: EvolutionSpec) => await peer.call<RunId>('run.start', spec),
    // `status` is synchronous in the seam because callers poll it from render
    // paths. The engine's own status is a cached projection of the progress
    // notifications, which is what makes that possible.
    status: (runId: RunId) => statuses.get(runId),
    cancel: async (runId: RunId, reason?: string) => {
      await peer.call('run.cancel', { runId, reason })
    },
    head: (artifact: ArtifactId) => heads.get(artifact),
    history: async (artifact: ArtifactId, limit?: number) =>
      await peer.call<readonly CommitRecord[]>('ledger.history', { artifact, limit }),
  }

  peer.handle('log.progress', (progress: any) => {
    const previous = statuses.get(progress.runId)
    statuses.set(progress.runId, {
      runId: progress.runId,
      artifact: progress.artifact ?? previous?.artifact ?? '',
      phase: 'running',
      round: Number(progress.round ?? 0),
      accepted: Number(progress.accepted ?? 0),
      rejected: Number(progress.rejected ?? 0),
      tokensSpent: Number(progress.tokensSpent ?? 0),
    })
    return null
  })

  peer.handle('artifact.publish', async (params: any): Promise<PublishOutcome> => {
    const adapter = options.evolution.artifact(params.artifact)
    if (adapter === undefined) {
      throw new Error(`cannot publish unknown artifact "${String(params.artifact)}"`)
    }
    const version = Number(params.version ?? 0)
    const outcome = await queue.submit(adapter, params.state as ArtifactState, version,
                                       params.commit as CommitRecord | undefined)
    if (outcome.status === 'pending') {
      // No head, no commit announcement. A staged change is not live, and
      // recording it as one would have the engine start its next run from a
      // state the harness is not serving.
      ctx.logger.info(`${String(params.artifact)}: ${outcome.reason} (${outcome.pendingId})`)
      return outcome
    }
    heads.set(params.artifact, {
      artifact: params.artifact,
      version,
      heldOut: Number(params.commit?.heldOutAfter ?? 0),
      state: params.state as ArtifactState,
    })
    if (params.commit !== undefined) options.evolution.commit(params.commit as CommitRecord)
    return outcome
  })

  ctx.effect(() => () => { child.kill() }, 'evolution.sidecar')
  return engine
}

/** The methods the optimiser calls back into. */
function serveHarnessHalf(ctx: Context, peer: Peer, options: EngineOptions): void {
  const { evolution } = options

  peer.handle('artifact.load', async (params: any) => {
    const adapter = evolution.artifact(params.artifact)
    if (adapter === undefined) {
      throw new Error(`unknown artifact "${String(params.artifact)}"`)
    }
    return await adapter.load()
  })

  peer.handle('tasks.fetch', async (params: any) => {
    const source = evolution.taskSource(params.source)
    if (source === undefined) {
      throw new Error(`unknown task source "${String(params.source)}"`)
    }
    return await source.fetch(Number(params.count ?? 16))
  })

  peer.handle('rollout.execute', async (params: any) =>
    await options.rollout(params.artifact, params.state as ArtifactState,
                          params.task as EvolutionTask))

  peer.handle('reward.score', async (params: any) => {
    // The scorer is resolved per call rather than captured at start, so a run
    // never outlives the registration it depends on without saying so.
    const scorer = evolution.scorer(params.scorer ?? currentScorer(evolution, params))
    if (scorer === undefined) {
      throw new Error(`unknown scorer "${String(params.scorer)}"`)
    }
    return await scorer.score(params.task as EvolutionTask,
                              params.run as RolloutResult,
                              params.reference as RolloutResult | undefined)
  })

  peer.handle('llm.complete', async (params: any) => {
    const answer = await options.complete(String(params.prompt ?? ''))
    return typeof answer === 'string' ? { text: answer } : answer
  })

  void ctx
}

/**
 * Which scorer a run uses, when the engine did not name one per call.
 *
 * The spec names it once at `run.start`; carrying it on every score message
 * would be redundant, so the single-registered-scorer case resolves without it
 * and anything ambiguous is an error rather than a guess.
 */
function currentScorer(evolution: EvolutionRegistry, params: any): string {
  const names = evolution.listScorers()
  if (typeof params.scorer === 'string') return params.scorer
  if (names.length === 1 && names[0] !== undefined) return names[0]
  throw new Error(
    `reward.score did not name a scorer and ${names.length} are registered ` +
    `(${names.join(', ')}); the engine must send one`)
}

function defaultSpawn(python?: string, cwd?: string): () => Sidecar {
  return () => {
    // stderr is inherited on purpose: the sidecar's tracebacks belong in the
    // harness's own log, not swallowed into a pipe nobody drains.
    const child = spawn(python ?? 'python3', ['-m', 'agentdescent.dsh.daemon'],
                        { stdio: ['pipe', 'pipe', 'inherit'], ...(cwd === undefined ? {} : { cwd }) })
    return { stdin: child.stdin, stdout: child.stdout, kill: () => { child.kill() } }
  }
}
