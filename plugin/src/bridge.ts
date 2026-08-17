/**
 * The harness's half of the wire to the engine.
 *
 * The mirror of `agentdescent/dsh/bridge.py`: one JSON object per line, and
 * **both** sides may call. The harness starts runs and reads the ledger; the
 * engine calls back for every rollout, every reflection and every score,
 * because what the design measures is this harness -- real tools, real sandbox,
 * real approval policy -- and not a replica of it.
 *
 * @module dsh-plugin-agentdescent/bridge
 */

import type { Readable, Writable } from 'node:stream'

/** Bumped when a method's shape changes incompatibly; the handshake compares it. */
export const PROTOCOL_VERSION = 1

const PARSE_ERROR = -32700
const METHOD_NOT_FOUND = -32601
const INTERNAL_ERROR = -32603

/** The peer answered with an error. */
export class RpcError extends Error {
  constructor(readonly code: number, message: string, readonly data?: unknown) {
    super(`[${code}] ${message}`)
    this.name = 'RpcError'
  }
}

/** The peer went away. Every outstanding call fails with this rather than hanging. */
export class PeerGone extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'PeerGone'
  }
}

export type Handler = (params: any) => unknown | Promise<unknown>

interface Pending {
  resolve: (value: any) => void
  reject: (error: Error) => void
}

/**
 * One end of the bridge over a pair of streams.
 *
 * Unlike the Python side this needs no thread per inbound call -- a handler that
 * calls back while it is being served is ordinary async here -- but the rest of
 * the contract is identical, deliberately, so one protocol document describes
 * both ends.
 */
export class Peer {
  private readonly handlers = new Map<string, Handler>()
  private readonly pending = new Map<number, Pending>()
  private buffer = ''
  private nextId = 0
  private gone: PeerGone | undefined

  constructor(private readonly input: Readable, private readonly output: Writable,
              private readonly onError: (message: string) => void = () => {}) {
    input.setEncoding('utf8')
    input.on('data', (chunk: string) => { this.consume(chunk) })
    input.on('end', () => { this.close('the peer closed the bridge') })
    input.on('error', (error: Error) => { this.close(`bridge read failed: ${error.message}`) })
  }

  handle(method: string, handler: Handler): this {
    this.handlers.set(method, handler)
    return this
  }

  get closed(): boolean { return this.gone !== undefined }

  /** Stop, failing every outstanding call instead of leaving it waiting. */
  close(reason = 'bridge closed locally'): void {
    if (this.gone !== undefined) return
    this.gone = new PeerGone(reason)
    const outstanding = [...this.pending.values()]
    this.pending.clear()
    for (const call of outstanding) call.reject(this.gone)
  }

  /** Fire and forget. Progress uses this, so it can never block a rollout. */
  notify(method: string, params?: unknown): void {
    this.send({ method, params })
  }

  /**
   * Call the peer and await its answer.
   *
   * `timeoutMs` is a local give-up, not a cancellation -- the peer is not told,
   * so a late reply must be dropped rather than matched onto the next call.
   */
  async call<T = unknown>(method: string, params?: unknown, timeoutMs?: number): Promise<T> {
    if (this.gone !== undefined) throw this.gone
    const id = (this.nextId += 1)
    return await new Promise<T>((resolve, reject) => {
      let timer: NodeJS.Timeout | undefined
      const settle = (fn: (value: any) => void) => (value: any) => {
        if (timer !== undefined) clearTimeout(timer)
        this.pending.delete(id)
        fn(value)
      }
      this.pending.set(id, { resolve: settle(resolve), reject: settle(reject) })
      if (timeoutMs !== undefined) {
        timer = setTimeout(() => {
          this.pending.delete(id)
          reject(new Error(`${method} did not answer within ${timeoutMs}ms`))
        }, timeoutMs)
        timer.unref?.()
      }
      try {
        this.send({ id, method, params })
      } catch (error) {
        this.pending.delete(id)
        reject(error as Error)
      }
    })
  }

  private consume(chunk: string): void {
    this.buffer += chunk
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (line.length > 0) this.dispatch(line)
      newline = this.buffer.indexOf('\n')
    }
  }

  private dispatch(line: string): void {
    let message: any
    try {
      message = JSON.parse(line)
    } catch {
      this.send({ error: { code: PARSE_ERROR, message: 'not JSON' } })
      return
    }
    if (message === null || typeof message !== 'object') return
    if (typeof message.method === 'string') void this.serve(message)
    else this.settle(message)
  }

  private async serve(message: any): Promise<void> {
    const { method, id } = message
    const handler = this.handlers.get(method)
    if (handler === undefined) {
      if (id !== undefined) {
        this.send({ id, error: { code: METHOD_NOT_FOUND, message: `no handler for ${method}` } })
      }
      return
    }
    try {
      const result = await handler(message.params)
      // A handler with no id is a notification: answering one would be a reply
      // to a message that carries no correlation, which the peer drops anyway.
      if (id !== undefined) this.send({ id, result: result ?? null })
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      this.onError(`handler ${String(method)} threw: ${detail}`)
      if (id !== undefined) {
        this.send({ id, error: { code: INTERNAL_ERROR, message: detail } })
      }
    }
  }

  private settle(message: any): void {
    const call = this.pending.get(message.id)
    if (call === undefined) return          // a reply to a call that gave up
    if (message.error !== undefined && message.error !== null) {
      const { code, message: text, data } = message.error
      call.reject(new RpcError(Number(code ?? INTERNAL_ERROR), String(text ?? ''), data))
    } else {
      call.resolve(message.result)
    }
  }

  private send(message: unknown): void {
    if (this.gone !== undefined) throw this.gone
    this.output.write(`${JSON.stringify(message)}\n`)
  }
}
