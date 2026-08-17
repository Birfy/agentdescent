/**
 * The TypeScript half of the wire, against itself.
 *
 * The Python side has the same contract and its own tests; what matters here is
 * that this end behaves identically, because one protocol document describes
 * both and a divergence would only show up as a wrong measurement.
 */

import { PassThrough } from 'node:stream'
import { describe, expect, it } from 'vitest'

import { Peer, PeerGone, PROTOCOL_VERSION, RpcError } from '../src/bridge.js'

function connect(): { a: Peer; b: Peer; errors: string[] } {
  const toB = new PassThrough()
  const toA = new PassThrough()
  const errors: string[] = []
  const a = new Peer(toA, toB, (m) => errors.push(m))
  const b = new Peer(toB, toA, (m) => errors.push(m))
  return { a, b, errors }
}

describe('the bridge', () => {
  it('carries a call and its result', async () => {
    const { a, b } = connect()
    b.handle('echo', (params) => ({ saw: params }))
    await expect(a.call('echo', { x: 1 })).resolves.toEqual({ saw: { x: 1 } })
  })

  it('works in both directions, because neither side is the client', async () => {
    const { a, b } = connect()
    a.handle('rollout.execute', (p: any) => ({ output: `ran ${p.task}` }))
    b.handle('run.start', () => 'run-1')
    await expect(a.call('run.start', {})).resolves.toBe('run-1')
    await expect(b.call('rollout.execute', { task: 't0' })).resolves.toEqual({ output: 'ran t0' })
  })

  it('lets a handler call back while it is being served', async () => {
    // The shape the design depends on: run.start asks for a rollout before it
    // returns. It must not deadlock against the reader.
    const { a, b } = connect()
    a.handle('rollout.execute', () => ({ output: 'done' }))
    b.handle('run.start', async () => {
      const inner = await b.call<{ output: string }>('rollout.execute', { task: 't0' })
      return { first: inner.output }
    })
    await expect(a.call('run.start', {})).resolves.toEqual({ first: 'done' })
  })

  it('settles concurrent calls against their own replies', async () => {
    const { a, b } = connect()
    b.handle('double', (p: any) => p.n * 2)
    const answers = await Promise.all([1, 2, 3, 4, 5].map((n) => a.call('double', { n })))
    expect(answers).toEqual([2, 4, 6, 8, 10])
  })

  it('turns a throwing handler into an error response, not a dead connection', async () => {
    const { a, b, errors } = connect()
    b.handle('boom', () => { throw new Error('that candidate exploded') })
    b.handle('fine', () => 'ok')
    await expect(a.call('boom', {})).rejects.toThrow(RpcError)
    await expect(a.call('boom', {})).rejects.toThrow(/that candidate exploded/)
    await expect(a.call('fine', {})).resolves.toBe('ok')   // still usable
    expect(errors.some((line) => line.includes('boom'))).toBe(true)
  })

  it('answers an unknown method instead of hanging', async () => {
    const { a } = connect()
    await expect(a.call('nope', {})).rejects.toThrow(/no handler for nope/)
  })

  it('delivers a notification and sends no reply', async () => {
    const { a, b } = connect()
    const seen: unknown[] = []
    b.handle('log.progress', (p) => { seen.push(p); return null })
    a.notify('log.progress', { round: 1 })
    await new Promise((resolve) => setImmediate(resolve))
    expect(seen).toEqual([{ round: 1 }])
  })

  it('fails outstanding calls when the peer goes away rather than hanging forever', async () => {
    const { a, b } = connect()
    b.handle('hang', async () => await new Promise(() => {}))
    const inflight = a.call('hang', {})
    a.close('peer died')
    await expect(inflight).rejects.toThrow(PeerGone)
  })

  it('refuses a call on a closed bridge immediately', async () => {
    const { a } = connect()
    a.close()
    await expect(a.call('anything')).rejects.toThrow(PeerGone)
  })

  it('drops a late reply instead of matching it onto the next call', async () => {
    const { a, b } = connect()
    let release: (value: unknown) => void = () => {}
    b.handle('slow', async () => await new Promise((resolve) => { release = resolve }))
    b.handle('quick', () => 'prompt')
    await expect(a.call('slow', {}, 20)).rejects.toThrow(/did not answer/)
    release('late')
    await new Promise((resolve) => setTimeout(resolve, 10))
    await expect(a.call('quick', {})).resolves.toBe('prompt')
  })

  it('survives garbage on the wire', async () => {
    const toB = new PassThrough()
    const toA = new PassThrough()
    const a = new Peer(toA, toB)
    const b = new Peer(toB, toA)
    b.handle('fine', () => 'ok')
    toB.write('this is not json\n')
    await expect(a.call('fine', {})).resolves.toBe('ok')
  })

  it('handles a message split across chunk boundaries', async () => {
    // Streams do not respect message boundaries; a naive parser that assumed one
    // chunk was one line would drop calls under load and only under load.
    const toB = new PassThrough()
    const toA = new PassThrough()
    const a = new Peer(toA, toB)
    const b = new Peer(toB, toA)
    b.handle('echo', (p) => p)
    const waiting = a.call('echo', { half: true })
    await new Promise((resolve) => setImmediate(resolve))
    await expect(waiting).resolves.toEqual({ half: true })
    const line = JSON.stringify({ id: 999, result: 'ignored' })
    toA.write(line.slice(0, 5))
    toA.write(`${line.slice(5)}\n`)
    await expect(a.call('echo', { again: 1 })).resolves.toEqual({ again: 1 })
  })

  it('exposes a protocol version the handshake can compare', () => {
    expect(PROTOCOL_VERSION).toBeGreaterThanOrEqual(1)
  })
})
