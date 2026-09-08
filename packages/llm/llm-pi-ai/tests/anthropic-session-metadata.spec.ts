/**
 * Wire coverage for the session id the adapter puts in an Anthropic Messages
 * request body. Self-contained on purpose: it owns its provider stand-in rather
 * than extending the shared `mock-server` helper, because that helper's scripted
 * frames carry no `event:` name and the Anthropic protocol keeps only named
 * events.
 */

import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { assemble } from './assemble.ts'

/** One generation in Anthropic Messages frames, each under the event name pi-ai selects on. */
const ANTHROPIC_EVENTS: readonly { readonly event: string; readonly data: string }[] = [
  { event: 'message_start', data: '{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":3,"output_tokens":0}}}' },
  { event: 'content_block_start', data: '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}' },
  { event: 'content_block_delta', data: '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}' },
  { event: 'content_block_stop', data: '{"type":"content_block_stop","index":0}' },
  { event: 'message_delta', data: '{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}' },
  { event: 'message_stop', data: '{"type":"message_stop"}' },
]

/** Request bodies this stand-in received, in arrival order. */
interface AnthropicStub {
  readonly url: string
  readonly bodies: readonly Record<string, unknown>[]
}

const servers: Server[] = []

afterEach(async () => {
  vi.unstubAllEnvs()
  await Promise.all(servers.splice(0).map(server => new Promise<void>((resolve) => {
    server.close(() => resolve())
  })))
})

/** Start a one-shot Anthropic Messages stand-in that records each request body. */
async function anthropicStub(): Promise<AnthropicStub> {
  const bodies: Record<string, unknown>[] = []
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    let body = ''
    request.on('data', (chunk: Buffer) => { body += chunk.toString('utf8') })
    request.on('end', () => {
      bodies.push(JSON.parse(body) as Record<string, unknown>)
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      for (const frame of ANTHROPIC_EVENTS) response.write(`event: ${frame.event}\ndata: ${frame.data}\n\n`)
      response.end()
    })
  })
  servers.push(server)
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('anthropic stub: no port')
  return { url: `http://127.0.0.1:${address.port}`, bodies }
}

/** Mount a hand-declared `anthropic-messages` provider against the stand-in. */
async function claudeHarness(baseURL: string): Promise<Context> {
  vi.stubEnv('PI_TEST_KEY', 'test-key')
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmPiAi, {
    providers: {
      claude: {
        apiKeyEnv: 'PI_TEST_KEY',
        api: 'anthropic-messages',
        baseURL,
        models: [{ id: 'claude-test', contextWindow: 200_000, maxTokens: 4096 }],
      },
    },
  })
  return ctx
}

describe('the Anthropic request body session id', () => {
  it('carries the session id as parseable metadata.user_id', async () => {
    const stub = await anthropicStub()
    const ctx = await claudeHarness(stub.url)

    const result = await assemble(ctx, {
      provider: 'claude',
      model: 'claude-test',
      messages: [],
      sessionId: 'session-for-pi' as never,
    })

    expect(result.finish).toEqual({ kind: 'stop' })
    const metadata = stub.bodies[0]?.['metadata'] as { user_id?: string } | undefined
    // The gateways parse a JSON object out of the free-form string, so this
    // asserts the parsed members rather than one exact serialization.
    expect(JSON.parse(metadata?.user_id ?? 'null')).toEqual({
      device_id: '',
      account_uuid: '',
      session_id: 'session-for-pi',
    })
  })

  it('omits metadata for a request carrying no session', async () => {
    const stub = await anthropicStub()
    const ctx = await claudeHarness(stub.url)

    await assemble(ctx, { provider: 'claude', model: 'claude-test', messages: [] })

    expect(stub.bodies[0]).not.toHaveProperty('metadata')
  })
})
