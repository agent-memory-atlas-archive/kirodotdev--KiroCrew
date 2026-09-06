/**
 * refreshSlot across a corpus change (dashboard.replay_from_acp).
 *
 * A count-matched page cut from a different corpus than the view's cursor cannot
 * be stitched, so the thunk retries unbounded. That retry is a whole-history read
 * only while nothing is rotated: after a size rotation the handler still answers
 * `has_more` with a cursor in the OTHER corpus, and handing that page to the
 * replacing reducer would hide the scrollback the view already holds. The thunk
 * must decline (keep the view) in that case and accept the retry only when it
 * reaches the start of history.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { configureStore } from '@reduxjs/toolkit'

type Row = {
  role: 'user' | 'assistant'
  content: string
  cls: string
  ts: string
  meta?: { mid?: string }
}

const TOTAL = 200
const rows = (n: number, from = 0): Row[] =>
  Array.from({ length: n }, (_, i) => ({
    role: i % 2 === 0 ? 'user' : 'assistant',
    content: `m${from + i}`,
    cls: 'msg',
    ts: new Date(Date.UTC(2026, 0, 1, 0, 0, from + i)).toISOString(),
    meta: { mid: `mid-${from + i}` },
  }))

let HISTORY: Row[] = rows(TOTAL)
/** Rows a size rotation moved into `archive/`: the unbounded read no longer
 *  reaches them and reports `has_more` for them. 0 = nothing rotated. */
let ROTATED = 0

vi.mock('../api/client', () => ({
  api: {
    chatSlotDetail: vi.fn((_slot: string, limit?: number) => {
      const corpus = [...HISTORY]
      const total = corpus.length
      if (limit === undefined) {
        // Unbounded: everything still in the live file, in the JSONL corpus.
        return Promise.resolve({
          key: _slot,
          messages: corpus.slice(ROTATED),
          has_more: ROTATED > 0,
          next_before: ROTATED,
          total,
          running: false,
          queue: [],
          cursor_space: 'jsonl',
        })
      }
      // Bounded: the provider came up between fetches, so this page is cut from
      // the replay corpus while the view's cursor still counts JSONL rows.
      const start = Math.max(0, total - limit)
      return Promise.resolve({
        key: _slot,
        messages: corpus.slice(start),
        has_more: start > 0,
        next_before: start,
        total,
        running: false,
        queue: [],
        cursor_space: 'acp_replay',
        transcript_source: 'acp_replay',
      })
    }),
  },
}))

import chatReducer, { refreshSlot, warmSlotCache } from './chatSlice'
import { api } from '../api/client'

const SLOT = 'slot-1'

/** A JSONL-cursored view that has paged back to the newest `held` rows. */
function pagedBack(held: number) {
  const base = chatReducer(undefined, { type: '@@INIT' })
  const oldest = TOTAL - held
  return configureStore({
    reducer: { chat: chatReducer },
    preloadedState: {
      chat: {
        ...base,
        activeSlot: SLOT,
        messages: HISTORY.slice(oldest),
        slotHasMore: oldest > 0,
        slotOldestIndex: oldest,
        slotCursorKey: SLOT,
        slotCursorSpace: 'jsonl',
      },
    },
    middleware: (getDefault) => getDefault({ serializableCheck: false, immutableCheck: false }),
  })
}

const limits = () =>
  (api.chatSlotDetail as unknown as { mock: { calls: unknown[][] } }).mock.calls.map(c => c[1])

describe('refreshSlot across a corpus change', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    HISTORY = rows(TOTAL)
    ROTATED = 0
  })

  it('keeps the loaded scrollback when the unbounded retry is still partial', async () => {
    const held = 120
    ROTATED = 50
    const store = pagedBack(held)
    const before = store.getState().chat.messages

    await store.dispatch(refreshSlot(SLOT) as never)

    // Bounded page, then the unbounded retry -- which came back `has_more`.
    expect(limits()).toEqual([held, undefined])
    // Declined: the view and its cursor are exactly what they were.
    const after = store.getState().chat
    expect(after.messages).toEqual(before)
    expect(after.messages[0].content).toBe(`m${TOTAL - held}`)
    expect(after.slotCursorSpace).toBe('jsonl')
  })

  it('accepts the unbounded retry once it reaches the start of history', async () => {
    const held = 120
    const store = pagedBack(held)

    await store.dispatch(refreshSlot(SLOT) as never)

    expect(limits()).toEqual([held, undefined])
    const after = store.getState().chat.messages
    expect(after).toHaveLength(TOTAL)
    expect(after[0].content).toBe('m0')
  })
})

describe('warmSlotCache carries the transcript source', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    HISTORY = rows(TOTAL)
    ROTATED = 0
  })

  it('records a replay-backed source for a background slot, and clears it when a later warm carries none', async () => {
    // switchSlot reads the CACHED corpus before stitching its page onto the kept
    // head; a warm that left the source unset would read as JSONL and drop the
    // scrollback a replay-backed cache holds.
    const store = pagedBack(120)
    // Cold background slot: the bounded warm comes back from the replay corpus.
    await store.dispatch(warmSlotCache('bg') as never)
    expect(store.getState().chat.slotTranscriptSource?.bg?.source).toBe('acp_replay')
    // Warm again with rows cached: the unbounded read carries no source, so the
    // per-response rule clears it, exactly as switchSlot/refreshSlot do.
    await store.dispatch(warmSlotCache('bg') as never)
    expect(store.getState().chat.slotTranscriptSource?.bg).toBeUndefined()
  })
})
