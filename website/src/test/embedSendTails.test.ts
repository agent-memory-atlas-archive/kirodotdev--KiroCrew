/**
 * `retireSendTail`: ChatEmbed's per-send tail records, retired one at a time.
 *
 * Pinned:
 * - retiring a record removes THAT record only; others keep their ids.
 * - the retired record's text is taken back only when the composer still
 *   holds exactly what the restore chain produced; an edit since leaves it.
 * - retiring the FIRST of two restores rebuilds the composer around the
 *   second's text and re-bases the second's before/after, so retiring it next
 *   empties the composer.
 * - a restore that started from something other than the previous restore's
 *   result (the user typed in between) breaks the chain: notice retires, text
 *   stays.
 * - a record that restored nothing (a chip send) retires without touching
 *   the composer; an unknown id changes nothing.
 */
import { describe, it, expect } from 'vitest'
import { retireSendTail, type SendTail } from '../app-sdk/embedSendTails'

const notice = (sendId: string, restore?: SendTail['restore']): SendTail =>
  ({ role: 'notice', content: 'Delivery not confirmed', seenCount: 0, sendId, restore })

describe('retireSendTail', () => {
  it('retires only the named record and gives back its text when the composer is untouched', () => {
    const a = notice('a', { text: 'hello', before: '', after: 'hello' })
    const r = retireSendTail([a], 'a', 'hello')
    expect(r.tails).toEqual([])
    expect(r.draft).toBe('')
  })

  it('leaves an edited composer alone', () => {
    const a = notice('a', { text: 'hello', before: '', after: 'hello' })
    const r = retireSendTail([a], 'a', 'hello, and more')
    expect(r.tails).toEqual([])
    expect(r.draft).toBeNull()
  })

  it('retiring the first of two chained restores rebuilds the composer around the second', () => {
    const a = notice('a', { text: 'first', before: '', after: 'first' })
    const b = notice('b', { text: 'second', before: 'first', after: 'first\n\nsecond' })
    const r = retireSendTail([a, b], 'a', 'first\n\nsecond')
    expect(r.draft).toBe('second')
    expect(r.tails).toHaveLength(1)
    expect(r.tails[0].sendId).toBe('b')
    expect(r.tails[0].restore).toEqual({ text: 'second', before: '', after: 'second' })
    // Then retiring B from the rebuilt state empties the composer.
    const r2 = retireSendTail(r.tails, 'b', r.draft!)
    expect(r2.tails).toEqual([])
    expect(r2.draft).toBe('')
  })

  it('retiring the second of two chained restores takes back only its own text', () => {
    const a = notice('a', { text: 'first', before: '', after: 'first' })
    const b = notice('b', { text: 'second', before: 'first', after: 'first\n\nsecond' })
    const r = retireSendTail([a, b], 'b', 'first\n\nsecond')
    expect(r.draft).toBe('first')
    expect(r.tails.map(t => t.sendId)).toEqual(['a'])
    expect(r.tails[0].restore).toEqual({ text: 'first', before: '', after: 'first' })
  })

  it('text typed BETWEEN two restores breaks the chain: the notice retires, the text stays', () => {
    const a = notice('a', { text: 'first', before: '', after: 'first' })
    // B's restore started from "first typed", not from A's result.
    const b = notice('b', { text: 'second', before: 'first typed', after: 'first typed\n\nsecond' })
    const r = retireSendTail([a, b], 'a', 'first typed\n\nsecond')
    expect(r.tails.map(t => t.sendId)).toEqual(['b'])
    expect(r.draft).toBeNull()
  })

  it('a record that restored nothing retires without touching the composer', () => {
    const chip = notice('c')
    const a = notice('a', { text: 'hello', before: '', after: 'hello' })
    const r = retireSendTail([chip, a], 'c', 'hello')
    expect(r.tails.map(t => t.sendId)).toEqual(['a'])
    expect(r.draft).toBeNull()
  })

  it('an unknown id changes nothing', () => {
    const a = notice('a', { text: 'hello', before: '', after: 'hello' })
    const r = retireSendTail([a], 'zzz', 'hello')
    expect(r.tails).toEqual([a])
    expect(r.draft).toBeNull()
  })
})
