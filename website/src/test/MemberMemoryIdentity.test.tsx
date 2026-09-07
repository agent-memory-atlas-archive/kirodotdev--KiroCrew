import { describe, expect, it, vi } from 'vitest'
import { fireEvent, screen, waitFor } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import { MemoryStoreField, memberMemoryState } from '../pages/KiroCrewAgentsPage'
import JobForm from '../components/JobForm'
import { wakesCrew } from '../components/crew/wakesCrew'
import type { CronJob } from '../types'

const calls = vi.hoisted(() => ({ update: vi.fn() }))
vi.mock('../api/client', () => ({ api: {
  models: vi.fn().mockResolvedValue([]),
  updateCron: calls.update,
} }))

describe('private member memory controls', () => {
  it('does not label a lost or mismatched private pointer as usable V1', () => {
    const stores = {
      default: {},
      'reviewer-private': { memory_version: 2, owner_member: 'reviewer' },
      'reviewer-archived': { memory_version: 2, owner_member: 'reviewer' },
      'writer-private': { memory_version: 2, owner_member: 'writer' },
      'legacy-notes': { memory_version: 1 },
    }
    expect(memberMemoryState('reviewer', 'default', stores)).toBe('unavailable')
    expect(memberMemoryState('reviewer', 'legacy-notes', stores)).toBe('unavailable')
    expect(memberMemoryState('reviewer', 'writer-private', stores)).toBe('unavailable')
    expect(memberMemoryState('reviewer', 'reviewer-private', stores)).toBe('private')
    expect(memberMemoryState('old-member', 'missing', stores)).toBe('unavailable')
    expect(memberMemoryState('old-member', 'legacy-notes', stores)).toBe('legacy')
    expect(memberMemoryState('old-member', 'default', stores)).toBe('legacy')
  })

  it('creates private memory automatically without a store picker', () => {
    renderWithProviders(<MemoryStoreField />)
    expect(screen.getByText(/own empty private memory/i)).toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })

  it('keeps a legacy member on usable V1 until its owner explicitly creates V2', () => {
    const initialize = vi.fn()
    renderWithProviders(<MemoryStoreField member="reviewer" value="default" memoryState="legacy" onInitialize={initialize} />)
    expect(screen.getByText('default', { exact: true })).toBeVisible()
    expect(screen.getByText(/Memory V1\. This member can keep working/)).toHaveTextContent('A new private Memory V2 starts empty')
    expect(screen.getByText(/Memory V1\. This member can keep working/)).toHaveTextContent('copy only what this member needs')
    expect(screen.getByText(/Memory V1\. This member can keep working/)).toHaveTextContent('Memory V1 stays unchanged')
    fireEvent.click(screen.getByRole('button', { name: 'Create private memory', exact: true }))
    expect(initialize).toHaveBeenCalledOnce()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })

  it('displays an immutable private identity and protects unsaved edits before navigation', () => {
    const manage = vi.fn()
    renderWithProviders(<MemoryStoreField member="reviewer" value="member-reviewer-123" memoryState="private" onManage={manage} manageDisabled />)
    expect(screen.getByText('member-reviewer-123')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Manage memory' })).toBeDisabled()
    expect(manage).not.toHaveBeenCalled()
  })

  it('does not offer V1 creation or V2 management for an unavailable binding', () => {
    renderWithProviders(<MemoryStoreField member="reviewer" value="missing-store" memoryState="unavailable" onInitialize={() => {}} onManage={() => {}} />)
    expect(screen.getByText(/configured memory store is unavailable/i)).toBeVisible()
    expect(screen.queryByText(/can keep working/i)).toBeNull()
    expect(screen.queryByRole('button', { name: 'Create private memory' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Manage memory' })).toBeNull()
  })
})

describe('scheduled member identity', () => {
  it('preserves legacy display attribution while private jobs follow their durable member', () => {
    const legacy = { agent: 'reviewer' } as CronJob
    // Displaying a legacy schedule beside its agent does not grant V2 memory.
    expect(wakesCrew(legacy, 'reviewer', true)).toBe(true)
    expect(wakesCrew(legacy, 'default', false)).toBe(false)
    const unbound = { agent: '' } as CronJob
    expect(wakesCrew(unbound, 'default', true)).toBe(true)
    expect(wakesCrew(unbound, 'reviewer', false)).toBe(false)
    const member = { agent: 'shared-template', member_id: 'reviewer' } as CronJob
    expect(wakesCrew(member, 'reviewer', false)).toBe(true)
    expect(wakesCrew(member, 'shared-template', false)).toBe(false)
    expect(wakesCrew(member, 'default', true)).toBe(false)
  })

  it('keeps the member immutable while editing a scheduled task', async () => {
    calls.update.mockResolvedValue({})
    const job = { id: 'job-one', name: 'Review', message: 'Review changes', agent: 'shared-template', member_id: 'reviewer', enabled: true, schedule: 'every 1h' } as CronJob
    renderWithProviders(<JobForm job={job} agents={[]} defaultAgent="default" onSaved={() => {}} />)
    expect(screen.getByTestId('jobform-locked-agent')).toHaveTextContent('reviewer')
    expect(screen.queryByLabelText('Switch agent')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Save/i }))
    await waitFor(() => expect(calls.update).toHaveBeenCalledWith('job-one', expect.objectContaining({ member_id: 'reviewer', agent: 'shared-template' })))
  })
})
