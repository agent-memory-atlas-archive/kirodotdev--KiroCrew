import { randomUUID } from 'node:crypto'
import { writeFile } from 'node:fs/promises'
import { test, expect, type APIRequestContext, type Page } from '@playwright/test'

// Retain successful recordings only for this focused memory spec, not the
// whole browser suite or its token-exchanging authentication setup project.
// These captures document settled controls. The shared Tabs/SegmentedControl
// honor reduced motion; CSS-only screenshot disabling does not stop JS springs.
// The rest of the browser suite continues to exercise the default animation.
test.use({ video: 'on' })

test.afterEach(async ({ page }, testInfo) => {
  const video = page.video()
  await page.close()
  if (video) await video.saveAs(testInfo.outputPath('member-memory-walkthrough.webm'))
  await writeFile(testInfo.outputPath('member-memory-evidence.json'), JSON.stringify({
    title: testInfo.title,
    status: testInfo.status,
    expectedStatus: testInfo.expectedStatus,
    retry: testInfo.retry,
    checkoutSha: process.env.GITHUB_SHA || 'unavailable',
    runId: process.env.GITHUB_RUN_ID || 'unavailable',
    runAttempt: process.env.GITHUB_RUN_ATTEMPT || 'unavailable',
    scope: 'Real dashboard and ephemeral gateway, synthetic data; no live provider or restart activation claim.',
    video: video ? 'member-memory-walkthrough.webm' : null,
  }, null, 2), 'utf8')
})

// These writes must never target an operator gateway. The offline E2E harness
// owns the seeded home and deletes it, including retained member archives.
test.beforeEach(async ({ page }) => {
  expect(process.env.KIROCREW_E2E_EPHEMERAL, 'Use the isolated gateway E2E harness').toBe('1')
  await page.emulateMedia({ reducedMotion: 'reduce' })
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true)
})

async function member(request: APIRequestContext, role: string) {
  const name = `memory-e2e-${role}-${randomUUID().slice(0, 8)}`
  const response = await request.post('/api/agents', { data: { name, kiro_agent: 'kirocrew' } })
  expect(response.ok(), await response.text()).toBeTruthy()
  const created = await response.json()
  expect(created.memory_store).toMatch(/^member-/)
  return { name, store: created.memory_store as string }
}

async function rows(request: APIRequestContext, store: string, q = '') {
  const response = await request.get('/api/memory/semantic', { params: { store, limit: 100, q } })
  expect(response.ok(), await response.text()).toBeTruthy()
  return (await response.json()).entries as { key: string; value_json: string; derived_from?: string }[]
}

async function writeFact(request: APIRequestContext, store: string, key: string, value: string) {
  const response = await request.put('/api/memory/semantic', {
    headers: { 'X-Session-Key': 'dashboard:ui' },
    params: { store }, data: { key, value, source: 'user_explicit' },
  })
  expect(response.ok(), await response.text()).toBeTruthy()
}

async function openMemory(page: Page, owner: { name: string; store: string }) {
  await page.goto(`/settings/overview?view=memory&store=${encodeURIComponent(owner.store)}`)
  await expect(page.getByText(`Memory for ${owner.name}`, { exact: true })).toBeVisible()
}

async function recordEditor(page: Page, lineage: 'V1' | 'V2') {
  if (lineage === 'V1') {
    const summary = page.locator('summary').filter({ hasText: /^Manage memory$/ })
    await expect(summary).toBeVisible()
    if (await summary.locator('..').getAttribute('open') === null) await summary.click()
  }
  const editor = page.getByTestId('memory-records-editor')
  await expect(editor).toBeVisible()
  return editor
}

test('member memory copy, correction and forgetting persist without changing V1 or another member', async ({ page, request }, testInfo) => {
  test.setTimeout(90000)
  const target = await member(request, 'reviewer')
  const other = await member(request, 'writer')
  expect(target.store).not.toBe(other.store)
  expect(await rows(request, target.store)).toEqual([])
  expect(await rows(request, other.store)).toEqual([])

  const key = `user.e2e_${randomUUID().replaceAll('-', '')}`
  const original = `Lighthouse ${key.slice(-8)} uses the blue review checklist.`
  const corrected = `Lighthouse ${key.slice(-8)} uses the green review checklist.`
  await writeFact(request, 'default', key, original)
  const globalBefore = (await rows(request, 'default', key)).find(row => row.key === key)
  expect(globalBefore).toBeDefined()

  await openMemory(page, target)
  await page.screenshot({ path: testInfo.outputPath('member-memory-owner-empty.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('button', { name: 'Copy knowledge' }).click()
  const copy = page.getByRole('dialog', { name: `Copy knowledge to “${target.name}”`, exact: true })
  await copy.getByRole('textbox').fill(key)
  await copy.getByRole('checkbox', { name: original, exact: true }).check()
  await page.screenshot({ path: testInfo.outputPath('member-memory-copy-dialog.png'), fullPage: false, animations: 'disabled' })
  await copy.getByRole('button', { name: 'Copy selected (1)' }).click()
  await expect(copy.getByRole('status')).toContainText('1')
  await copy.getByRole('button', { name: 'Close', exact: true }).click()
  await expect(page.getByText(original, { exact: true })).toBeVisible()
  const seeded = (await rows(request, target.store)).find(row => row.key === key)
  expect(JSON.parse(seeded!.derived_from!)).toMatchObject({ store: 'default' })

  const search = page.getByRole('textbox', { name: 'Search memory', exact: true })
  await search.fill('blue review checklist')
  await expect(page.getByText('This is the context the member would receive for this task. Search results below are for browsing and editing.', { exact: true })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('member-memory-recall-before.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('button', { name: 'Recall for this task', exact: true }).click()
  await expect(page.getByText('Memory selected for this task', { exact: true })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('member-memory-recall-evidence.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('button', { name: 'Clear search', exact: true }).click()
  await expect(page.getByText('Memory selected for this task', { exact: true })).toHaveCount(0)

  await page.getByRole('button', { name: 'Edit', exact: true }).click()
  const correction = page.getByRole('dialog', { name: 'Edit memory' })
  await correction.getByRole('textbox', { name: 'Memory text', exact: true }).fill(corrected)
  await correction.getByRole('button', { name: 'Preview changes', exact: true }).click()
  await expect(correction.getByText('Will change: 1', { exact: true })).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('member-memory-correction-preview.png'), fullPage: false, animations: 'disabled' })
  await correction.getByRole('button', { name: 'Apply changes (1)', exact: true }).click()
  await expect(correction).toBeHidden()
  await page.reload()
  await expect(page.getByText(corrected, { exact: true })).toBeVisible()
  await expect(page.getByText(`Memory for ${target.name}`, { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Copy knowledge', exact: true })).toBeEnabled()
  expect((await rows(request, target.store)).find(row => row.key === key)?.derived_from).toBe(seeded?.derived_from)
  expect((await rows(request, 'default', key)).find(row => row.key === key)).toEqual(globalBefore)
  expect(await rows(request, other.store)).toEqual([])

  await page.getByRole('tab', { name: 'Profile', exact: true }).click()
  await expect(page.getByRole('textbox', { name: 'Preferences', exact: true })).toBeEnabled()
  await expect(page.getByRole('textbox', { name: 'Projects', exact: true })).toBeEnabled()
  await page.screenshot({ path: testInfo.outputPath('member-memory-profile.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('tab', { name: 'Memories', exact: true }).click()
  await page.screenshot({ path: testInfo.outputPath('member-memory-desktop.png'), fullPage: true, animations: 'disabled' })
  await page.setViewportSize({ width: 390, height: 844 })
  await expect(page.getByText(corrected, { exact: true })).toBeVisible()
  await expect(page.getByText('Facts are saved details, Rules guide how the member works, and Experiences are events the member can recall.', { exact: true })).toBeVisible()
  for (const label of ['All', 'Facts', 'Rules', 'Experiences']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
    await expect(page.getByText(label, { exact: true })).toBeInViewport({ ratio: 1 })
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
  await page.screenshot({ path: testInfo.outputPath('member-memory-mobile.png'), fullPage: true, animations: 'disabled' })
  await page.setViewportSize({ width: 1280, height: 720 })

  await openMemory(page, other)
  await expect(page.getByText(corrected, { exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Open member conversation', exact: true })).toBeVisible()
  expect(await rows(request, other.store)).toEqual([])
  await page.screenshot({ path: testInfo.outputPath('member-memory-separate-empty-member.png'), fullPage: false, animations: 'disabled' })
  await openMemory(page, target)

  await page.getByRole('button', { name: 'View details', exact: true }).click()
  const detail = page.getByRole('dialog', { name: 'Memory details' })
  await expect(detail.getByText('Revision history', { exact: true })).toBeVisible()
  await expect(detail.getByRole('status')).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('member-memory-details-desktop.png'), fullPage: false, animations: 'disabled' })
  await detail.getByRole('button', { name: 'Forget', exact: true }).click()
  await page.getByRole('dialog', { name: 'Forget' }).getByRole('button', { name: 'Preview changes', exact: true }).click()
  await page.getByRole('dialog', { name: 'Forget' }).getByRole('button', { name: 'Forget selected memories: 1', exact: true }).click()
  await expect.poll(async () => (await rows(request, target.store)).length).toBe(0)
  await page.reload()
  await expect(page.getByText(corrected, { exact: true })).toHaveCount(0)
  expect((await rows(request, 'default', key)).find(row => row.key === key)).toEqual(globalBefore)
})

test('private backup staging survives navigation and can be cancelled without replacing active memory', async ({ page, request }, testInfo) => {
  test.setTimeout(90000)
  const owner = await member(request, 'recovery')
  const key = 'user.e2e_restore'
  await writeFact(request, owner.store, key, 'The backup contains the earlier decision.')
  await openMemory(page, owner)
  await page.getByRole('tab', { name: 'Recovery', exact: true }).click()
  await page.getByRole('button', { name: 'Back up now', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Restore backup', exact: true })).toHaveCount(1)
  await writeFact(request, owner.store, key, 'The active member keeps the newer decision.')
  const active = await rows(request, owner.store)
  const restoreBackup = page.getByRole('button', { name: 'Restore backup', exact: true })
  await restoreBackup.click()
  await expect(restoreBackup).toBeVisible()
  await expect(restoreBackup).toBeDisabled()
  await expect(restoreBackup).toHaveAttribute('aria-expanded', 'true')
  await expect(page.getByText(/Stage this memory backup for the next Kiro Crew gateway restart/)).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('member-memory-restore-confirm.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('button', { name: 'Confirm restore', exact: true }).click()
  await expect(page.getByText('Restore is ready. Restart Kiro Crew (the gateway) to finish restoring this backup. The current memory store stays active until then.', { exact: true })).toBeVisible()
  await page.reload()
  await page.getByRole('tab', { name: 'Recovery', exact: true }).click()
  await expect(page.getByText('Restore is ready. Restart Kiro Crew (the gateway) to finish restoring this backup. The current memory store stays active until then.', { exact: true })).toBeVisible()
  expect(await rows(request, owner.store)).toEqual(active)
  // With reduced motion the indicator must stay inside the selected tab,
  // including after the restore refresh that previously obscured Profile.
  const recoveryTab = page.getByRole('tab', { name: 'Recovery', exact: true })
  await expect(recoveryTab).toHaveAttribute('aria-selected', 'true')
  expect(await recoveryTab.evaluate(tab => {
    const indicator = tab.querySelector('span[aria-hidden="true"]')!
    const bounds = tab.getBoundingClientRect()
    const pill = indicator.getBoundingClientRect()
    return pill.left >= bounds.left - 1 && pill.right <= bounds.right + 1
      && pill.top >= bounds.top - 1 && pill.bottom <= bounds.bottom + 1
      && getComputedStyle(indicator).transform === 'none'
  })).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('member-memory-restore-pending.png'), fullPage: false, animations: 'disabled' })
  await page.getByRole('button', { name: 'Cancel staged restore', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Restore backup', exact: true })).toBeEnabled()
  expect(await rows(request, owner.store)).toEqual(active)
  await page.screenshot({ path: testInfo.outputPath('member-memory-restore-cancelled.png'), fullPage: false, animations: 'disabled' })
  const advanced = page.locator('summary').filter({ hasText: /^Advanced memory analysis$/ })
  await expect(advanced).toBeVisible()
  await advanced.click()
  const countBy = page.getByRole('combobox', { name: 'Count by', exact: true })
  await expect(countBy).toBeVisible()
  await countBy.scrollIntoViewIfNeeded()
  await expect(countBy).toBeInViewport({ ratio: 1 })
  await page.screenshot({ path: testInfo.outputPath('member-memory-advanced-analysis.png'), fullPage: false, animations: 'disabled' })
})

test('an empty private memory opens its exact member conversation and reuses the persisted thread after reload', async ({ page, request }, testInfo) => {
  test.setTimeout(90000)
  const owner = await member(request, 'conversation')
  const readOwner = async () => {
    const response = await request.get('/api/members')
    expect(response.ok(), await response.text()).toBeTruthy()
    const roster = (await response.json()).members as {
      name: string; slug: string; slot_key: string; memory_store: string; memory_version: number; memory_owner: string
    }[]
    const row = roster.find(candidate => candidate.name === owner.name)
    expect(row, 'The exact created member must be present in the real roster').toBeDefined()
    return row!
  }
  const initial = await readOwner()
  expect(initial).toMatchObject({
    slot_key: '', memory_store: owner.store, memory_version: 2, memory_owner: owner.name,
  })
  expect(await rows(request, owner.store)).toEqual([])

  const threadPath = `/api/members/${encodeURIComponent(initial.slug)}/thread`
  const waitForThread = () => page.waitForResponse(response =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === threadPath,
  )
  await openMemory(page, owner)
  const opened = waitForThread()
  await page.getByRole('button', { name: 'Open member conversation', exact: true }).click()
  const response = await opened
  expect(response.ok(), await response.text()).toBeTruthy()
  const binding = await response.json() as { member: string; slug: string; slot_key: string }
  expect(binding).toMatchObject({ member: owner.name, slug: initial.slug })
  expect(binding.slot_key).not.toBe('')
  await expect(page).toHaveURL(url => url.pathname === '/members' && url.searchParams.get('member') === owner.name)
  const memberHeader = page.getByTestId('member-thread-header')
  await expect(memberHeader.getByText(owner.name, { exact: true })).toBeVisible()
  await expect(memberHeader.getByRole('button', { name: 'Edit member', exact: true })).toBeAttached()
  await expect(page.getByPlaceholder(/message/i)).toBeVisible()

  const panelToggle = page.getByTestId('member-panel-toggle')
  if (await panelToggle.isVisible()) await panelToggle.click()
  await page.getByTestId('side-panel-leading-tab').click()
  const summary = page.getByTestId('member-crew-summary')
  const memoryStatus = summary.getByText('Private Memory V2 — only this member can use it.', { exact: true })
  await expect(memoryStatus).toBeVisible()
  const manageMemory = summary.getByRole('button', { name: 'Manage memory', exact: true })
  await manageMemory.scrollIntoViewIfNeeded()
  await expect(memoryStatus).toBeInViewport({ ratio: 1 })
  await expect(manageMemory).toBeInViewport({ ratio: 1 })
  await page.screenshot({ path: testInfo.outputPath('member-memory-members-status.png'), fullPage: false, animations: 'disabled' })

  // The roster reads dm.json from disk, so this checks the saved binding rather
  // than inferring persistence from the currently mounted chat pane.
  await expect.poll(readOwner).toMatchObject({
    slot_key: binding.slot_key, memory_store: owner.store, memory_owner: owner.name,
  })
  const reopened = waitForThread()
  await page.reload()
  const reloadedResponse = await reopened
  expect(reloadedResponse.ok(), await reloadedResponse.text()).toBeTruthy()
  expect(await reloadedResponse.json()).toEqual(binding)
  await expect(memberHeader.getByText(owner.name, { exact: true })).toBeVisible()
  await expect(page.getByPlaceholder(/message/i)).toBeVisible()
  expect(await rows(request, owner.store)).toEqual([])
})


test('a legacy configured default member keeps V1 until its owner creates empty private memory', async ({ page, request }, testInfo) => {
  test.setTimeout(90000)
  const legacyMembers = JSON.parse(process.env.KIROCREW_E2E_LEGACY_MEMBERS || '[]') as string[]
  const name = legacyMembers[testInfo.retry]
  expect(name, 'The isolated harness must seed a fresh legacy member for this attempt').toMatch(/^memory-e2e-legacy-\d+$/)
  const currentDefault = await request.get('/api/config/default-agent')
  expect(currentDefault.ok(), await currentDefault.text()).toBeTruthy()
  const priorDefault = (await currentDefault.json()).default_agent as string
  const readOwner = async () => {
    const response = await request.get('/api/members')
    expect(response.ok(), await response.text()).toBeTruthy()
    const roster = (await response.json()).members as {
      name: string; slug: string; memory_store: string; memory_version: number; memory_owner: string
    }[]
    const owner = roster.find(candidate => candidate.name === name)
    expect(owner, 'The legacy alias must exist in the real configured roster').toBeDefined()
    return owner!
  }
  expect(await readOwner()).toMatchObject({ memory_store: 'default', memory_version: 1, memory_owner: '' })
  const key = `user.legacy_${randomUUID().replaceAll('-', '')}`
  await writeFact(request, 'default', key, 'Existing Global knowledge requires an explicit copy.')
  const globalBefore = await rows(request, 'default', key)
  expect(globalBefore).toHaveLength(1)
  const other = await member(request, 'migration-peer')
  let slotKey = ''
  try {
    const promoted = await request.put('/api/config/default-agent', { data: { agent: name } })
    expect(promoted.ok(), await promoted.text()).toBeTruthy()
    expect(await promoted.json()).toMatchObject({ default_agent: name })
    // No agent in this request: exercise the configured default selection,
    // rather than pinning a member and accidentally bypassing that resolver.
    const created = await request.post('/api/chat/slots', { data: { title: `Legacy memory setup ${name}` } })
    expect(created.ok(), await created.text()).toBeTruthy()
    const slot = await created.json() as { key: string; agent: string }
    expect(slot.agent).toBe(name)
    slotKey = slot.key
    expect(slotKey).not.toBe('')
    await page.goto(`/chat?sid=${encodeURIComponent(slotKey)}`)
    const input = page.getByPlaceholder(/message/i)
    await expect(input).toBeVisible()
    const prompt = 'Explain the next step for this project.'
    await input.fill(prompt)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(input).toHaveValue('')
    await expect(page.getByText(prompt, { exact: true }).first()).toBeVisible()
    await expect(page.getByText('pong from the fake ACP backend', { exact: true })).toBeVisible({ timeout: 15000 })
    await expect(page.getByTestId('error-card')).toHaveCount(0)
    await page.screenshot({ path: testInfo.outputPath('legacy-member-memory-v1-working.png'), fullPage: false, animations: 'disabled' })
    expect(await rows(request, 'default', key)).toEqual(globalBefore)

    await page.goto(`/capabilities?tab=crews&crew=${encodeURIComponent(name)}`)
    const editor = page.getByRole('dialog').filter({ has: page.getByTestId('crew-editor-identity') })
    await expect(editor).toBeVisible()
    const memoryTab = editor.getByRole('tab', { name: /^Workspace · Memory(?: Shared)?$/ })
    await expect(memoryTab).toBeVisible()
    await memoryTab.click()
    const guidance = editor.getByText(/Memory V1\. This member can keep working/)
    await expect(guidance).toContainText('A new private Memory V2 starts empty')
    await expect(guidance).toContainText('copy only what this member needs')
    await expect(guidance).toContainText('Memory V1 stays unchanged')
    await expect(editor.getByRole('button', { name: 'Create private memory', exact: true })).toBeEnabled()
    await expect(editor.getByText(/Also used by/)).toHaveCount(0)
    await page.screenshot({ path: testInfo.outputPath('legacy-member-memory-v1-choice.png'), fullPage: false, animations: 'disabled' })
    const provisioned = page.waitForResponse(response =>
      response.request().method() === 'PUT' && new URL(response.url()).pathname === `/api/agents/${name}`,
    )
    await editor.getByRole('button', { name: 'Create private memory', exact: true }).click()
    const initialized = await provisioned
    expect(initialized.ok(), await initialized.text()).toBeTruthy()
    const store = (await initialized.json()).memory_store as string
    expect(store).toMatch(/^member-/)
    expect(store).not.toBe(other.store)
    await expect.poll(readOwner).toMatchObject({ memory_store: store, memory_version: 2, memory_owner: name })
    await expect(editor.getByRole('button', { name: 'Create private memory', exact: true })).toHaveCount(0)
    await expect(editor.getByRole('button', { name: 'Manage memory', exact: true })).toBeEnabled()
    await expect(editor.getByText(/Also used by/)).toHaveCount(0)
    await editor.getByRole('button', { name: 'Manage memory', exact: true }).click()
    await expect(page.getByText(`Memory for ${name}`, { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Copy knowledge', exact: true })).toBeEnabled()
    await page.screenshot({ path: testInfo.outputPath('legacy-member-memory-v2-empty.png'), fullPage: false, animations: 'disabled' })
    const freshThread = page.waitForResponse(response =>
      response.request().method() === 'POST' && /\/api\/members\/[^/]+\/thread$/.test(new URL(response.url()).pathname),
    )
    await page.getByRole('button', { name: 'Open member conversation', exact: true }).click()
    const fresh = await freshThread.then(response => response.json() as Promise<{ slot_key: string }>)
    expect(fresh.slot_key).not.toBe(slotKey)
    await expect.poll(() => new URL(page.url()).searchParams.get('sid')).toBe(fresh.slot_key)
    await page.goto(`/settings/overview?view=memory&store=${encodeURIComponent(store)}`)
    await page.reload()
    await expect(page.getByText(`Memory for ${name}`, { exact: true })).toBeVisible()
    expect(await rows(request, store)).toEqual([])
    expect(await rows(request, other.store)).toEqual([])
    expect(await rows(request, 'default', key)).toEqual(globalBefore)
    expect(await readOwner()).toMatchObject({ memory_store: store, memory_version: 2, memory_owner: name })
  } finally {
    // The gateway's serial browser suite shares the default setting. Restore
    // it before deleting only this attempt's synthetic alias and chat slot.
    const restored = await request.put('/api/config/default-agent', { data: { agent: priorDefault } })
    expect(restored.ok(), await restored.text()).toBeTruthy()
    if (slotKey) {
      const removedSlot = await request.delete(`/api/chat/slots/${encodeURIComponent(slotKey)}`)
      expect(removedSlot.ok(), await removedSlot.text()).toBeTruthy()
    }
    const removedMember = await request.delete(`/api/agents/${encodeURIComponent(name)}`)
    expect(removedMember.ok(), await removedMember.text()).toBeTruthy()
  }
})

for (const lineage of ['V1', 'V2'] as const) {
  test(`${lineage} previews and atomically edits 65 email memories across pages, preserving a concurrent correction`, async ({ page, request }, testInfo) => {
    test.setTimeout(120000)
    const owner = await member(request, `bulk-${lineage.toLowerCase()}`)
    const store = lineage === 'V1' ? 'default' : owner.store
    const untouchedStore = lineage === 'V1' ? owner.store : 'default'
    const token = randomUUID().replaceAll('-', '')
    const oldEmail = `old-${token}@example.com`
    const newEmail = `new-${token}@example.com`
    const key = (index: number) => `user.bulk_${token}_${index}`
    // Bounded concurrent fixture writes use only this harness's ephemeral home.
    for (let first = 0; first < 65; first += 5) {
      await Promise.all(Array.from({ length: Math.min(5, 65 - first) }, (_, step) => {
        const index = first + step
        return writeFact(request, store, key(index), `Contact ${index}: ${oldEmail}`)
      }))
    }
    const records = async (scope: string, q: string) => {
      const response = await request.get('/api/memory/records', { params: { store: scope, q, kind: 'all', limit: 100 } })
      expect(response.ok(), await response.text()).toBeTruthy()
      return await response.json() as { total: number; entries: { id: string; text: string; metadata: { email_addresses: string[] } }[] }
    }
    expect((await records(store, oldEmail)).total).toBe(65)
    if (lineage === 'V2') await openMemory(page, owner)
    else {
      await page.goto('/settings/overview?view=memory')
      await expect(page.locator('summary').filter({ hasText: /^Manage memory$/ })).toBeVisible()
      await expect(page.getByRole('combobox', { name: 'Memory store', exact: true })).toBeEnabled()
      await page.screenshot({ path: testInfo.outputPath('memory-v1-global-overview.png'), fullPage: true, animations: 'disabled' })
    }
    const editor = await recordEditor(page, lineage)
    await editor.getByRole('textbox', { name: 'Search memory', exact: true }).fill(oldEmail)
    await expect(editor.getByRole('button', { name: 'Email', exact: true })).toHaveCount(0)
    await expect(editor.getByText('Matching memories: 65', { exact: true })).toBeVisible()
    await editor.getByRole('checkbox', { name: 'Select this page', exact: true }).check()
    await editor.getByRole('button', { name: 'Select all 65 matching memories', exact: true }).click()
    await editor.getByRole('button', { name: 'Next page', exact: true }).click()
    await expect(editor.getByText('51–65 of 65', { exact: true })).toBeVisible()
    const selectedCount = editor.getByText('Selected: 65', { exact: true })
    await expect(selectedCount).toBeVisible()
    await selectedCount.scrollIntoViewIfNeeded()
    await expect(selectedCount).toBeInViewport({ ratio: 1 })
    await expect(editor.getByRole('button', { name: 'Find and replace', exact: true })).toBeInViewport({ ratio: 1 })
    await expect(editor.getByRole('button', { name: 'Forget selected memories: 65', exact: true })).toBeInViewport({ ratio: 1 })
    await page.screenshot({ path: testInfo.outputPath(`memory-${lineage.toLowerCase()}-bulk-selection.png`), fullPage: false, animations: 'disabled' })
    await editor.getByRole('button', { name: 'Find and replace', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Edit selected memories', exact: true })
    await expect(dialog).toBeVisible()
    await dialog.getByLabel('Find text', { exact: true }).fill(oldEmail)
    await dialog.getByLabel('Replace with', { exact: true }).fill(newEmail)
    await expect(dialog.getByLabel('Find text', { exact: true })).toHaveValue(oldEmail)
    await expect(dialog.getByLabel('Replace with', { exact: true })).toHaveValue(newEmail)
    await expect(dialog.getByRole('button', { name: 'Preview changes', exact: true })).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`memory-${lineage.toLowerCase()}-bulk-edit-dialog.png`), fullPage: false, animations: 'disabled' })
    await dialog.getByRole('button', { name: 'Preview changes', exact: true }).click()
    await expect(dialog.getByText('Will change: 65', { exact: true })).toBeVisible()
    await expect(dialog.getByText('1–25 of 65', { exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: 'Next page', exact: true }).click()
    await expect(dialog.getByText('26–50 of 65', { exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: 'Next page', exact: true }).click()
    await expect(dialog.getByText('51–65 of 65', { exact: true })).toBeVisible()
    await expect(dialog.getByRole('button', { name: 'Next page', exact: true })).toBeDisabled()
    expect((await records(store, oldEmail)).total).toBe(65)
    expect((await records(store, newEmail)).total).toBe(0)

    // A real concurrent write invalidates the whole batch; no row may be partly changed.
    const concurrent = `Concurrent owner correction: ${oldEmail}`
    await writeFact(request, store, key(0), concurrent)
    const rejected = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/memory/bulk/apply')
    await dialog.getByRole('button', { name: 'Apply changes (65)', exact: true }).click()
    expect((await rejected).status()).toBe(409)
    await expect(dialog.getByRole('button', { name: 'Refresh selected records', exact: true })).toBeVisible()
    expect((await records(store, oldEmail)).total).toBe(65)
    expect((await records(store, newEmail)).total).toBe(0)
    await dialog.getByRole('button', { name: 'Refresh selected records', exact: true }).click()
    await expect(dialog.getByLabel('Replace with', { exact: true })).toHaveValue(newEmail)
    await dialog.getByRole('button', { name: 'Preview changes', exact: true }).click()
    await expect(dialog.getByText('Will change: 65', { exact: true })).toBeVisible()
    await page.setViewportSize({ width: 390, height: 844 })
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390)
    await page.screenshot({ path: testInfo.outputPath(`memory-${lineage.toLowerCase()}-bulk-preview-mobile.png`), fullPage: false, animations: 'disabled' })
    await dialog.getByRole('button', { name: 'Apply changes (65)', exact: true }).click()
    await expect(dialog).toBeHidden()
    await expect(editor.getByText('Updated records: 65', { exact: true })).toBeVisible()
    const updated = await records(store, newEmail)
    expect(updated.total).toBe(65)
    expect(updated.entries.find(row => row.id === key(0))?.text).toContain('Concurrent owner correction:')
    expect(updated.entries.every(row => row.metadata.email_addresses.includes(newEmail))).toBeTruthy()
    expect((await records(store, oldEmail)).total).toBe(0)
    expect((await records(untouchedStore, newEmail)).total).toBe(0)
    await page.reload()
    await (await recordEditor(page, lineage)).getByRole('textbox', { name: 'Search memory', exact: true }).fill(newEmail)
    await expect(page.getByText('Matching memories: 65', { exact: true })).toBeVisible()
    await editor.getByRole('checkbox', { name: 'Select this page', exact: true }).check()
    await editor.getByRole('button', { name: 'Forget selected memories: 50', exact: true }).click()
    const forget = page.getByRole('dialog', { name: 'Forget selected memories: 50', exact: true })
    await forget.getByRole('button', { name: 'Preview changes', exact: true }).click()
    await expect(forget.getByText('Will change: 50', { exact: true })).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`memory-${lineage.toLowerCase()}-forget-preview-mobile.png`), fullPage: false, animations: 'disabled' })
    await forget.getByRole('button', { name: 'Close', exact: true }).click()
    expect((await records(store, newEmail)).total).toBe(65)
  })
}


for (const lineage of ['V1', 'V2'] as const) {
  const scenario = lineage === 'V1'
    ? 'refuses an automated overwrite and persists an explicit owner correction'
    : 'reviews proposals by keeping the current value and accepting a later proposal'
  test(`${lineage} ${scenario}`, async ({ page, request }, testInfo) => {
    test.setTimeout(90000)
    const owner = await member(request, `review-${lineage.toLowerCase()}`)
    const store = lineage === 'V1' ? 'default' : owner.store
    const token = randomUUID().replaceAll('-', '')
    const key = `user.proposal_${token}`
    const currentValue = `Current contact: owner-${token}@example.com`
    const firstProposal = `Unconfirmed contact: wrong-${token}@example.com`
    const acceptedValue = `Confirmed contact: team-${token}@example.com`
    type RecordState = { id: string; value_json: string; source: string; revision: string; metadata: { revision: number; pending_conflicts: number } }
    const readRecord = async () => {
      const response = await request.get('/api/memory/records', { params: { store, q: key, kind: 'fact', limit: 50 } })
      expect(response.ok(), await response.text()).toBeTruthy()
      const result = await response.json() as { total: number; entries: RecordState[] }
      expect(result.total).toBe(1)
      expect(result.entries[0].id).toBe(key)
      return result.entries[0]
    }
    const readHistory = async () => {
      const response = await request.get('/api/memory/records/history', { params: { store, kind: 'fact', id: key, limit: 25 } })
      expect(response.ok(), await response.text()).toBeTruthy()
      return await response.json() as { current_revision: number; entries: { id: number; base_revision: number; operation: string; status: string; after_json: string }[] }
    }
    const propose = async (value: string) => {
      const response = await request.put('/api/memory/semantic', {
        headers: { 'X-Session-Key': 'dashboard:ui' }, params: { store },
        data: { key, value, source: 'consolidation', confidence: 0.9 },
      })
      expect(response.status(), await response.text()).toBe(409)
      const error = (await response.json()).error
      if (lineage === 'V1') {
        expect(error).toBe('Existing entry set by user cannot be overwritten by automated source')
        expect((await readRecord()).metadata.pending_conflicts).toBe(0)
      } else {
        expect(error).toContain('saved for review')
        await expect.poll(async () => (await readRecord()).metadata.pending_conflicts).toBe(1)
      }
    }
    const filterRecord = async () => {
      const editor = await recordEditor(page, lineage)
      await editor.getByRole('textbox', { name: 'Search memory', exact: true }).fill(key)
      await expect(editor.getByText('Matching memories: 1', { exact: true })).toBeVisible()
      return editor
    }

    await writeFact(request, store, key, currentValue)
    const initial = await readRecord()
    await propose(firstProposal)
    expect((await readRecord()).value_json).toBe(initial.value_json)
    if (lineage === 'V2') await openMemory(page, owner)
    else await page.goto('/settings/overview?view=memory')
    let editor = await filterRecord()
    if (lineage === 'V1') {
      expect(await readRecord()).toEqual(initial)
      await expect(editor.getByRole('button', { name: /Review proposals/ })).toHaveCount(0)
      await editor.getByRole('button', { name: 'Edit', exact: true }).click()
      const correction = page.getByRole('dialog', { name: 'Edit memory', exact: true })
      await correction.getByRole('textbox', { name: 'Memory text', exact: true }).fill(acceptedValue)
      await correction.getByRole('button', { name: 'Preview changes', exact: true }).click()
      await expect(correction.getByText('Will change: 1', { exact: true })).toBeVisible()
      expect(await readRecord()).toEqual(initial)
      await correction.getByRole('button', { name: 'Apply changes (1)', exact: true }).click()
      await expect(correction).toBeHidden()
      const corrected = await readRecord()
      expect(JSON.parse(corrected.value_json)).toBe(acceptedValue)
      expect(corrected.source).toBe('user_explicit')
      expect(corrected.metadata).toMatchObject({ revision: initial.metadata.revision + 1, pending_conflicts: 0 })
      const history = await readHistory()
      expect(history.current_revision).toBe(corrected.metadata.revision)
      expect(history.entries[0]).toMatchObject({ operation: 'correct', status: 'accepted' })
      await page.reload()
      editor = await filterRecord()
      await expect(editor.getByText(acceptedValue, { exact: true })).toBeVisible()
      await expect(editor.getByRole('button', { name: /Review proposals/ })).toHaveCount(0)
      expect(await readRecord()).toEqual(corrected)
      return
    }
    await editor.getByRole('button', { name: 'Review proposals (1)', exact: true }).click()
    let detail = page.getByRole('dialog', { name: 'Memory details', exact: true })
    await expect(detail.getByText(firstProposal, { exact: true })).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath('member-memory-review-proposals.png'), fullPage: false, animations: 'disabled' })
    const resolutionPreview = page.waitForResponse(response => response.request().method() === 'POST' && new URL(response.url()).pathname === '/api/memory/bulk/preview')
    await detail.getByRole('button', { name: 'Keep current value', exact: true }).click()
    const previewResponse = await resolutionPreview
    expect(previewResponse.ok(), await previewResponse.text()).toBeTruthy()
    const resolution = await previewResponse.json()
    expect(resolution.changed_count).toBe(1)
    expect(resolution.entries[0].operation).toBe('resolve')
    expect(resolution.entries[0].before).toEqual(resolution.entries[0].after)
    let correction = page.getByRole('dialog', { name: 'Edit memory', exact: true })
    await expect(correction.getByText('The current value stays unchanged. Applying records your decision and closes its pending proposals.', { exact: true })).toBeVisible()
    // Preview alone must leave the proposal pending and the record version intact.
    expect((await readRecord()).metadata).toMatchObject({ revision: initial.metadata.revision, pending_conflicts: 1 })
    await correction.getByRole('button', { name: 'Apply changes (1)', exact: true }).click()
    await expect(correction).toBeHidden()
    await expect.poll(async () => (await readRecord()).metadata.pending_conflicts).toBe(0)
    const kept = await readRecord()
    expect(kept.value_json).toBe(initial.value_json)
    expect(kept.source).toBe(initial.source)
    expect(kept.metadata.revision).toBe(initial.metadata.revision + 1)
    const keptHistory = await readHistory()
    expect(keptHistory.current_revision).toBe(kept.metadata.revision)
    expect(keptHistory.entries[0]).toMatchObject({ operation: 'resolve', status: 'accepted' })
    expect(keptHistory.entries.some(entry => entry.status === 'conflict' && entry.after_json.includes(firstProposal))).toBeTruthy()
    await page.reload()
    editor = await filterRecord()
    await expect(editor.getByText(currentValue, { exact: true })).toBeVisible()
    await expect(editor.getByRole('button', { name: /Review proposals/ })).toHaveCount(0)

    await propose(acceptedValue)
    await page.reload()
    editor = await filterRecord()
    await editor.getByRole('button', { name: 'Review proposals (1)', exact: true }).click()
    detail = page.getByRole('dialog', { name: 'Memory details', exact: true })
    await expect(detail.getByText(acceptedValue, { exact: true })).toBeVisible()
    await expect(detail.getByRole('button', { name: 'Use proposed value', exact: true })).toHaveCount(1)
    await detail.getByRole('button', { name: 'Use proposed value', exact: true }).click()
    correction = page.getByRole('dialog', { name: 'Edit memory', exact: true })
    await expect(correction.getByRole('textbox', { name: 'Memory text', exact: true })).toHaveValue(acceptedValue)
    await correction.getByRole('button', { name: 'Preview changes', exact: true }).click()
    await expect(correction.getByText('Will change: 1', { exact: true })).toBeVisible()
    expect((await readRecord()).value_json).toBe(initial.value_json)
    await correction.getByRole('button', { name: 'Apply changes (1)', exact: true }).click()
    await expect(correction).toBeHidden()
    await expect.poll(async () => (await readRecord()).metadata.pending_conflicts).toBe(0)
    const accepted = await readRecord()
    expect(JSON.parse(accepted.value_json)).toBe(acceptedValue)
    expect(accepted.metadata.revision).toBe(kept.metadata.revision + 1)
    const acceptedHistory = await readHistory()
    expect(acceptedHistory.current_revision).toBe(accepted.metadata.revision)
    expect(acceptedHistory.entries[0].status).toBe('accepted')
    expect(acceptedHistory.entries[0].after_json).toContain(acceptedValue)
    await page.reload()
    editor = await filterRecord()
    await expect(editor.getByText(acceptedValue, { exact: true })).toBeVisible()
    await expect(editor.getByRole('button', { name: /Review proposals/ })).toHaveCount(0)
  })
}
