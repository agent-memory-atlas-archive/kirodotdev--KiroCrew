# Memory V2 design and operating handoff

Memory V2 gives each Crew Member an exclusive memory store. Global Memory V1
remains the ordinary assistant's memory. There is no automatic migration between
them. The [memory specification](../../../../system-specs/modules/memory-skills-hooks.md),
[security specification](../../../../system-specs/modules/security.md) and
[dashboard specification](../../../../system-specs/modules/learn-cron-dashboard.md)
own the current contracts.

The [algorithm report](algorithm-effectiveness-report.md) and its
[visual edition](https://github.com/kirodotdev/KiroCrew/blob/568abd2faf45db6ae36dd62164b8f5a85af74a71/docs/task-specs/2026/09/memory-v2/algorithm-effectiveness-report.html) distinguish measurements
from design intent. Historical source provenance, model outputs, media and review records
are retained in the [evidence inventory](https://github.com/kirodotdev/KiroCrew/blob/23853b45f433ca0f0369498c46d4a0d212568b79/inventory.json).
That commit publishes evidence; each original receipt identifies the revision
that actually ran. An older passing run does not validate later source changes.

## Member experience

| Flow | Memory used |
| --- | --- |
| Ordinary assistant | Existing Global V1 context and retrieval |
| Existing member before owner opt-in | Its exact Global or named V1 binding |
| New or opted-in member chat, with Crew Mode off | That member's private V2 store |
| Crew Mode assignment | Each delegate keeps its binding; a V2 delegate cannot access Global, V1 or a peer's store by proxy |
| Member child run or Agent schedule | The verified parent's member identity |
| Restart, resume or supported continuation | The same protected persisted identity |
| Missing, unreadable or mismatched private memory | An explicit refusal before provider work, with no Global fallback |

New members start with an automatically created empty store. Existing members
keep their declared V1 memory, including when selected as the default assistant.
The owner may choose **Crew Manager → the member → Workspace · Memory → Create
private memory**. This creates empty V2 memory and leaves the old V1 store intact.
Old V1 records remain available for explicit copying of selected knowledge.
An existing V2 member never falls back to V1 when private identity is damaged.

The V2 choice starts a new conversation without importing old V1 messages or
provider context. Finish or stop the member's visible work and attached children
before choosing it. Old transcripts and previously assigned schedules retain
their original memory identity. Selective copying remains an explicit owner step.

Setup validates the complete edit before allocating private ownership. If the
subsequent configuration publication fails or has an uncertain outcome, surviving
private ownership evidence fences that member until the owner repairs its binding.
It is not safe to infer that such a store is disposable or fall back to V1.

The release PR's **Before you upgrade** section should explain this opt-in. This
feature does not edit `CHANGELOG.md`, which the repository reserves for a version
bump. The installation guide describes the choice and V2 execution requirements.

The memory store uses its member's avatar. Memories, Profile and Recovery have
separate tabs, and Advanced contains Explore memory. Owners can search records,
inspect source and revision history, preview corrections, review proposals,
forget selected memories, and copy up to 50 explicitly selected source records.
Copying retains provenance and preserves source and existing target content.
Visited tabs preserve drafts. Errors retain affected records and offer retry.
The standalone Email action is removed; addresses use ordinary text search.
The copy dialog names the destination member. Restore experience names the
single-record action separately from whole-store Restore backup. A pending store
switch offers Keep editing without losing the draft. Legacy setup-error metadata
remains readable as diagnostic history but no longer hides Continue or adds a
mandatory-setup link. Crew Manager uses member terminology for creation and chat.

Forgetting removes records from recall but does not erase audit snapshots. The
editor has no direct Undo for Forget. A backup containing the records can recover
them by replacing the whole store after restart. The replaced-experience browser
is not a general trash browser. Deleting a member archives its identity and files
and releases cached handles. Reusing its name creates a fresh store. Archive
reattachment is not implemented.

## Algorithm boundary

V1 preserves first-turn preference/project context, decayed history, semantic and
query-ranked episodic retrieval, scoped lessons, automatic conflict precedence,
0.55/0.42 admission floors and live-record capacity rules. Warm follow-ups retain
their existing conversation behavior.

V2 injects complete admitted identity, Soul, permanent rules, owner-managed
preferences and applicable project guidance on every turn. Essential documents
have a separate 64,000-character budget. Profile writes validate before committing;
external project changes are rechecked before the next turn. Unreadable, unsafe
or oversized essential sources refuse rather than being silently truncated.

Memory fragments are recalled on demand. V2 message construction performs no
fragment recall or embedding. Ranking combines semantic similarity and meaningful
query-term coverage, importance and diversity. A missing vector contributes zero
semantic relevance. The reference Qwen3 operating points, 0.62 for short fragments
and 0.57 above 300 characters, are provisional in-sample choices. Custom models
are supported but are not validated by those reference-model measurements.

V2 has no time decay, age downweighting or automatic age/capacity eviction.
Explicit forgetting and reviewed replacement control retention. Inferred
corrections become proposals unless revision and transcript evidence authorize
the change. Recall packs identified snippets into bounded character and serialized
transport budgets, preserving stored content and marking truncation. Candidate
retrieval still scales with the population; output bounds are not latency bounds.

## Storage and execution

Private storage is a separate database and Markdown root, not a tenant column in
Global. The database lineage and protected owner marker must agree. Compatibility
views do not confer V2 ownership. A named legacy store retains V1 policy unless
its validated private identity selects V2. Restores verify both identity and data.

Protected assignments, process incarnation, session and store proofs carry member
authority. Sandbox-writable transcript labels cannot create it. Reads, writes,
run continuation, steering, listing, history and schedules check the actual caller
and target. A member cannot obtain Global or peer memory through a delegated proxy.
Owner-selected work and automatic trigger matching remain separate policies.

Private execution requires an enforceable Linux/WSL namespace or macOS outer
Seatbelt sandbox and the private MCP capability of a supported provider. Kiro
internal delegation must be disabled on macOS. Native Windows and the current
public Codex ACP adapter refuse private execution. Owner management remains
available. Private views withhold Global, peer stores, snapshots and shared
transcripts. A privileged host writer can deliberately publish a copy into shared
project data; the spawn-time hardlink check does not constrain that host authority.

Linux's cross-user-namespace capability check denies access through a host process's
`/proc/<pid>/root` link. Capability dropping and seccomp are not the cause of that
specific denial. Actual namespace and Seatbelt canaries remain required evidence.
The static memory seam scanner is a bounded regression aid, not the security fence.

## Shared infrastructure and recovery limits

- Both versions receive owner editing, revision metadata and broader security
  auditing. Revision/proposal history has no automatic retention limit or purge UI.
- One bounded embedding worker serves all stores. Interactive threads default to
  four, bulk threads to one; explicit operator settings are preserved up to host
  CPU availability. No measured latency or cost improvement is claimed.
- Model identity prevents comparisons between incompatible vector spaces. The
  bounded repair sweep visits Global and already-open named stores without opening
  idle stores or loading an absent model. Re-embedding can still take substantial
  time. FAISS eligibility and incremental repair preserve V1 acceleration.
- Automatic backups visit active V2 stores only. V1 backups remain manual. Staged
  restores activate after restart and coordinate with admitted product writers.
  External programs bypassing product locks are outside that writer contract.
- A failed restore fences the affected store while healthy stores remain usable.
  Structural preparation failures can still block memory installation-wide. Owner
  status, cancellation and valid-backup recovery remain available where safe.
  A turn waits at most 30 seconds for preparation, then receives a retryable
  refusal without starting a provider or recording a session failure. Stop does
  not cancel shared recovery. Owner shutdown remains available during preparation;
  its bounded cleanup does not promise to interrupt a filesystem or SQLite call.
- Preserved restore copies and archived stores require explicit cleanup. They do
  not count toward ordinary backup rotation. Do not infer safe deletion from a
  missing journal, timestamp or directory name.
- Startup restricts the data home to its owner. Use a dedicated installation home.
  Structured tools are the supported named-store access path, including named V1.
- Unverifiable internal callers are refused once private boundaries exist. Local
  owner-token bootstrap also needs process attribution; cross-host, WSL or container
  setups may need a login link from the gateway-host CLI. Proof publication requires
  a filesystem supporting atomic hard links. Per-request attribution uses shared
  executor capacity without a new deadline; saturation remains an operating limit.

## Withdrawing V2

The feature lands as one commit. Reverting it also removes shared fixes. There is
no live kill-switch or tested emergency withdrawal build. A selective withdrawal
requires a reviewed maintenance patch and temporary private-memory unavailability.

Stop writers and preserve an owner-only data-home copy first. A withdrawal patch
must refuse private provisioning/execution, direct private administrative mutations
and pending private restore activation while keeping owner status reads and V1
branches available. Keep schemas, protected identity, sandbox fences and shared
repairs intact. Do not open V2 with an old V1 binary or rewrite its lineage markers.
The shared fixes that must survive withdrawal include credential-presence-only
channel diagnostics, verified-interpreter launcher logging, bounded cron
assignment scanning, FAISS eligibility after deletion, vector-space validation,
and restore isolation for healthy stores. Reverting the feature commit wholesale
would remove these repairs and is not the withdrawal procedure.
CI must exercise these refusals, ordinary/named V1 work and preservation of every
private store and journal before such a patch ships. Re-enable only those guards
after the defect is fixed; do not replace the whole home and discard newer V1 work.

## Evidence and remaining acceptance

CI owns product tests, builds, lint, browser and model execution. Source review
does not prove kernel behavior, browser geometry, model quality or production impact.
The algorithm report identifies the real-model runs, source hashes and failed as
well as passing structural results. Its 50-topic corpus is not a held-out V1/V2
comparison. Browser fixtures use synthetic data and do not establish live model
answers, Crew delegation or restore activation unless a recording actually does so.

The retained [first review record](https://github.com/kirodotdev/KiroCrew/blob/09968d7f5bcae6080b0f9df7066335ba0be93812/docs/task-specs/2026/09/memory-v2/adversarial-review-resolution.md)
and [second review record](https://github.com/kirodotdev/KiroCrew/blob/09968d7f5bcae6080b0f9df7066335ba0be93812/docs/task-specs/2026/09/memory-v2/fable-round-2-resolution.md)
preserve the supplied synthesis mappings. Full per-lane reports and unlisted nits
were not supplied. Current readiness belongs to the PR's exact source revision and
completed checks, not this handoff. Final acceptance includes fresh CI, inspected
current UI evidence, answered review concerns and the three requested reports.

Generated HTML, figures and historical UI originals are preserved in the
[immutable evidence archive](https://github.com/kirodotdev/KiroCrew/tree/568abd2faf45db6ae36dd62164b8f5a85af74a71/temp-screenshots/memory-v2), outside the feature tree. The Markdown report remains
the written evidence record. The committed hybrid-result JSON remains a test
fixture for corpus, arithmetic and snapshot integrity; it is excluded from wheels
and source distributions and is not a new measurement of later candidates.
