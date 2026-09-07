# Crew Mode

Two related things are spelled *crew*, and this spec owns both:

1. **A crew** is a named entry in the config's `agents` map. It binds a kiro-cli
   agent template plus a workspace, a memory store, a model and a reasoning
   effort, and it carries free-text `triggers` that decide whether the
   orchestrator may route work to it. The selection path is the `select_crew`
   MCP tool.
2. **Crew Mode** is the `"crew"` chat-slot mode. Its control plane is
   `crew_chat.py`: a durable per-slot ingress queue, a single-flight decision
   agent, and one continuable sub-session per topic, all running on the slot's
   crew.

Neither is *Remote Instances* (see [instances.md](instances.md)), and neither is
an Issue Radar *crew*, which is that app's own repository work crew
(see [issue-radar.md](issue-radar.md)).

## Components

Legacy topic respawn requires its original run identity or surviving legacy
run state. If pruning removed both, continuation refuses with a named memory
error and leaves the queued request retryable; the owner must start a new topic.
Missing history must never silently turn a private topic into Global memory.

| File | Role |
|---|---|
| `src/kiro_crew/config/sections.py` | `KiroCrewAgentConfig` — the crew record: `kiro_agent`, `workspace`, `memory_store`, `model`, `reasoning_effort`, `description`, `triggers`, `source`, `session_color`, `avatar`, per-crew watchdog overrides |
| `src/kiro_crew/config/loader.py` | `resolve_agent_bindings` (crew to workspace / memory store / template) and `resolve_effective_model` (the default-model precedence) |
| `src/kiro_crew/mcp_core.py` | `_do_select_crew` — the roster and bind bodies |
| `src/kiro_crew/mcp_tools/control.py` | The `select_crew` tool declaration and dispatch |
| `src/kiro_crew/validation.py` | `SELECT_CREW_SCHEMA` — argument validation for that tool |
| `src/kiro_crew/members.py` | Per-crew member space: activity log, DM-thread binding, permanent rules, self-maintained briefing, the member turn chokepoint |
| `src/kiro_crew/crew_chat.py` | `CrewStore` and `CrewOrchestrator` — the Crew Mode control plane |
| `src/kiro_crew/subagent.py` | `_validate_agent` — what an `agent=` name is checked against, and `UNADVERTISED_AGENTS` |
| `src/kiro_crew/config/prompt-orchestrator.md` | The orchestrator prompt that names `select_crew` and the delegation rule |
| `src/kiro_crew/dashboard/handlers/agents.py` | Crew CRUD on `/api/agents`, and the roster row serializer |
| `src/kiro_crew/dashboard/handlers/members.py` | `/api/members` roster, thread get-or-create, rules, activity |
| `src/kiro_crew/dashboard/chat_folders.py` | `api_chat_slot_mode` — the mode switch and its refusals |
| `website/src/pages/KiroCrewAgentsPage.tsx` | The Crews UI, mounted as the **Crews** tab of `CapabilitiesPage` (Agent Capabilities) |
| `website/src/components/crew/crewEditorSections.ts` | The crew editor's pane registry, including the Routing pane that edits `triggers` |
| `website/src/components/CrewWakeSection.tsx` | "What wakes this agent" — schedules, deliberately distinct from `triggers` |

## Crew records and binding

A crew lives only in `config.json` under `agents.<name>`. It is not a kiro-cli
agent file: `kiro_agent` points at one. `resolve_agent_bindings` turns a crew
name into `ResolvedBindings`, in this order:

1. the named crew, when it is a key of `config.agents`;
2. otherwise a **materialized** kiro agent of that name (an app-registered agent
   under the user's `~/.kiro/agents/`, or a project agent), which keeps
   dispatching itself with the default workspace and Global Memory V1;
3. otherwise `default_agent`, with `requested_resolved` set to `False` so a
   caller never advertises a binding that is not running.

An unresolvable workspace falls back to `default_workspace`. Memory identity
resolves exactly: the reserved `default` assistant uses Global Memory V1;
existing members keep their declared V1 binding until the owner chooses V2.
New and opted-in members own unique private V2 stores. Missing, unreadable,
shared or mismatched private identity stops execution with an actionable error.
Selecting a member as `default_agent` preserves that member's memory version and
binding. With no agents configured, the resolver returns the existing defaults.

Member creation automatically provisions empty private memory. Members cannot
choose a shared store or rebind their private store. Legacy members may continue
using V1 or explicitly choose empty V2 memory from their settings; former Global
or named V1 contents remain untouched. Config fields, atomic publication, ownership manifest and
recovery semantics are owned by [config](config.md#named-memory-stores-memory_storespy).

Opting into V2 opens a fresh member conversation. Existing V1 conversation and
native provider context cannot become private context by changing the config.
The member-thread binding records its private store generation and reuses that
conversation across later opens and restarts. An already protected V2 thread
keeps its existing key. Old schedules and child runs retain their recorded store;
the opt-in does not relabel past or already assigned work.

Private memory also pins an active dashboard turn to its member in ordinary
chat slots. A provider-side agent switch stops the stream with a visible notice
and resets the provider before another turn; later events cannot continue under
another agent while using that member's memory. This covers member DMs and
ordinary V2 chats. Ordinary V1 chats keep their existing switch behavior. The
validation and reset contract is owned by
[session](session.md#private-member-session-ownership).

The member side panel's Crew summary tab and the editor link to
`/settings/overview?view=memory&store=<name>`. The private memory workspace has
Memories, Profile and Recovery tabs: browsing/search/correction/copy stay in
Memories, preferences and project anchors stay in Profile, and backups plus
retired experiences stay in Recovery. Advanced facet analysis is collapsed.
Profile and Recovery load on first visit; visited Profile stays mounted so tab
changes cannot discard its drafts. Changing the selected member requires explicit
discard while a profile draft or memory mutation dialog is open. Source references
are rendered as origin labels and item references rather than JSON payloads.

The workspace header, store picker and copy-source picker reuse the owning
member's exact avatar descriptor and name, including uploaded pictures. Returning
from the member editor refreshes that identity. Empty memory can open
`/members?member=<exact-name>` directly; this link selects the member by name,
then uses the existing verified thread-opening endpoint. A failed thread open
retains its localized error heading and structured diagnostic report. Details
reveals the redacted reason on demand; Ask the agent receives the same report
when navigation permits. The cached conversation and its drafts remain available.

Facts, rules and experiences all support correction and explicit forgetting.
Experience correction keeps the same record identity and provenance. A store
marked unavailable still makes a scoped read to obtain its actual refusal, with
Retry and Recovery actions; it never displays cached records as a successful
read. Recovery paginates retired memories and refreshes live recall after an
item is restored. Complete snapshot restoration stays visibly staged across
page visits until gateway restart, and the owner can cancel the pending stage
without changing current memory or its saved backup.

Inline schedules created inside the editor persist `member_id` separately from
the provider template. A legacy schedule carrying only `agent_id` stays in Global
Memory V1 even when that string matches a member alias. The editor lists private
member jobs by exact `member_id`, and an existing job's member is immutable.
Legacy jobs retain their previous template/sequence display attribution and show
Global Memory V1 in the member's Schedules pane. Displaying an old schedule there
does not migrate it or grant access to that member's private store.

`resolve_effective_model` is the single source of truth for what model a new
session on a crew starts with, highest tier first: the crew's own `model`, the
bound kiro agent's pinned model (skipped for the built-in `kirocrew` agent), the
global `agent.model`, then the installed agent file's model. A per-session pick
outranks all four and is not considered there.

The loader is defensive about hand-edited config: a non-string `model` or
`triggers` collapses to `""`, an unknown `reasoning_effort` collapses to inherit,
and a junk watchdog override collapses to `0`.

## Selection: the `select_crew` contract

`select_crew` has two modes, both answered as JSON by `_do_select_crew`.

`route_crew` resolves each trigger-matched member independently. Healthy matches
retain their rank and owned store. Matching members whose memory cannot be
resolved appear in `unavailable` with a bounded, path- and credential-redacted
reason. No healthy match and no trigger match are distinct outcomes; unavailable
memory never authorizes substitution with Global memory. A named `select_crew`
refusal returns `crew` and `error` without a bound store or routing activity.

**Roster** (`crew` omitted or empty):

```json
{"default_agent": "default",
 "crews": [{"name": "oncall", "triggers": "incident, prod outage"}],
 "guidance": "Select a crew ONLY when its triggers clearly and specifically match…"}
```

Three rules define that list, and each is load-bearing:

- A crew whose `triggers` is empty or whitespace is **omitted entirely**. There
  is no fallback to `description`: no triggers means not a routing candidate.
- `default_agent` is omitted, because it is the caller.
- The response carries `default_agent` and `guidance` so the model has an
  explicit fallback and a high-confidence bar rather than inferring one.

**Bind** (`crew` names a roster entry):

```json
{"crew": "oncall",
 "bound": {"kiro_agent": "oncall-agent", "workspace": "/…/oncall",
           "memory_store": "oncall-mem", "model": ""}}
```

An unknown name answers `{"error": "unknown crew '…'", "available": "…"}`. The
membership test against `cfg.agents` is the deny-by-default gate;
`SELECT_CREW_SCHEMA` deliberately does not impose a name grammar, because crew
creation only strips the name, so a stricter schema would list a crew in the
roster and then refuse to bind it.

A bind also records a routing-decision pointer through
`members.record_activity` with `via="select_crew"`. Two properties of that write
matter:

- The entry keys the session under `decided_in`, not `session`, because the
  decision is made in the parent session while the crew runs somewhere else. A
  consumer counting sessions a crew took part in therefore cannot miscount a
  session the crew never ran in.
- The caller's memory mode is resolved at the call, and only `persistent`
  sessions are recorded. An unreadable session degrades to the private spelling,
  so the failure mode is a missing entry, never a durably logged private session
  key.

These entries are **intent, not execution**: binding a crew does not oblige the
model to delegate to it, and no `via="spawn"` execution entry exists today.

## Delegating to a bound crew

Explicit member delegation uses `spawn_run(crew=<member>)`. The member alias
resolves its provider template and private memory together. The separate
`agent=` argument identifies a provider template, not a durable member identity;
it must not be used to infer access to a member's private memory.

A private member's own sub-tasks and schedules retain its store. It cannot select
Global V1 or a peer through `spawn_run` or `cron_add`. The trusted owner or Crew
coordinator assigns cross-member work; named tool delegation respects the
recipient's routing opt-in. HTTP spawning verifies the actual calling process
before accepting a parent session, and the run primitive checks the boundary
again before allocating a provider.

A named-but-unknown agent is **refused**, never silently answered by the default
agent, with the machine-readable code `agent_not_found`. That refusal is a
privilege boundary: the default agent frequently runs at broader approval, so a
typo'd or injected name falling back to it would be an escalation at the manager
primitive. An empty `agent` still means "use the default".

Crew Mode resolves the alias itself instead of relying on the coincidence:
`CrewOrchestrator._dispatch_agent` calls `resolve_agent_bindings` per dispatch
and passes `bindings.kiro_agent`. It returns the raw crew name when
`requested_resolved` is `False`, so an unknown crew is refused by
`_validate_agent` rather than quietly running the default agent under a stale
name, and it resolves an empty crew too so the concrete template stays inside
`capabilities.spawn.scopes.agents`.

## Crew Mode: the slot mode

`api_chat_slot_mode` admits `""`, `"orchestrator"` and `"crew"`
(`chat_folders._VALID_MODES`). `"crew"` is the only one with a capability gate,
and four refusals guard the switch:

| Code | Condition |
|---|---|
| `crew_unsupported_slot` | `crew_chat.is_crew_capable_slot_key(slot.key)` is false: the folded key is nothing but dots, ends in a dot (Win32 strips it, merging two slots onto one store), or is a Win32 reserved device basename |
| `member_mode_locked` | The slot is a member DM thread; member slots are born and retired only through the members thread endpoint |
| `remote_mode_unsupported` | `slot.executor == "remote"`: a mode-specific dispatch runs its tools on this machine, not on the bound peer |
| `crew_app_session_unsupported` | Refused at ingest for an app-owned slot, because dispatch does not preserve the app identity and the subagents would run outside the app's profile |

The switch is additionally refused while the slot has pending subagent work, in
either direction, keyed on `effective_session_key(slot)` so a channel-linked
slot is checked under the key its spawns actually ran on. The busy check runs
inside the state-wide slot-metadata transaction lock and fails closed.

The mode is preview-gated in the UI: `PREVIEW_CREW` (Settings → Developer →
Feature Previews) gates both doors, the Crew Members rail item and the sidebar's
"New Crew Mode chat" entry. The gate is on ingress only, so a session already in
crew mode keeps running when the flag goes off.

## Crew Mode data flow

A topic records its execution template and memory store. Generic topics inherit
the coordinator's store; explicit delegation resolves the chosen member's store
and refuses an unknown or unavailable member. Continuing or respawning a topic
retains that topic's identity, including after restart. A memory failure leaves
the message pending with an explanation and never respawns it under the
coordinator or Global Memory V1.

Store layout is `<data home>/crew/<folded slot key>-<8 hex digest>/` holding
`queue.json`, `topics.json`, `forwards.json` and `slot_key`. The digest is taken
over the exact key, so two keys differing only in case get distinct directories
on a case-insensitive filesystem; the readable half is capped, so an overlong
slot name cannot produce a path the filesystem refuses. Because the directory
name is a fold plus a digest it cannot be decoded back, which is why the exact
key is written into `slot_key` and read from there on resume.

One request walks this path:

1. `CrewOrchestrator.ingest` takes a per-slot `asyncio.Lock` before any await,
   so the order requests reach the lock is the order they surface.
2. The entry is appended and `queue.json` is awaited **by name** before anything
   the user can see. A failed write rolls the in-memory entry back, so a 500
   leaves nothing queued.
3. The echoed user message and a templated ack are posted durably. A
   non-durable transcript row is logged loudly and the request still runs: the
   queue is the record, and the transcript is its mirror.
4. A decision pass is scheduled. It is single-flight per slot, prompts a
   tool-free background call with a rendered state snapshot, and accepts only a
   dict carrying an `actions` list, so braced JSON in surrounding prose is not
   mistaken for the payload.
5. Each action is applied: `route` an existing topic, `spawn` a new one, `hold`
   until a running topic finishes, `steer` a droppable advisory correction into
   in-flight work, `ask` one short disambiguating question, or `meta` answer a
   question about the topics themselves.
6. A `spawn` persists a stable `dispatch_id` and awaits a queue-only barrier
   before starting the run, so a crash between the spawn and the accepted-state
   write is adopted on restart instead of re-executing the task. Continuations
   and digest respawns use the same barrier. Agent or memory-binding resolution
   failures before dispatch reopen the previous queue state and remove the
   unmatched dispatch identity, allowing a retry without a gateway restart.
   Completion-triggered held dispatch awaits its final queue/topic save even
   when resolution fails, so the reopened request is durable before return.
7. Sub-tasks carry `_SUB_TASK_SUFFIX`, which forbids nested subagents and
   requires a `<<<SUMMARY … >>>` block; `on_subagent_done` extracts it with
   `_SUMMARY_RE` and forwards it verbatim with mechanical attribution. A result
   arriving while the tab is closed is persisted to `forwards.json` and
   re-delivered on reopen.

Caps and timeouts, all named constants in `crew_chat.py`: `_DECISION_TIMEOUT`
45s, `_DECIDE_MAX_ATTEMPTS` 3 (a pass that settles nothing eventually fails the
entry visibly, because silence is the one outcome the user cannot act on),
`_QUEUE_TERMINAL_CAP` 200 and `_TOPIC_IDLE_CAP` 200 (a running or held topic is
never pruned).

## Failure modes

- **A broken store file is fatal, not empty.** `CrewStore._load` treats only
  `FileNotFoundError` as "nothing enqueued yet". Collapsing every read error to
  `[]` made the next `save()` write that emptiness back, erasing pending
  requests and undelivered forwards. A wedged slot is recoverable; an erased
  queue is not.
- **Writes never block the loop.** A two-lock split keeps the sequence
  bookkeeping off the disk path, so a slow filesystem cannot stall unrelated
  chats; `wait_writes` and `wait_for` are how a caller awaits a specific write.
- **A permanent delete is remembered.** `purge_slot` marks the key so an
  in-flight decision pass cannot dispatch into a deleted conversation, and only
  a fresh request clears the mark.
- **Restart reconciliation adopts, it does not re-run.** `resume_persisted_slots`
  reads store directories, recovers each exact key, and reconciles claimed
  entries against live runs and durable evidence.
- **A temporary slot does not leak memory.** Crew dispatch forwards the slot's
  `blocks_reads` flag, so a temporary crew slot injects no stored memory or
  lessons into the subagents it spawns.
- **An unreadable member rules file stops the member.** `read_member_rules`
  raises `MemberRulesUnreadable` for a file that exists but cannot be parsed, and
  `member_turn_context` guarantees every member turn either delivers the current
  section or runs the standalone fail-closed rules read.

## Boundaries

- A crew's `triggers` is free text read by a model. It is not a matcher, and no
  regex interprets it.
- `POST /api/agents` requires an explicit `kiro_agent`; the silent `"kirocrew"`
  default is refused, because it made every template-less crew an alias for the
  default agent. A template absent from the installed listing is accepted with a
  warning rather than refused, since an edition may resolve a row the listing
  cannot see.
- A credential-shaped crew name is refused at creation, and roster values are
  masked for every caller but the owner. An already-stored name is not renamed
  retroactively, which is why the owner keeps reading it verbatim: a name must
  be legible to be renamed.
- `kirocrew`, `kirocrew-conductor`, `kirocrew-pipeline-conductor` and
  `kirocrew-security-conductor` are in `UNADVERTISED_AGENTS`, so they never
  appear in a rendered roster.

## Tests that pin this

| Test | What it holds |
|---|---|
| `test/test_select_crew.py` | Roster excludes the default crew and every triggerless crew, carries `default_agent` plus guidance; a named crew returns its bindings; an unknown name returns `error` plus `available`; the schema accepts spaces and dots in a crew name |
| `test/test_crew_chat.py` | Store round-trip and persistence, dots-only and dot-bearing key handling, the durable-read boundary, ingest enqueue/ack/schedule ordering, write-failure rollback, and that an unreadable store file is not read as empty |
| `test/test_crew_http.py` | Interleaved messages end to end, completion forwarding and held dispatch, mode-switch refusals (bad mode, app-owned slot, unstorable name), the busy/idle switch guard, and that a refused ingest is not a 200 |
| `test/test_crew_store_read_retry.py`, `test/test_crew_store_write_retry.py` | Retrying reader and writer behaviour around the atomic store files |
| `test/test_crew_reasoning_effort.py` | Per-crew effort reaches a crew dispatch |
| `test/test_members.py`, `test/test_members_dm_thread.py` | Slug validation and containment, activity recording and dedupe, DM-binding canonicality, rules and briefing reads |
| `test/test_chat_send_agent_model_default.py` | The crew model default a new session starts on |

## Design of record

[`../../request-for-change/rfc-orchestrator-chat-sessions.md`](../../request-for-change/rfc-orchestrator-chat-sessions.md)
carries the intent behind the Crew Mode control plane. This spec is the contract
for what ships.
