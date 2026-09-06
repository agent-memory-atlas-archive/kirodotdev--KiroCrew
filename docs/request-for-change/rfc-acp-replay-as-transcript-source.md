---
title: ACP replay as the transcript source — demote Kiro Crew's JSONL to a sidecar
status: in-progress
author: zezhexu
created: 2026-09-06
last-audited: 2026-09-06
audited-at: d4c2cbf22
doc-pr:
implementation-prs: ["#8862"]
tracking-issues: []
supersedes: []
superseded-by: []
---
# RFC: ACP replay as the transcript source — demote Kiro Crew's JSONL to a sidecar

- Status: **in-progress** — P0 is [PR #8862](https://github.com/kirodotdev/KiroCrew/pull/8862) (`dashboard.replay_from_acp`, default off). P1–P3 are proposals.
- Author: zezhexu (drafted with Kiro)
- Created: 2026-09-06 · Audited at main `d4c2cbf22` (the P0 branch base; the branch itself is #8862)
- Related: [`rfc-append-only-session-transcript.md`](rfc-append-only-session-transcript.md) (fixes how the JSONL is *written*; this RFC changes what the JSONL is *for*), [`rfc-crew-agent-sdk-boundary.md`](rfc-crew-agent-sdk-boundary.md) (the replay fold lives behind the SDK driver seam), `docs/system-specs/modules/acp-client.md` § "Transcript-replay capture".
- Evidence: the fidelity measurement summarized in §9; the full report and probe scripts live in the operator workspace at `research/acp-replay/acp-replay-fidelity.md` (not in this repository).

## 1. Summary

Kiro Crew keeps two records of every conversation. kiro-cli persists the whole session and replays it on `session/load`; Kiro Crew independently splits the live `session/update` stream into JSONL rows and renders *those*, while the runtime's reader loop counts and discards the replay. This RFC proposes making the agent's replay the source of truth for everything the agent saw, and shrinking Kiro Crew's JSONL to a **sidecar** carrying only what the agent never saw — approval decisions, notices, provenance, session metadata — anchored to the replay by message and tool-call identity. P0 (the flagged read-path prototype) is measured and in review; P1–P3 phase out the duplicate write.

## 2. Problem — the transcript is dual-written and the agent's copy is thrown away

Measured on the P0 branch (#8862) at main `d4c2cbf22`:

- `AcpRuntime.load_session` (`src/kiro_crew/acp/runtime.py`) issues `session/load`. kiro-cli answers by streaming the entire prior conversation as `session/update` notifications. With the flag off the reader loop routes every one of them to `AcpRuntime._note_dropped_frame` — "Counted, not logged per frame: this is the measured flood (transcript replay during session/load …)".
- Meanwhile the live path parses each `session/update` chunk into a row and appends it to the slot's JSONL (`_save_slot_to_history`, see `rfc-append-only-session-transcript.md` §2 for that write path). A resumed session renders that JSONL.
- kiro-cli's own store (`~/.kiro/sessions/cli/<sid>.jsonl`) is never read by Kiro Crew. One development host measured 1,719 files / 5.41 GB of it on 2026-08-19.

Four costs follow from holding two truths:

1. **Drift.** Every operation that rewrites history — compaction, rewind, regenerate, edit-and-resend, switch-variant, `/clear` — must be mirrored on both sides. PR #8862's review found the `/clear` case leaking retained replay frames back into a wiped transcript (fixed in round 8: the `EVENT_CLEAR_STATUS` branch of `dashboard/chat_runner.py` now calls `chat_utils.discard_slot_replay`); the same class of bug exists wherever the two records are edited independently.
2. **Lossy reconstruction.** Rebuilding rows from streamed fragments is inherently worse than the agent's record. On the v1/v2 engines chunks carry no `messageId`; thinking is broadcast on a separate topic and never persisted, so after a restart the JSONL has no reasoning to show, and its position relative to tool calls was only ever array order.
3. **Two parsers per protocol change.** `parse_session_update` (live) and the JSONL reader both have to understand every field kiro-cli adds; each `_meta.kiro.*` extension is implemented twice or not at all.
4. **Unused capability.** The agent already stores, compacts, forks and replays the conversation. Kiro Crew re-implements storage and gets none of the fork / checkpoint / history-pull features the v3 engine advertises.

## 3. Why Kiro Crew's transcript must stop being the source of truth

Continuing to dual-write is not a neutral default:

- **The drift surface grows with every feature.** Each of the six rewrite operations above is an invalidation point that must call `chat_utils.discard_slot_replay` (regenerate, edit-and-resend's `_commit_live_state` and switch-variant in `chat_regenerate.py`; rewind's `_commit_live_state` in `chat_rewind.py`; the `EVENT_CLEAR_STATUS` branch in `chat_runner.py`). P0 needs all five call sites *because* two records exist; a single record needs none.
- **The reconstruction cannot be repaired from the client side.** Thinking segmentation, message identity and turn boundaries are facts the agent knows and the wire fragments do not carry on v1/v2. No amount of client-side re-anchoring recovers an identity that was never sent; only the agent's record has it.
- **Protocol upgrades cost double.** The v3 engine already stamps `_meta.kiro.{messageId,timestamp,replay}` on every replayed frame and preserves `title`/`locations`/`content`. Consuming that requires touching one reducer if replay is the source, two if JSONL stays authoritative.
- **Storage is paid twice.** kiro-cli's store grows unbounded regardless; Kiro Crew's JSONL duplicates the bulk (assistant text, tool rawInput/rawOutput) that the replay already provides.

## 4. Why the agent should own it — the protocol already says so

ACP's `session/load` contract (protocol version 1, the integer every kiro-cli engine answers; see §9 table 1) obliges the agent to replay the prior conversation as `session/update` notifications in the **same shape as live** before responding. The ACP v2 draft generalizes this as `session/resume` + `replayFrom: start`. Either way the client needs exactly one reducer, and message identity is the agent's to assign.

Measured fidelity (kiro-cli 2.21.0, §9):

- Synthetic probe, all three engines: assistant text identical to live (386/405/398 chars = live), thinking text identical on v2 (115 = 115 chars), tool `toolCallId` / `rawInput` / `rawOutput` / final `status` identical, kind order within a turn identical.
- Real dashboard session (690 JSONL rows, 395 tool calls, 100 nudge cycles, one auto-compaction) replayed in 1.8 s: **395/395** tool rows matched by `tool_call_id`, **173/177** assistant rows exact (3 minor drift, 1 was a compaction notice), **113/115** prompts found wrapped inside the replayed `user_message_chunk`. No truncation at the compaction point.
- v1/v2 vs v3 (§9 table 2): v3 adds `_meta.kiro.messageId` + `timestamp` + `replay:true` on every frame, keeps humanized tool `title`, `locations`, shell `content` preview and `_meta.kiro.toolName`; v1/v2 drop all of those. Nothing in v3's extra fidelity is something Kiro Crew's JSONL could have supplied better — it is the agent's own presentation layer being preserved.

## 5. The dividing line — seen by the agent vs. not

The split is **not** "messages vs. non-messages". It is:

| The agent saw it → replay owns it | The agent never saw it → sidecar must record it |
|---|---|
| assistant text, thinking, their order | approval / permission decisions and who made them |
| every tool call: id, rawInput (incl. `__tool_use_purpose`), rawOutput, final status | `role=notice` / `error` rows (transient retry, tool-blocked, recovery) |
| the fact and order of user turns (as the full assembled prompt) | compaction markers and context-usage samples (`_kiro.dev/metadata`) |
| on v3: message identity, timestamps, humanized tool presentation | session organization: title, folder, tags, JSONL header line |
| | cross-channel identity, `sendId`, steer / nudge / `injectKind` provenance, cron and subagent labels |
| | the CLEAN user text (the replay carries the wrapper; the JSONL row is exact) |
| | `turn_stats`, `file_changes`, `mid` (until P1 replaces it) |
| | OAuth cards (should become live state, not history — P3) |
| on v1/v2 only: humanized tool `title`, `kind` for non-read/execute tools, `locations`, stdout preview | |

Everything in the right-hand column is a fact Kiro Crew produced or observed on the client side of the wire. Nothing in the left-hand column is.

## 6. Proposal and phasing

The JSONL is demoted from transcript to **sidecar + index**: rows the agent never saw, plus per-turn anchors (`tool_call_id` today, `_meta.kiro.messageId` on v3) so they can be re-inserted at the right position of the replayed stream.

### P0 — flagged read path (this PR, #8862)

- `dashboard.replay_from_acp` (default off). `AcpRuntime(capture_replay=True)` retains the load-time `session/update` frames per sid (`_REPLAY_CAPTURE_MAX_FRAMES = 20_000`, `_REPLAY_CAPTURE_MAX_BYTES = 64 MiB`; overflow discards the whole capture, never a truncated prefix) on `AcpSessionHandle.replay_updates` (`_capture_replay_frame` / `_take_replay_capture` in `runtime.py`). `dashboard/chat_replay.py` folds them through the same `parse_session_update` the live path uses and overlays the JSONL sidecar (`merge_replay_transcript`).
- **Nothing stops being written.** Only what a resumed session *reads* changes.
- Exit criteria: flag off is byte-identical to main (no new fields on default installs); flag on renders the real 690-row session with every sidecar row present and in order; all six rewrite operations invalidate the retained frames; CI green.

### P1 — identity anchor on v3

- When frames carry `_meta.kiro.messageId`, anchor sidecar rows and the merge on it instead of prompt-text containment and `tool_call_id` ordinals; persist `messageId` in the JSONL row and retire Kiro Crew's `mid` for rows that have one.
- Drop the humanized-title / `locations` / `kind` sidecar when the engine preserves them.
- Exit criteria: on v3 the merge performs zero containment matches; a rewind by `messageId` and a rewind by JSONL row agree on the same cut point.

### P2 — stop writing the body

- Stop persisting assistant text, thinking and tool rawInput/rawOutput to the JSONL; write only sidecar rows and anchors.
- **Prerequisites (blocking):** (a) search and memory consolidation, which read the JSONL body today, switch to a lazily built index over the replay or a retained body cache; (b) a replay must be obtainable for a session the gateway does not currently hold — via the gateway's own `session/load` at resume (P0 mechanism), or the v3 `_kiro/session/history` pull API once its paging is understood (§9 table 1).
- Exit criteria: a fresh install running a full day writes no `role=assistant`/`role=tool` body rows; search results and consolidation output are unchanged against a recorded corpus.

### P3 — OAuth and other live state out of history

- MCP OAuth cards and similar interactive state move to a live-state channel; the transcript no longer carries them.
- Exit criteria: no `mcp_oauth` rows are written; an in-flight OAuth survives a tab reload via live state.

## 7. Benefits

- **One truth, guaranteed by the protocol** rather than by five invalidation call sites kept in sync by review.
- **One parser.** `parse_session_update` is the only reducer for live and replayed frames.
- **Thinking and segmentation stop being a client-side reconstruction problem** — the agent's record carries them, and on v3 carries their identity.
- **JSONL shrinks by an order of magnitude** in both bytes and schema (sidecar rows are small and few: in the 690-row session, 8 rows had no replay counterpart).
- **v3 capabilities become usable directly**: `fork` (with `messageId`), checkpoints, `_kiro/session/history`, `session/list`.

## 8. Alternatives considered

**(a) Keep dual-writing.** Zero migration cost, but §3 applies in full: every new rewrite path is a new drift bug, every protocol field is implemented twice, and thinking after a restart stays unrecoverable on v1/v2. This is the status quo the P0 review already found wanting (`/clear` leak).

**(b) Trust only the JSONL; never read the replay.** Simplest today, but it forfeits `messageId` forever — the JSONL cannot be given an identity the wire never sent it. It also has no future: the v3 engine keeps its own store and writes nothing under `sessions/cli`, so there is no file to fall back to reading.

**(c) Move everything into the agent's store, no sidecar.** The right-hand column of §5 has nowhere to live: the agent never saw an approval decision, a notice, a cron label or a folder assignment. And `session/list` has no search, so session discovery would regress.

**(d) Push the sidecar into `_meta` and let kiro-cli store it.** The ACP spec only says an agent SHOULD round-trip `_meta`; nothing guarantees it survives compaction or fork. More fundamentally, Kiro Crew's product metadata would live in a store it does not control — snapshot/restore, incognito sessions and write-boundary redaction (`_redact_at_write_boundary`) all depend on owning the sidecar file.

## 9. Evidence

Full report: operator workspace `research/acp-replay/acp-replay-fidelity.md` (kiro-cli 2.21.0, 2026-09-06; scripts `acp_replay_probe.py`, `analyze_replay.py`, `align_kirocrew_session.py`, `probe_live_sid.py`, raw frames under `out/`). The load-bearing rows are reproduced here so the document stands alone.

**Table 1 — protocol reality per engine**

| question | v1 | v2 (kiro-cli default; what Kiro Crew runs) | v3 |
|---|---|---|---|
| `initialize.protocolVersion` answered | `1` | `1` | `1` (rejects a dated string with `-32602`) |
| `session/resume` + `replayFrom` (ACP v2 draft) | `-32601` | `-32601` | `-32601` |
| `agentCapabilities.loadSession` | true | true | true |
| `sessionCapabilities` | `{}` | `{}` | `list`, `fork` (with `messageId`), `_meta.kiro.replayMarking`, `checkpoints`, `_kiro/session/history` |
| per-frame `_meta.kiro.messageId` / `timestamp` / `replay:true` | none | none | every replayed frame |
| transcript on disk | `~/.kiro/sessions/cli/<sid>.jsonl` | same | none under `sessions/cli` |
| `session/load` for a sid another process holds | refused, 0 frames | same | same |

**Table 2 — live stream vs. replay, synthetic probe**

| field | v1 | v2 | v3 |
|---|---|---|---|
| agent text identical | ✅ 386 = 386 | ✅ 405 = 405 | ✅ 398 = 398 |
| thinking text identical | n/a | ✅ 115 = 115 (18 chunks → 2) | n/a |
| text chunks merged | 32 → 7 | 34 → 8 | 33 → 4 |
| user prompt replayed as `user_message_chunk` | ✅ | ✅ | ✅ |
| tool id / rawInput / rawOutput / final status | ✅ | ✅ | ✅ |
| tool `kind` | read/execute ✅, `other` → null | same | ✅ |
| tool `title` | raw tool name | raw tool name | humanized ✅ |
| tool `locations`, `content` preview, `_meta.kiro.toolName` | dropped | dropped | ✅ |
| `messageId` on chunks | 0 | 0 | 8 / 8 |

**Table 3 — real Kiro Crew session (690 JSONL rows, v2) aligned with its replay (1,557 frames, 1.8 s)**

| JSONL row class | count | in replay | how |
|---|---|---|---|
| tool | 395 | 395 / 395 | exact `tool_call_id` == `toolCallId` |
| assistant text | 177 | 173 exact, 3 minor drift, 1 sidecar (compaction notice) | merged `agent_message_chunk` |
| user | 11 | 10 (wrapped), 1 missing (a steer after the last turn) | inside `user_message_chunk` after `[CURRENT USER REQUEST -- respond to this]` |
| nudge | 100 | 100 / 100 (wrapped) | same; JSONL `—` vs wire `--` |
| inject (`synthesis`, `user_replay`, `recovery`) | 3 | 3 (wrapped) | they were prompts |
| inject `[Tool blocked …]` | 2 | 0 | delivered as a tool result |
| error (`transient_retry`) | 2 | 0 | never reached the agent |
| compaction notice | 1 | 0 | replay still covers everything before the compaction point |

**Screenshots** — same session, same pod, flag on vs. off, from the P0 PR body: `temp-screenshots/acp-replay/replay-on.png`, `replay-off.png`, `settings-toggle.png` (committed on the branch). ON shows the model's thought-process block and the tool card's purpose line and file chip; OFF shows the JSONL's collapsed "Worked through 1 step" and no reasoning.

**On/off comparison demo** — TODO: a scripted capture exists in the operator workspace (`mockups/acp-replay-demo/`) but has no README yet; link it here once written.

## 10. Non-goals and risks

Non-goals:

- P0 deletes no JSONL write and changes no persistence format.
- This RFC does not change how the JSONL is written (that is `rfc-append-only-session-transcript.md`), nor the model-context ownership, which stays with kiro-cli.

Risks and known limits:

- **v1/v2 presentation loss.** Humanized `title`, `locations`, stdout preview and `kind` for non-read/execute tools must keep coming from the JSONL until the engine preserves them (§9 table 2). Rendering tool cards from replay alone regresses the card header.
- **Replay availability.** `session/load` refuses a sid another process holds and requires the agent process to be alive; frames exist only at the gateway's own resume. A tab opened before the first post-restart prompt still renders JSONL; the flip happens on the next detail fetch.
- **User text is wrapped.** The replayed prompt is the full assembled context (2–144 KB per turn in the real session). The clean text needs the JSONL row or the marker strip (113/115 prompts in the real session).
- **kiro-cli's store has no pruning** and its `Compaction` records snapshot the whole history; making it authoritative does not make that disk problem go away.
- **Harness parity.** `replay_updates` / `discard_replay` are declared on `LLMProvider` with `None`/no-op defaults so non-ACP harnesses stay unaffected; any P2 change to the write path must keep that symmetric.
- **Search and consolidation trade-off (P2).** Lazy indexing over replay costs a `session/load` per indexed session; a retained body cache re-introduces a second copy. Which one is chosen is an open question for P2, not for this document.

## 11. Open questions

1. P2 prerequisite (b): can `_kiro/session/history` on v3 page a whole session for a sid the gateway does not hold? Its cursor shape was not probed.
2. Should the v1/v2 engines be asked to stamp `replayMarking`-style `_meta.kiro.{messageId,timestamp,replay}` and preserve `title`/`locations`? That delta is the entire difference between "replay + thin sidecar" and today's parallel JSONL on the default engine.
3. Where does the v3-only `fetch_cloud_config` pre-turn tool call (§9, one extra tool per replay) get filtered — in the fold, or in the sidecar merge?
