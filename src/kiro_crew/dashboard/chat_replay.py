"""Render a resumed session's transcript from kiro-cli's ``session/load`` replay.

Prototype behind ``dashboard.replay_from_acp`` (default off).

Today a resumed dashboard session shows the JSONL Kiro Crew wrote turn by turn,
while the frames kiro-cli replays during ``session/load`` -- the agent's OWN
record of the conversation -- are counted and dropped by the runtime's reader.
With the flag on, :class:`~kiro_crew.acp.runtime.AcpRuntime` keeps those frames
(``AcpSessionHandle.replay_updates``) and this module turns them into transcript
rows through the SAME parser the live path uses -- reached through
:func:`kiro_crew.agent_sdk.fold_replay_updates`, the SDK's driver seam, so this
dashboard module never imports the ACP layer -- then overlays from the JSONL only
what the agent never saw.

Measured split (research/acp-replay, kiro-cli 2.21.0, engines v1/v2/v3):

* replay is authoritative for assistant text, thinking text, and every tool
  call's id / rawInput / rawOutput / final status;
* replay carries each user turn as the FULL assembled prompt (system prompt,
  context blocks, the user's words after ``[CURRENT USER REQUEST -- respond to
  this]``), so the clean user text comes from the JSONL row when one matches;
  a turn the DASHBOARD dispatched (a cron notification, a sub-agent completion,
  a recovery re-send, the synthesis prompt) is persisted as an ``inject`` /
  ``subagent`` row rather than ``user``, and is matched as that turn's prompt
  the same way (see ``_is_prompt_row``);
* on the v1/v2 engines a replayed tool call's ``title`` collapses to the raw
  tool name and ``locations`` are gone, so the humanized title / purpose / kind
  are borrowed from the JSONL tool row with the same ``tool_call_id``;
* notices, errors, approvals, steer provenance, in-turn injects, compaction
  banners and cron/subagent labels exist only in the JSONL -- they are the
  sidecar and are re-inserted at their turn, anchored on the last replayed row
  (tool call or answer text) they followed in the JSONL, if any.

Every emitted row is tagged ``meta.source`` (``acp_replay`` | ``jsonl``) so the
dashboard can show where it came from. The JSONL keeps being written -- this
changes what a resumed session READS, never what it persists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

from kiro_crew.agent_sdk import fold_replay_updates
from kiro_crew.context import _MULTIBYTE_TABLE, USER_REQUEST_HEADER
from kiro_crew.dashboard.chat_utils import (
    _MAX_TOOL_PURPOSE,
    _redact_tool_field,
    effective_session_key,
)
from kiro_crew.quick_prompts import quick_prompt_header
from kiro_crew.security import redact_credentials, redact_exfiltration_urls

logger = logging.getLogger(__name__)

SOURCE_REPLAY = "acp_replay"
SOURCE_JSONL = "jsonl"

#: The assembled prompt marks the human's words with :data:`USER_REQUEST_HEADER`
#: (owned by ``context.py``, imported rather than respelled); the replayed
#: ``user_message_chunk`` is the whole prompt, so the clean text is whatever
#: follows the LAST occurrence. Compared dash-folded because the outbound
#: sanitizer rewrites the header's em dash to ASCII on the wire. Measured present
#: in 113/115 prompts of a real session; the rest were short raw prompts.
#: Bound on the raw prompt kept when no marker is present and no JSONL row
#: matches: a multi-KB context blob is not a chat bubble.
_RAW_PROMPT_KEEP = 4_000
#: JSONL roles that mean "this row started a prompt the agent received".
_PROMPT_ROLES = frozenset({"user", "nudge"})
#: ``inject`` rows that START a turn rather than annotate one. The queue drain
#: writes exactly these ``meta.injectKind`` values on the row it appends right
#: before dispatching the prompt (``chat_runner``: a cron notification, a
#: recovery re-send, a user-message replay, the subagent synthesis prompt).
#: Every other ``inject`` -- a policy notice, a reconcile note, a mid-turn
#: annotation -- is delivered INSIDE a turn and stays sidecar. Provenance, not
#: text: ``cls`` is not persisted for this role, ``meta`` is.
_TURN_STARTING_INJECT_KINDS = frozenset({"cron", "recovery", "user_replay", "synthesis"})


def _fold_dashes(text: str) -> str:
    """ASCII-fold punctuation with the SAME table the outbound prompt goes through.

    ``build_message`` translates the assembled prompt with ``_MULTIBYTE_TABLE``
    (dashes, curly quotes, ellipsis, arrows, non-breaking space, ``x``), while
    the JSONL ``user`` row keeps the text as typed. Folding only dashes here
    would leave a prompt with a smart apostrophe failing both the exact and the
    containment test, and its turn would render twice -- once from the replay
    and once as an unmatched JSONL group.
    """
    return (text or "").translate(_MULTIBYTE_TABLE)


def _norm(text: str) -> str:
    """Whitespace-fold and ASCII-fold punctuation so JSONL text matches the wire.

    Kiro Crew rewrites multi-byte punctuation to ASCII before a prompt reaches
    kiro-cli, so without folding both sides every such row looks unmatched.
    """
    return " ".join(_fold_dashes(text).split())


def _redact_text(text: str) -> str:
    safe, _ = redact_exfiltration_urls(text or "")
    safe, _ = redact_credentials(safe)
    return safe


def clean_user_text(prompt: str) -> str:
    """The human's words out of an assembled prompt (see :data:`USER_REQUEST_HEADER`)."""
    if not prompt:
        return ""
    # The marker arrives either as typed (em dash) or ASCII-folded by
    # ``build_message``. Search the ORIGINAL prompt for each spelling and slice
    # with the matched spelling's own length: ``_MULTIBYTE_TABLE`` changes
    # character counts (``…`` -> ``...``, ``✓`` -> ``[x]``), so an index found
    # in a folded copy is not an index into the prompt. The rightmost match
    # wins, as before -- the last header is the live request.
    idx, width = -1, 0
    for marker in {USER_REQUEST_HEADER, _fold_dashes(USER_REQUEST_HEADER)}:
        found = prompt.rfind(marker)
        if found > idx:
            idx, width = found, len(marker)
    if idx >= 0:
        return prompt[idx + width :].strip()
    if len(prompt) > _RAW_PROMPT_KEEP:
        return prompt[-_RAW_PROMPT_KEEP:].strip()
    return prompt.strip()


def _meta_of(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def _row_text(row: dict[str, Any]) -> str:
    content = row.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return ""


def _is_prompt_row(row: dict[str, Any]) -> bool:
    """Did this JSONL row start a prompt the agent received?

    ``user`` / ``nudge`` rows (not steers, which land INSIDE the running turn),
    plus the automation-dispatched prompts the queue drain writes under other
    roles: a ``subagent`` row (a sub-agent completion handed to the agent as its
    next turn) and an ``inject`` row whose ``injectKind`` names a turn-starting
    dispatch. Without these, a cron / subagent / recovery turn's JSONL prompt
    stays in the PREVIOUS group while the replay opens a new turn for it -- the
    prompt then renders twice, once in each place, and its answer under the
    wrong prompt.
    """
    role = row.get("role")
    meta = _meta_of(row)
    if role in _PROMPT_ROLES:
        # A steer is delivered INSIDE the running turn, not as its own prompt.
        return not meta.get("steer")
    if role == "subagent":
        return True
    if role == "inject":
        return str(meta.get("injectKind") or "") in _TURN_STARTING_INJECT_KINDS
    return False


def _jsonl_turns(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, list[dict[str, Any]]]]:
    """Group JSONL rows as (prompt_row | None, rows_after_it_until_next_prompt)."""
    groups: list[tuple[dict[str, Any] | None, list[dict[str, Any]]]] = []
    pre: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] | None = None
    for row in rows:
        if _is_prompt_row(row):
            cur = []
            groups.append((row, cur))
            continue
        if cur is None:
            pre.append(row)
        else:
            cur.append(row)
    if pre:
        groups.insert(0, (None, pre))
    return groups


#: Below this many normalized characters a JSONL prompt is matched by EQUALITY
#: only: a short prompt ("a", "ok", "/plain") is a substring of almost any
#: later assembled prompt, so containment would pair the first answer with a
#: later question and reorder the transcript.
_CONTAINMENT_MIN_CHARS = 40

#: :func:`_match_prompt`'s answer when two or more JSONL turns are equally the
#: replayed turn (identical prompts, tied identity). Distinct from ``-1`` (no
#: JSONL prompt at all, the replay renders its own): the caller drops the
#: replayed turn and lets the JSONL, which records both, stand.
_AMBIGUOUS = -2


def _assistant_texts_match(replay_text: str, jsonl_text: str) -> bool:
    """One normalized assistant text is the other (equal, or one contains the
    other once it is long enough that containment is not a coincidence)."""
    if not jsonl_text:
        return False
    return replay_text == jsonl_text or (
        len(jsonl_text) > 40 and (jsonl_text in replay_text or replay_text in jsonl_text)
    )


def _turn_identity_score(turn: dict[str, Any], rest: list[dict[str, Any]]) -> int:
    """How many of ``rest``'s rows (a JSONL turn body) name something this
    replayed turn actually carries: a tool row with one of its ``tool_call_id``s,
    or an assistant row whose text is one of its answers. 0 = nothing in common."""
    score = 0
    replay_answers = [_norm(r["content"]) for r in turn["rows"] if r["role"] == "assistant"]
    for row in rest:
        role = row.get("role")
        meta = _meta_of(row)
        if role == "tool":
            if str(meta.get("tool_call_id") or "") in turn["tool_index"]:
                score += 1
        elif role == "assistant":
            text = _norm(_row_text(row))
            if any(_assistant_texts_match(rt, text) for rt in replay_answers):
                score += 1
    return score


def _match_prompt(
    groups: list[tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    start: int,
    turn: dict[str, Any],
) -> int:
    """Index of the JSONL prompt group at/after ``start`` that is this turn's prompt.

    Two passes, both on dash-folded / whitespace-folded text. First an EXACT
    match of the JSONL row against the human's slice of the replayed prompt
    (:func:`clean_user_text`), which is what the writer sent and is exact for
    plain prompts and nudges. Only when nothing is equal -- a steer appended to
    the prompt, a quick-prompt macro expanded in place -- is containment inside
    the whole replayed prompt tried, and only for rows long enough that a
    substring hit cannot be a coincidence.

    The prompt text alone cannot tell two identical prompts apart ("continue",
    a repeated question): when a pass yields several candidates, the one whose
    JSONL body shares an identity with this turn -- a ``tool_call_id`` the
    replay carries, an answer the replay has -- wins, so a replay that omits
    the FIRST of two identical turns does not hand the second turn's thinking
    and tools to the first. When the best candidates TIE there is no evidence
    for either, and guessing would put replay-only rows under the wrong
    prompt: the turn is reported :data:`_AMBIGUOUS` and the caller keeps the
    JSONL for it.
    """
    prompt_text = turn["prompt_text"]
    hay = _norm(prompt_text)
    if not hay:
        return -1
    clean_raw = _fold_dashes(clean_user_text(prompt_text)).strip()
    clean = _norm(clean_raw)
    exact: list[int] = []
    for j in range(start, len(groups)):
        prow = groups[j][0]
        if prow is None:
            continue
        needle_raw = _fold_dashes(_row_text(prow)).strip()
        if not needle_raw:
            continue
        # Equal, or the human's words followed by a LINE BREAK: the writer
        # appends per-turn guidance after the request on its own lines, so a
        # newline boundary is exact where a bare prefix would let "a" claim
        # "a longer question".
        if _norm(needle_raw) == clean or clean_raw.startswith(needle_raw + "\n"):
            exact.append(j)
            continue
        # A persisted quick prompt ("/plain") reached the agent as its expanded
        # macro, which opens with the [QUICK PROMPT <token>] header both derive
        # from; the token itself is never in the wire text.
        qp_header = quick_prompt_header(needle_raw)
        if qp_header and clean_raw.startswith(qp_header):
            exact.append(j)
    candidates = exact
    if not candidates:
        for j in range(start, len(groups)):
            prow = groups[j][0]
            if prow is None:
                continue
            needle = _norm(_row_text(prow))
            if len(needle) >= _CONTAINMENT_MIN_CHARS and needle[:120] in hay:
                candidates.append(j)
    if not candidates:
        return -1
    if len(candidates) == 1:
        return candidates[0]
    scored = sorted(
        ((_turn_identity_score(turn, groups[j][1]), j) for j in candidates), reverse=True
    )
    if scored[0][0] == scored[1][0]:
        # Two JSONL turns are equally this turn: nothing distinguishes them, so
        # picking either would render replay-only rows (thinking, tool detail)
        # under a prompt that may not be theirs. The JSONL is the record of
        # BOTH turns; the caller renders it and drops this replayed turn.
        return _AMBIGUOUS
    return scored[0][1]


def _tag(row: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    meta = dict(out.get("meta") or {}) if isinstance(out.get("meta"), dict) else {}
    meta["source"] = source
    out["meta"] = meta
    return out


def _overlay_turn(
    turn: dict[str, Any], prompt_row: dict[str, Any] | None, jsonl_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One turn's final rows: replay rows enriched + sidecar rows anchored.

    ``turn`` is one element of :func:`fold_replay_updates`'s result:
    ``{"prompt_text", "rows", "tool_index"}``.
    """
    # 1. the prompt bubble: JSONL row (clean text, ts, mid, steer/nudge flags) wins
    if prompt_row is not None:
        head = _tag(prompt_row, SOURCE_JSONL)
    else:
        head = {
            "role": "user",
            "content": _redact_text(clean_user_text(turn["prompt_text"])),
            "cls": "msg msg-u",
            "meta": {"source": SOURCE_REPLAY},
        }
    out: list[dict[str, Any]] = [head]

    rows = [dict(r, meta=dict(r["meta"])) for r in turn["rows"]]
    tool_pos = {tcid: i for tcid, i in turn["tool_index"].items()}

    # 2. enrich replayed tool rows from their JSONL twin; collect sidecar rows
    #    with an anchor: the index of the last replay row (tool OR assistant
    #    text) the JSONL matched before them. Anchoring on tools alone put a
    #    compaction banner that FOLLOWED the answer in the JSONL before it.
    sidecar: list[tuple[int | None, dict[str, Any]]] = []
    last_anchor: int | None = None
    matched_assistant: set[int] = set()
    for row in jsonl_rows:
        role = row.get("role")
        meta = _meta_of(row)
        if role == "tool":
            tcid = str(meta.get("tool_call_id") or "")
            idx = tool_pos.get(tcid)
            if idx is not None:
                last_anchor = idx
                rmeta = rows[idx]["meta"]
                # v1/v2 replay: title is the raw tool name, purpose/kind may be
                # thinner than what the live path persisted. Borrow, do not
                # overwrite what the replay did carry.
                jsonl_content = _row_text(row)
                if jsonl_content and "replay_title" not in rmeta:
                    # The JSONL persists two rows per approved tool (🔧 then ✅);
                    # the first carries the humanized title the live UI groups
                    # on, so borrow that one and keep the replay's own for audit.
                    rmeta["replay_title"] = rows[idx]["content"]
                    rows[idx]["content"] = jsonl_content
                for key in ("purpose", "kind"):
                    if not rmeta.get(key) and meta.get(key):
                        rmeta[key] = meta[key]
                if "output" not in rmeta and meta.get("output"):
                    rmeta["output"] = meta["output"]
                if meta.get("done"):
                    rmeta["done"] = True
                if row.get("ts"):
                    rows[idx]["ts"] = row["ts"]
                if meta.get("mid"):
                    rmeta["mid"] = meta["mid"]
                continue
            # A tool row the replay lacks: keep it as sidecar so nothing vanishes.
            sidecar.append((last_anchor, _tag(row, SOURCE_JSONL)))
            continue
        if role in ("assistant", "chunk", "streaming", "thinking"):
            text = _norm(_row_text(row))
            if role == "assistant" and text:
                for i, r in enumerate(rows):
                    if i in matched_assistant or r["role"] != "assistant":
                        continue
                    rt = _norm(r["content"])
                    if _assistant_texts_match(rt, text):
                        matched_assistant.add(i)
                        last_anchor = i
                        if row.get("ts"):
                            r["ts"] = row["ts"]
                        if meta.get("mid"):
                            r["meta"]["mid"] = meta["mid"]
                        # Kiro Crew-side facts riding an assistant row.
                        for key in ("turn_stats", "file_changes"):
                            if key in meta:
                                r["meta"][key] = meta[key]
                        break
                else:
                    # Notice-shaped assistant rows (compaction banner) and text
                    # the replay does not have: sidecar.
                    sidecar.append((last_anchor, _tag(row, SOURCE_JSONL)))
            # chunk/streaming/thinking rows are transient wire rows; the replay
            # carries the finished text.
            continue
        # user (steer), inject, notice, error, permission, system, done, ...
        sidecar.append((last_anchor, _tag(row, SOURCE_JSONL)))

    # 3. A REWRITTEN turn: the JSONL has assistant text for this turn and none
    #    of it is what the replay carries (regenerate / edit-and-resend / variant
    #    switch happened after the resume). The replay is then a record of a
    #    response that does not exist any more, so the JSONL wins the WHOLE turn --
    #    rendering both would resurrect the old answer beside the new one.
    #    Self-correcting even if the rewrite endpoint's invalidation did not run.
    jsonl_assistant = sum(
        1 for r in jsonl_rows if r.get("role") == "assistant" and _row_text(r).strip()
    )
    replay_assistant = sum(1 for r in rows if r["role"] == "assistant")
    if jsonl_assistant and replay_assistant and not matched_assistant:
        return [head] + [_tag(r, SOURCE_JSONL) for r in jsonl_rows]

    # 4. splice sidecar rows after their anchor row (or right after the prompt)
    by_anchor: dict[int | None, list[dict[str, Any]]] = {}
    for anchor, row in sidecar:
        by_anchor.setdefault(anchor, []).append(row)
    out.extend(by_anchor.pop(None, []))
    for i, r in enumerate(rows):
        out.append(r)
        out.extend(by_anchor.pop(i, []))
    for leftovers in by_anchor.values():
        out.extend(leftovers)
    return out


def merge_replay_transcript(
    updates: list[dict[str, Any]], jsonl_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build the transcript from replay frames, overlaying JSONL sidecar rows.

    Returns ``(rows, report)``; ``report`` counts rows by source so the caller
    can surface how much of the transcript the replay carried. When the replay
    holds no turn at all the JSONL rows are returned untouched (tagged), so a
    caller never loses history to an empty replay.
    """
    turns = fold_replay_updates(
        updates,
        redact_text=_redact_text,
        redact_field=_redact_tool_field,
        purpose_limit=_MAX_TOOL_PURPOSE,
    )
    if not turns:
        rows = [_tag(r, SOURCE_JSONL) for r in jsonl_rows]
        return rows, {"replay": 0, "jsonl": len(rows), "turns": 0}

    groups = _jsonl_turns(jsonl_rows)
    out: list[dict[str, Any]] = []
    cursor = 0
    # Rows before the first JSONL prompt (welcome notices etc.) lead the transcript.
    if groups and groups[0][0] is None:
        out.extend(_tag(r, SOURCE_JSONL) for r in groups[0][1])
        cursor = 1
    for turn in turns:
        j = _match_prompt(groups, cursor, turn)
        if j == _AMBIGUOUS:
            # Duplicate prompts with nothing to tell them apart: rendering this
            # turn's replay rows under either would misattribute them, so the
            # turn is dropped and the JSONL groups -- which record both turns
            # -- flow through below as sidecar, in order. The cursor does not
            # move: a later, unambiguous turn still matches past them.
            continue
        if j >= 0:
            # JSONL prompts the replay skipped (steer-only turns, dropped
            # prompts) are still shown, in order, as sidecar.
            for k in range(cursor, j):
                prow, rest = groups[k]
                if prow is not None:
                    out.append(_tag(prow, SOURCE_JSONL))
                out.extend(_tag(r, SOURCE_JSONL) for r in rest)
            prow, rest = groups[j]
            cursor = j + 1
        else:
            prow, rest = None, []
        out.extend(_overlay_turn(turn, prow, rest))
    for k in range(cursor, len(groups)):
        prow, rest = groups[k]
        if prow is not None:
            out.append(_tag(prow, SOURCE_JSONL))
        out.extend(_tag(r, SOURCE_JSONL) for r in rest)

    report = {
        "replay": sum(1 for r in out if r.get("meta", {}).get("source") == SOURCE_REPLAY),
        "jsonl": sum(1 for r in out if r.get("meta", {}).get("source") == SOURCE_JSONL),
        "turns": len(turns),
    }
    return out, report


# ── Merge cache + rewrite invalidation ─────────────────────────────────────────
#
# The frames never change for a resumed session, so one merge is reused while
# the JSONL corpus is unchanged -- the detail endpoint is polled on every
# reconnect and turn end, and re-walking a multi-thousand-frame replay each time
# is a real cost. Two things break the reuse and are handled here:
#
# * rows mutate IN PLACE after they are appended (a permission row gets
#   ``resolved``, a tool row gets ``done``/``output``, an assistant row gets
#   ``turn_stats``), so the cache witness folds those fields in per row;
# * the transcript is REWRITTEN (rewind, regenerate, edit-and-resend, variant
#   switch): the replay then describes a conversation that does not exist any more, so
#   the rewrite endpoints call :func:`discard_replay_for_slot`, which drops both
#   the cached rows and the provider's frames -- the next render is the JSONL.

#: slot key -> ((frame count, corpus witness), rows, report, bytes). Bounded on
#: BOTH axes: an entry count, so an operator with many slots does not
#: accumulate one merged transcript per slot, and an aggregate byte ceiling,
#: because 64 entries of a multi-MB transcript each is gateway memory a count
#: alone does not bound. A single merge past ``_MERGE_CACHE_ENTRY_MAX_BYTES``
#: is served but never cached -- re-merging it on the next fetch is cheaper
#: than pinning that much memory for one slot. Oldest entries go first.
#: Every read, eviction, invalidation and write goes through ``_MERGE_CACHE_LOCK``:
#: the merge runs off-loop (``asyncio.to_thread``), so two detail requests for
#: different slots evict concurrently, and an unsynchronized
#: ``pop(next(iter(...)))`` on a dict another thread is mutating raises
#: ``KeyError``/``RuntimeError`` mid-request.
_MERGE_CACHE: dict[str, tuple[tuple[Any, ...], list[dict[str, Any]], dict[str, int], int]] = {}
_MERGE_CACHE_LOCK = threading.Lock()
_MERGE_CACHE_MAX = 64
_MERGE_CACHE_MAX_BYTES = 64 * 1024 * 1024
_MERGE_CACHE_ENTRY_MAX_BYTES = 16 * 1024 * 1024


def _rows_bytes(rows: list[dict[str, Any]]) -> int:
    """Serialized size of ``rows`` -- the memory a cache entry pins, to the order
    that matters for a budget (the JSON is what the endpoint ships anyway)."""
    return sum(
        len(json.dumps(r, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8"))
        for r in rows
    )


def _merge_cache_bytes() -> int:
    return sum(entry[3] for entry in _MERGE_CACHE.values())


def corpus_witness(corpus: list[dict[str, Any]]) -> str:
    """A fingerprint of a JSONL corpus over EVERY value the merge consumes.

    Every row is folded in whole -- role, ts, content and the full ``meta`` --
    because the overlay copies sidecar rows verbatim and borrows ``purpose``,
    ``input``, ``kind``, ``output``, ``done``, ``resolved``, ``mid``,
    ``turn_stats`` and ``file_changes`` from tool and assistant rows, and rows
    mutate IN PLACE after they are appended (a tool refinement can rewrite
    ``input`` or ``purpose`` to a string of the same length). A length- or
    flag-based witness therefore reuses a stale merge; the digest cannot. One
    linear pass over the corpus bytes -- the same order as the redaction pass
    the detail endpoint already pays on every response, and far cheaper than
    re-walking a multi-thousand-frame replay.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(len(corpus)).encode())
    for row in corpus:
        digest.update(
            json.dumps(row, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\x00")
    return digest.hexdigest()


def merge_replay_transcript_cached(
    slot_key: str, updates: list[dict[str, Any]], jsonl_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """:func:`merge_replay_transcript`, reusing the last result for an unchanged corpus.

    Returns fresh copies so a caller mutating its response cannot poison the
    cache. Synchronous and CPU-bound like the merge -- call it off-loop.
    """
    key = (len(updates), corpus_witness(jsonl_rows))
    with _MERGE_CACHE_LOCK:
        cached = _MERGE_CACHE.get(slot_key)
        if cached is not None and cached[0] == key:
            return list(cached[1]), dict(cached[2])
    # The merge itself runs unlocked: it is the expensive part, and holding the
    # lock across it would serialize every slot's detail fetch on one CPU.
    merged, report = merge_replay_transcript(updates, jsonl_rows)
    size = _rows_bytes(merged)
    with _MERGE_CACHE_LOCK:
        # A stale entry for this slot is dropped either way, so a slot whose merge
        # outgrew the per-entry cap does not keep pinning its previous, smaller one.
        _MERGE_CACHE.pop(slot_key, None)
        if size <= _MERGE_CACHE_ENTRY_MAX_BYTES:
            while _MERGE_CACHE and (
                len(_MERGE_CACHE) >= _MERGE_CACHE_MAX
                or _merge_cache_bytes() + size > _MERGE_CACHE_MAX_BYTES
            ):
                _MERGE_CACHE.pop(next(iter(_MERGE_CACHE)))
            _MERGE_CACHE[slot_key] = (key, merged, report, size)
    return list(merged), dict(report)


def discard_replay_for_slot(sessions: Any, slot: Any) -> None:
    """Forget the resume replay for ``slot`` because its transcript was rewritten.

    Drops the cached merge AND asks the live provider to drop its frames
    (``LLMProvider.discard_replay``, a no-op on backends without a replay), so the
    next detail fetch renders the JSONL instead of resurrecting the pre-rewrite
    response. ``sessions`` is the session manager (``state.sessions``); resolution
    failures are swallowed because a rewrite must never fail on a cleanup step.
    """
    with _MERGE_CACHE_LOCK:
        _MERGE_CACHE.pop(str(getattr(slot, "key", "") or ""), None)
    try:
        provider = sessions.get_provider(effective_session_key(slot))
    except Exception:
        return
    if provider is not None:
        try:
            provider.discard_replay()
        except Exception:
            logger.debug("discard_replay failed", exc_info=True)
