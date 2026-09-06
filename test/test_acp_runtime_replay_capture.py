"""AcpRuntime transcript-replay capture (``dashboard.replay_from_acp``).

During ``session/load`` kiro-cli replays the whole prior transcript as
``session/update`` frames tagged with the resumed sid. That sid has no
registered queue yet, so the reader's default is the counted-drop path. With
``capture_replay=True`` the runtime retains those frames for the sid that
``load_session()`` armed and hands them to the handle; with it off (the
default) nothing changes.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kiro_crew.acp import runtime as runtime_mod
from kiro_crew.acp.runtime import AcpRuntime
from kiro_crew.acp.types import JsonRpcMessage

SID = "resume-sid-1"


def _make_runtime(capture: bool) -> tuple[AcpRuntime, asyncio.StreamReader]:
    rt = AcpRuntime(work_dir="/tmp", capture_replay=capture)
    reader = asyncio.StreamReader()
    proc = MagicMock()
    proc.stdout = reader
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.returncode = None
    proc.pid = 4242
    rt._process = proc
    rt._pid = 4242
    rt._initialized = True
    rt._can_load_session = True
    return rt, reader


def _update(sid: str, text: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": sid,
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        },
    }


def _msg(sid: str, text: str) -> JsonRpcMessage:
    return JsonRpcMessage.from_dict(_update(sid, text))


# ── helper-level behaviour ─────────────────────────────────────────────────


def test_capture_off_never_retains() -> None:
    rt, _ = _make_runtime(capture=False)
    rt._arm_replay_capture(SID)  # no-op when capture is off
    assert rt._capture_replay_frame(SID, _msg(SID, "a")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert frames == [] and overflow == 0


def test_capture_on_retains_only_armed_sid() -> None:
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "a")) is True
    assert rt._capture_replay_frame("other-sid", _msg("other-sid", "b")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert [f["update"]["content"]["text"] for f in frames] == ["a"]
    assert overflow == 0
    # popped: a second take is empty and the sid is disarmed
    assert rt._take_replay_capture(SID) == ([], 0)
    assert rt._capture_replay_frame(SID, _msg(SID, "late")) is False


def test_capture_ignores_non_dict_params() -> None:
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    bad = JsonRpcMessage.from_dict(
        {"jsonrpc": "2.0", "method": "session/update", "params": ["not", "a", "dict"]}
    )
    assert rt._capture_replay_frame(SID, bad) is False
    assert rt._capture_replay_frame(None, _msg(SID, "x")) is False


def test_capture_cap_discards_whole_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_FRAMES", 3)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    kept = [rt._capture_replay_frame(SID, _msg(SID, str(i))) for i in range(5)]
    assert kept == [True, True, True, False, False]
    frames, overflow = rt._take_replay_capture(SID)
    # Same discipline as the byte ceiling: a partial replay is not served.
    assert frames == []
    assert overflow == 2


def test_rearm_clears_previous_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_FRAMES", 1)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    rt._capture_replay_frame(SID, _msg(SID, "a"))
    rt._capture_replay_frame(SID, _msg(SID, "b"))
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "c")) is True
    frames, overflow = rt._take_replay_capture(SID)
    assert len(frames) == 1 and overflow == 0


def test_byte_ceiling_discards_whole_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two small frames fit; a large tool-result frame blows the cumulative cap.
    # The capture is then discarded OUTRIGHT (not truncated) and stays off for
    # the rest of the load, so the consumer renders its own transcript.
    monkeypatch.setattr(runtime_mod, "_REPLAY_CAPTURE_MAX_BYTES", 600)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "small one")) is True
    assert rt._capture_replay_frame(SID, _msg(SID, "small two")) is True
    assert rt._capture_replay_frame(SID, _msg(SID, "x" * 2000)) is False
    assert rt._capture_replay_frame(SID, _msg(SID, "after")) is False
    frames, overflow = rt._take_replay_capture(SID)
    assert frames == [] and overflow == 0
    assert SID not in rt._replay_capture_discarded
    assert SID not in rt._replay_capture_bytes
    # Re-arming the same sid starts clean.
    rt._arm_replay_capture(SID)
    assert rt._capture_replay_frame(SID, _msg(SID, "fresh")) is True
    frames, _ = rt._take_replay_capture(SID)
    assert len(frames) == 1


def test_process_wide_budget_discards_a_new_capture_and_frees_on_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The per-session ceiling bounds ONE resume; this bounds all of them. With a
    # budget that fits roughly one small capture, a second concurrent session's
    # capture is discarded whole (its consumer keeps the JSONL) while the first
    # keeps its frames; unregistering the first returns its bytes, and the next
    # capture fits again.
    ledger = runtime_mod._ReplayRetentionLedger(600)
    monkeypatch.setattr(runtime_mod, "_REPLAY_LEDGER", ledger)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture("sid-a")
    assert rt._capture_replay_frame("sid-a", _msg("sid-a", "x" * 300)) is True
    frames_a, _ = rt._take_replay_capture("sid-a")
    assert len(frames_a) == 1 and ledger.total() > 0
    held_by_a = ledger.total()
    rt._arm_replay_capture("sid-b")
    assert rt._capture_replay_frame("sid-b", _msg("sid-b", "y" * 300)) is False
    frames_b, overflow_b = rt._take_replay_capture("sid-b")
    assert frames_b == [] and overflow_b == 0
    # b's discard released only b's bytes; a's retention is untouched.
    assert ledger.total() == held_by_a
    rt.unregister_session("sid-a")
    assert ledger.total() == 0
    rt._arm_replay_capture("sid-c")
    assert rt._capture_replay_frame("sid-c", _msg("sid-c", "z" * 300)) is True


def test_runtime_death_releases_in_flight_captures_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # A capture still in flight lives only in the runtime's buckets, which death
    # clears, so its bytes go back to the ledger. Frames already handed to a
    # handle keep serving the dashboard after the death, so their reservation
    # stays until the handle drops them (discard_replay / unregister_session).
    ledger = runtime_mod._ReplayRetentionLedger(10_000)
    monkeypatch.setattr(runtime_mod, "_REPLAY_LEDGER", ledger)
    rt, _ = _make_runtime(capture=True)
    rt._arm_replay_capture("sid-live")
    assert rt._capture_replay_frame("sid-live", _msg("sid-live", "kept")) is True
    rt._take_replay_capture("sid-live")  # frames handed to a handle; bytes stay reserved
    held = ledger.total()
    assert held > 0
    rt._session_queues["sid-live"] = asyncio.Queue()
    rt._arm_replay_capture("sid-inflight")
    assert rt._capture_replay_frame("sid-inflight", _msg("sid-inflight", "mid-load")) is True
    assert ledger.total() > held
    rt._mark_dead("test death")
    assert ledger.total() == held
    assert rt._replay_capture == {} and rt._replay_capture_bytes == {}
    rt.release_replay_retention("sid-live")
    assert ledger.total() == 0


def test_concurrent_same_sid_resumes_release_only_their_own_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two multiplexed runtimes resume the SAME sid at once (an eager resume
    # racing a first message). Each keys the shared ledger by its own owner
    # token, so the loser's discard, re-arm and unregister pop only ITS bytes:
    # the winner's frames stay counted and the process budget is never
    # undercounted (the OOM path GPT flagged on a sid-only key).
    ledger = runtime_mod._ReplayRetentionLedger(10_000)
    monkeypatch.setattr(runtime_mod, "_REPLAY_LEDGER", ledger)
    winner, _ = _make_runtime(capture=True)
    loser, _ = _make_runtime(capture=True)
    assert winner._replay_owner != loser._replay_owner
    winner._arm_replay_capture(SID)
    assert winner._capture_replay_frame(SID, _msg(SID, "w" * 300)) is True
    held_by_winner = ledger.total()
    assert held_by_winner > 0
    # The loser re-arms the same sid: a sid-only ledger would drop the winner here.
    loser._arm_replay_capture(SID)
    assert ledger.total() == held_by_winner
    assert loser._capture_replay_frame(SID, _msg(SID, "l" * 300)) is True
    assert ledger.total() > held_by_winner
    # Loser's load fails: its frames are dropped and only its bytes return.
    loser._take_replay_capture(SID)
    loser.unregister_session(SID)
    assert ledger.total() == held_by_winner
    # Loser's death releases nothing further; the winner's retention is intact.
    loser._mark_dead("lost the race")
    assert ledger.total() == held_by_winner
    winner.unregister_session(SID)
    assert ledger.total() == 0


# ── reader-loop routing ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reader_loop_retains_replay_for_armed_sid_and_drops_otherwise() -> None:
    rt, reader = _make_runtime(capture=True)
    rt._arm_replay_capture(SID)
    dropped: list[tuple[object, object]] = []
    rt._note_dropped_frame = lambda sid, method: dropped.append((sid, method))  # type: ignore[method-assign, assignment]
    task = asyncio.ensure_future(rt._reader_loop())
    await asyncio.sleep(0)
    reader.feed_data((json.dumps(_update(SID, "kept")) + "\n").encode())
    reader.feed_data((json.dumps(_update("unknown", "dropped")) + "\n").encode())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    frames, _ = rt._take_replay_capture(SID)
    assert [f["update"]["content"]["text"] for f in frames] == ["kept"]
    assert ("unknown", "session/update") in dropped
    assert (SID, "session/update") not in dropped


@pytest.mark.asyncio
async def test_reader_loop_capture_off_keeps_counted_drop() -> None:
    rt, reader = _make_runtime(capture=False)
    dropped: list[tuple[object, object]] = []
    rt._note_dropped_frame = lambda sid, method: dropped.append((sid, method))  # type: ignore[method-assign, assignment]
    task = asyncio.ensure_future(rt._reader_loop())
    await asyncio.sleep(0)
    reader.feed_data((json.dumps(_update(SID, "x")) + "\n").encode())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    assert (SID, "session/update") in dropped
    assert rt._replay_capture == {}


# ── load_session hands the frames to the handle ────────────────────────────


@pytest.mark.asyncio
async def test_load_session_attaches_replay_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, _ = _make_runtime(capture=True)

    async def fake_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        assert method == "session/load"
        # Frames arrive WHILE the load request is in flight.
        assert rt._capture_replay_frame(SID, _msg(SID, "hello")) is True
        return {"modes": {"availableModes": [], "currentModeId": "kirocrew"}, "models": {}}

    monkeypatch.setattr(rt, "_send_and_await", fake_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])
    monkeypatch.setattr(runtime_mod, "_load_watchdog_settings", lambda *_a, **_k: None)

    handle = await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert [f["update"]["content"]["text"] for f in handle.replay_updates] == ["hello"]
    # the bucket is gone: a later frame for the same id takes the drop path
    assert rt._capture_replay_frame(SID, _msg(SID, "late")) is False


@pytest.mark.asyncio
async def test_load_session_failure_disarms_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    rt, _ = _make_runtime(capture=True)

    async def failing_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        rt._capture_replay_frame(SID, _msg(SID, "partial"))
        raise runtime_mod.AcpRuntimeError("Session is active in another process (PID 1)")

    monkeypatch.setattr(rt, "_send_and_await", failing_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])

    with pytest.raises(runtime_mod.AcpRuntimeError):
        await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert rt._replay_capture == {}
    assert rt._replay_capture_overflow == {}


@pytest.mark.asyncio
async def test_load_session_served_default_failure_releases_resumed_sid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # session/load succeeded and the frames are on the handle when the
    # served-default check raises on the wire. The caller falls back to a fresh
    # session, so the resumed sid must be terminated first: its queue
    # registration and its process-budget reservation go with it, the same way
    # the set_mode failure path already cleans up.
    ledger = runtime_mod._ReplayRetentionLedger(10_000)
    monkeypatch.setattr(runtime_mod, "_REPLAY_LEDGER", ledger)
    rt, _ = _make_runtime(capture=True)
    terminated: list[str] = []

    async def fake_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        assert method == "session/load"
        assert rt._capture_replay_frame(SID, _msg(SID, "hello")) is True
        return {"modes": {"availableModes": [], "currentModeId": "kirocrew"}, "models": {}}

    async def fake_terminate(session_id: str) -> None:
        terminated.append(session_id)
        rt.unregister_session(session_id)

    async def failing_served_default(self: object) -> None:
        raise runtime_mod.AcpRuntimeError("set_model failed on the wire")

    monkeypatch.setattr(rt, "_send_and_await", fake_send)
    monkeypatch.setattr(rt, "terminate_session", fake_terminate)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])
    monkeypatch.setattr(runtime_mod, "_load_watchdog_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runtime_mod.AcpSessionHandle, "ensure_served_default", failing_served_default
    )

    with pytest.raises(runtime_mod.AcpRuntimeError):
        await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert terminated == [SID]
    assert SID not in rt._session_queues
    assert ledger.total() == 0


@pytest.mark.asyncio
async def test_load_session_capture_off_leaves_handle_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt, _ = _make_runtime(capture=False)

    async def fake_send(method: str, params: dict, timeout: float = 0.0) -> dict:
        return {"modes": {"availableModes": [], "currentModeId": "kirocrew"}, "models": {}}

    monkeypatch.setattr(rt, "_send_and_await", fake_send)
    monkeypatch.setattr(rt, "_session_start_budget", AsyncMock(return_value=5.0))
    monkeypatch.setattr(runtime_mod, "pooled_session_servers", lambda *_a, **_k: [])
    monkeypatch.setattr(runtime_mod, "_load_watchdog_settings", lambda *_a, **_k: None)

    handle = await rt.load_session("/tmp/x.json", SID, cwd="/tmp")
    assert handle.replay_updates == []
