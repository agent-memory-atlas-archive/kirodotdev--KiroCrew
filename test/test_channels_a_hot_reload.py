"""Config hot-reload for the Telegram / Discord / Webex channels.

Each channel is asked the same three questions:

* a reloaded THRESHOLD or render toggle is read at point of use, with the
  loader's own clamp (Telegram, Discord) or pair normalization (Webex) re-run,
  so a reloaded value cannot make the soft nudge unreachable;
* a reloaded ALLOW-LIST reaches the live transport, so an added id is authorized
  and a removed one is refused without a restart;
* a DEGRADED section, or a value of the wrong shape, keeps the PREVIOUS
  authorization state -- these are fail-closed boundaries, and rebuilding them
  from a document the loader could not parse would lock out every intended
  sender or serve a room nobody approved.

Two Telegram/Discord/Webex specifics get their own coverage: Discord's runtime
thread promotions must survive a reload (they are not in ``config.json``, and
dropping them would strand every follow-up reply into a thread the bot just
made), and Webex's SECOND copy of the email roster -- the card-press path, which
does not flow through ``receive`` -- must follow the same reload as the first.

The appliers are driven both directly (a hand-built :class:`ConfigChange`, which
is what a dispatcher's subscriber actually receives) and through
``ConfigWatch.refresh_now()`` against a real temp config file, so the wiring from
a file write to a transport's frozenset is covered end to end.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kiro_crew.config import live
from kiro_crew.config.live import ConfigChange, ConfigWatch
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.discord.transport import DiscordTransport
from kiro_crew.discord.transport_dispatch import DiscordDispatcher
from kiro_crew.messaging.transport import InboundMessage
from kiro_crew.telegram.transport import TelegramTransport, forum_gate_outcome
from kiro_crew.telegram.transport_dispatch import TelegramDispatcher
from kiro_crew.webex.transport import WebexTransport
from kiro_crew.webex.transport_dispatch import WebexDispatcher

# ------------------------------------------------------------------
# Fakes + helpers
# ------------------------------------------------------------------


class FakeSessions:
    """Only the surface these dispatchers touch in these tests."""

    def __init__(self, pct: float = 0.0) -> None:
        self._pct = pct
        self.busy: set[str] = set()

    def is_busy(self, key) -> bool:
        return key in self.busy

    def check_context_usage(self, key, provider) -> float:
        return self._pct

    def max_generation(self, bucket: str) -> int:
        return 0


class FakeClient:
    """A transport client stand-in: nothing is sent in these tests."""


def _boot_cfg(section_name: str, **section_kw):
    """A boot-snapshot stand-in shaped like the fields the dispatchers read."""
    return SimpleNamespace(
        agent=SimpleNamespace(default_agent="", approval_mode="interactive"),
        messaging=SimpleNamespace(
            dm_scope="per-channel-peer",
            idle_reset_minutes=0,
            daily_reset_hour=-1,
            queue_mode="steer",
        ),
        **{section_name: SimpleNamespace(**section_kw)},
    )


def _change(new, *changed: str, old=None) -> ConfigChange:
    """The shape a subscriber receives from one reload."""
    return ConfigChange(old=old, new=new, changed=frozenset(changed))


def _cfg_with(section_name: str, degraded: frozenset[str] = frozenset(), **section_kw):
    """A real :class:`KiroCrewConfig` carrying one replaced channel section."""
    cfg = KiroCrewConfig()
    section = replace(getattr(cfg, section_name), **section_kw)
    return replace(cfg, _degraded_sections=degraded, **{section_name: section})


def _outcomes(audited: list[dict], operation: str) -> list[str]:
    """Outcomes for one operation only.

    A monkeypatched module-level ``sel`` also captures ``authorize``'s own
    ``denied`` rows, which are a different event: asserting on the raw list would
    couple these tests to how many times the test happened to authorize.
    """
    return [a["outcome"] for a in audited if a.get("operation") == operation]


@pytest.fixture(autouse=True)
def _fresh_watcher():
    """Every test gets an unstarted process watcher, and leaves none behind."""
    live.reset_for_tests()
    yield
    live.reset_for_tests()


# ------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------


class TestTelegramAllowLists:
    def _transport(self, *, users=(7,), allow_forum=False, chats=()) -> TelegramTransport:
        return TelegramTransport(
            FakeClient(),
            allowed_user_ids=list(users),
            allow_forum=allow_forum,
            allowed_forum_chat_ids=list(chats),
        )

    def _msg(self, user_id: str) -> InboundMessage:
        return InboundMessage(
            channel_type="telegram", user_id=user_id, conversation_id=user_id, text="hi"
        )

    def test_added_id_is_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("7"))
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=[9], allow_forum=False, allowed_forum_chat_ids=[])
        )
        assert t.authorize(self._msg("9"))
        assert not t.authorize(self._msg("7"))

    def test_user_ids_are_coerced_to_strings_like_the_constructor(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=[9, "11", " "], allow_forum=False, allowed_forum_chat_ids=[]
            )
        )
        assert t._allowed == frozenset({"9", "11"})

    def test_the_forum_gate_stays_a_conjunction_after_a_reload(self):
        t = self._transport(users=(7,))
        gate = lambda: forum_gate_outcome(  # noqa: E731 - one call site, read as data
            "supergroup",
            -100,
            5,
            allow_forum=t._allow_forum,
            allowed_forum_chat_ids=t._allowed_forum_chat_ids,
        )
        assert gate() == "denied_forum_not_allowed"

        # The chat id alone is not enough: allow_forum is still off.
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=[7], allow_forum=False, allowed_forum_chat_ids=[-100])
        )
        assert gate() == "denied_forum_not_allowed"

        # The switch alone is not enough either: a different chat is listed.
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=[7], allow_forum=True, allowed_forum_chat_ids=[-999])
        )
        assert gate() == "denied_forum_not_allowed"

        # Both halves satisfied.
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=[7], allow_forum=True, allowed_forum_chat_ids=[-100])
        )
        assert gate() is None

    def test_an_uncoercible_forum_chat_id_keeps_the_previous_set(self):
        t = self._transport(allow_forum=True, chats=(-100,))
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=[7], allow_forum=True, allowed_forum_chat_ids=["not-a-number"]
            )
        )
        assert t._allowed_forum_chat_ids == frozenset({-100})

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(allowed_user_ids="9", allow_forum=False, allowed_forum_chat_ids=[])
        )
        assert t.authorize(self._msg("7"))

    def test_a_non_bool_allow_forum_keeps_the_previous_value(self):
        t = self._transport(allow_forum=False)
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=[7], allow_forum="yes", allowed_forum_chat_ids=[])
        )
        assert t._allow_forum is False

    def test_changes_are_audited_by_count_and_ids_never_logged(self, monkeypatch, caplog):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.telegram.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.telegram.transport"):
            t.reconfigure(
                SimpleNamespace(
                    allowed_user_ids=[424242], allow_forum=True, allowed_forum_chat_ids=[-100]
                )
            )
        assert _outcomes(audited, "telegram_transport.reconfigure") == [
            "allow_list_changed",
            "forum_allow_list_changed",
            "allow_forum_enabled",
        ]
        assert "424242" not in caplog.text
        assert all("424242" not in str(a.get("resources", "")) for a in audited)


class TestTelegramDispatcherApplier:
    def _dispatcher(self, transport=None) -> TelegramDispatcher:
        d = TelegramDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("telegram", soft_threshold_pct=80, show_thinking=False),
            allowed_user_ids={"7"},
        )
        d.client = FakeClient()
        d.transport = transport
        return d

    def _transport(self) -> TelegramTransport:
        return TelegramTransport(FakeClient(), allowed_user_ids=[7])

    def test_applier_pushes_the_reloaded_roster(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(_cfg_with("telegram", allowed_user_ids=[9]), "telegram")
        )
        assert t._allowed == frozenset({"9"})

    def test_a_degraded_telegram_section_keeps_the_previous_roster(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with("telegram", degraded=frozenset({"telegram"}), allowed_user_ids=[9]),
                "telegram",
            )
        )
        assert t._allowed == frozenset({"7"})

    def test_a_whole_config_degrade_keeps_the_previous_roster(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with("telegram", degraded=frozenset({"*"}), allowed_user_ids=[9]),
                "telegram",
            )
        )
        assert t._allowed == frozenset({"7"})

    def test_a_messaging_only_change_does_not_touch_the_roster(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(_cfg_with("telegram", allowed_user_ids=[]), "messaging.dm_scope")
        )
        assert t._allowed == frozenset({"7"})

    def test_the_applier_no_ops_without_a_transport(self):
        self._dispatcher(None)._on_config_change(_change(_cfg_with("telegram"), "telegram"))

    def test_the_subscription_is_registered_and_held(self):
        d = self._dispatcher()
        assert "TelegramDispatcher" in [s.name for s in live.watch().subscriptions()]
        assert d._config_sub.callback() is not None

    def test_the_threshold_follows_the_live_snapshot_and_is_clamped(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("telegram", soft_threshold_pct=40))
        assert d._soft_threshold() == 40
        live.watch().prime(_cfg_with("telegram", soft_threshold_pct=0))
        assert d._soft_threshold() == 1

    def test_show_thinking_follows_the_live_snapshot(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("telegram", show_thinking=True))
        assert bool(d._live_cfg().telegram.show_thinking) is True

    def test_an_unreadable_config_falls_back_to_the_boot_copy(self, monkeypatch):
        d = self._dispatcher()
        monkeypatch.setattr(
            "kiro_crew.config.loader.KiroCrewConfig.load",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))),
        )
        assert d._soft_threshold() == 80

    def test_dm_scope_is_pinned_until_the_generation_advances(self):
        d = self._dispatcher()
        route = ("direct", "7")
        first = d._session_key(route)
        live.watch().prime(_cfg_with("messaging", dm_scope="unified"))
        assert d._session_key(route) == first
        d._conv.bump_gen(route)
        assert d._session_key(route) != first


# ------------------------------------------------------------------
# Discord
# ------------------------------------------------------------------


class TestDiscordAllowLists:
    def _transport(self, *, users=("11",), threads=(), channels=()) -> DiscordTransport:
        return DiscordTransport(
            FakeClient(),
            allowed_user_ids=list(users),
            allowed_thread_ids=list(threads),
            allowed_channel_ids=list(channels),
        )

    def _msg(self, user_id: str) -> InboundMessage:
        return InboundMessage(
            channel_type="discord", user_id=user_id, conversation_id="c1", text="hi"
        )

    def test_added_id_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("11"))
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["22"],
                allowed_thread_ids=[],
                allowed_channel_ids=[],
                auto_thread=True,
            )
        )
        assert t.authorize(self._msg("22"))
        assert not t.authorize(self._msg("11"))

    def test_a_runtime_promoted_thread_survives_the_reload(self):
        """A thread the bot created is not in config.json.

        Dropping it on a reload would strand every follow-up reply the user sends
        into the thread the bot just made, which is exactly why the set is
        mutable in the first place.
        """
        t = self._transport(threads=("t-configured",))
        t._allowed_threads.add("t-runtime")
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["11"],
                allowed_thread_ids=["t-configured"],
                allowed_channel_ids=[],
                auto_thread=True,
            )
        )
        assert t._allowed_threads == {"t-configured", "t-runtime"}

    def test_a_thread_removed_from_the_config_is_dropped(self):
        t = self._transport(threads=("t-configured",))
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["11"],
                allowed_thread_ids=[],
                allowed_channel_ids=[],
                auto_thread=True,
            )
        )
        assert t._allowed_threads == set()

    def test_the_channel_allow_list_follows_the_reload(self):
        t = self._transport(channels=("c-old",))
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["11"],
                allowed_thread_ids=[],
                allowed_channel_ids=["c-new"],
                auto_thread=True,
            )
        )
        assert t._allowed_channels == frozenset({"c-new"})

    def test_auto_thread_follows_the_reload_and_rejects_a_non_bool(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["11"],
                allowed_thread_ids=[],
                allowed_channel_ids=[],
                auto_thread=False,
            )
        )
        assert t._auto_thread is False
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["11"],
                allowed_thread_ids=[],
                allowed_channel_ids=[],
                auto_thread="yes",
            )
        )
        assert t._auto_thread is False

    def test_a_non_list_field_keeps_the_previous_set(self):
        t = self._transport(channels=("c-old",))
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids="22",
                allowed_thread_ids=None,
                allowed_channel_ids="c-new",
                auto_thread=True,
            )
        )
        assert t._allowed == frozenset({"11"})
        assert t._allowed_channels == frozenset({"c-old"})

    def test_changes_are_audited_by_count(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.discord.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["22"],
                allowed_thread_ids=["t1"],
                allowed_channel_ids=["c1"],
                auto_thread=True,
            )
        )
        assert _outcomes(audited, "discord_transport.reconfigure") == [
            "allow_list_changed",
            "channel_allow_list_changed",
            "thread_allow_list_changed",
        ]


class TestDiscordDispatcherApplier:
    def _dispatcher(self, transport=None) -> DiscordDispatcher:
        d = DiscordDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg(
                "discord",
                soft_threshold_pct=80,
                reactions_enabled=True,
                show_thinking=False,
            ),
            allowed_user_ids={"11"},
        )
        d.client = FakeClient()
        d.transport = transport
        return d

    def test_applier_pushes_the_reloaded_roster(self):
        t = DiscordTransport(FakeClient(), allowed_user_ids=["11"])
        self._dispatcher(t)._on_config_change(
            _change(_cfg_with("discord", allowed_user_ids=["22"]), "discord")
        )
        assert t._allowed == frozenset({"22"})

    def test_a_degraded_discord_section_keeps_the_previous_roster(self):
        t = DiscordTransport(FakeClient(), allowed_user_ids=["11"])
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with("discord", degraded=frozenset({"discord"}), allowed_user_ids=["22"]),
                "discord",
            )
        )
        assert t._allowed == frozenset({"11"})

    def test_the_subscription_is_registered(self):
        self._dispatcher()
        assert "DiscordDispatcher" in [s.name for s in live.watch().subscriptions()]

    def test_the_threshold_follows_the_live_snapshot_and_is_clamped(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("discord", soft_threshold_pct=45))
        assert d._soft_threshold() == 45
        live.watch().prime(_cfg_with("discord", soft_threshold_pct=999))
        assert d._soft_threshold() == 100

    def test_the_render_toggles_follow_the_live_snapshot(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("discord", reactions_enabled=False, show_thinking=True))
        assert d._render_config() == (False, True)

    def test_dm_scope_is_pinned_until_the_generation_advances(self):
        d = self._dispatcher()
        first = d._session_key("11", "")
        live.watch().prime(_cfg_with("messaging", dm_scope="unified"))
        assert d._session_key("11", "") == first
        d._conv.bump_gen("user:11")
        assert d._session_key("11", "") != first


# ------------------------------------------------------------------
# Webex
# ------------------------------------------------------------------


class TestWebexAllowLists:
    def _transport(self, *, emails=("a@example.com",), group=False, rooms=()) -> WebexTransport:
        return WebexTransport(
            FakeClient(),
            allowed_emails=list(emails),
            allow_group_rooms=group,
            allowed_room_ids=list(rooms),
        )

    def _msg(self, email: str) -> InboundMessage:
        return InboundMessage(channel_type="webex", user_id=email, conversation_id="r1", text="hi")

    def test_added_email_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("a@example.com"))
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["B@Example.com"], allow_group_rooms=False, allowed_room_ids=[]
            )
        )
        assert t.authorize(self._msg("b@example.com"))
        assert not t.authorize(self._msg("a@example.com"))

    def test_emails_are_lowercased_and_room_ids_are_not(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["Mixed@Case.COM"],
                allow_group_rooms=True,
                allowed_room_ids=["Y2lzY29:RoOm"],
            )
        )
        assert t._allowed == frozenset({"mixed@case.com"})
        assert t._allowed_rooms == frozenset({"Y2lzY29:RoOm"})

    def _space_ok(self, t: WebexTransport, room: str) -> bool:
        """The space gate as ``room_permitted`` and ``may_send_to`` both spell it."""
        return t._allow_group_rooms and room in t._allowed_rooms

    def test_the_space_gate_stays_a_conjunction_after_a_reload(self):
        t = self._transport()
        assert not self._space_ok(t, "oc_room")

        # The room id alone is not enough: allow_group_rooms is still off.
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["a@example.com"],
                allow_group_rooms=False,
                allowed_room_ids=["oc_room"],
            )
        )
        assert not self._space_ok(t, "oc_room")

        # The switch alone is not enough either: a different room is listed.
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["a@example.com"],
                allow_group_rooms=True,
                allowed_room_ids=["oc_other"],
            )
        )
        assert not self._space_ok(t, "oc_room")

        # Both halves satisfied.
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["a@example.com"],
                allow_group_rooms=True,
                allowed_room_ids=["oc_room"],
            )
        )
        assert self._space_ok(t, "oc_room")
        assert t.may_send_to("oc_room")

    def test_an_emptied_room_list_still_denies_every_space(self):
        t = self._transport(group=True, rooms=("oc_room",))
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["a@example.com"], allow_group_rooms=True, allowed_room_ids=[]
            )
        )
        assert not self._space_ok(t, "oc_room")
        assert not t.may_send_to("oc_room")

    def test_a_non_list_field_keeps_the_previous_set(self):
        t = self._transport(group=True, rooms=("oc_room",))
        t.reconfigure(
            SimpleNamespace(
                allowed_emails="b@example.com",
                allow_group_rooms=True,
                allowed_room_ids="oc_new",
            )
        )
        assert t._allowed == frozenset({"a@example.com"})
        assert t._allowed_rooms == frozenset({"oc_room"})

    def test_a_non_bool_allow_group_rooms_keeps_the_previous_value(self):
        t = self._transport(group=False)
        t.reconfigure(
            SimpleNamespace(
                allowed_emails=["a@example.com"],
                allow_group_rooms="yes",
                allowed_room_ids=[],
            )
        )
        assert t._allow_group_rooms is False

    def test_changes_are_audited_by_count_and_addresses_never_logged(self, monkeypatch, caplog):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.webex.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.webex.transport"):
            t.reconfigure(
                SimpleNamespace(
                    allowed_emails=["secret@example.com"],
                    allow_group_rooms=True,
                    allowed_room_ids=["oc_room"],
                )
            )
        assert _outcomes(audited, "webex_transport.reconfigure") == [
            "allow_list_changed",
            "room_allow_list_changed",
            "allow_group_enabled",
        ]
        assert "secret@example.com" not in caplog.text


class TestWebexDispatcherApplier:
    def _dispatcher(self, transport=None, *, pct: float = 0.0) -> WebexDispatcher:
        d = WebexDispatcher(
            sessions=FakeSessions(pct=pct),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg(
                "webex",
                soft_threshold_pct=80,
                hard_threshold_pct=95,
                reply_in_thread=False,
                allowed_emails=["a@example.com"],
            ),
        )
        d.client = FakeClient()
        d.transport = transport
        return d

    def test_applier_pushes_the_reloaded_lists(self):
        t = WebexTransport(FakeClient(), allowed_emails=["a@example.com"])
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with(
                    "webex",
                    allowed_emails=["b@example.com"],
                    allow_group_rooms=True,
                    allowed_room_ids=["oc_room"],
                ),
                "webex",
            )
        )
        assert t._allowed == frozenset({"b@example.com"})
        assert t._allow_group_rooms is True
        assert t._allowed_rooms == frozenset({"oc_room"})

    def test_a_degraded_webex_section_keeps_the_previous_lists(self):
        t = WebexTransport(FakeClient(), allowed_emails=["a@example.com"])
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with("webex", degraded=frozenset({"webex"}), allowed_emails=["b@example.com"]),
                "webex",
            )
        )
        assert t._allowed == frozenset({"a@example.com"})

    def test_the_card_press_copy_follows_the_same_reload(self):
        """Webex's SECOND roster copy: a press does not flow through receive."""
        d = self._dispatcher()
        live.watch().prime(_cfg_with("webex", allowed_emails=["a@example.com"]))
        assert d._sender_allowed("a@example.com")
        live.watch().prime(_cfg_with("webex", allowed_emails=["b@example.com"]))
        assert d._sender_allowed("b@example.com")
        assert not d._sender_allowed("a@example.com")

    def test_the_card_press_copy_denies_on_an_emptied_roster(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("webex", allowed_emails=["a@example.com"]))
        assert d._sender_allowed("a@example.com")
        live.watch().prime(_cfg_with("webex", allowed_emails=[]))
        assert not d._sender_allowed("a@example.com")

    def test_an_inverted_reloaded_pair_is_normalized_not_trusted(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("webex", soft_threshold_pct=95, hard_threshold_pct=50))
        assert d._thresholds() == (50, 50)

    def test_reply_in_thread_follows_the_live_snapshot(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("webex", reply_in_thread=True))
        inbound = SimpleNamespace(parent_id="p1")
        assert d._reply_parent(inbound) == "p1"

    def test_the_subscription_is_registered(self):
        self._dispatcher()
        assert "WebexDispatcher" in [s.name for s in live.watch().subscriptions()]


# ------------------------------------------------------------------
# End to end: a file write reaches each transport
# ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["telegram", "discord", "webex"])
async def test_a_config_file_write_reaches_the_transport(channel, tmp_path, monkeypatch):
    """The whole path: write config.json -> ConfigWatch -> transport frozenset.

    Driven through ``refresh_now`` rather than the poll interval so the test does
    not sleep, and pointed at temp files exactly as the loader's own tests do.
    """
    cfg_path = tmp_path / "config.json"
    local = tmp_path / "config.local.json"
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_path)
    monkeypatch.setattr("kiro_crew.config.loader.config_local_path", lambda: local)

    if channel == "telegram":
        cfg_path.write_text(json.dumps({"telegram": {"allowed_user_ids": [7]}}))
        transport = TelegramTransport(FakeClient(), allowed_user_ids=[7])
        dispatcher = TelegramDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("telegram", soft_threshold_pct=80),
            allowed_user_ids={"7"},
        )
        after, expected = {"telegram": {"allowed_user_ids": [9]}}, frozenset({"9"})
    elif channel == "discord":
        cfg_path.write_text(json.dumps({"discord": {"allowed_user_ids": ["11"]}}))
        transport = DiscordTransport(FakeClient(), allowed_user_ids=["11"])
        dispatcher = DiscordDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("discord", soft_threshold_pct=80),
            allowed_user_ids={"11"},
        )
        after, expected = {"discord": {"allowed_user_ids": ["22"]}}, frozenset({"22"})
    else:
        cfg_path.write_text(json.dumps({"webex": {"allowed_emails": ["a@example.com"]}}))
        transport = WebexTransport(FakeClient(), allowed_emails=["a@example.com"])
        dispatcher = WebexDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("webex", soft_threshold_pct=80, hard_threshold_pct=95),
        )
        after, expected = (
            {"webex": {"allowed_emails": ["b@example.com"]}},
            frozenset({"b@example.com"}),
        )

    dispatcher.client = FakeClient()
    dispatcher.transport = transport

    watch = ConfigWatch(poll_interval_secs=0.05)
    sub = dispatcher._config_sub
    watch.subscribe(*sub.prefixes, callback=dispatcher._on_config_change, name=sub.name)
    await watch.refresh_now()  # establish the baseline

    cfg_path.write_text(json.dumps(after))
    change = await watch.refresh_now()
    assert change is not None and change.touched(channel)
    assert transport._allowed == expected
