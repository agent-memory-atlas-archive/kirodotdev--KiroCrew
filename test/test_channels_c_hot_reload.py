"""Config hot-reload for the WeCom / Weixin / Feishu channels.

Each channel gets the same three questions asked of it:

* a reloaded THRESHOLD is read at point of use, and the loader's pair
  normalization is re-run so an inverted pair cannot make the soft nudge
  unreachable;
* a reloaded ALLOW-LIST reaches the live transport, so an added id is
  authorized and a removed one is refused without a restart;
* a DEGRADED section, or a value of the wrong shape, keeps the PREVIOUS
  authorization state -- these are fail-closed boundaries, and rebuilding them
  from a document the loader could not parse would lock out every intended
  sender (or, for WeCom's allow-all, widen to the whole tenant).

The appliers are driven both directly (a hand-built :class:`ConfigChange`, which
is what a dispatcher's subscriber actually receives) and through
``ConfigWatch.refresh_now()`` against a real temp config file, so the wiring from
a file write to a transport's frozenset is covered end to end rather than only
the halves.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kiro_crew.config import live
from kiro_crew.config.live import ConfigChange, ConfigWatch
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.feishu.client import CHAT_GROUP, CHAT_P2P, LarkInbound
from kiro_crew.feishu.transport import FeishuTransport
from kiro_crew.feishu.transport_dispatch import FeishuDispatcher
from kiro_crew.messaging.transport import InboundMessage
from kiro_crew.wecom.client import WeComInbound
from kiro_crew.wecom.transport import WeComTransport, allowed_userids_from_config
from kiro_crew.wecom.transport_dispatch import WeComDispatcher
from kiro_crew.weixin.transport import WeixinTransport
from kiro_crew.weixin.transport_dispatch import WeixinDispatcher

# ------------------------------------------------------------------
# Fakes + helpers
# ------------------------------------------------------------------


class FakeSessions:
    """Only the surface the dispatchers touch in these tests."""

    def __init__(self, pct: float = 0.0) -> None:
        self._pct = pct
        self.busy: set[str] = set()

    def is_busy(self, key) -> bool:
        return key in self.busy

    def check_context_usage(self, key, provider) -> float:
        return self._pct

    def max_generation(self, bucket: str) -> int:
        return 0


class FakeProvider:
    def __init__(self) -> None:
        self.compacted = False

    async def compact(self) -> None:
        self.compacted = True

    async def wait_for_compaction(self, timeout: float = 0.0) -> dict:
        return {"type": "completed", "summary": ""}


class FakeWeComClient:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, str]] = []

    async def send_proactive(self, chat_id: str, text: str) -> bool:
        self.pushed.append((chat_id, text))
        return True

    def already_delivered(self, msgid: str) -> bool:
        return False

    def forget_msgid(self, msgid: str) -> None:
        pass


class FakeLarkClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []

    async def send_reply(self, message_id: str, text: str) -> bool:
        self.replies.append((message_id, text))
        return True


class FakeWeixinClient:
    pass


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


def _reconfigure_outcomes(audited: list[dict], operation: str) -> list[str]:
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
# WeCom
# ------------------------------------------------------------------


class TestWeComAllowList:
    def test_flattener_skips_non_dict_and_missing_userid(self):
        assert allowed_userids_from_config([{"userid": "Wei"}, {"name": "x"}, "junk"]) == ["Wei"]

    def test_flattener_reports_a_non_list_as_unusable(self):
        assert allowed_userids_from_config("Wei") is None
        assert allowed_userids_from_config(None) is None

    def _transport(self, *, allowed=("Wei",), allow_all=False) -> WeComTransport:
        return WeComTransport(
            FakeWeComClient(), allowed_users=list(allowed), allow_all=allow_all, owner_id=""
        )

    def _msg(self, userid: str) -> InboundMessage:
        return InboundMessage(
            channel_type="wecom", user_id=userid, conversation_id=userid, text="hi"
        )

    def test_added_userid_is_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("Wei"))
        assert not t.authorize(self._msg("Ming"))
        t.reconfigure(SimpleNamespace(allowed_users=[{"userid": "Ming"}], allow_all_users=False))
        assert t.authorize(self._msg("Ming"))
        assert not t.authorize(self._msg("Wei"))

    def test_allow_all_flip_widens_and_is_audited(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.wecom.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport(allowed=())
        assert not t.authorize(self._msg("Stranger"))
        t.reconfigure(SimpleNamespace(allowed_users=[], allow_all_users=True))
        assert t.authorize(self._msg("Stranger"))
        assert _reconfigure_outcomes(audited, "wecom_transport.reconfigure") == [
            "allow_all_enabled"
        ]

    def test_allow_all_flip_off_is_audited_and_narrows(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.wecom.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport(allowed=(), allow_all=True)
        t.reconfigure(SimpleNamespace(allowed_users=[], allow_all_users=False))
        assert not t.authorize(self._msg("Stranger"))
        assert _reconfigure_outcomes(audited, "wecom_transport.reconfigure") == [
            "allow_all_disabled"
        ]

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_users="Ming", allow_all_users=False))
        assert t.authorize(self._msg("Wei"))
        assert not t.authorize(self._msg("Ming"))

    def test_a_non_bool_allow_all_keeps_the_previous_value(self):
        t = self._transport(allowed=(), allow_all=False)
        t.reconfigure(SimpleNamespace(allowed_users=[], allow_all_users="yes"))
        assert not t.authorize(self._msg("Stranger"))

    def test_values_are_never_logged_only_counts(self, caplog):
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.wecom.transport"):
            t.reconfigure(
                SimpleNamespace(allowed_users=[{"userid": "Secret"}], allow_all_users=False)
            )
        assert "Secret" not in caplog.text


class TestWeComDispatcherApplier:
    def _dispatcher(self, transport=None) -> WeComDispatcher:
        d = WeComDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("wecom", soft_threshold_pct=80, hard_threshold_pct=95),
            owner_id="",
        )
        d.client = FakeWeComClient()
        d.transport = transport
        return d

    def test_applier_pushes_the_reloaded_roster_at_the_transport(self):
        t = WeComTransport(FakeWeComClient(), allowed_users=["Wei"], owner_id="")
        d = self._dispatcher(t)
        d._on_config_change(
            _change(_cfg_with("wecom", allowed_users=[{"userid": "Ming"}]), "wecom")
        )
        assert t._allowed == frozenset({"Ming"})

    def test_a_degraded_wecom_section_keeps_the_previous_roster(self):
        t = WeComTransport(FakeWeComClient(), allowed_users=["Wei"], owner_id="")
        d = self._dispatcher(t)
        d._on_config_change(
            _change(
                _cfg_with(
                    "wecom", degraded=frozenset({"wecom"}), allowed_users=[{"userid": "Ming"}]
                ),
                "wecom",
            )
        )
        assert t._allowed == frozenset({"Wei"})

    def test_a_whole_config_degrade_keeps_the_previous_roster(self):
        t = WeComTransport(FakeWeComClient(), allowed_users=["Wei"], owner_id="")
        d = self._dispatcher(t)
        d._on_config_change(
            _change(
                _cfg_with("wecom", degraded=frozenset({"*"}), allowed_users=[{"userid": "Ming"}]),
                "wecom",
            )
        )
        assert t._allowed == frozenset({"Wei"})

    def test_a_messaging_only_change_does_not_touch_the_roster(self):
        t = WeComTransport(FakeWeComClient(), allowed_users=["Wei"], owner_id="")
        d = self._dispatcher(t)
        d._on_config_change(_change(_cfg_with("wecom", allowed_users=[]), "messaging.dm_scope"))
        assert t._allowed == frozenset({"Wei"})

    def test_the_applier_no_ops_without_a_transport(self):
        self._dispatcher(None)._on_config_change(_change(_cfg_with("wecom"), "wecom"))

    def test_the_subscription_is_registered_and_held(self):
        d = self._dispatcher()
        names = [s.name for s in live.watch().subscriptions()]
        assert "WeComDispatcher" in names
        assert d._config_sub.callback() is not None


class TestWeComThresholds:
    def _dispatcher(self, *, soft: int, hard: int) -> WeComDispatcher:
        d = WeComDispatcher(
            sessions=FakeSessions(pct=90.0),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("wecom", soft_threshold_pct=soft, hard_threshold_pct=hard),
            owner_id="",
        )
        d.client = FakeWeComClient()
        return d

    def test_thresholds_read_the_config_file_when_no_watcher_is_armed(self, tmp_path, monkeypatch):
        """No snapshot yet -> the file, not the boot copy.

        ``load()`` is fingerprint-cached, so this is the cheap and CORRECT
        fallback: the file is the truth, and the boot copy is only reached when
        the file cannot be read at all (see the next test).
        """
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({"wecom": {"soft_threshold_pct": 33, "hard_threshold_pct": 66}})
        )
        monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_path)
        monkeypatch.setattr(
            "kiro_crew.config.loader.config_local_path", lambda: tmp_path / "local.json"
        )
        assert self._dispatcher(soft=70, hard=90)._thresholds() == (33, 66)

    def test_an_unreadable_config_falls_back_to_the_boot_copy(self, monkeypatch):
        d = self._dispatcher(soft=70, hard=90)
        monkeypatch.setattr(
            "kiro_crew.config.loader.KiroCrewConfig.load",
            staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))),
        )
        assert d._thresholds() == (70, 90)

    def test_thresholds_follow_the_live_snapshot(self):
        d = self._dispatcher(soft=70, hard=90)
        live.watch().prime(_cfg_with("wecom", soft_threshold_pct=40, hard_threshold_pct=60))
        assert d._thresholds() == (40, 60)

    def test_an_inverted_reloaded_pair_is_normalized_not_trusted(self):
        d = self._dispatcher(soft=70, hard=90)
        live.watch().prime(_cfg_with("wecom", soft_threshold_pct=95, hard_threshold_pct=50))
        assert d._thresholds() == (50, 50)

    @pytest.mark.asyncio
    async def test_a_reloaded_hard_threshold_forces_a_compaction_this_turn(self):
        d = self._dispatcher(soft=70, hard=99)
        provider = FakeProvider()
        inbound = WeComInbound(userid="Wei", text="hi", response_url="", req_id="r1", chatid="")
        await d._maybe_notice(inbound, "wecom:k", provider)
        assert not provider.compacted
        live.watch().prime(_cfg_with("wecom", soft_threshold_pct=70, hard_threshold_pct=80))
        await d._maybe_notice(inbound, "wecom:k", provider)
        assert provider.compacted


# ------------------------------------------------------------------
# Weixin
# ------------------------------------------------------------------


class TestWeixinAllowListAndPolicy:
    def _transport(self, *, allowed=("wxid_abc",), policy="allowlist") -> WeixinTransport:
        return WeixinTransport(
            FakeWeixinClient(),
            account_id="acct",
            ctx_store=SimpleNamespace(),
            allowed_user_ids=list(allowed),
            dm_policy=policy,
        )

    def _msg(self, user_id: str) -> InboundMessage:
        return InboundMessage(
            channel_type="weixin", user_id=user_id, conversation_id=user_id, text="hi"
        )

    def test_opaque_ids_survive_the_reload(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(allowed_user_ids=["wxid_new", "deadbeef@im.bot"], dm_policy="allowlist")
        )
        assert t._allowed == frozenset({"wxid_new", "deadbeef@im.bot"})
        assert t.authorize(self._msg("wxid_new"))
        assert t.authorize(self._msg("deadbeef@im.bot"))
        assert not t.authorize(self._msg("wxid_abc"))

    def test_blank_entries_are_dropped_and_duplicates_deduped(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_user_ids=["wxid_a", " ", "wxid_a", "wxid_b"], dm_policy="allowlist"
            )
        )
        assert t._allowed == frozenset({"wxid_a", "wxid_b"})

    def test_an_unknown_policy_keeps_the_previous_one(self):
        t = self._transport(policy="allowlist")
        t.reconfigure(SimpleNamespace(allowed_user_ids=["wxid_abc"], dm_policy="everyone"))
        assert t._dm_policy == "allowlist"
        assert not t.authorize(self._msg("stranger"))

    def test_a_policy_change_is_audited_by_name(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.weixin.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport(policy="allowlist")
        t.reconfigure(SimpleNamespace(allowed_user_ids=["wxid_abc"], dm_policy="disabled"))
        assert _reconfigure_outcomes(audited, "weixin_transport.reconfigure") == [
            "dm_policy_changed"
        ]
        assert audited[0]["resources"] == "from=allowlist to=disabled"
        assert not t.authorize(self._msg("wxid_abc"))

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_user_ids="wxid_new", dm_policy="allowlist"))
        assert t._allowed == frozenset({"wxid_abc"})

    def test_outbound_authorization_follows_the_reload(self):
        t = self._transport()
        assert t.may_send_to("wxid_abc")
        t.reconfigure(SimpleNamespace(allowed_user_ids=["wxid_new"], dm_policy="allowlist"))
        assert not t.may_send_to("wxid_abc")
        assert t.may_send_to("wxid_new")


class TestWeixinDispatcherApplier:
    def _dispatcher(self, transport=None) -> WeixinDispatcher:
        d = WeixinDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("weixin", soft_threshold_pct=80, hard_threshold_pct=95),
            account_id="acct",
            ctx_store=SimpleNamespace(),
        )
        d.client = FakeWeixinClient()
        d.transport = transport
        return d

    def _transport(self) -> WeixinTransport:
        return WeixinTransport(
            FakeWeixinClient(),
            account_id="acct",
            ctx_store=SimpleNamespace(),
            allowed_user_ids=["wxid_abc"],
            dm_policy="allowlist",
        )

    def test_applier_pushes_the_reloaded_roster(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(_cfg_with("weixin", allowed_user_ids=["wxid_new"]), "weixin")
        )
        assert t._allowed == frozenset({"wxid_new"})

    def test_a_degraded_weixin_section_keeps_roster_and_policy(self):
        t = self._transport()
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with(
                    "weixin",
                    degraded=frozenset({"weixin"}),
                    allowed_user_ids=["wxid_new"],
                    dm_policy="open",
                ),
                "weixin",
            )
        )
        assert t._allowed == frozenset({"wxid_abc"})
        assert t._dm_policy == "allowlist"

    def test_the_subscription_is_registered(self):
        self._dispatcher()
        assert "WeixinDispatcher" in [s.name for s in live.watch().subscriptions()]

    def test_thresholds_follow_the_live_snapshot_and_normalize(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("weixin", soft_threshold_pct=95, hard_threshold_pct=50))
        assert d._thresholds() == (50, 50)


# ------------------------------------------------------------------
# Feishu
# ------------------------------------------------------------------


class TestFeishuAllowLists:
    def _transport(
        self, *, open_ids=("ou_abc",), allow_group=False, group_ids=()
    ) -> FeishuTransport:
        return FeishuTransport(
            FakeLarkClient(),
            allowed_open_ids=list(open_ids),
            allow_group=allow_group,
            allowed_group_ids=list(group_ids),
        )

    def _msg(self, open_id: str) -> InboundMessage:
        return InboundMessage(
            channel_type="feishu", user_id=open_id, conversation_id="msg1", text="hi"
        )

    def test_added_open_id_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("ou_abc"))
        t.reconfigure(
            SimpleNamespace(allowed_open_ids=["ou_new"], allow_group=False, allowed_group_ids=[])
        )
        assert t.authorize(self._msg("ou_new"))
        assert not t.authorize(self._msg("ou_abc"))

    @pytest.mark.asyncio
    async def test_the_group_gate_stays_a_conjunction_after_a_reload(self):
        dispatched: list[LarkInbound] = []

        async def _dispatch(inbound):
            dispatched.append(inbound)

        t = FeishuTransport(FakeLarkClient(), allowed_open_ids=["ou_abc"], dispatch=_dispatch)
        group = LarkInbound(
            open_id="ou_abc",
            text="hi",
            message_id="m1",
            chat_type=CHAT_GROUP,
            chat_id="oc_room",
        )
        await t.receive(group)
        assert dispatched == []

        # The group id alone is not enough: allow_group is still off.
        t.reconfigure(
            SimpleNamespace(
                allowed_open_ids=["ou_abc"], allow_group=False, allowed_group_ids=["oc_room"]
            )
        )
        await t.receive(group)
        assert dispatched == []

        # The switch alone is not enough either: a different room is listed.
        t.reconfigure(
            SimpleNamespace(
                allowed_open_ids=["ou_abc"], allow_group=True, allowed_group_ids=["oc_other"]
            )
        )
        await t.receive(group)
        assert dispatched == []

        # Both halves satisfied.
        t.reconfigure(
            SimpleNamespace(
                allowed_open_ids=["ou_abc"], allow_group=True, allowed_group_ids=["oc_room"]
            )
        )
        await t.receive(group)
        assert [m.chat_id for m in dispatched] == ["oc_room"]

    @pytest.mark.asyncio
    async def test_a_p2p_turn_is_unaffected_by_the_group_gate(self):
        dispatched: list[LarkInbound] = []

        async def _dispatch(inbound):
            dispatched.append(inbound)

        t = FeishuTransport(FakeLarkClient(), allowed_open_ids=["ou_abc"], dispatch=_dispatch)
        t.reconfigure(
            SimpleNamespace(allowed_open_ids=["ou_abc"], allow_group=False, allowed_group_ids=[])
        )
        await t.receive(
            LarkInbound(
                open_id="ou_abc", text="hi", message_id="m2", chat_type=CHAT_P2P, chat_id=""
            )
        )
        assert [m.message_id for m in dispatched] == ["m2"]

    def test_a_non_list_field_keeps_the_previous_set(self):
        t = self._transport(open_ids=("ou_abc",), allow_group=True, group_ids=("oc_room",))
        t.reconfigure(
            SimpleNamespace(allowed_open_ids="ou_new", allow_group=True, allowed_group_ids="oc_new")
        )
        assert t._allowed == frozenset({"ou_abc"})
        assert t._allowed_group_ids == frozenset({"oc_room"})

    def test_a_non_bool_allow_group_keeps_the_previous_value(self):
        t = self._transport(allow_group=False)
        t.reconfigure(
            SimpleNamespace(allowed_open_ids=["ou_abc"], allow_group="yes", allowed_group_ids=[])
        )
        assert t._allow_group is False

    def test_both_allow_list_changes_are_audited(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.feishu.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(
                allowed_open_ids=["ou_new"], allow_group=True, allowed_group_ids=["oc_room"]
            )
        )
        assert _reconfigure_outcomes(audited, "feishu_transport.reconfigure") == [
            "allow_list_changed",
            "group_allow_list_changed",
            "allow_group_enabled",
        ]

    def test_configured_targets_follow_the_reload(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(allowed_open_ids=["ou_new"], allow_group=False, allowed_group_ids=[])
        )
        assert [x.target_id for x in t.configured_targets()] == ["user:ou_new"]


class TestFeishuDispatcherApplier:
    def _dispatcher(self, transport=None) -> FeishuDispatcher:
        d = FeishuDispatcher(
            sessions=FakeSessions(pct=90.0),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("feishu", soft_threshold_pct=80, hard_threshold_pct=95),
        )
        d.client = FakeLarkClient()
        d.transport = transport
        return d

    def test_applier_pushes_the_reloaded_allow_lists(self):
        t = FeishuTransport(FakeLarkClient(), allowed_open_ids=["ou_abc"])
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with(
                    "feishu",
                    allowed_open_ids=["ou_new"],
                    allow_group=True,
                    allowed_group_ids=["oc_room"],
                ),
                "feishu",
            )
        )
        assert t._allowed == frozenset({"ou_new"})
        assert t._allow_group is True
        assert t._allowed_group_ids == frozenset({"oc_room"})

    def test_a_degraded_feishu_section_keeps_the_previous_lists(self):
        t = FeishuTransport(FakeLarkClient(), allowed_open_ids=["ou_abc"])
        self._dispatcher(t)._on_config_change(
            _change(
                _cfg_with("feishu", degraded=frozenset({"feishu"}), allowed_open_ids=["ou_new"]),
                "feishu",
            )
        )
        assert t._allowed == frozenset({"ou_abc"})

    def test_the_subscription_is_registered(self):
        self._dispatcher()
        assert "FeishuDispatcher" in [s.name for s in live.watch().subscriptions()]

    def test_the_threshold_floor_is_one_not_zero(self):
        d = self._dispatcher()
        live.watch().prime(_cfg_with("feishu", soft_threshold_pct=0, hard_threshold_pct=0))
        assert d._thresholds() == (1, 1)

    @pytest.mark.asyncio
    async def test_a_reloaded_hard_threshold_compacts_this_turn(self):
        d = self._dispatcher()
        provider = FakeProvider()
        inbound = LarkInbound(
            open_id="ou_abc", text="hi", message_id="m1", chat_type=CHAT_P2P, chat_id=""
        )
        await d._maybe_notice(inbound, "feishu:k", provider)
        assert not provider.compacted
        live.watch().prime(_cfg_with("feishu", soft_threshold_pct=70, hard_threshold_pct=80))
        await d._maybe_notice(inbound, "feishu:k", provider)
        assert provider.compacted


# ------------------------------------------------------------------
# dm_scope is pinned per generation
# ------------------------------------------------------------------


class TestDmScopeIsPinnedPerGeneration:
    """A dm_scope flip must not re-key a conversation that is already running.

    dm_scope selects the session-key NAMESPACE, so adopting a new value between
    two turns of one conversation would mint a different key and jump the running
    DM into another session. The value is therefore read live but pinned for the
    life of a generation -- and ``/new``, the idle reset and the daily reset all
    advance the generation, which is the boundary where the new value is safe.
    """

    def test_wecom_pins_the_scope_until_the_generation_advances(self):
        d = WeComDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("wecom", soft_threshold_pct=80, hard_threshold_pct=95),
            owner_id="",
        )
        first = d._session_key("Wei")
        live.watch().prime(_cfg_with("messaging", dm_scope="unified"))
        assert d._session_key("Wei") == first
        d._conv.bump_gen("Wei")
        assert d._session_key("Wei") != first

    def test_feishu_pins_the_scope_per_route(self):
        d = FeishuDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("feishu", soft_threshold_pct=80, hard_threshold_pct=95),
        )
        route = ("direct", "ou_abc")
        first = d._session_key(route)
        live.watch().prime(_cfg_with("messaging", dm_scope="unified"))
        assert d._session_key(route) == first
        d._conv.bump_gen(route)
        assert d._session_key(route) != first


# ------------------------------------------------------------------
# End to end: a file write reaches the transport
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_config_file_write_reaches_the_wecom_transport(tmp_path, monkeypatch):
    """The whole path: write config.json -> ConfigWatch -> transport frozenset.

    Driven through ``refresh_now`` rather than the poll interval so the test does
    not sleep, and pointed at temp files exactly as the loader's own tests do.
    """
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"wecom": {"allowed_users": [{"userid": "Wei"}]}}))
    local = tmp_path / "config.local.json"
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_path)
    monkeypatch.setattr("kiro_crew.config.loader.config_local_path", lambda: local)

    transport = WeComTransport(FakeWeComClient(), allowed_users=["Wei"], owner_id="")
    dispatcher = WeComDispatcher(
        sessions=FakeSessions(),
        ctx_builder=SimpleNamespace(),
        cfg=_boot_cfg("wecom", soft_threshold_pct=80, hard_threshold_pct=95),
        owner_id="",
    )
    dispatcher.client = FakeWeComClient()
    dispatcher.transport = transport

    watch = ConfigWatch(poll_interval_secs=0.05)
    sub = dispatcher._config_sub
    watch.subscribe(*sub.prefixes, callback=dispatcher._on_config_change, name=sub.name)
    await watch.refresh_now()  # establish the baseline

    cfg_path.write_text(
        json.dumps({"wecom": {"allowed_users": [{"userid": "Ming"}], "allow_all_users": False}})
    )
    change = await watch.refresh_now()
    assert change is not None and change.touched("wecom")
    assert transport._allowed == frozenset({"Ming"})
