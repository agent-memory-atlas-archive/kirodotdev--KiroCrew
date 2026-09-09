"""Config hot-reload for the Teams / iMessage / WhatsApp channels.

Each channel gets the same three questions asked of it:

* a reloaded THRESHOLD is read at point of use, and the loader's pair
  normalization is re-run so an inverted pair cannot make the soft nudge
  unreachable;
* a reloaded ALLOW-LIST (or DM policy, or group rule set) reaches the live
  transport, so an added identity is authorized and a removed one is refused
  without a restart;
* a DEGRADED section, or a value of the wrong shape, keeps the PREVIOUS
  authorization state -- these are fail-closed boundaries, and rebuilding them
  from a document the loader could not parse would lock out every intended
  sender.

Teams gets a fourth question, because its allow-list has THREE holders: the
transport's frozen roster, the dispatcher's copy and the session-resume owner.
Two of them agreeing and the third stale is exactly the state that lets a
removed identity keep listing dashboard sessions, so the applier is checked for
agreement across all three.

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
from kiro_crew.config.schema import requires_restart
from kiro_crew.imessage.transport import IMessageTransport, allowed_handles_from_config
from kiro_crew.imessage.transport_dispatch import IMessageDispatcher
from kiro_crew.messaging.transport import InboundMessage
from kiro_crew.teams.transport import TeamsTransport, allowed_emails_from_config
from kiro_crew.teams.transport_dispatch import TeamsDispatcher
from kiro_crew.whatsapp.jids import OwnIdentity
from kiro_crew.whatsapp.transport import WhatsAppTransport
from kiro_crew.whatsapp.transport_dispatch import WhatsAppDispatcher

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


class FakeTeamsClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, conversation_id: str, text: str, **kw) -> str:
        self.sent.append((conversation_id, text))
        return "m1"


class FakeIMessageClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send(self, handle: str, text: str) -> str:
        self.sent.append((handle, text))
        return "m1"

    def set_message_handler(self, fn) -> None:
        pass


class FakeWhatsAppClient:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.on_message = None
        # ``may_send_to`` asks the linked account whether a JID is its own thread,
        # which is the ``self`` policy's whole answer.
        self.me = OwnIdentity(jid="15559999999@s.whatsapp.net", lid="")

    async def send_text(self, jid: str, text: str) -> str:
        self.sent.append((jid, text))
        return "m1"


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


def _prime_live(section_name: str, **section_kw) -> None:
    """Make the process watcher's snapshot carry one replaced section.

    Thresholds and the ``messaging.*`` rotation fields are read at point of use
    from the live snapshot, so a test exercising one has to put the value THERE,
    not only in the dispatcher's boot copy.
    """
    live.watch().prime(_cfg_with(section_name, **section_kw))


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
# Teams
# ------------------------------------------------------------------


class TestTeamsAllowList:
    def test_normalizer_lowercases_and_drops_non_strings(self):
        assert allowed_emails_from_config(["A@x.com", "", 7, "b@X.com"]) == ["a@x.com", "b@x.com"]

    def test_normalizer_reports_a_non_list_as_unusable(self):
        assert allowed_emails_from_config("a@x.com") is None
        assert allowed_emails_from_config(None) is None

    def _transport(self, *, allowed=("a@x.com",)) -> TeamsTransport:
        return TeamsTransport(FakeTeamsClient(), allowed_emails=list(allowed))

    def _msg(self, email: str) -> InboundMessage:
        return InboundMessage(
            channel_type="teams", user_id=email, conversation_id="conv1", text="hi"
        )

    def test_added_identity_is_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("a@x.com"))
        assert not t.authorize(self._msg("b@x.com"))
        t.reconfigure(SimpleNamespace(allowed_emails=["b@x.com"]))
        assert t.authorize(self._msg("b@x.com"))
        assert not t.authorize(self._msg("a@x.com"))

    def test_a_reloaded_identity_matches_case_insensitively(self):
        t = self._transport(allowed=())
        t.reconfigure(SimpleNamespace(allowed_emails=["MiXeD@X.com"]))
        assert t.authorize(self._msg("mixed@x.com"))

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_emails="b@x.com"))
        assert t.authorize(self._msg("a@x.com"))
        assert not t.authorize(self._msg("b@x.com"))

    def test_a_roster_change_is_audited_by_count(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.teams.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_emails=["b@x.com"]))
        rows = [a for a in audited if a.get("operation") == "teams_transport.reconfigure"]
        assert [r["outcome"] for r in rows] == ["allow_list_changed"]
        assert "b@x.com" not in rows[0]["resources"]

    def test_an_unchanged_roster_is_not_audited(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.teams.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_emails=["A@X.com"]))
        assert _reconfigure_outcomes(audited, "teams_transport.reconfigure") == []

    def test_values_are_never_logged_only_counts(self, caplog):
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.teams.transport"):
            t.reconfigure(SimpleNamespace(allowed_emails=["secret@x.com"]))
        assert "secret@x.com" not in caplog.text


class TestTeamsDispatcherApplier:
    def _dispatcher(self, transport=None, *, allowed=("a@x.com",)) -> TeamsDispatcher:
        d = TeamsDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("teams", soft_threshold_pct=80, hard_threshold_pct=95),
            allowed_emails=set(allowed),
        )
        d.client = FakeTeamsClient()
        d.transport = transport
        return d

    def test_all_three_allow_list_copies_agree_after_one_apply(self):
        transport = TeamsTransport(FakeTeamsClient(), allowed_emails=["a@x.com"])
        d = self._dispatcher(transport)
        assert d._session_resume.owner_id == "a@x.com"

        d._on_config_change(_change(_cfg_with("teams", allowed_emails=["b@x.com"]), "teams"))

        assert transport._allowed == frozenset({"b@x.com"})
        assert d._allowed_emails == frozenset({"b@x.com"})
        assert d._session_resume.owner_id == "b@x.com"
        assert d._session_resume.is_owner("B@X.com")
        assert not d._session_resume.is_owner("a@x.com")

    def test_a_second_identity_removes_the_session_resume_owner(self):
        d = self._dispatcher(TeamsTransport(FakeTeamsClient(), allowed_emails=["a@x.com"]))
        d._on_config_change(
            _change(_cfg_with("teams", allowed_emails=["a@x.com", "b@x.com"]), "teams")
        )
        assert d._session_resume.owner_id == ""
        assert not d._session_resume.is_owner("a@x.com")

    def test_a_degraded_section_keeps_every_copy(self):
        transport = TeamsTransport(FakeTeamsClient(), allowed_emails=["a@x.com"])
        d = self._dispatcher(transport)
        d._on_config_change(
            _change(
                _cfg_with("teams", degraded=frozenset({"teams"}), allowed_emails=["b@x.com"]),
                "teams",
            )
        )
        assert transport._allowed == frozenset({"a@x.com"})
        assert d._allowed_emails == frozenset({"a@x.com"})
        assert d._session_resume.owner_id == "a@x.com"

    def test_a_change_outside_teams_is_ignored(self):
        transport = TeamsTransport(FakeTeamsClient(), allowed_emails=["a@x.com"])
        d = self._dispatcher(transport)
        d._on_config_change(_change(_cfg_with("teams", allowed_emails=[]), "messaging.dm_scope"))
        assert transport._allowed == frozenset({"a@x.com"})

    def test_thresholds_follow_the_live_config_and_are_normalized(self):
        d = self._dispatcher()
        _prime_live("teams", soft_threshold_pct=10, hard_threshold_pct=20)
        assert d._thresholds() == (10, 20)
        # Inverted pair: the loader's normalization keeps the soft nudge reachable.
        _prime_live("teams", soft_threshold_pct=90, hard_threshold_pct=50)
        soft, hard = d._thresholds()
        assert soft <= hard

    def test_thresholds_fall_back_to_the_boot_copy_when_the_live_read_fails(self, monkeypatch):
        d = self._dispatcher()
        monkeypatch.setattr(
            "kiro_crew.imessage.transport_dispatch.live.snapshot",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            raising=False,
        )
        monkeypatch.setattr(
            "kiro_crew.teams.transport_dispatch.live.snapshot",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert d._thresholds() == (80, 95)

    def test_dm_scope_is_pinned_for_a_generation_and_adopted_at_the_boundary(self):
        d = self._dispatcher()
        _prime_live("teams")  # messaging.dm_scope default
        first = d._dm_scope("a@x.com", 0)
        _prime_live("teams")
        cfg = KiroCrewConfig()
        live.watch().prime(replace(cfg, messaging=replace(cfg.messaging, dm_scope="unified")))
        assert d._dm_scope("a@x.com", 0) == first  # same generation: pinned
        assert d._dm_scope("a@x.com", 1) == "unified"  # new generation: adopted


# ------------------------------------------------------------------
# iMessage
# ------------------------------------------------------------------


class TestIMessageAllowList:
    def test_normalizer_reuses_handle_normalization(self):
        assert allowed_handles_from_config(["+1 (555) 010-0000", "", 7]) == ["+15550100000"]

    def test_normalizer_reports_a_non_list_as_unusable(self):
        assert allowed_handles_from_config("+15550100000") is None
        assert allowed_handles_from_config(None) is None

    def _transport(self, *, allowed=("+15550100000",)) -> IMessageTransport:
        return IMessageTransport(FakeIMessageClient(), allowed_handles=list(allowed))

    def _msg(self, handle: str) -> InboundMessage:
        return InboundMessage(
            channel_type="imessage", user_id=handle, conversation_id=handle, text="hi"
        )

    def test_added_handle_is_authorized_and_removed_one_refused(self):
        t = self._transport()
        assert t.authorize(self._msg("+15550100000"))
        assert not t.authorize(self._msg("+15550100001"))
        t.reconfigure(SimpleNamespace(allowed_handles=["+1 (555) 010-0001"]))
        assert t.authorize(self._msg("+15550100001"))
        assert not t.authorize(self._msg("+15550100000"))

    def test_an_empty_roster_is_adopted_and_denies_everyone(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_handles=[]))
        assert not t.authorize(self._msg("+15550100000"))

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_handles="+15550100001"))
        assert t.authorize(self._msg("+15550100000"))

    def test_a_roster_change_is_audited_without_the_handle(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            "kiro_crew.imessage.transport.sel",
            lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw)),
        )
        t = self._transport()
        t.reconfigure(SimpleNamespace(allowed_handles=["+15550100001"]))
        rows = [a for a in audited if a.get("operation") == "imessage_transport.reconfigure"]
        assert [r["outcome"] for r in rows] == ["allow_list_changed"]
        assert "5550100001" not in rows[0]["resources"]

    def test_values_are_never_logged(self, caplog):
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.imessage.transport"):
            t.reconfigure(SimpleNamespace(allowed_handles=["+15550109999"]))
        assert "5550109999" not in caplog.text


class TestIMessageDispatcherApplier:
    def _dispatcher(self, transport=None) -> IMessageDispatcher:
        d = IMessageDispatcher(
            sessions=FakeSessions(),
            ctx_builder=SimpleNamespace(),
            cfg=_boot_cfg("imessage", soft_threshold_pct=80, hard_threshold_pct=95),
        )
        d.client = FakeIMessageClient()
        d.transport = transport
        return d

    def test_a_reloaded_allow_list_reaches_the_transport(self):
        transport = IMessageTransport(FakeIMessageClient(), allowed_handles=["+15550100000"])
        d = self._dispatcher(transport)
        d._on_config_change(
            _change(_cfg_with("imessage", allowed_handles=["+15550100001"]), "imessage")
        )
        assert transport._allowed == frozenset({"+15550100001"})

    def test_a_degraded_section_keeps_the_previous_allow_list(self):
        transport = IMessageTransport(FakeIMessageClient(), allowed_handles=["+15550100000"])
        d = self._dispatcher(transport)
        d._on_config_change(
            _change(
                _cfg_with(
                    "imessage", degraded=frozenset({"imessage"}), allowed_handles=["+15550100001"]
                ),
                "imessage",
            )
        )
        assert transport._allowed == frozenset({"+15550100000"})

    def test_thresholds_follow_the_live_config(self):
        d = self._dispatcher()
        _prime_live("imessage", soft_threshold_pct=30, hard_threshold_pct=40)
        assert d._thresholds() == (30, 40)

    def test_an_inverted_threshold_pair_is_normalized(self):
        d = self._dispatcher()
        _prime_live("imessage", soft_threshold_pct=99, hard_threshold_pct=10)
        soft, hard = d._thresholds()
        assert soft <= hard


# ------------------------------------------------------------------
# WhatsApp
# ------------------------------------------------------------------


class TestWhatsAppAuthorizationReload:
    def _transport(self, *, policy="allowlist", wa_ids=("15550100000",), groups=None):
        async def _dispatch(msg):
            return None

        return WhatsAppTransport(
            FakeWhatsAppClient(),
            _dispatch,
            dm_policy=policy,
            allowed_wa_ids=list(wa_ids),
            groups=list(groups or []),
        )

    def test_added_wa_id_may_be_sent_to_and_removed_one_refused(self):
        t = self._transport()
        assert t.may_send_to("15550100000@s.whatsapp.net")
        assert not t.may_send_to("15550100001@s.whatsapp.net")
        t.reconfigure(
            SimpleNamespace(dm_policy="allowlist", allowed_wa_ids=["15550100001"], groups=[])
        )
        assert t.may_send_to("15550100001@s.whatsapp.net")
        assert not t.may_send_to("15550100000@s.whatsapp.net")

    def test_a_policy_flip_to_disabled_narrows_immediately(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(dm_policy="disabled", allowed_wa_ids=["15550100000"], groups=[])
        )
        assert not t.may_send_to("15550100000@s.whatsapp.net")

    def test_an_unknown_policy_string_is_adopted_and_denies_everyone(self):
        """Adopted, because ``_dm_policy`` fails closed on what it cannot name.

        Keeping the previous (wider) policy would be the unsafe direction here.
        """
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(dm_policy="whatever", allowed_wa_ids=["15550100000"], groups=[])
        )
        assert not t.may_send_to("15550100000@s.whatsapp.net")

    def test_a_non_string_policy_keeps_the_previous_one(self):
        t = self._transport()
        t.reconfigure(SimpleNamespace(dm_policy=None, allowed_wa_ids=["15550100000"], groups=[]))
        assert t.may_send_to("15550100000@s.whatsapp.net")

    def test_a_non_list_roster_keeps_the_previous_allow_list(self):
        t = self._transport()
        t.reconfigure(
            SimpleNamespace(dm_policy="allowlist", allowed_wa_ids="15550100001", groups=[])
        )
        assert t.may_send_to("15550100000@s.whatsapp.net")

    def test_a_reloaded_group_becomes_configured(self):
        t = self._transport(groups=[])
        assert not t.group_gate.configured("123@g.us")
        t.reconfigure(
            SimpleNamespace(
                dm_policy="allowlist",
                allowed_wa_ids=["15550100000"],
                groups=[{"jid": "123@g.us", "mode": "mention"}],
            )
        )
        assert t.group_gate.configured("123@g.us")

    def test_a_removed_group_stops_being_configured(self):
        t = self._transport(groups=[{"jid": "123@g.us", "mode": "mention"}])
        t.reconfigure(
            SimpleNamespace(dm_policy="allowlist", allowed_wa_ids=["15550100000"], groups=[])
        )
        assert not t.group_gate.configured("123@g.us")

    def test_an_unknown_group_mode_falls_back_to_mention(self):
        t = self._transport(groups=[])
        t.reconfigure(
            SimpleNamespace(
                dm_policy="allowlist",
                allowed_wa_ids=[],
                groups=[{"jid": "123@g.us", "mode": "shout"}],
            )
        )
        assert t._group_rules[0]["mode"] == "mention"

    def test_an_unchanged_group_set_keeps_the_same_gate(self):
        """The gate holds the unprompted-reply cooldown clock, so it is not churned."""
        rules = [{"jid": "123@g.us", "mode": "mention"}]
        t = self._transport(groups=rules)
        gate = t.group_gate
        t.reconfigure(
            SimpleNamespace(dm_policy="allowlist", allowed_wa_ids=["15550100000"], groups=rules)
        )
        assert t.group_gate is gate

    def test_values_are_never_logged_only_counts(self, caplog):
        t = self._transport()
        with caplog.at_level("INFO", logger="kiro_crew.whatsapp.transport"):
            t.reconfigure(
                SimpleNamespace(dm_policy="allowlist", allowed_wa_ids=["15550109999"], groups=[])
            )
        assert "5550109999" not in caplog.text


class TestWhatsAppDispatcherApplier:
    def _dispatcher(self, transport=None) -> WhatsAppDispatcher:
        d = WhatsAppDispatcher(
            _boot_cfg("whatsapp", soft_threshold_pct=80, hard_threshold_pct=95),
            FakeSessions(),
            SimpleNamespace(),
            approval_mode="interactive",
        )
        d.client = FakeWhatsAppClient()
        d.transport = transport
        return d

    def _transport(self, **kw):
        async def _dispatch(msg):
            return None

        return WhatsAppTransport(FakeWhatsAppClient(), _dispatch, **kw)

    def test_a_reloaded_policy_and_roster_reach_the_transport(self):
        transport = self._transport(dm_policy="allowlist", allowed_wa_ids=["15550100000"])
        d = self._dispatcher(transport)
        d._on_config_change(
            _change(
                _cfg_with("whatsapp", dm_policy="disabled", allowed_wa_ids=["15550100001"]),
                "whatsapp",
            )
        )
        assert transport._dm_policy == "disabled"
        assert transport._allowed == frozenset({"15550100001@s.whatsapp.net"})

    def test_a_degraded_section_keeps_policy_roster_and_groups(self):
        transport = self._transport(
            dm_policy="allowlist",
            allowed_wa_ids=["15550100000"],
            groups=[{"jid": "123@g.us", "mode": "mention"}],
        )
        d = self._dispatcher(transport)
        d._on_config_change(
            _change(
                _cfg_with(
                    "whatsapp",
                    degraded=frozenset({"whatsapp"}),
                    dm_policy="open",
                    allowed_wa_ids=[],
                    groups=[],
                ),
                "whatsapp",
            )
        )
        assert transport._dm_policy == "allowlist"
        assert transport._allowed == frozenset({"15550100000@s.whatsapp.net"})
        assert transport.group_gate.configured("123@g.us")

    def test_thresholds_follow_the_live_config_and_are_normalized(self):
        d = self._dispatcher()
        _prime_live("whatsapp", soft_threshold_pct=25, hard_threshold_pct=35)
        assert d._thresholds() == (25, 35)
        _prime_live("whatsapp", soft_threshold_pct=95, hard_threshold_pct=5)
        soft, hard = d._thresholds()
        assert soft <= hard

    def test_db_path_is_declared_boot_only(self):
        """The device store is opened once at connect and holds the account key."""
        assert requires_restart("whatsapp.db_path") is True


# ------------------------------------------------------------------
# End to end: a config file write reaches the transports
# ------------------------------------------------------------------


def _point_loader_at(tmp_path, monkeypatch, doc: dict):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(doc))
    local = tmp_path / "config.local.json"
    monkeypatch.setattr("kiro_crew.config.loader.config_path", lambda: cfg_path)
    monkeypatch.setattr("kiro_crew.config.loader.config_local_path", lambda: local)
    return cfg_path


async def _drive(dispatcher, cfg_path, doc: dict) -> None:
    """Baseline, rewrite, then one forced reload -- no sleeping on the poll."""
    watch = ConfigWatch(poll_interval_secs=0.05)
    sub = dispatcher._config_sub
    watch.subscribe(*sub.prefixes, callback=dispatcher._on_config_change, name=sub.name)
    await watch.refresh_now()
    cfg_path.write_text(json.dumps(doc))
    await watch.refresh_now()


@pytest.mark.asyncio
async def test_a_config_file_write_reaches_the_teams_transport(tmp_path, monkeypatch):
    cfg_path = _point_loader_at(tmp_path, monkeypatch, {"teams": {"allowed_emails": ["a@x.com"]}})
    transport = TeamsTransport(FakeTeamsClient(), allowed_emails=["a@x.com"])
    d = TeamsDispatcher(
        sessions=FakeSessions(),
        ctx_builder=SimpleNamespace(),
        cfg=_boot_cfg("teams", soft_threshold_pct=80, hard_threshold_pct=95),
        allowed_emails={"a@x.com"},
    )
    d.client = FakeTeamsClient()
    d.transport = transport

    await _drive(d, cfg_path, {"teams": {"allowed_emails": ["b@x.com"]}})

    assert transport._allowed == frozenset({"b@x.com"})
    assert d._session_resume.owner_id == "b@x.com"


@pytest.mark.asyncio
async def test_a_config_file_write_reaches_the_imessage_transport(tmp_path, monkeypatch):
    cfg_path = _point_loader_at(
        tmp_path, monkeypatch, {"imessage": {"allowed_handles": ["+15550100000"]}}
    )
    transport = IMessageTransport(FakeIMessageClient(), allowed_handles=["+15550100000"])
    d = IMessageDispatcher(
        sessions=FakeSessions(),
        ctx_builder=SimpleNamespace(),
        cfg=_boot_cfg("imessage", soft_threshold_pct=80, hard_threshold_pct=95),
    )
    d.client = FakeIMessageClient()
    d.transport = transport

    await _drive(d, cfg_path, {"imessage": {"allowed_handles": ["+15550100001"]}})

    assert transport._allowed == frozenset({"+15550100001"})


@pytest.mark.asyncio
async def test_a_config_file_write_reaches_the_whatsapp_transport(tmp_path, monkeypatch):
    cfg_path = _point_loader_at(
        tmp_path, monkeypatch, {"whatsapp": {"dm_policy": "allowlist", "allowed_wa_ids": []}}
    )

    async def _dispatch(msg):
        return None

    transport = WhatsAppTransport(
        FakeWhatsAppClient(), _dispatch, dm_policy="allowlist", allowed_wa_ids=[]
    )
    d = WhatsAppDispatcher(
        _boot_cfg("whatsapp", soft_threshold_pct=80, hard_threshold_pct=95),
        FakeSessions(),
        SimpleNamespace(),
        approval_mode="interactive",
    )
    d.client = FakeWhatsAppClient()
    d.transport = transport

    await _drive(
        d,
        cfg_path,
        {
            "whatsapp": {
                "dm_policy": "allowlist",
                "allowed_wa_ids": ["15550100001"],
                "groups": [{"jid": "123@g.us", "mode": "mention"}],
            }
        },
    )

    assert transport.may_send_to("15550100001@s.whatsapp.net")
    assert transport.group_gate.configured("123@g.us")
