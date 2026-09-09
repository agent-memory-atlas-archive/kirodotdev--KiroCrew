"""Config hot-reload for the CHANNEL-BOOT area.

The claim under test is that a channel's CONNECTION parameters are applied by
restarting that one channel in process, and that nothing else about a config
reload touches a live socket:

* ``GatewayOrchestrator.__init__`` hoists each channel through its own
  ``_hoist_<channel>`` method rather than one 130-line block, and every attribute
  the channel's own boot factory consumes is still set (both halves of that
  comparison are DERIVED from the source by AST, so a dropped attribute goes
  red -- a hardcoded list would only pin a copy that goes stale);
* :meth:`GatewayOrchestrator.restart_channel` closes the previous handle,
  re-hoists from the RELOADED section and starts a new one, and the factory's
  ``dispatcher.transport`` re-point is what keeps the allow-list appliers
  pushing at the live transport instead of the closed one;
* a ``boot_keys`` path restarts the channel and any other path in the same
  section does NOT (the live appliers rely on that, or their fields would
  bounce a connection);
* the Slack applier reconciles tracked/open channels, activations and the
  phase-emoji table in place, and never restarts Slack -- its socket is
  host-managed and its tokens are not in ``config.json``;
* ``enterprise.reload_allowed_team_ids`` keeps the validated read as the sole
  source, so it cannot widen from a raw load and fails closed on a degraded one;
* ``channel_restart_required`` answers from the config SCHEMA, and the WhatsApp
  saver reports it per FIELD rather than unconditionally.

Reloads are simulated with ``live.watch().prime(cfg)`` plus a direct applier
call -- the shape a subscriber really receives. The poll task is never started.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kiro_crew.channels import builtin_channel_descriptors
from kiro_crew.config import live
from kiro_crew.config.live import ConfigChange
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.dashboard.channel_folders import channel_restart_required
from kiro_crew.messaging import registry
from kiro_crew.slack import gateway as gw
from kiro_crew.slack import handler as slack_handler
from kiro_crew.slack.gateway import GatewayOrchestrator

# ------------------------------------------------------------------
# Fixtures + helpers
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_watcher():
    """Every test gets an unstarted process watcher, and leaves none behind."""
    live.reset_for_tests()
    yield
    live.reset_for_tests()


def _make_orchestrator() -> GatewayOrchestrator:
    """A Slack-less orchestrator: the hoists and appliers need nothing more."""
    cfg = KiroCrewConfig()
    with patch.object(cfg, "load_credentials", return_value={"KIROCREW_OWNER_ID": "U_OWNER"}):
        return GatewayOrchestrator(cfg, no_dashboard=True, no_crons=True, no_open=True)


def _cfg_with(section_name: str, degraded: frozenset[str] = frozenset(), **section_kw):
    """A real :class:`KiroCrewConfig` carrying one replaced section."""
    cfg = KiroCrewConfig()
    section = replace(getattr(cfg, section_name), **section_kw)
    return replace(cfg, _degraded_sections=degraded, **{section_name: section})


def _change(new, *changed: str, old=None) -> ConfigChange:
    """The shape a subscriber receives from one reload."""
    return ConfigChange(old=old, new=new, changed=frozenset(changed))


def _bootable_types() -> tuple[str, ...]:
    return tuple(d.channel_type for d in registry.bootable(builtin_channel_descriptors()))


# ------------------------------------------------------------------
# 1. Hoist-attribute parity
# ------------------------------------------------------------------


def _self_attrs_assigned(func) -> set[str]:
    """Every ``self.<name>`` assigned (or annotated-assigned) in *func*."""
    src = inspect.getsource(func)
    tree = ast.parse(inspect.cleandoc(src) if src.startswith("def ") else _dedent(src))
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                names.add(t.attr)
    return names


def _dedent(src: str) -> str:
    lines = src.splitlines()
    pad = min((len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()), default=0)
    return "\n".join(ln[pad:] for ln in lines)


def _orch_attrs_consumed(channel_type: str) -> set[str]:
    """``orch._<channel>_*`` names the channel's own boot factory reads.

    Derived from the factory's source rather than listed here, so an attribute
    the extraction dropped surfaces as a name nothing hoists. Both access
    shapes the factories use are collected: a plain ``orch._x`` attribute load
    and a ``getattr(orch, "_x", default)`` with a literal name.
    """
    prefix = f"_{channel_type}_"
    paths = [
        Path(gw.__file__).with_name("gateway.py").parent.parent / channel_type / "gateway.py",
        Path(gw.__file__),
    ]
    found: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr.startswith(prefix):
                found.add(node.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and node.args[1].value.startswith(prefix)
            ):
                found.add(node.args[1].value)
    return found


#: Attributes a channel's boot factory reads that NO ``_hoist_*`` sets, and did
#: not set before the extraction either -- the factory supplies its own default,
#: so they are optional hooks rather than hoist outputs. Kept as a short, named
#: exemption list (never a copy of the attribute list) so a NEWLY unhoisted
#: attribute fails and has to be justified here on purpose.
_HOIST_EXEMPT: dict[str, frozenset[str]] = {
    # ``getattr(orch, "_weixin_home", "") or str(data_home())``.
    "weixin": frozenset({"_weixin_home"}),
}


class TestHoistAttributeParity:
    def test_every_bootable_channel_has_a_hoist(self):
        missing = [c for c in _bootable_types() if not hasattr(GatewayOrchestrator, f"_hoist_{c}")]
        assert missing == [], f"bootable channels without a _hoist_<channel>: {missing}"

    def test_hoist_sets_the_enabled_flag_the_factory_gates_on(self):
        for channel_type in _bootable_types():
            hoist = getattr(GatewayOrchestrator, f"_hoist_{channel_type}")
            assert f"_{channel_type}_enabled" in _self_attrs_assigned(hoist), channel_type

    def test_hoist_sets_every_channel_attribute_its_boot_factory_consumes(self):
        """The parity claim: nothing the old block set was lost in the split.

        Both sides are DERIVED -- the consumed set from each channel factory's
        own reads, the produced set from the hoist's assignments -- so a
        moved-but-forgotten attribute cannot be papered over by editing a list
        in this test. The only list here is ``_HOIST_EXEMPT``, and an addition to
        it is a deliberate statement that the factory owns that default.
        """
        gaps: dict[str, set[str]] = {}
        for channel_type in _bootable_types():
            hoist = getattr(GatewayOrchestrator, f"_hoist_{channel_type}")
            produced = _self_attrs_assigned(hoist)
            # The client mirror is lifecycle, not hoist: restart_channel clears it
            # through setattr and the factory writes the live handle back.
            exempt = _HOIST_EXEMPT.get(channel_type, frozenset()) | {f"_{channel_type}_client"}
            missing = _orch_attrs_consumed(channel_type) - produced - exempt
            if missing:
                gaps[channel_type] = missing
        assert gaps == {}, f"attributes consumed at boot but no longer hoisted: {gaps}"

    def test_the_exemption_list_names_nothing_a_hoist_already_sets(self):
        """A stale exemption would hide a later real deletion."""
        for channel_type, exempt in _HOIST_EXEMPT.items():
            hoist = getattr(GatewayOrchestrator, f"_hoist_{channel_type}")
            assert exempt & _self_attrs_assigned(hoist) == set(), channel_type

    def test_a_hoist_only_writes_its_own_channels_attributes(self):
        """A cross-write would make a one-channel restart mutate another channel."""
        others = _bootable_types()
        for channel_type in others:
            hoist = getattr(GatewayOrchestrator, f"_hoist_{channel_type}")
            foreign = {
                a
                for a in _self_attrs_assigned(hoist)
                for c in others
                if c != channel_type and a.startswith(f"_{c}_")
            }
            assert foreign == set(), f"_hoist_{channel_type} writes {foreign}"

    def test_init_no_longer_hoists_any_channel_attribute_itself(self):
        """__init__ delegates to the hoists; a re-added inline assignment is a fork.

        Two writers for one attribute is how a restart silently stops applying
        a field: ``restart_channel`` re-runs only the hoist. The ``_<channel>_client``
        mirrors are exempt -- ``restart_channel`` resets those itself.
        """
        assigned = _self_attrs_assigned(GatewayOrchestrator.__init__)
        prefixes = tuple(f"_{c}_" for c in _bootable_types())
        mirrors = {f"_{c}_client" for c in _bootable_types()}
        inline = sorted(a for a in assigned if a.startswith(prefixes) and a not in mirrors)
        assert inline == [], f"__init__ still hoists channel attributes inline: {inline}"

    def test_init_calls_exactly_one_hoist_per_bootable_channel(self):
        """A channel whose hoist is never called boots with no config at all."""
        src = inspect.getsource(GatewayOrchestrator.__init__)
        called = [
            node.func.attr
            for node in ast.walk(ast.parse(_dedent(src)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("_hoist_")
        ]
        assert sorted(called) == sorted(f"_hoist_{c}" for c in _bootable_types())
        assert len(called) == len(set(called))


# ------------------------------------------------------------------
# 2. restart_channel
# ------------------------------------------------------------------


class FakeTransport:
    """Stands in for the object a dispatcher's allow-list applier pushes at."""

    def __init__(self, allowed: list[str]) -> None:
        self.allowed = list(allowed)
        self.reconfigured: list[list[str]] = []

    def reconfigure(self, section) -> None:
        self.allowed = list(section.allowed_wa_ids)
        self.reconfigured.append(self.allowed)


class FakeClient:
    def __init__(self, transport: FakeTransport) -> None:
        self.transport = transport
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Recorder:
    """A channel whose start factory behaves like the real three-object wiring."""

    def __init__(self) -> None:
        self.dispatcher = SimpleNamespace(transport=None)
        self.started: list[list[str]] = []
        self.clients: list[FakeClient] = []
        self.fail = False

    async def start(self, orch) -> FakeClient | None:
        if self.fail:
            raise RuntimeError("factory blew up")
        allowed = list(orch._cfg.whatsapp.allowed_wa_ids)
        self.started.append(allowed)
        transport = FakeTransport(allowed)
        client = FakeClient(transport)
        # The contract CHANNELS-C documented: the factory re-points the shared
        # dispatcher at the transport it just built.
        self.dispatcher.transport = transport
        self.clients.append(client)
        return client


@pytest.fixture
def restartable(monkeypatch):
    """One bootable descriptor ('whatsapp') whose start factory is observable."""
    rec = _Recorder()
    desc = registry.ChannelDescriptor(
        channel_type="whatsapp",
        start=rec.start,
        boot_keys=frozenset({"enabled"}),
    )
    monkeypatch.setattr(gw, "builtin_channel_descriptors", lambda: (desc,))
    monkeypatch.setattr(gw, "_channel_transport_permitted", lambda member: True)
    monkeypatch.setattr(GatewayOrchestrator, "_badge_unready_channels", lambda self, descs: None)
    return rec


class TestRestartChannel:
    def _orch(self, rec, *, enabled: bool = True, allowed=("wa-1",)) -> GatewayOrchestrator:
        orch = _make_orchestrator()
        orch._channel_transports_started = True
        orch._cfg = replace(
            orch._cfg,
            whatsapp=replace(orch._cfg.whatsapp, enabled=enabled, allowed_wa_ids=list(allowed)),
        )
        return orch

    def test_unknown_channel_is_refused(self, restartable):
        orch = self._orch(restartable)
        with pytest.raises(ValueError, match="not a restartable channel"):
            asyncio.run(orch.restart_channel("nosuchchannel"))

    def test_closes_the_previous_client_and_builds_a_new_one(self, restartable):
        orch = self._orch(restartable)
        old_transport = FakeTransport(["wa-old"])
        old = FakeClient(old_transport)
        orch._channel_handles["whatsapp"] = old
        restartable.dispatcher.transport = old_transport

        cfg = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-new"])
        with patch.object(cfg, "load_credentials", return_value={}):
            client = asyncio.run(orch.restart_channel("whatsapp", cfg=cfg))

        assert old.closed is True
        assert client is not None and client is not old
        assert orch._channel_handles["whatsapp"] is client

    def test_new_transport_is_built_from_the_reloaded_section(self, restartable):
        """The factory must see the RELOADED allow-list, not the boot-time one."""
        orch = self._orch(restartable, allowed=("wa-boot",))
        cfg = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-reloaded"])
        with patch.object(cfg, "load_credentials", return_value={}):
            asyncio.run(orch.restart_channel("whatsapp", cfg=cfg))

        assert restartable.started == [["wa-reloaded"]]
        # The section on the orchestrator is replaced too, because the factories
        # and the dispatchers they build read their options off ``orch._cfg``.
        assert orch._cfg.whatsapp.allowed_wa_ids == ["wa-reloaded"]

    def test_dispatcher_transport_is_repointed_at_the_new_transport(self, restartable):
        """Otherwise the live applier keeps pushing at the CLOSED transport.

        (See ``hotreload/requests/channels_c.md``: the applier no-ops on a None
        transport, so the dangerous state is a stale non-None one.)
        """
        orch = self._orch(restartable)
        old_transport = FakeTransport(["wa-old"])
        orch._channel_handles["whatsapp"] = FakeClient(old_transport)
        restartable.dispatcher.transport = old_transport

        cfg = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-new"])
        with patch.object(cfg, "load_credentials", return_value={}):
            client = asyncio.run(orch.restart_channel("whatsapp", cfg=cfg))

        assert restartable.dispatcher.transport is not old_transport
        assert restartable.dispatcher.transport is client.transport

        # And a push after the restart lands on the live transport.
        restartable.dispatcher.transport.reconfigure(SimpleNamespace(allowed_wa_ids=["wa-pushed"]))
        assert client.transport.allowed == ["wa-pushed"]
        assert old_transport.allowed == ["wa-old"]

    def test_a_channel_disabled_by_the_reload_ends_closed(self, restartable):
        orch = self._orch(restartable)
        old = FakeClient(FakeTransport([]))
        orch._channel_handles["whatsapp"] = old

        cfg = _cfg_with("whatsapp", enabled=False)
        with patch.object(cfg, "load_credentials", return_value={}):
            assert asyncio.run(orch.restart_channel("whatsapp", cfg=cfg)) is None

        assert old.closed is True
        assert "whatsapp" not in orch._channel_handles
        assert restartable.started == []

    def test_a_governance_deny_ends_closed_without_starting(self, restartable, monkeypatch):
        monkeypatch.setattr(gw, "_channel_transport_permitted", lambda member: False)
        orch = self._orch(restartable)
        cfg = _cfg_with("whatsapp", enabled=True)
        with patch.object(cfg, "load_credentials", return_value={}):
            assert asyncio.run(orch.restart_channel("whatsapp", cfg=cfg)) is None
        assert restartable.started == []

    def test_a_close_that_raises_does_not_stop_the_restart(self, restartable):
        orch = self._orch(restartable)

        class Angry(FakeClient):
            async def close(self) -> None:
                raise RuntimeError("close failed")

        orch._channel_handles["whatsapp"] = Angry(FakeTransport([]))
        cfg = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-new"])
        with patch.object(cfg, "load_credentials", return_value={}):
            client = asyncio.run(orch.restart_channel("whatsapp", cfg=cfg))
        assert client is not None
        assert restartable.started == [["wa-new"]]

    def test_cfg_defaults_to_the_watcher_snapshot(self, restartable):
        """No explicit cfg: the primed snapshot is what gets hoisted."""
        orch = self._orch(restartable, allowed=("wa-boot",))
        primed = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-primed"])
        with patch.object(primed, "load_credentials", return_value={}):
            live.watch().prime(primed)
            asyncio.run(orch.restart_channel("whatsapp"))
        assert restartable.started == [["wa-primed"]]


# ------------------------------------------------------------------
# 3. boot-key vs live-key discrimination
# ------------------------------------------------------------------


class TestBootKeyVsLiveKey:
    def test_changed_boot_keys_matches_on_the_first_segment_under_the_section(self):
        desc = registry.ChannelDescriptor(
            channel_type="telegram", boot_keys=frozenset({"bot_token", "accounts"})
        )
        assert registry.changed_boot_keys(desc, {"telegram.bot_token"}) == frozenset({"bot_token"})
        # Nested leaf under a boot key still counts.
        assert registry.changed_boot_keys(desc, {"telegram.accounts.main.token"}) == frozenset(
            {"accounts"}
        )
        # A live field of the same section, and another section entirely.
        assert registry.changed_boot_keys(desc, {"telegram.allowed_user_ids"}) == frozenset()
        assert registry.changed_boot_keys(desc, {"discord.bot_token"}) == frozenset()

    def test_every_bootable_descriptor_declares_boot_keys(self):
        bare = [
            d.channel_type
            for d in registry.bootable(builtin_channel_descriptors())
            if not d.boot_keys
        ]
        assert bare == [], f"bootable channels with no boot_keys (never restart): {bare}"

    def test_slack_declares_none_because_its_lifecycle_is_host_managed(self):
        slack = next(d for d in builtin_channel_descriptors() if d.channel_type == "slack")
        assert slack.start is None
        assert slack.boot_keys == frozenset()

    def _orch_with_recorder(self, restartable):
        orch = _make_orchestrator()
        orch._channel_transports_started = True
        calls: list[tuple[str, object]] = []

        async def fake_restart(channel_type, *, cfg=None):
            calls.append((channel_type, cfg))
            return None

        orch.restart_channel = fake_restart  # type: ignore[method-assign]
        return orch, calls

    def test_a_boot_key_change_restarts_the_channel(self, restartable):
        orch, calls = self._orch_with_recorder(restartable)
        cfg = _cfg_with("whatsapp", enabled=False)
        asyncio.run(orch._on_channel_config_change(_change(cfg, "whatsapp.enabled")))
        assert [c[0] for c in calls] == ["whatsapp"]
        assert calls[0][1] is cfg

    def test_a_live_key_change_leaves_the_transport_alone(self, restartable):
        orch, calls = self._orch_with_recorder(restartable)
        cfg = _cfg_with("whatsapp", allowed_wa_ids=["wa-9"])
        asyncio.run(
            orch._on_channel_config_change(
                _change(cfg, "whatsapp.allowed_wa_ids", "whatsapp.dm_policy")
            )
        )
        assert calls == []

    def test_a_mixed_change_restarts_once(self, restartable):
        orch, calls = self._orch_with_recorder(restartable)
        cfg = _cfg_with("whatsapp", enabled=True, allowed_wa_ids=["wa-9"])
        asyncio.run(
            orch._on_channel_config_change(
                _change(cfg, "whatsapp.enabled", "whatsapp.allowed_wa_ids")
            )
        )
        assert [c[0] for c in calls] == ["whatsapp"]

    def test_a_change_before_the_boot_loop_is_retained_not_dropped(self, restartable):
        """The watcher is armed at dashboard init, several awaited steps before the
        transports start, so a CLI write in that window is a real path: it must
        not restart a channel that has not booted, and it must not be lost."""
        orch, calls = self._orch_with_recorder(restartable)
        orch._channel_transports_started = False
        first = _cfg_with("whatsapp", enabled=True)
        second = _cfg_with("whatsapp", enabled=False)
        asyncio.run(orch._on_channel_config_change(_change(first, "whatsapp.enabled")))
        asyncio.run(orch._on_channel_config_change(_change(second, "whatsapp.enabled")))
        assert calls == []
        # Consecutive changes collapse to the newest, whose `new` is the current file.
        assert orch._pending_channel_change is not None
        assert orch._pending_channel_change.new is second

    def test_the_retained_change_is_replayed_once_transports_are_up(self, restartable):
        orch, calls = self._orch_with_recorder(restartable)
        orch._channel_transports_started = False
        cfg = _cfg_with("whatsapp", enabled=False)
        asyncio.run(orch._on_channel_config_change(_change(cfg, "whatsapp.enabled")))
        assert calls == []
        # What the boot loop does right after registry.start_channels returns.
        orch._channel_transports_started = True
        pending, orch._pending_channel_change = orch._pending_channel_change, None
        asyncio.run(orch._on_channel_config_change(pending))
        assert [c[0] for c in calls] == ["whatsapp"]
        assert calls[0][1] is cfg
        assert orch._pending_channel_change is None

    def test_the_boot_loop_replays_the_retained_change_right_after_starting(self):
        """Pin the ORDER in the real boot loop: the flag flips after start_channels
        returns and the retained change is dispatched immediately after it."""
        import inspect

        src = inspect.getsource(gw.GatewayOrchestrator._start_channel_transports)
        started = src.index("self._channel_transports_started = True")
        replay = src.index("await self._on_channel_config_change(pending)")
        assert src.index("registry.start_channels(") < started < replay

    def test_one_channel_failing_does_not_skip_the_others(self, monkeypatch):
        descs = (
            registry.ChannelDescriptor(
                channel_type="whatsapp", start=_Recorder().start, boot_keys=frozenset({"enabled"})
            ),
            registry.ChannelDescriptor(
                channel_type="discord", start=_Recorder().start, boot_keys=frozenset({"enabled"})
            ),
        )
        monkeypatch.setattr(gw, "builtin_channel_descriptors", lambda: descs)
        orch = _make_orchestrator()
        orch._channel_transports_started = True
        seen: list[str] = []

        async def fake_restart(channel_type, *, cfg=None):
            seen.append(channel_type)
            if channel_type == "whatsapp":
                raise RuntimeError("nope")
            return None

        orch.restart_channel = fake_restart  # type: ignore[method-assign]
        asyncio.run(
            orch._on_channel_config_change(
                _change(KiroCrewConfig(), "whatsapp.enabled", "discord.enabled")
            )
        )
        assert seen == ["whatsapp", "discord"]


# ------------------------------------------------------------------
# 4. the Slack applier
# ------------------------------------------------------------------


class TestSlackApplierReconciles:
    def _orch(self) -> GatewayOrchestrator:
        orch = _make_orchestrator()
        orch._tracking_channels = {"C_OLD"}
        orch._open_channels = {"C_OPEN_OLD"}
        return orch

    def test_tracking_channels_are_reconciled_in_place(self, monkeypatch):
        orch = self._orch()
        tracked = orch._tracking_channels
        pushed: list[set[str]] = []
        monkeypatch.setattr(slack_handler, "set_tracking_channels", pushed.append)
        cfg = _cfg_with("slack", tracking_channels=[{"channel_id": "C_NEW"}, {"nope": 1}, "junk"])
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.tracking_channels")))
        # Same set object -- the Slack-native modal edits this one.
        assert orch._tracking_channels is tracked
        assert tracked == {"C_NEW"}
        assert pushed == [{"C_NEW"}]

    def test_open_channels_are_reconciled_in_place(self, monkeypatch):
        orch = self._orch()
        opened = orch._open_channels
        pushed: list[set[str]] = []
        monkeypatch.setattr(slack_handler, "set_open_channels", pushed.append)
        cfg = _cfg_with("slack", open_channels=["C_OPEN_NEW"])
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.open_channels")))
        assert orch._open_channels is opened
        assert opened == {"C_OPEN_NEW"}
        assert pushed == [{"C_OPEN_NEW"}]

    def test_a_degraded_slack_section_keeps_the_previous_sets(self, monkeypatch):
        orch = self._orch()
        pushed: list[set[str]] = []
        monkeypatch.setattr(slack_handler, "set_tracking_channels", pushed.append)
        cfg = _cfg_with(
            "slack",
            degraded=frozenset({"slack"}),
            tracking_channels=[{"channel_id": "C_NEW"}],
            open_channels=["C_OPEN_NEW"],
        )
        asyncio.run(
            orch._on_slack_config_change(
                _change(cfg, "slack.tracking_channels", "slack.open_channels")
            )
        )
        assert orch._tracking_channels == {"C_OLD"}
        assert orch._open_channels == {"C_OPEN_OLD"}
        assert pushed == []

    def test_widening_an_authorization_set_is_audited(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            gw, "sel", lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw))
        )
        monkeypatch.setattr(slack_handler, "set_open_channels", lambda ids: None)
        orch = self._orch()
        cfg = _cfg_with("slack", open_channels=["C_OPEN_NEW"])
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.open_channels")))
        rows = [a for a in audited if a.get("operation") == "slack.authorization_config_change"]
        assert [r["outcome"] for r in rows] == ["allowed"]
        assert rows[0]["resources"] == "slack.open_channels"

    def test_a_live_field_change_alone_is_not_audited_as_authorization(self, monkeypatch):
        audited: list[dict] = []
        monkeypatch.setattr(
            gw, "sel", lambda: SimpleNamespace(log_api_access=lambda **kw: audited.append(kw))
        )
        orch = self._orch()
        cfg = _cfg_with("slack")
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.reactions")))
        assert [
            a for a in audited if a.get("operation") == "slack.authorization_config_change"
        ] == []

    def test_no_slack_restart_and_no_session_teardown(self, monkeypatch):
        """Slack's socket is host-managed: the applier must not touch it."""
        orch = self._orch()
        orch._channel_handles["whatsapp"] = FakeClient(FakeTransport([]))
        restarts: list[str] = []

        async def boom(channel_type, *, cfg=None):
            restarts.append(channel_type)

        orch.restart_channel = boom  # type: ignore[method-assign]
        monkeypatch.setattr(slack_handler, "set_open_channels", lambda ids: None)
        monkeypatch.setattr(slack_handler, "set_tracking_channels", lambda ids: None)
        cfg = _cfg_with("slack", open_channels=["C_X"], tracking_channels=[{"channel_id": "C_Y"}])
        asyncio.run(
            orch._on_slack_config_change(
                _change(cfg, "slack.open_channels", "slack.tracking_channels")
            )
        )
        assert restarts == []
        assert set(orch._channel_handles) == {"whatsapp"}
        assert orch._channel_handles["whatsapp"].closed is False

    def test_observe_registration_follows_the_new_activations(self):
        from kiro_crew.config.loader import ACTIVATION_OBSERVE

        class FakeHistory:
            def __init__(self) -> None:
                self._observe_channels = {"C_STALE"}

            def set_observe(self, ch_id: str) -> None:
                self._observe_channels.add(ch_id)

            def unset_observe(self, ch_id: str) -> None:
                self._observe_channels.discard(ch_id)

        orch = self._orch()
        history = FakeHistory()
        orch.channel_history = history
        cfg = _cfg_with("slack")
        cfg.slack_channels = {
            "C_WATCH": SimpleNamespace(activation=ACTIVATION_OBSERVE),
            "C_MENTION": SimpleNamespace(activation="mention"),
        }
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.channels")))
        assert history._observe_channels == {"C_WATCH"}

    def test_observe_caps_are_pushed_to_the_live_history(self):
        class FakeHistory:
            def __init__(self) -> None:
                self._observe_channels: set[str] = set()
                self._observe_max_entries = 1
                self._observe_ttl_secs = 1

        orch = self._orch()
        history = FakeHistory()
        orch.channel_history = history
        cfg = replace(KiroCrewConfig(), observe_max_messages=250, observe_ttl_hours=3.0)
        asyncio.run(
            orch._on_slack_config_change(
                _change(cfg, "slack.observe_max_messages", "slack.observe_ttl_hours")
            )
        )
        assert (history._observe_max_entries, history._observe_ttl_secs) == (250, 10800)

    def test_observe_caps_prefer_the_historys_own_setter(self):
        class FakeHistory:
            def __init__(self) -> None:
                self._observe_channels: set[str] = set()
                self.calls: list[tuple[int, int]] = []

            def set_observe_limits(self, max_entries: int, ttl_secs: int) -> None:
                self.calls.append((max_entries, ttl_secs))

        orch = self._orch()
        history = FakeHistory()
        orch.channel_history = history
        cfg = replace(KiroCrewConfig(), observe_max_messages=99, observe_ttl_hours=1.0)
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.observe_ttl_hours")))
        assert history.calls == [(99, 3600)]

    def test_allowed_enterprise_ids_change_reloads_the_validated_allowlist(self, monkeypatch):
        from kiro_crew.slack import enterprise as slack_enterprise

        called: list[bool] = []
        monkeypatch.setattr(
            slack_enterprise, "reload_allowed_team_ids", lambda: called.append(True)
        )
        orch = self._orch()
        orch._slack_enabled = True
        cfg = _cfg_with("slack", allowed_enterprise_ids=["T123"])
        asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.allowed_enterprise_ids")))
        assert called == [True]

    def test_the_applier_is_subscribed_on_the_process_watcher(self):
        orch = _make_orchestrator()
        names = {s.name for s in orch._config_subs}
        assert names == {
            "GatewayOrchestrator.channel_restart",
            "GatewayOrchestrator.slack",
        }


class TestPhaseEmojisFollowConfig:
    def test_a_reactions_change_rebuilds_the_table_in_place(self):
        table = slack_handler.phase_emojis()
        before = dict(table)
        phase = next(iter(before))
        orch = _make_orchestrator()
        orch._tracking_channels = set()
        orch._open_channels = set()
        try:
            cfg = _cfg_with("slack", reactions={phase: "rocket"})
            asyncio.run(orch._on_slack_config_change(_change(cfg, "slack.reactions")))
            # Same object: the reaction controllers hold the table, not a copy.
            assert slack_handler.phase_emojis() is table
            assert table[phase] == "rocket"
        finally:
            slack_handler.refresh_phase_emojis(None)
        assert slack_handler.phase_emojis() == before

    def test_a_null_override_suppresses_that_phase(self):
        table = slack_handler.phase_emojis()
        phase = next(iter(table))
        try:
            assert slack_handler.refresh_phase_emojis({phase: None}) == []
            assert slack_handler.phase_emojis()[phase] is None
        finally:
            slack_handler.refresh_phase_emojis(None)

    def test_unknown_keys_are_reported_not_applied(self):
        try:
            unknown = slack_handler.refresh_phase_emojis({"not_a_phase": "rocket"})
            assert unknown == ["not_a_phase"]
            assert "not_a_phase" not in slack_handler.phase_emojis()
        finally:
            slack_handler.refresh_phase_emojis(None)


# ------------------------------------------------------------------
# 5. enterprise allowlist reload
# ------------------------------------------------------------------


class TestReloadAllowedTeamIds:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from kiro_crew.slack import enterprise as ent

        monkeypatch.setattr(ent, "_validated_team_id", "", raising=False)
        monkeypatch.setattr(ent, "_validated_enterprise_id", "", raising=False)
        monkeypatch.setattr(ent, "_allowed_team_ids", set(), raising=False)
        monkeypatch.setattr(ent, "_allowlist_configured", False, raising=False)
        self.audited: list[dict] = []
        monkeypatch.setattr(
            ent, "sel", lambda: SimpleNamespace(log_api_access=lambda **kw: self.audited.append(kw))
        )
        self.ent = ent

    def _rows(self, operation: str) -> list[dict]:
        return [a for a in self.audited if a.get("operation") == operation]

    def test_noop_before_the_workspace_is_validated(self, monkeypatch):
        loads: list[bool] = []
        monkeypatch.setattr(self.ent, "_load_allowed_team_ids", lambda: loads.append(True) or False)
        assert self.ent.reload_allowed_team_ids() is False
        assert loads == []
        assert self._rows("slack.allowed_team_ids_reload") == []

    def test_reload_re_runs_the_validated_read_not_a_raw_load(self, monkeypatch):
        monkeypatch.setattr(self.ent, "_validated_team_id", "T_VALID", raising=False)
        monkeypatch.setattr(self.ent, "_read_allowlist", lambda: ({"T_CHILD"}, ""))
        assert self.ent.reload_allowed_team_ids() is False
        assert self.ent._allowed_team_ids == {"T_VALID", "T_CHILD"}
        assert self.ent._allowlist_configured is True
        assert [r["outcome"] for r in self._rows("slack.allowed_team_ids_reload")] == ["allowed"]

    def test_a_degraded_read_fails_closed_and_admits_nothing(self, monkeypatch):
        monkeypatch.setattr(self.ent, "_validated_team_id", "T_VALID", raising=False)
        monkeypatch.setattr(self.ent, "_allowed_team_ids", {"T_VALID"}, raising=False)
        monkeypatch.setattr(self.ent, "_read_allowlist", lambda: (None, "unreadable"))
        assert self.ent.reload_allowed_team_ids() is True
        # Not even the validated team_id: that is the question the allowlist answers.
        assert self.ent._allowed_team_ids == set()
        assert self.ent._allowlist_configured is True
        rows = self._rows("slack.allowed_team_ids_reload")
        assert [r["outcome"] for r in rows] == ["denied"]
        assert rows[0]["error"] == "config_load_degraded_fail_closed"

    def test_a_raising_read_also_fails_closed(self, monkeypatch):
        monkeypatch.setattr(self.ent, "_validated_team_id", "T_VALID", raising=False)

        def boom():
            raise RuntimeError("disk gone")

        monkeypatch.setattr(self.ent, "_read_allowlist", boom)
        assert self.ent.reload_allowed_team_ids() is True
        assert self.ent._allowed_team_ids == set()

    def test_an_unconfigured_allowlist_stays_default_open(self, monkeypatch):
        monkeypatch.setattr(self.ent, "_validated_team_id", "T_VALID", raising=False)
        monkeypatch.setattr(self.ent, "_read_allowlist", lambda: (set(), ""))
        assert self.ent.reload_allowed_team_ids() is False
        assert self.ent._allowlist_configured is False
        assert self.ent._allowed_team_ids == {"T_VALID"}

    def test_a_narrowing_write_takes_effect_without_a_restart(self, monkeypatch):
        monkeypatch.setattr(self.ent, "_validated_team_id", "T_VALID", raising=False)
        monkeypatch.setattr(self.ent, "_read_allowlist", lambda: ({"T_A", "T_B"}, ""))
        self.ent.reload_allowed_team_ids()
        assert "T_B" in self.ent._allowed_team_ids
        monkeypatch.setattr(self.ent, "_read_allowlist", lambda: ({"T_A"}, ""))
        self.ent.reload_allowed_team_ids()
        assert self.ent._allowed_team_ids == {"T_VALID", "T_A"}


# ------------------------------------------------------------------
# 6. channel_restart_required + the WhatsApp saver flag
# ------------------------------------------------------------------


class TestChannelRestartRequired:
    def test_a_restart_marked_field_requires_a_restart(self):
        from kiro_crew.config.schema import requires_restart

        assert requires_restart("slack.command") is True
        assert channel_restart_required("slack", ["command"]) is True

    def test_a_live_field_does_not(self):
        assert channel_restart_required("slack", ["open_channels"]) is False
        assert channel_restart_required("discord", ["reactions_enabled"]) is False
        assert channel_restart_required("whatsapp", ["session_folder"]) is False

    def test_a_connection_field_is_applied_in_process_so_it_is_not_marked(self):
        """``restart_channel`` applies these, so the schema must not claim otherwise."""
        for section, field in (
            ("whatsapp", "enabled"),
            ("telegram", "bot_token"),
            ("discord", "bot_token"),
            ("wecom", "ws_url"),
        ):
            assert channel_restart_required(section, [field]) is False, f"{section}.{field}"

    def test_a_mixed_save_requires_a_restart_if_any_field_does(self):
        assert channel_restart_required("slack", ["open_channels", "command"]) is True

    def test_an_env_write_always_requires_a_restart(self):
        assert channel_restart_required("whatsapp", [], env_updates={"WA_TOKEN": "x"}) is True
        assert channel_restart_required("slack", ["open_channels"], env_updates={"X": "1"}) is True

    def test_an_empty_save_requires_nothing(self):
        assert channel_restart_required("whatsapp", []) is False

    def test_the_answer_comes_from_the_schema_not_a_hand_list(self, monkeypatch):
        import kiro_crew.config.schema as schema

        monkeypatch.setattr(
            schema, "_RESTART_PATHS", frozenset({"whatsapp.dm_policy"}), raising=False
        )
        assert channel_restart_required("whatsapp", ["dm_policy"]) is True
        assert channel_restart_required("whatsapp", ["session_folder"]) is False

    def test_the_schema_is_the_only_answer(self):
        """No per-channel fallback table exists: a second list is a second place to drift.

        The helper consults ``config.schema.requires_restart`` and nothing else, so
        the module carries no ``LIVE_RELOAD_FIELDS`` / ``_FALLBACK_*`` tables and the
        import is unconditional rather than guarded.
        """
        from kiro_crew.config.schema import requires_restart
        from kiro_crew.dashboard import channel_folders

        for name in ("LIVE_RELOAD_FIELDS", "_FALLBACK_LIVE_FIELDS", "_FALLBACK_BOOT_FIELDS"):
            assert not hasattr(channel_folders, name), name
        assert channel_folders.requires_restart is requires_restart


class TestWhatsappSaverFlagIsPerField:
    def test_the_saver_asks_the_helper_rather_than_answering_True(self):
        """The bug this replaced: an unconditional ``restart_required: True``."""
        src = inspect.getsource(
            __import__(
                "kiro_crew.dashboard.handlers.whatsapp_setup", fromlist=["whatsapp_config_save"]
            ).whatsapp_config_save
        )
        assert 'channel_restart_required("whatsapp", body.keys())' in src
        assert '"restart_required": True' not in src

    def test_a_folder_only_save_reports_no_restart(self):
        assert channel_restart_required("whatsapp", ["session_folder"]) is False

    def test_the_flag_varies_with_the_saved_fields(self, monkeypatch):
        import kiro_crew.config.schema as schema

        monkeypatch.setattr(
            schema, "_RESTART_PATHS", frozenset({"whatsapp.enabled"}), raising=False
        )
        assert channel_restart_required("whatsapp", ["session_folder"]) is False
        assert channel_restart_required("whatsapp", ["enabled"]) is True
        assert channel_restart_required("whatsapp", ["session_folder", "enabled"]) is True
