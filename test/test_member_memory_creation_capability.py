"""Member creation refuses unusable isolation before changing ownership."""

from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew import member_memory_auth as auth
from kiro_crew import sandbox
from kiro_crew.agent_discovery import AgentInfo
from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig, config_dir
from kiro_crew.dashboard.handlers import agents as handlers
from kiro_crew.memory_stores import (
    memory_stores_root,
    provision_member_memory,
    require_member_memory_store,
)

_UNSUPPORTED = [
    ("win32", "kas", "auto", "namespace", False, "WSL/Linux gateway"),
    ("linux", "codex", "auto", "namespace", False, "Use Kiro, Claude Code or KAS"),
    ("linux", "kas", "off", "namespace", False, "Enable agent.sandbox"),
    ("linux", "kas", "auto", "none", False, "Restore OS sandbox support"),
    ("darwin", "", "auto", "sandbox-exec", True, "Disable that delegation"),
]


def _environment(monkeypatch, platform, backend, mode, mechanism, delegates):
    cfg = KiroCrewConfig.load()
    cfg.agent.acp_backend = "kas"
    cfg.agent.member_acp_backend = backend
    cfg.agent.sandbox = mode
    cfg.agents["reviewer"] = KiroCrewAgentConfig()
    cfg.save()
    monkeypatch.setattr(auth, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(sandbox, "_clamp_sandbox_mode", lambda value: value)
    monkeypatch.setattr(sandbox, "detect_backend", lambda **kwargs: mechanism)
    monkeypatch.setattr(sandbox, "kiro_internal_sandbox_enabled", lambda: delegates)
    return cfg


def _snapshot():
    root = memory_stores_root()
    return (
        (config_dir() / "config.json").read_bytes(),
        sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else [],
    )


def _app(monkeypatch):
    monkeypatch.setattr(
        "kiro_crew.dashboard.handlers.source_providers.is_owner_dashboard_request", lambda _: True
    )
    monkeypatch.setattr(handlers, "list_agents", lambda: [])
    app = web.Application()
    app.router.add_post("/api/agents", handlers.api_kirocrew_agents_create)
    app.router.add_post("/api/agents/sync", handlers.api_kirocrew_agents_sync)
    app.router.add_put("/api/agents/{name}", handlers.api_kirocrew_agent_update)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["create", "opt_in", "sync"])
@pytest.mark.parametrize("platform,backend,mode,mechanism,delegates,remedy", _UNSUPPORTED)
async def test_dashboard_refuses_unsupported_allocation_without_side_effects(
    monkeypatch, entrypoint, platform, backend, mode, mechanism, delegates, remedy
):
    await asyncio.to_thread(
        _environment, monkeypatch, platform, backend, mode, mechanism, delegates
    )
    app = _app(monkeypatch)
    retire = AsyncMock()
    monkeypatch.setattr(handlers, "_retire_legacy_member_contexts", retire)
    checked = []
    supported = auth.private_memory_execution_supported

    def check(*, session_key):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        checked.append(session_key)
        return supported(session_key=session_key)

    monkeypatch.setattr(auth, "private_memory_execution_supported", check)
    if entrypoint == "sync":
        discovered = AgentInfo(
            name="new-member",
            filename="new-member.json",
            description="",
            model="auto",
            source="package",
        )
        monkeypatch.setattr(handlers, "list_agents", lambda: [discovered])
    before = await asyncio.to_thread(_snapshot)
    async with TestClient(TestServer(app)) as client:
        if entrypoint == "create":
            response = await client.post(
                "/api/agents", json={"name": "new-member", "kiro_agent": "kirocrew"}
            )
        elif entrypoint == "sync":
            response = await client.post("/api/agents/sync")
        else:
            response = await client.put(
                "/api/agents/reviewer", json={"provision_memory": True, "description": "Changed"}
            )
        assert response.status == 409, await response.text()
        result = await response.json()
        assert result["code"] == "member_memory_unavailable"
        assert remedy in result["error"]
    assert checked == [
        "dashboard:member-reviewer" if entrypoint == "opt_in" else "dashboard:member-new-member"
    ]
    retire.assert_not_awaited()
    assert await asyncio.to_thread(_snapshot) == before
    loaded = await asyncio.to_thread(KiroCrewConfig.load)
    assert await asyncio.to_thread(require_member_memory_store, loaded, "reviewer") == "default"


@pytest.mark.parametrize("action", ["create", "update"])
@pytest.mark.parametrize("platform,backend,mode,mechanism,delegates,remedy", _UNSUPPORTED)
def test_cli_refuses_unsupported_allocation_before_persisting(
    monkeypatch, capsys, action, platform, backend, mode, mechanism, delegates, remedy
):
    from kiro_crew.cli_commands import _handle_agent

    _environment(monkeypatch, platform, backend, mode, mechanism, delegates)
    before = _snapshot()
    with pytest.raises(SystemExit) as exc:
        _handle_agent(
            argparse.Namespace(
                agent_action=action,
                name="new-member" if action == "create" else "reviewer",
                kiro_agent="kirocrew" if action == "create" else None,
                workspace="default" if action == "create" else None,
                memory_store="default" if action == "create" else None,
                provision_memory=True,
            )
        )
    assert exc.value.code == 1
    assert remedy in capsys.readouterr().err
    assert _snapshot() == before
    assert require_member_memory_store(KiroCrewConfig.load(), "reviewer") == "default"


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["create", "opt_in"])
async def test_supported_member_backend_can_allocate_with_a_different_default_backend(
    monkeypatch, entrypoint
):
    cfg = await asyncio.to_thread(
        _environment, monkeypatch, "linux", "kas", "auto", "namespace", False
    )
    cfg.agent.acp_backend = "codex"
    await asyncio.to_thread(cfg.save)
    async with TestClient(TestServer(_app(monkeypatch))) as client:
        if entrypoint == "create":
            response = await client.post(
                "/api/agents", json={"name": "new-member", "kiro_agent": "kirocrew"}
            )
            name = "new-member"
        else:
            response = await client.put("/api/agents/reviewer", json={"provision_memory": True})
            name = "reviewer"
        assert response.status == 200, await response.text()
        store = (await response.json())["memory_store"]
    loaded = await asyncio.to_thread(KiroCrewConfig.load)
    assert store != "default"
    assert loaded.memory_stores[store].owner_member == name
    assert await asyncio.to_thread(require_member_memory_store, loaded, name) == store


@pytest.mark.asyncio
async def test_unsupported_gateway_keeps_v1_edits_and_owned_v2_management(monkeypatch):
    cfg = await asyncio.to_thread(
        _environment, monkeypatch, "win32", "kas", "auto", "namespace", False
    )
    cfg.agents["private-member"] = KiroCrewAgentConfig()
    store = await asyncio.to_thread(provision_member_memory, cfg, "private-member")
    await asyncio.to_thread(cfg.save)
    async with TestClient(TestServer(_app(monkeypatch))) as client:
        response = await client.put("/api/agents/reviewer", json={"description": "Still V1"})
        assert response.status == 200, await response.text()
        response = await client.put("/api/agents/private-member", json={"provision_memory": True})
        assert response.status == 200, await response.text()
        assert (await response.json())["memory_store"] == store
    loaded = await asyncio.to_thread(KiroCrewConfig.load)
    assert loaded.agents["reviewer"].description == "Still V1"
    assert await asyncio.to_thread(require_member_memory_store, loaded, "reviewer") == "default"
    assert await asyncio.to_thread(require_member_memory_store, loaded, "private-member") == store
