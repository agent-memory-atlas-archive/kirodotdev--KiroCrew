"""Bulk routes cross the production owner and store resolution gates."""

import asyncio
import json
import os
from datetime import date, timedelta

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from test_member_memory_api import env as _memory_env
from test_member_memory_api import request

from kiro_crew.dashboard.handlers import memory_edit

env = _memory_env


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_agent_and_nonowner_cannot_reach_bulk_even_without_store(env, internal):
    for handler in (
        memory_edit.api_memory_records,
        memory_edit.api_memory_records_refresh,
        memory_edit.api_memory_bulk_preview,
        memory_edit.api_memory_bulk_apply,
    ):
        response = await handler(
            request(
                env,
                body={} if handler != memory_edit.api_memory_records else None,
                internal=internal,
            )
        )
        assert response.status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["default", "member-alice"])
async def test_owner_bulk_selects_only_requested_store_and_is_retryable(env, name):
    for tier in env.tiers.values():
        assert tier.set_semantic("user.email", "owner@old.example", 1.0, "user_explicit") is None
    for key in ("default", "member-alice", "member-bob"):
        response = await memory_edit.api_memory_records(
            request(env, query={"store": key, "q": "email"}, owner=True)
        )
        assert response.status == 200
        assert json.loads(response.text)["total"] == 1
    response = await memory_edit.api_memory_bulk_preview(
        request(
            env,
            owner=True,
            session="dashboard:ui",
            body={
                "store": name,
                "selection": {"query": {"q": "email"}},
                "operation": {"type": "replace_text", "find": "old", "replacement": "new"},
            },
        )
    )
    assert response.status == 200, response.text
    preview = json.loads(response.text)
    for _ in range(2):
        response = await memory_edit.api_memory_bulk_apply(
            request(
                env,
                owner=True,
                session="dashboard:ui",
                body={"store": name, "preview_id": preview["preview_id"]},
            )
        )
        assert response.status == 200, response.text
        assert json.loads(response.text)["changed_count"] == 1
    for key, tier in env.tiers.items():
        expected = "new" if key == ("" if name == "default" else name) else "old"
        assert (
            json.loads(tier.get_semantic("user.email")["value_json"]) == f"owner@{expected}.example"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["default", "member-alice"])
async def test_owner_record_listing_accepts_query_auth_token_without_widening_filters(env, name):
    tier = env.tiers["" if name == "default" else name]
    tier.set_semantic("user.contact", "selected contact", 1.0, "user_explicit")
    tier.set_semantic("user.language", "unrelated language", 1.0, "user_explicit")
    query = {"store": name, "q": "contact", "kind": "fact", "token": "fixture-link-token"}
    response = await memory_edit.api_memory_records(request(env, query=query, owner=True))
    assert response.status == 200, response.text
    payload = json.loads(response.text)
    assert payload["total"] == 1
    assert [entry["id"] for entry in payload["entries"]] == ["user.contact"]
    assert "fixture-link-token" not in response.text

    query["topic"] = "contact"
    invalid = await memory_edit.api_memory_records(request(env, query=query, owner=True))
    assert invalid.status == 400
    denied = await memory_edit.api_memory_records(request(env, query=query))
    assert denied.status == 403


@pytest.mark.asyncio
async def test_bad_store_and_invalid_filters_fail_explicitly(env):
    for query, status in [
        ({"store": "unknown"}, 404),
        ({"store": "default", "kind": "typo"}, 400),
        ({"store": "default", "limit": "-1"}, 400),
        ({"store": "default", "topic": "typo"}, 400),
        ({"store": "default", "topic": "email"}, 400),
    ]:
        response = await memory_edit.api_memory_records(request(env, query=query, owner=True))
        assert response.status == status


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["default", "member-alice"])
async def test_http_query_refresh_counts_current_membership_and_inactive_exclusions(env, name):
    tier = env.tiers["" if name == "default" else name]

    def seed():
        for candidate in env.tiers.values():
            for index in range(4):
                assert (
                    candidate.set_semantic(
                        f"user.contact{index}", "Straße user@example.org", 1.0, "user_explicit"
                    )
                    is None
                )

    await asyncio.to_thread(seed)

    @web.middleware
    async def owner_principal(request, handler):
        # Authentication is a fixture principal; the actual owner/store gates,
        # bounded JSON reader and selection service all run over HTTP below.
        request["user"] = "owner"
        request["app"] = ""
        return await handler(request)

    app = web.Application(middlewares=[owner_principal])
    app["state"] = env.state
    app.router.add_post("/api/memory/records/refresh", memory_edit.api_memory_records_refresh)
    selected = {
        "query": {"q": "STRASSE", "kind": "fact"},
        "exclude": [
            {"kind": "fact", "id": "user.contact0"},
            {"kind": "fact", "id": "user.contact1"},
        ],
    }
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/api/memory/records/refresh", json={"store": name, "selection": selected}
        )
        assert response.status == 200, await response.text()
        assert await response.json() == {"matched_count": 2}

        def change_membership():
            tier.delete_semantic("user.contact0", source="user_explicit")
            tier.set_semantic("user.contact1", "No matching phrase", 1.0, "user_explicit")
            tier.set_semantic(
                "user.contact4", "Ｓｔｒａｓｓｅ user@example.org", 1.0, "user_explicit"
            )

        await asyncio.to_thread(change_membership)
        before = {key: candidate.db.total_changes for key, candidate in env.tiers.items()}
        response = await client.post(
            "/api/memory/records/refresh", json={"store": name, "selection": selected}
        )
        assert response.status == 200, await response.text()
        assert await response.json() == {"matched_count": 3}
        for peer in {"default", "member-alice", "member-bob"} - {name}:
            response = await client.post(
                "/api/memory/records/refresh", json={"store": peer, "selection": selected}
            )
            assert response.status == 200, await response.text()
            assert await response.json() == {"matched_count": 2}
        response = await client.post(
            "/api/memory/records/refresh",
            json={"store": name, "items": selected["exclude"]},
        )
        explicit = await response.json()
        assert response.status == 200
        assert set(explicit) == {"entries", "missing"}
        assert [row["id"] for row in explicit["entries"]] == ["user.contact1"]
        assert explicit["missing"] == [{"kind": "fact", "id": "user.contact0"}]
        assert {key: candidate.db.total_changes for key, candidate in env.tiers.items()} == before


@pytest.mark.asyncio
async def test_private_store_unavailable_never_returns_global_records(env, monkeypatch):
    async def unavailable(state, name):
        return None

    monkeypatch.setattr(memory_edit, "vector_memory_for_store", unavailable)
    response = await memory_edit.api_memory_records(
        request(env, query={"store": "member-alice"}, owner=True)
    )
    assert response.status == 503
    assert json.loads(response.text)["code"] == "store_unavailable"


@pytest.mark.asyncio
async def test_owner_markdown_loader_retains_private_history_without_vector_attachment(env):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    assert memory.vector_store is None
    path = env.home / "memory_stores" / "member-alice" / "memory" / "history" / "2000-01-01.md"
    path.write_text(
        "# 2000-01-01\n#### decision\nKeep the original project contract.", encoding="utf-8"
    )
    assert memory.prune_history(keep_days=1) == 0
    assert "original project contract" in memory.read_recent_history()
    assert await markdown_memory_for_store(env.state, "member-alice") is memory


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", ["hardlink", "opened_path"])
async def test_private_history_reader_refuses_inodes_outside_its_binding(env, monkeypatch, escape):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    path = env.home / "memory_stores" / "member-alice" / "memory" / "history" / "2000-01-01.md"
    other = env.home / "other-member-evidence.txt"
    other.write_text("Private evidence belonging elsewhere.", encoding="utf-8")
    if escape == "hardlink":
        os.link(other, path)
    else:
        path.write_text("This member's own content.", encoding="utf-8")
        monkeypatch.setattr("kiro_crew.memory.fd_real_path", lambda descriptor: str(other))
    assert memory.read_recent_history() == ""
    assert other.read_text(encoding="utf-8") == "Private evidence belonging elsewhere."


@pytest.mark.asyncio
@pytest.mark.parametrize("document", ["preferences", "projects"])
@pytest.mark.parametrize("failure", ["hardlink", "opened_path", "invalid_utf8", "oversize"])
async def test_private_anchor_read_refuses_unsafe_present_files(
    env, monkeypatch, document, failure
):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    path = memory._memory_dir / f"{document}.md"
    other = env.home / "memory_stores" / "member-bob" / "private-evidence.txt"
    other.write_text("Evidence belongs only to Bob.", encoding="utf-8")
    if failure == "hardlink":
        path.unlink()
        os.link(other, path)
        reason = "hard links"
    elif failure == "opened_path":
        monkeypatch.setattr("kiro_crew.memory.fd_real_path", lambda descriptor: str(other))
        reason = "bound path"
    elif failure == "invalid_utf8":
        path.write_bytes(b"\xff")
        reason = "UTF-8"
    else:
        monkeypatch.setattr(memory, "_HISTORY_SNAPSHOT_MAX_BYTES", 128)
        path.write_text("x" * 129, encoding="utf-8")
        reason = "size cap"
    with pytest.raises(OSError, match=reason):
        getattr(memory, f"read_{document}")()
    assert other.read_text(encoding="utf-8") == "Evidence belongs only to Bob."


@pytest.mark.asyncio
@pytest.mark.parametrize("document", ["preferences", "projects"])
async def test_private_anchor_missing_and_empty_remain_valid_initial_states(env, document):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    path = memory._memory_dir / f"{document}.md"
    path.unlink()
    reader = getattr(memory, f"read_{document}")
    assert reader() == ""
    path.write_text("", encoding="utf-8")
    assert reader() == ""
    getattr(memory, f"write_{document}")("# Current guidance\nKeep the actual project contract.")
    assert "actual project contract" in reader()


@pytest.mark.asyncio
@pytest.mark.parametrize("document", ["preferences", "projects"])
async def test_private_anchor_get_returns_scoped_unavailable_reason_without_contents(env, document):
    from kiro_crew.dashboard.handlers import memory as memory_handlers
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    path = memory._memory_dir / f"{document}.md"
    other = env.home / "memory_stores" / "member-bob" / "private-evidence.txt"
    other.write_text("DO-NOT-EXPOSE-THIS-PRIVATE-CONTENT", encoding="utf-8")
    path.unlink()
    os.link(other, path)
    handler = getattr(memory_handlers, f"api_memory_{document}")
    response = await handler(request(env, query={"store": "member-alice"}, owner=True))
    assert response.status == 503
    body = json.loads(response.text)
    assert body["code"] == "store_unavailable"
    assert "member-alice" in body["error"] and "hard links" in body["error"]
    assert "DO-NOT-EXPOSE-THIS-PRIVATE-CONTENT" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["hardlink", "invalid_utf8", "vanished", "unsafe_root", "linked_root"]
)
async def test_private_index_rebuild_refusal_preserves_previous_search(env, monkeypatch, failure):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    memory.write_preferences("# Preferences\nOriginal searchable sentinel.")
    memory.rebuild_index()
    before = memory.search("sentinel")
    count = memory.index_row_count()
    assert before
    path = memory._history_dir / "2000-01-01.md"
    if failure == "hardlink":
        other = env.home / "memory_stores" / "member-bob" / "secret-evidence.txt"
        other.write_text("foreignclassifiedrecord", encoding="utf-8")
        os.link(other, path)
    elif failure == "invalid_utf8":
        path.write_bytes(b"\xff")
    elif failure == "vanished":
        path.write_text("A source about to disappear.", encoding="utf-8")
        original = memory._guarded_entry

        def read_then_disappear(candidate, **kwargs):
            if candidate == path:
                path.unlink()
            return original(candidate, **kwargs)

        monkeypatch.setattr(memory, "_guarded_entry", read_then_disappear)
    elif failure == "linked_root":
        from conftest import make_dir_link

        other = env.home / "memory_stores" / "member-bob" / "history"
        other.mkdir()
        (other / path.name).write_text("foreignclassifiedrecord", encoding="utf-8")
        memory._history_dir.rmdir()
        make_dir_link(memory._history_dir, other)
    else:
        monkeypatch.setattr(memory, "_read_root_guard", lambda: False)
    with monkeypatch.context() as traversal_guard:
        if failure in {"unsafe_root", "linked_root"}:
            traversal_guard.setattr(
                "kiro_crew.memory.os.scandir",
                lambda *args: pytest.fail("unsafe root was traversed"),
            )
        with pytest.raises(OSError, match="refused"):
            memory.rebuild_index()
    assert memory.search("sentinel") == before
    assert memory.search("foreignclassifiedrecord") == []
    assert memory.index_row_count() == count


@pytest.mark.asyncio
async def test_private_index_database_failure_rolls_back_and_is_explicit(env, monkeypatch):
    from kiro_crew._sqlite_compat import sqlite3
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    memory.write_preferences("# Preferences\nOriginal searchable sentinel.")
    memory.rebuild_index()
    before = memory.search("sentinel")
    original_get_db = memory._get_db

    class FailingInsert:
        def __init__(self):
            self.connection = original_get_db()

        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO memory_fts"):
                raise sqlite3.OperationalError("injected disk write failure")
            return self.connection.execute(sql, *args)

        def commit(self):
            self.connection.commit()

        def close(self):
            self.connection.close()

        def rollback(self):
            self.connection.rollback()

    with monkeypatch.context() as database_guard:
        database_guard.setattr(memory, "_get_db", FailingInsert)
        with pytest.raises(sqlite3.OperationalError, match="disk write failure"):
            memory.rebuild_index()
    assert memory.search("sentinel") == before


@pytest.mark.asyncio
async def test_private_index_readers_keep_previous_index_until_all_sources_are_read(
    env, monkeypatch
):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    memory.write_preferences("# Preferences\nOriginal searchable sentinel.")
    memory.rebuild_index()
    previous = memory.search("sentinel")
    memory._preferences_file.write_text("Replacement searchable record.", encoding="utf-8")
    later = memory._history_dir / "2000-01-01.md"
    later.write_bytes(b"\xff")
    original_read = memory._guarded_entry
    checked = []

    def read_with_concurrent_query(path, **kwargs):
        if path == later:
            checked.append(path)
            assert memory.search("sentinel") == previous
            assert memory.search("Replacement") == []
        return original_read(path, **kwargs)

    monkeypatch.setattr(memory, "_guarded_entry", read_with_concurrent_query)
    with pytest.raises(OSError, match="UTF-8"):
        memory.rebuild_index()
    assert checked == [later]
    assert memory.search("sentinel") == previous
    assert memory.search("Replacement") == []


@pytest.mark.asyncio
async def test_private_index_covers_retained_days_beyond_snapshot_limit(env):
    from kiro_crew.dashboard.handlers._shared import markdown_memory_for_store

    memory = await markdown_memory_for_store(env.state, "member-alice")
    first_day = date(2000, 1, 1)
    total_days = memory._HISTORY_SNAPSHOT_MAX_ENTRIES + 1
    for index in range(total_days):
        path = memory._history_dir / f"{(first_day + timedelta(days=index)).isoformat()}.md"
        text = "oldestevidencesentinel" if index == 0 else f"Retained decision number {index}."
        path.write_text(text, encoding="utf-8")
    (memory._history_dir / "notes.md").write_text("notadailysentinel", encoding="utf-8")
    assert memory.rebuild_index() == total_days + 2
    assert memory.search("oldestevidencesentinel")
    assert memory.search("notadailysentinel") == []
    assert len(memory.read_history_entries()) == memory._HISTORY_SNAPSHOT_MAX_ENTRIES
    assert (memory._history_dir / "2000-01-01.md").read_text(
        encoding="utf-8"
    ) == "oldestevidencesentinel"
