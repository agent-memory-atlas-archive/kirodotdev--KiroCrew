"""Read-only knowledge stats: the store aggregate, the CLI verb, the MCP twin.

Every case runs against ONE seeded library whose numbers are fixed by
:func:`_seed_library` -- the fixture contract the CLI, the MCP tool and the pod
check are all asserted against:

===================  =========  =====
source               documents  items
===================  =========  =====
alpha                        2      4
beta                         1      2
gamma (registered)           0      0
(no source)                  1      2
-------------------  ---------  -----
TOTAL                        4      8
===================  =========  =====

``alpha`` also holds one non-active row, which no total counts. The sourceless
bucket holds one hashless item, which counts as an item and as no document.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kiro_crew.knowledge.store import KnowledgeStore

# The fixture contract, named once so a drifting seed fails as a contract
# mismatch rather than as an arithmetic surprise in each assertion.
EXPECTED_SOURCES = 3
EXPECTED_DOCUMENTS = 4
EXPECTED_ITEMS = 8


def _library_fingerprint(db_path) -> tuple:
    """Every row identity the stats surfaces read, via a read-only connection.

    Opened ``mode=ro`` so taking the fingerprint cannot itself be the write a
    read-only assertion is looking for. Byte-comparing the file would instead
    measure SQLite bookkeeping -- opening a store runs ``CREATE TABLE IF NOT
    EXISTS`` and the migrations -- which is not the claim being made.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        items = conn.execute(
            "SELECT id, status, source_id, content_hash FROM items ORDER BY id"
        ).fetchall()
        sources = conn.execute("SELECT id, name FROM sources ORDER BY id").fetchall()
    finally:
        conn.close()
    return (tuple(items), tuple(sources))


def _add_document(store: KnowledgeStore, source_id: str | None, title: str, chunks: int) -> str:
    """Write one document as *chunks* item rows sharing its whole-text hash."""
    content_hash = hashlib.sha256(title.encode()).hexdigest()
    for index in range(chunks):
        store.add_item(
            title=title,
            content=f"{title} part {index}",
            item_type="document",
            source_id=source_id,
            chunk_index=index,
            content_hash=content_hash,
        )
    return content_hash


def _seed_library(db_path) -> KnowledgeStore:
    store = KnowledgeStore(str(db_path))
    # local_folder, not an invented type: the store's constructor reaps an
    # itemless source of any other type, so `gamma` would not survive the next
    # open and the zero-item row this contract pins would vanish.
    alpha = store.add_source("alpha", "local_folder", "/tmp/alpha")
    beta = store.add_source("beta", "local_folder", "/tmp/beta")
    store.add_source("gamma", "local_folder", "/tmp/gamma")
    _add_document(store, alpha, "alpha-one", chunks=3)
    _add_document(store, alpha, "alpha-two", chunks=1)
    _add_document(store, beta, "beta-one", chunks=2)
    _add_document(store, None, "orphan-one", chunks=1)
    store.add_item(
        title="hashless",
        content="no document identity",
        item_type="document",
        source_id=None,
        chunk_index=0,
    )
    superseded = store.add_item(
        title="alpha-old",
        content="replaced",
        item_type="document",
        source_id=alpha,
        chunk_index=0,
        content_hash=hashlib.sha256(b"alpha-old").hexdigest(),
    )
    store.db.execute("UPDATE items SET status = 'superseded' WHERE id = ?", (superseded,))
    store.db.commit()
    return store


@pytest.fixture
def seeded_home(tmp_path):
    """A config dir holding the seeded library at its canonical path."""
    db_path = tmp_path / "workspace" / "knowledge" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    store = _seed_library(db_path)
    store.db.close()
    return tmp_path


@pytest.fixture
def seeded_store(tmp_path):
    store = _seed_library(tmp_path / "knowledge.db")
    yield store
    store.db.close()


class TestStoreAggregate:
    def test_totals_match_the_fixture_contract(self, seeded_store):
        stats = seeded_store.aggregate_stats()
        assert (stats.sources, stats.documents, stats.items) == (
            EXPECTED_SOURCES,
            EXPECTED_DOCUMENTS,
            EXPECTED_ITEMS,
        )

    def test_per_source_reconciles_with_the_totals(self, seeded_store):
        stats = seeded_store.aggregate_stats()
        assert sum(s.items for s in stats.per_source) == stats.items
        assert sum(s.documents for s in stats.per_source) == stats.documents

    def test_per_source_rows_carry_the_seeded_split(self, seeded_store):
        stats = seeded_store.aggregate_stats()
        by_name = {s.name: (s.documents, s.items) for s in stats.per_source}
        assert by_name == {
            "alpha": (2, 4),
            "beta": (1, 2),
            "gamma": (0, 0),
            "(no source)": (1, 2),
        }

    def test_registered_source_with_no_items_is_listed_at_zero(self, seeded_store):
        stats = seeded_store.aggregate_stats()
        gamma = next(s for s in stats.per_source if s.name == "gamma")
        assert (gamma.documents, gamma.items) == (0, 0)
        assert gamma.source_id is not None

    def test_sourceless_bucket_reports_no_source_id(self, seeded_store):
        stats = seeded_store.aggregate_stats()
        orphan = next(s for s in stats.per_source if s.source_id is None)
        assert orphan.items == 2, "one hashed document chunk plus one hashless item"
        assert orphan.documents == 1, "the hashless item belongs to no document"

    def test_sourceless_bucket_is_absent_when_every_item_has_a_source(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "owned.db"))
        try:
            source_id = store.add_source("only", "folder", "/tmp/only")
            _add_document(store, source_id, "doc", chunks=2)
            stats = store.aggregate_stats()
        finally:
            store.db.close()
        assert [s.source_id for s in stats.per_source] == [source_id]

    def test_non_active_items_are_excluded(self, seeded_store):
        raw = seeded_store.db.execute("SELECT COUNT(*) AS n FROM items").fetchone()["n"]
        assert raw == EXPECTED_ITEMS + 1
        assert seeded_store.aggregate_stats().items == EXPECTED_ITEMS

    def test_empty_library_reports_zeroes(self, tmp_path):
        store = KnowledgeStore(str(tmp_path / "empty.db"))
        try:
            stats = store.aggregate_stats()
        finally:
            store.db.close()
        assert (stats.sources, stats.documents, stats.items, stats.per_source) == (0, 0, 0, ())

    def test_reading_the_aggregate_leaves_every_row_untouched(self, tmp_path):
        db_path = tmp_path / "knowledge.db"
        store = _seed_library(db_path)
        store.db.close()
        before = _library_fingerprint(db_path)
        reader = KnowledgeStore(str(db_path))
        try:
            reader.aggregate_stats()
        finally:
            reader.db.close()
        assert _library_fingerprint(db_path) == before


class TestCliStatsVerb:
    def _run(self, home, as_json: bool = False):
        from kiro_crew import cli

        args = SimpleNamespace(knowledge_action="stats", json=as_json)
        with (
            patch("kiro_crew.cli.config_dir", return_value=home),
            patch("kiro_crew.cli.sel", return_value=MagicMock()),
        ):
            cli._knowledge(args)

    def test_human_output_carries_totals_and_a_row_per_source(self, seeded_home, capsys):
        self._run(seeded_home)
        out = capsys.readouterr().out
        assert f"{EXPECTED_SOURCES} source(s)" in out
        assert f"{EXPECTED_DOCUMENTS} document(s)" in out
        assert f"{EXPECTED_ITEMS} item(s)" in out
        for name in ("alpha", "beta", "gamma", "(no source)"):
            assert name in out

    def test_json_output_matches_the_fixture_contract(self, seeded_home, capsys):
        self._run(seeded_home, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["sources"] == EXPECTED_SOURCES
        assert payload["documents"] == EXPECTED_DOCUMENTS
        assert payload["items"] == EXPECTED_ITEMS
        assert sum(row["items"] for row in payload["per_source"]) == EXPECTED_ITEMS
        orphan = next(row for row in payload["per_source"] if row["id"] is None)
        assert orphan["items"] == 2

    def test_unconfigured_library_is_reported_not_crashed(self, tmp_path, capsys):
        self._run(tmp_path)
        assert "not configured" in capsys.readouterr().out

    def test_unconfigured_library_stays_machine_readable_under_json(self, tmp_path, capsys):
        self._run(tmp_path, as_json=True)
        assert json.loads(capsys.readouterr().out) == {"error": "not_configured"}

    def test_stats_leaves_every_library_row_untouched(self, seeded_home):
        db_path = seeded_home / "workspace" / "knowledge" / "knowledge.db"
        before = _library_fingerprint(db_path)
        self._run(seeded_home)
        assert _library_fingerprint(db_path) == before

    def test_usage_names_both_verbs_and_nothing_else(self, seeded_home, capsys):
        from kiro_crew import cli

        with (
            patch("kiro_crew.cli.config_dir", return_value=seeded_home),
            patch("kiro_crew.cli.sel", return_value=MagicMock()),
        ):
            cli._knowledge(SimpleNamespace(knowledge_action=None))
        usage = capsys.readouterr().out
        assert "knowledge dedup" in usage and "knowledge stats" in usage

    @pytest.mark.parametrize("refused", ["flush", "rebuild", "repair", "reindex"])
    def test_a_repair_action_reaches_no_library(self, seeded_home, capsys, refused):
        from kiro_crew import cli

        db_path = seeded_home / "workspace" / "knowledge" / "knowledge.db"
        before = _library_fingerprint(db_path)
        with (
            patch("kiro_crew.cli.config_dir", return_value=seeded_home),
            patch("kiro_crew.cli.sel", return_value=MagicMock()),
        ):
            cli._knowledge(SimpleNamespace(knowledge_action=refused))
        assert "Usage:" in capsys.readouterr().out
        assert _library_fingerprint(db_path) == before


class TestMcpParity:
    """The MCP twin answers the same question from the same aggregate."""

    def _call(self, home):
        from kiro_crew.mcp_core import _call_tool_inner

        with (
            patch("kiro_crew.mcp_core.config_dir", return_value=home),
            patch("kiro_crew.mcp_core.sel", return_value=MagicMock()),
        ):
            return _call_tool_inner("knowledge_list_sources", {})

    def test_tool_reports_the_fixture_contract_totals(self, seeded_home):
        result = self._call(seeded_home)
        assert (
            f"Knowledge library: {EXPECTED_SOURCES} source(s), "
            f"{EXPECTED_DOCUMENTS} document(s), {EXPECTED_ITEMS} item(s)."
        ) in result

    def test_tool_totals_equal_the_cli_json(self, seeded_home, capsys):
        from kiro_crew import cli

        with (
            patch("kiro_crew.cli.config_dir", return_value=seeded_home),
            patch("kiro_crew.cli.sel", return_value=MagicMock()),
        ):
            cli._knowledge(SimpleNamespace(knowledge_action="stats", json=True))
        payload = json.loads(capsys.readouterr().out)
        result = self._call(seeded_home)
        assert (
            f"{payload['sources']} source(s), {payload['documents']} document(s), "
            f"{payload['items']} item(s)."
        ) in result

    def test_tool_still_lists_every_source_id_for_scoping(self, seeded_home):
        result = self._call(seeded_home)
        assert "Sources (3):" in result
        for name in ("alpha", "beta", "gamma"):
            assert f"- {name} — id: " in result

    def test_descriptor_advertises_the_counts_and_the_read_only_boundary(self):
        from kiro_crew.mcp_tools.knowledge import schemas

        descriptor = next(d for d in schemas() if d["name"] == "knowledge_list_sources")
        description = descriptor["description"]
        assert "documents" in description and "items" in description
        assert "never rebuilds" in description

    def test_tool_leaves_every_library_row_untouched(self, seeded_home):
        db_path = seeded_home / "workspace" / "knowledge" / "knowledge.db"
        before = _library_fingerprint(db_path)
        self._call(seeded_home)
        assert _library_fingerprint(db_path) == before

    def test_unconfigured_library_is_reported_not_crashed(self, tmp_path):
        assert "not configured" in self._call(tmp_path)
