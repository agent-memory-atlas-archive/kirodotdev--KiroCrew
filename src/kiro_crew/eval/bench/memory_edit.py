"""Offline memory-editor scale probe, with isolated synthetic V1/V2 databases.

Run ``python -m kiro_crew.eval.bench.memory_edit --json /output/scale.json``.
Measures one warm synthetic pass with tracemalloc enabled, not P50/P95, native
RSS, generation throughput, semantic extraction quality or answer accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

from kiro_crew import memory_edit, memory_record_metadata, memory_schema, memory_stores
from kiro_crew.config import loader
from kiro_crew.vector_memory import VectorMemoryStore


def measure(home: Path, *, rows: int = 10000) -> list[dict]:
    """The caller owns an empty temporary home. No provider/model is allocated."""
    name = "member-scale"
    owner = {"memory_version": 2, "owner_member": "scale"}
    (home / "config.json").write_text(
        json.dumps(
            {
                "memory_stores": {"default": {}, name: owner},
                "agents": {"scale": {"memory_store": name}},
            }
        ),
        encoding="utf-8",
    )
    member = home / "memory_stores" / name
    member.mkdir(parents=True)
    (member / "member-memory.json").write_text(json.dumps(owner), encoding="utf-8")
    results = []
    with patch.dict(os.environ, {"KIROCREW_HOME": str(home), "KIROCREW_SKIP_MODEL_DOWNLOAD": "1"}):
        loader._invalidate_config_cache()
        with patch.object(memory_stores, "_DECLARED_MEMO", None):
            try:
                for version, directory in (("v1", home), ("v2", member)):
                    store = VectorMemoryStore(db_path=directory / "memory.db")
                    store.init()
                    try:
                        stamp = "2026-09-07T00:00:00+00:00"
                        # One deterministic indexed dataset; this deliberately
                        # does not measure conversation extraction or embedding.
                        with store._db_lock:
                            for index in range(rows):
                                value = (
                                    f"Contact account{index}@old.example for invoices."
                                    if index % 5 == 0
                                    else f"Project setting number {index} uses the north deployment."
                                )
                                store.db.execute(
                                    memory_schema.semantic_upsert(store._lineage),
                                    memory_schema.semantic_upsert_params(
                                        store._lineage,
                                        f"user.scale{index:05}",
                                        json.dumps(value),
                                        1.0,
                                        "user_explicit",
                                        stamp,
                                    ),
                                )
                            memory_record_metadata.reconcile(store.db)
                            store.db.commit()
                        expected = (rows + 4) // 5
                        timings = {}
                        tracemalloc.start()
                        try:
                            started = time.perf_counter()
                            listing = memory_edit.list_records(
                                store, {"q": "@old.example"}, offset=max(0, expected - 50)
                            )
                            timings["last_page_ms"] = round(
                                (time.perf_counter() - started) * 1000, 2
                            )
                            started = time.perf_counter()
                            preview = memory_edit.preview_edit(
                                store,
                                version,
                                b"isolated-fixture-key",
                                {
                                    "selection": {"query": {"q": "@old.example"}},
                                    "operation": {
                                        "type": "replace_text",
                                        "find": "@old.example",
                                        "replacement": "@new.example",
                                    },
                                },
                            )
                            timings["preview_ms"] = round((time.perf_counter() - started) * 1000, 2)
                            started = time.perf_counter()
                            applied = memory_edit.apply_edit(
                                store, version, b"isolated-fixture-key", preview["preview_id"]
                            )
                            timings["apply_ms"] = round((time.perf_counter() - started) * 1000, 2)
                            _, peak = tracemalloc.get_traced_memory()
                        finally:
                            tracemalloc.stop()
                        after = memory_edit.list_records(store, {"q": "@new.example"})
                        assert listing["total"] == expected and not listing["has_more"]
                        assert (
                            preview["changed_count"]
                            == applied["changed_count"]
                            == after["total"]
                            == expected
                        )
                        results.append(
                            {
                                "version": version,
                                "store_rows": rows,
                                "email_matches": expected,
                                "edited": applied["changed_count"],
                                "preview_sample": len(preview["entries"]),
                                "peak_python_allocation_mib": round(peak / 1048576, 2),
                                "timings_with_tracemalloc": timings,
                                "embedding_calls": 0,
                            }
                        )
                    finally:
                        store.close()
            finally:
                loader._invalidate_config_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10000)
    args = parser.parse_args()
    if not 1 <= args.rows <= 50000:
        parser.error("--rows must be between 1 and 50000")
    with tempfile.TemporaryDirectory(prefix="kirocrew-memory-edit-") as work:
        results = measure(Path(work), rows=args.rows)
    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "workload": "synthetic indexed facts; every fifth row contains one email; one pass with tracemalloc",
        "results": results,
    }
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
