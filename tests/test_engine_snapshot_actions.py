"""Pure engine/snapshot/reuse GUI loader tests (#424 P6 GUI presentation)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gui_qt.engine_snapshot_actions import (
    EngineSnapshotActionError,
    collect_engine_snapshot_overview,
    discover_snapshot_manifests,
    excerpt_text,
    load_reuse_candidates_overview,
    reconcile_snapshots,
    snapshot_locator_text,
)


def _occurrence(occurrence_id, file_rel_path, line, source_text="Hello"):
    return SimpleNamespace(
        occurrence_id=occurrence_id,
        locator={
            "engine": "renpy",
            "locator_schema_version": 1,
            "locator": {"file_rel_path": file_rel_path, "line_hint": line},
        },
        source_text=source_text,
    )


class EngineSnapshotActionsTests(unittest.TestCase):
    def test_snapshot_locator_text_uses_nested_locator(self):
        locator = {
            "engine": "renpy",
            "locator_schema_version": 1,
            "locator": {"file_rel_path": "game/tl/schinese/script.rpy", "line_hint": 7},
        }
        self.assertEqual(
            snapshot_locator_text(locator),
            "game/tl/schinese/script.rpy:7",
        )
        self.assertEqual(snapshot_locator_text(None), "(locator unavailable)")

    def test_excerpt_text_is_bounded(self):
        self.assertEqual(excerpt_text("abc"), "abc")
        self.assertTrue(excerpt_text("x" * 300).endswith("…"))
        self.assertEqual(len(excerpt_text("x" * 300)), 160)

    def test_discover_snapshot_manifests_reads_valid_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project_snapshots"
            older = root / "v1"
            newer = root / "v2"
            invalid = root / "broken"
            for directory in (older, newer, invalid):
                directory.mkdir(parents=True)
            older.joinpath("project_snapshot.json").write_text(
                json.dumps(
                    {
                        "kind": "project_snapshot",
                        "engine": "renpy",
                        "adapter_version": "1.1.8",
                        "game_version": {"version_id": "1.0.0"},
                        "coverage": {"coverage_status": "attention"},
                        "snapshot_digest": "abc",
                        "occurrence_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            newer.joinpath("project_snapshot.json").write_text(
                json.dumps(
                    {
                        "kind": "project_snapshot",
                        "engine": "renpy",
                        "adapter_version": "1.1.8",
                        "game_version": {"version_id": "1.1.0"},
                        "coverage": {"coverage_status": "block"},
                        "snapshot_digest": "def",
                        "occurrence_count": 3,
                    }
                ),
                encoding="utf-8",
            )
            invalid.joinpath("project_snapshot.json").write_text("{bad json", encoding="utf-8")
            os.utime(older / "project_snapshot.json", (1000, 1000))
            os.utime(newer / "project_snapshot.json", (2000, 2000))

            snapshots = discover_snapshot_manifests(str(root))

        self.assertEqual([item["version_id"] for item in snapshots], ["1.1.0", "1.0.0"])
        self.assertEqual(snapshots[0]["coverage_status"], "block")
        self.assertEqual(snapshots[1]["occurrence_count"], 2)

    def test_collect_overview_uses_adapter_capabilities_and_snapshot_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project_snapshots"
            package = root / "v1"
            package.mkdir(parents=True)
            package.joinpath("project_snapshot.json").write_text(
                json.dumps(
                    {
                        "kind": "project_snapshot",
                        "engine": "renpy",
                        "adapter_version": "1.1.8",
                        "game_version": {"version_id": "1.0.0"},
                        "coverage": {},
                    }
                ),
                encoding="utf-8",
            )
            import gemini_translate_batch as batch

            with mock.patch.object(batch, "PROJECT_SNAPSHOTS_DIR", str(root)):
                payload = collect_engine_snapshot_overview(game_root="")

        self.assertEqual(payload["engine"], "renpy")
        self.assertIn("text_span_replace", payload["capabilities"]["declarative_writeback"])
        self.assertEqual(payload["snapshot_count"], 1)
        self.assertEqual(payload["latest_snapshot"]["version_id"], "1.0.0")

    def test_reconcile_snapshots_formats_items_and_locators(self):
        base = SimpleNamespace(
            occurrences=(_occurrence("base:1", "old.rpy", 1),),
        )
        target = SimpleNamespace(
            occurrences=(
                _occurrence("target:1", "new.rpy", 2, "Hello there"),
                _occurrence("target:2", "new.rpy", 3, "Other"),
            ),
        )
        item = SimpleNamespace(
            item_id="item-1",
            disposition="ambiguous",
            match_kind="ambiguous",
            confidence=0.5,
            evidence={"source_equal": True},
            base_occurrence_id="base:1",
            target_occurrence_id="",
            candidate_target_occurrence_ids=("target:1", "target:2"),
        )
        report = SimpleNamespace(
            base_version_id="1.0.0",
            target_version_id="1.1.0",
            status="attention",
            summary={"matched": 0, "ambiguous": 1},
            coverage_changes={"added": 1},
            reconciliation_digest="digest",
            items=(item,),
        )
        with (
            mock.patch(
                "engine_adapters.versioning.load_project_snapshot",
                side_effect=[base, target],
            ),
            mock.patch(
                "engine_adapters.versioning.reconcile_project_snapshots",
                return_value=report,
            ),
        ):
            payload = reconcile_snapshots("/snap/base", "/snap/target")

        self.assertEqual(payload["status"], "attention")
        self.assertEqual(payload["item_count"], 1)
        row = payload["items"][0]
        self.assertTrue(row["ambiguous"])
        self.assertEqual(row["base_locator"], "old.rpy:1")
        self.assertEqual(row["candidate_locators"], ["new.rpy:2", "new.rpy:3"])
        self.assertEqual(row["evidence"], {"source_equal": True})

    def test_reconcile_snapshots_requires_both_paths(self):
        with self.assertRaises(EngineSnapshotActionError):
            reconcile_snapshots("", "/snap/target")

    def test_reconcile_snapshots_wraps_artifact_errors(self):
        from engine_adapters.versioning import VersioningArtifactError

        with mock.patch(
            "engine_adapters.versioning.load_project_snapshot",
            side_effect=VersioningArtifactError("bad snapshot"),
        ):
            with self.assertRaises(EngineSnapshotActionError) as raised:
                reconcile_snapshots("/snap/base", "/snap/target")
        self.assertIn("bad snapshot", str(raised.exception))

    def test_load_reuse_candidates_overview_formats_candidates(self):
        candidate = SimpleNamespace(
            candidate_id="reusecand1:abc",
            reuse_class="ambiguous",
            status="pending",
            confidence=0.42,
            reference_only=False,
            has_translation_record=True,
            reference_origin="model_initial",
            base_version_id="1.0.0",
            target_version_id="1.1.0",
            base_occurrence_id="base:1",
            target_occurrence_id="",
            candidate_target_occurrence_ids=("target:1", "target:2"),
            reference_translation="回忆",
            effective_translation="回忆",
            evidence={"source_equal": True},
            decision={},
            audit=(),
        )
        candidate_set = SimpleNamespace(
            status="attention",
            stale_reasons=(),
            summary={"ambiguous": 1},
            base_version_id="1.0.0",
            target_version_id="1.1.0",
            reconciliation_digest="digest",
            candidate_set_digest="setdigest",
            candidates=(candidate,),
        )
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "reuse"
            package.mkdir()
            report = package / "reuse_report.json"
            report.write_text("{}", encoding="utf-8")
            with mock.patch(
                "engine_adapters.reuse.load_reuse_candidates",
                return_value=candidate_set,
            ):
                payload = load_reuse_candidates_overview(str(package))

        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["summary"], {"ambiguous": 1})
        row = payload["candidates"][0]
        self.assertEqual(row["reuse_class"], "ambiguous")
        self.assertEqual(row["candidate_target_occurrence_ids"], ["target:1", "target:2"])

    def test_reuse_overview_pages_large_packages_and_keeps_full_detail(self):
        long_translation = "原创测试旧译文甲" * 30
        candidates = tuple(
            SimpleNamespace(
                candidate_id=f"reusecand1:{index}",
                reuse_class="exact_reuse",
                status="pending",
                confidence=0.95,
                reference_only=False,
                has_translation_record=True,
                reference_origin="model_initial",
                base_version_id="1.0.0",
                target_version_id="1.1.0",
                base_occurrence_id=f"base:{index}",
                target_occurrence_id=f"target:{index}",
                candidate_target_occurrence_ids=(),
                reference_translation=long_translation,
                effective_translation=long_translation,
                evidence={"source_equal": True},
                decision={},
                audit=({"action": "reject", "note": "原创审计条目"},),
            )
            for index in range(501)
        )
        candidate_set = SimpleNamespace(
            status="attention",
            stale_reasons=(),
            summary={"class_exact_reuse": 501},
            base_version_id="1.0.0",
            target_version_id="1.1.0",
            reconciliation_digest="digest",
            candidate_set_digest="setdigest",
            candidates=candidates,
        )
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "reuse"
            package.mkdir()
            (package / "reuse_report.json").write_text("{}", encoding="utf-8")
            with mock.patch(
                "engine_adapters.reuse.load_reuse_candidates",
                return_value=candidate_set,
            ):
                first_page = load_reuse_candidates_overview(str(package))
                last_page = load_reuse_candidates_overview(
                    str(package),
                    candidate_offset=500,
                )

        self.assertEqual(first_page["candidate_count"], 501)
        self.assertEqual(first_page["candidate_start"], 1)
        self.assertEqual(first_page["candidate_end"], 500)
        self.assertEqual(len(first_page["candidates"]), 500)
        first = first_page["candidates"][0]
        self.assertEqual(len(first["effective_translation"]), 160)
        self.assertEqual(first["effective_translation_full"], long_translation)
        self.assertEqual(first["audit"], [{"action": "reject", "note": "原创审计条目"}])
        self.assertEqual(last_page["candidate_start"], 501)
        self.assertEqual(last_page["candidate_end"], 501)
        self.assertEqual(last_page["candidates"][0]["candidate_id"], "reusecand1:500")
        self.assertEqual(last_page["candidates"][0]["effective_translation_full"], long_translation)

    def test_load_reuse_candidates_overview_requires_path(self):
        with self.assertRaises(EngineSnapshotActionError):
            load_reuse_candidates_overview("")


if __name__ == "__main__":
    unittest.main()
