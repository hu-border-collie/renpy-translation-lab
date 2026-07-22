# -*- coding: utf-8 -*-
"""Tests for Project Analysis Phase 1 contract (#256)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import project_analysis as pa


def _lineage(source_fp: str = "src-fp-1", upstream: str = "up-1") -> dict:
    return pa.empty_lineage(
        source_fingerprint=source_fp,
        upstream_dependency_digest=upstream,
        prompt_schema_version="prompt-v1",
        generated_at="2026-07-22T00:00:00Z",
    )


def _chunk(cid: str, item_ids: list[str], *, status: str = pa.STATUS_PUBLISHED) -> dict:
    return {
        "id": cid,
        "kind": pa.KIND_CHUNK,
        "status": status,
        "source_files": ["script.rpy"],
        "evidence_item_ids": item_ids,
        "line_span": [10, 20],
        "source_checksum": "chk-" + cid,
        "upstream_artifact_ids": [],
        "lineage": _lineage(),
        "summary": f"summary for {cid}",
    }


def _label(lid: str, upstream: list[str], *, status: str = pa.STATUS_PUBLISHED) -> dict:
    return {
        "id": lid,
        "kind": pa.KIND_LABEL,
        "status": status,
        "source_files": ["script.rpy"],
        "label_id": lid,
        "evidence_item_ids": [],
        "upstream_artifact_ids": upstream,
        "source_checksum": "chk-" + lid,
        "lineage": _lineage(),
        "summary": f"label {lid}",
    }


def _route(rid: str, upstream: list[str], *, status: str = pa.STATUS_PUBLISHED) -> dict:
    return {
        "id": rid,
        "kind": pa.KIND_ROUTE,
        "status": status,
        "source_files": ["script.rpy"],
        "route_id": rid,
        "evidence_item_ids": [],
        "upstream_artifact_ids": upstream,
        "source_checksum": "chk-" + rid,
        "lineage": _lineage(),
        "summary": f"route {rid}",
    }


def _brief(*, status: str = pa.STATUS_PUBLISHED) -> dict:
    return {
        "id": "project_brief",
        "kind": pa.KIND_PROJECT_BRIEF,
        "status": status,
        "source_files": ["script.rpy"],
        "evidence_item_ids": [],
        "upstream_artifact_ids": ["route-a", "route-b"],
        "lineage": _lineage(),
        "summary": "global brief",
    }


class SchemaAndFingerprintTests(unittest.TestCase):
    def test_normalize_summary_requires_id_and_kind(self):
        with self.assertRaises(pa.ProjectAnalysisSchemaError):
            pa.normalize_summary_record({"kind": "chunk", "status": "draft"})
        with self.assertRaises(pa.ProjectAnalysisSchemaError):
            pa.normalize_summary_record({"id": "c1", "kind": "nope", "status": "draft"})

    def test_unknown_schema_version_rejected(self):
        with self.assertRaises(pa.ProjectAnalysisSchemaError):
            pa.normalize_lineage({"schema_version": 99})
        with self.assertRaises(pa.ProjectAnalysisSchemaError):
            pa.normalize_manifest({"schema_version": 99})

    def test_published_becomes_stale_on_fingerprint_mismatch(self):
        record = _chunk("chunk-1", ["item-1"])
        self.assertEqual(
            pa.evaluate_record_status(record, expected_source_fingerprint="src-fp-1"),
            pa.STATUS_PUBLISHED,
        )
        self.assertEqual(
            pa.evaluate_record_status(record, expected_source_fingerprint="src-fp-OTHER"),
            pa.STATUS_STALE,
        )
        self.assertFalse(
            pa.is_injectable_record(record, expected_source_fingerprint="src-fp-OTHER")
        )
        self.assertTrue(
            pa.is_injectable_record(record, expected_source_fingerprint="src-fp-1")
        )

    def test_digest_source_items_stable(self):
        items = [
            {"id": "b", "source_checksum": "2"},
            {"id": "a", "source_checksum": "1", "file_rel_path": "a.rpy"},
        ]
        d1 = pa.digest_source_items(items)
        d2 = pa.digest_source_items(list(reversed(items)))
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 64)


class InvalidationPlannerTests(unittest.TestCase):
    def test_local_item_change_invalidates_same_route_only(self):
        chunks = [
            _chunk("chunk-a1", ["item-a"]),
            _chunk("chunk-b1", ["item-b"]),
        ]
        labels = [
            _label("label-a", ["chunk-a1"]),
            _label("label-b", ["chunk-b1"]),
        ]
        routes = [
            _route("route-a", ["label-a"]),
            _route("route-b", ["label-b"]),
        ]
        plan = pa.plan_invalidation(
            chunks=chunks,
            labels=labels,
            routes=routes,
            project_brief=_brief(),
            changed_item_ids=["item-a"],
        )
        self.assertIn("chunk-a1", plan.stale_artifact_ids)
        self.assertIn("label-a", plan.stale_artifact_ids)
        self.assertIn("route-a", plan.stale_artifact_ids)
        self.assertIn("project_brief", plan.stale_artifact_ids)
        self.assertNotIn("chunk-b1", plan.stale_artifact_ids)
        self.assertNotIn("label-b", plan.stale_artifact_ids)
        self.assertNotIn("route-b", plan.stale_artifact_ids)
        self.assertTrue(plan.brief_stale)

    def test_unrelated_route_survives_other_route_change(self):
        plan = pa.plan_invalidation(
            chunks=[_chunk("chunk-a1", ["item-a"]), _chunk("chunk-b1", ["item-b"])],
            labels=[_label("label-a", ["chunk-a1"]), _label("label-b", ["chunk-b1"])],
            routes=[_route("route-a", ["label-a"]), _route("route-b", ["label-b"])],
            project_brief=_brief(),
            changed_item_ids=["item-b"],
        )
        self.assertIn("route-b", plan.stale_artifact_ids)
        self.assertNotIn("route-a", plan.stale_artifact_ids)

    def test_apply_invalidation_marks_records_stale(self):
        records = [_chunk("chunk-a1", ["item-a"]), _chunk("chunk-b1", ["item-b"])]
        plan = pa.plan_invalidation(chunks=records, changed_item_ids=["item-a"])
        updated = pa.apply_invalidation_to_records(
            records, plan, default_kind=pa.KIND_CHUNK
        )
        by_id = {r["id"]: r for r in updated}
        self.assertEqual(by_id["chunk-a1"]["status"], pa.STATUS_STALE)
        self.assertEqual(by_id["chunk-b1"]["status"], pa.STATUS_PUBLISHED)


class StoreIoTests(unittest.TestCase):
    def test_atomic_roundtrip_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_summaries(
                pa.KIND_CHUNK,
                [_chunk("chunk-a1", ["item-a"]), _chunk("chunk-b1", ["item-b"])],
            )
            store.save_summaries(
                pa.KIND_LABEL,
                [_label("label-a", ["chunk-a1"]), _label("label-b", ["chunk-b1"])],
            )
            store.save_routes(
                [_route("route-a", ["label-a"]), _route("route-b", ["label-b"])]
            )
            store.save_brief_text("# draft\n", published=False)
            store.save_brief_text("# published\n", published=True)
            manifest = store.rebuild_manifest(
                project_identity={"game_root": "/game"},
                expected_source_fingerprint="src-fp-1",
            )
            # Force brief lineage so published stays fresh under expected fp.
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF]["status"] = pa.STATUS_PUBLISHED
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF]["lineage"] = _lineage()
            store.save_manifest(manifest)

            status = store.collect_status(expected_source_fingerprint="src-fp-1")
            self.assertTrue(status["store_exists"])
            self.assertEqual(status["chunk_count"], 2)
            self.assertEqual(status["route_count"], 2)
            self.assertEqual(status["brief_status"], pa.STATUS_PUBLISHED)
            self.assertEqual(status["overall_status"], pa.STATUS_PUBLISHED)
            self.assertTrue(status["injectable"])

            stale_status = store.collect_status(
                expected_source_fingerprint="src-fp-OTHER"
            )
            self.assertEqual(stale_status["overall_status"], pa.STATUS_STALE)
            self.assertFalse(stale_status["injectable"])

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            with self.assertRaises(pa.ProjectAnalysisPathEscapeError):
                store.path_for("../outside.json")
            with self.assertRaises(pa.ProjectAnalysisPathEscapeError):
                pa.resolve_under_store(tmp, "..\\secret.txt")
            if os.name == "nt":
                with self.assertRaises(pa.ProjectAnalysisPathEscapeError):
                    pa.resolve_under_store(tmp, "C:/Windows/system32/drivers")

    def test_corrupt_json_and_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            Path(store.manifest_path).write_text("{not-json", encoding="utf-8")
            status = store.collect_status()
            self.assertEqual(status["overall_status"], pa.STATUS_FAILED)
            self.assertIn("corrupt", status["error"].lower())

            Path(store.manifest_path).write_text(
                json.dumps({"schema_version": 99}), encoding="utf-8"
            )
            status = store.collect_status()
            self.assertEqual(status["overall_status"], pa.STATUS_FAILED)
            self.assertIn("schema_version", status["error"])

    def test_missing_store_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(os.path.join(tmp, "empty"))
            status = store.collect_status()
            self.assertEqual(status["overall_status"], pa.STATUS_MISSING)
            self.assertFalse(status["store_exists"])
            self.assertFalse(status["injectable"])

    def test_stale_published_not_injectable_via_helper(self):
        record = _chunk("chunk-1", ["item-1"], status=pa.STATUS_PUBLISHED)
        self.assertFalse(
            pa.is_injectable_record(record, expected_source_fingerprint="other")
        )


class CliStatusTests(unittest.TestCase):
    def test_project_analysis_status_command(self):
        import gemini_translate_batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_summaries(pa.KIND_CHUNK, [_chunk("chunk-1", ["item-1"])])
            store.rebuild_manifest(expected_source_fingerprint="src-fp-1")

            with mock.patch("builtins.print") as print_mock:
                batch_mod.main(
                    [
                        "project-analysis-status",
                        "--store-dir",
                        tmp,
                        "--source-fingerprint",
                        "src-fp-1",
                    ]
                )
            text = "\n".join(str(c.args[0]) for c in print_mock.call_args_list if c.args)
            self.assertIn("Project analysis status", text)
            self.assertIn("Overall:", text)

            with mock.patch("builtins.print") as print_mock:
                batch_mod.main(
                    ["project-analysis-status", "--store-dir", tmp, "--json"]
                )
            printed = [
                str(c.args[0]) for c in print_mock.call_args_list if c.args
            ]
            payload = None
            for chunk in printed:
                text = chunk.strip()
                if text.startswith("{"):
                    payload = json.loads(text)
                    break
            self.assertIsNotNone(payload, msg=f"no JSON status in prints: {printed!r}")
            self.assertEqual(payload["chunk_count"], 1)
            self.assertIn("overall_status", payload)

    def test_format_status_label_chinese(self):
        label = pa.format_status_label(
            {
                "overall_status": pa.STATUS_PUBLISHED,
                "store_exists": True,
                "chunk_count": 2,
                "label_count": 1,
                "route_count": 1,
                "brief_status": pa.STATUS_PUBLISHED,
                "injectable": True,
            }
        )
        self.assertIn("已发布", label)


class DoctorIntegrationTests(unittest.TestCase):
    def test_doctor_context_includes_project_analysis(self):
        import gemini_translate_batch as batch_mod

        with tempfile.TemporaryDirectory() as tmp:
            status = pa.collect_project_analysis_status(store_dir=tmp)
            self.assertEqual(status["overall_status"], pa.STATUS_MISSING)

            context = batch_mod.collect_doctor_context_status()
            self.assertIn("project_analysis", context)
            self.assertIn("overall_status", context["project_analysis"])


if __name__ == "__main__":
    unittest.main()
