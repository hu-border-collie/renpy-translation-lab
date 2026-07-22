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

    def test_missing_fingerprint_on_published_is_stale(self):
        """P1: expected fingerprint + empty lineage must not stay injectable."""
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_summaries(
                pa.KIND_CHUNK,
                [
                    {
                        **_chunk("chunk-1", ["item-1"]),
                        "lineage": pa.empty_lineage(),  # published but no fingerprint
                    }
                ],
            )
            manifest = store.rebuild_manifest()
            # Aggregate lineage is empty; claim published.
            manifest["artifacts"][pa.KIND_CHUNK]["status"] = pa.STATUS_PUBLISHED
            manifest["artifacts"][pa.KIND_CHUNK]["lineage"] = pa.empty_lineage()
            store.save_manifest(manifest)

            status = store.collect_status(expected_source_fingerprint="src-fp-1")
            self.assertEqual(status["overall_status"], pa.STATUS_STALE)
            self.assertFalse(status["injectable"])
            self.assertEqual(
                status["artifacts"][pa.KIND_CHUNK]["status"], pa.STATUS_STALE
            )

    def test_manifest_published_brief_without_file_not_injectable(self):
        """P1: manifest alone cannot claim published brief without the md file."""
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_summaries(
                pa.KIND_CHUNK, [_chunk("chunk-1", ["item-1"])]
            )
            manifest = store.rebuild_manifest(expected_source_fingerprint="src-fp-1")
            # Ensure chunk layer is published+fresh.
            manifest["artifacts"][pa.KIND_CHUNK]["status"] = pa.STATUS_PUBLISHED
            manifest["artifacts"][pa.KIND_CHUNK]["lineage"] = _lineage()
            # Claim brief published without writing project_brief.published.md
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF] = {
                "status": pa.STATUS_PUBLISHED,
                "draft_present": False,
                "published_present": True,
                "lineage": _lineage(),
                "id": "project_brief",
            }
            store.save_manifest(manifest)
            self.assertFalse(
                os.path.isfile(store.artifact_path(pa.PROJECT_BRIEF_PUBLISHED_FILENAME))
            )

            status = store.collect_status(expected_source_fingerprint="src-fp-1")
            self.assertEqual(status["brief_status"], pa.STATUS_STALE)
            self.assertFalse(status["brief_published_present"])
            self.assertNotEqual(status["overall_status"], pa.STATUS_PUBLISHED)
            self.assertFalse(status["injectable"])

    def test_kind_must_match_target_file(self):
        """P2: chunk_summaries.jsonl must not accept kind=route rows."""
        with self.assertRaises(pa.ProjectAnalysisSchemaError):
            pa.normalize_summary_record(
                {
                    "id": "r1",
                    "kind": pa.KIND_ROUTE,
                    "status": pa.STATUS_DRAFT,
                    "lineage": pa.empty_lineage(),
                },
                default_kind=pa.KIND_CHUNK,
            )
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            with self.assertRaises(pa.ProjectAnalysisSchemaError):
                store.save_summaries(
                    pa.KIND_CHUNK,
                    [
                        {
                            "id": "bad",
                            "kind": pa.KIND_ROUTE,
                            "status": pa.STATUS_DRAFT,
                            "source_files": [],
                            "evidence_item_ids": [],
                            "upstream_artifact_ids": [],
                            "lineage": pa.empty_lineage(),
                        }
                    ],
                )
            # Loader rejects mismatched rows already on disk.
            path = store.artifact_path(pa.CHUNK_SUMMARIES_FILENAME)
            os.makedirs(tmp, exist_ok=True)
            Path(path).write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "kind": "route",
                        "status": "draft",
                        "source_files": [],
                        "evidence_item_ids": [],
                        "upstream_artifact_ids": [],
                        "lineage": pa.empty_lineage(),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(pa.ProjectAnalysisSchemaError):
                store.load_summaries(pa.KIND_CHUNK)


class PathAndCliContractTests(unittest.TestCase):
    def test_base_dir_selects_distinct_tool_store_slugs(self):
        """P2: project switch must resolve different analysis store dirs."""
        import translator_runtime as runtime

        previous_location = runtime.CONTEXT_STORAGE_LOCATION
        try:
            runtime.CONTEXT_STORAGE_LOCATION = "tool"
            # Platform-native fixtures: .../ProjectA/work → slug ProjectA.
            root = Path(tempfile.gettempdir()) / "rtl-pa-slug-fixture"
            base_a = str(root / "ProjectA" / "work")
            base_b = str(root / "ProjectB" / "work")
            slug_a = runtime._project_slug_from_base_dir(base_a)
            slug_b = runtime._project_slug_from_base_dir(base_b)
            self.assertEqual(slug_a, "ProjectA")
            self.assertEqual(slug_b, "ProjectB")

            path_a = runtime.get_default_project_analysis_store_dir(base_a)
            path_b = runtime.get_default_project_analysis_store_dir(base_b)
            self.assertNotEqual(path_a, path_b)
            self.assertEqual(os.path.basename(path_a), slug_a)
            self.assertEqual(os.path.basename(path_b), slug_b)
            self.assertEqual(os.path.basename(os.path.dirname(path_a)), "project_analysis")
            status_a = pa.collect_project_analysis_status(base_dir=base_a)
            status_b = pa.collect_project_analysis_status(base_dir=base_b)
            self.assertEqual(status_a["store_dir"], path_a)
            self.assertEqual(status_b["store_dir"], path_b)
        finally:
            runtime.CONTEXT_STORAGE_LOCATION = previous_location

    def test_project_analysis_status_does_not_persist_game_root(self):
        """Readonly status must not rewrite translator_config.json."""
        import gemini_translate_batch as batch_mod
        import translator_runtime as runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "Game_Example"
            work = project / "work"
            original_game = project / "original" / "game"
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)
            config_path = root / "translator_config.json"
            # Point at project root; loader would normally normalize to work/ and persist.
            config_path.write_text(
                json.dumps({"game_root": str(project)}),
                encoding="utf-8",
            )
            before = config_path.read_text(encoding="utf-8")

            with mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(config_path)):
                with mock.patch("builtins.print"):
                    batch_mod.main(["project-analysis-status"])

            after = config_path.read_text(encoding="utf-8")
            self.assertEqual(after, before)
            saved = json.loads(after)
            self.assertEqual(
                runtime.canonical_abs_path(saved["game_root"]),
                runtime.canonical_abs_path(str(project)),
            )

    def test_json_status_stdout_is_pure_json_when_path_warns(self):
        """P2: --json must not mix path-resolution warnings into stdout."""
        import io
        from contextlib import redirect_stdout

        import gemini_translate_batch as batch_mod

        def noisy_collect(*_args, **_kwargs):
            print(
                "Warning: context_storage.location is 'game' but game_root is unset; "
                "using tool logs root with an 'unset' project slug."
            )
            return {
                "overall_status": pa.STATUS_MISSING,
                "store_dir": "C:/tmp/project_analysis/unset",
                "store_exists": False,
                "schema_version": pa.SCHEMA_VERSION,
                "injectable": False,
                "chunk_count": 0,
                "label_count": 0,
                "route_count": 0,
                "brief_status": pa.STATUS_MISSING,
                "error": "",
            }

        buf = io.StringIO()
        with mock.patch(
            "project_analysis.collect_project_analysis_status",
            side_effect=noisy_collect,
        ), redirect_stdout(buf):
            batch_mod.main(["project-analysis-status", "--json", "--store-dir", "C:/tmp"])
        raw = buf.getvalue()
        self.assertNotIn("Warning:", raw, msg=f"stdout mixed warning: {raw!r}")
        payload = json.loads(raw)
        self.assertEqual(payload["overall_status"], pa.STATUS_MISSING)
        self.assertIn("store_dir", payload)


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
