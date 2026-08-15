"""End-to-end P4 workflow tests through the CLI runner layer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
from tests.test_engine_adapter_p4 import TestEngineAdapterP4


class TranslationReuseWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TestEngineAdapterP4(
            "test_records_export_load_roundtrip_and_tamper_rejection"
        )
        self.fixture.setUp()

    def _write_manifest(self, package_dir, *, version, items, with_results):
        package = Path(package_dir)
        package.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 2,
            "mode": "translation",
            "created_at": "2026-08-15T00:00:00",
            "base_dir": str(package.parent),
            "tl_dir": str(package.parent / "tl"),
            "chunks": [
                {
                    "key": "chunk-1",
                    "chunk_index": 0,
                    "file_rel_path": "chapter.rpy",
                    "items": items,
                }
            ],
            "summary": {"chunk_count": 1, "item_count": len(items)},
        }
        if with_results:
            rows = [
                {
                    "key": "chunk-1",
                    "normalized_response": {
                        "items": [
                            {"id": item["id"], "translation": f"译-{item['id']}"}
                            for item in items
                        ]
                    },
                }
            ]
            results_path = package / "results.jsonl"
            results_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
        manifest_path = package / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    def test_full_reuse_workflow_feeds_check_apply_channel(self):
        base = self.fixture._snapshot(
            "1.0",
            [
                {
                    "key": "stable",
                    "source": "Stable line",
                    "locator": "stable-locator",
                    "content": "stable-content",
                },
                {
                    "key": "moved",
                    "source": "Moved line",
                    "file": "old.rpy",
                    "line": 2,
                    "content": "old-move-context",
                },
            ],
        )
        target = self.fixture._snapshot(
            "2.0",
            [
                {
                    "key": "stable",
                    "source": "Stable line",
                    "locator": "stable-locator",
                    "content": "stable-content",
                },
                {
                    "key": "moved",
                    "source": "Moved line",
                    "file": "new.rpy",
                    "line": 30,
                    "content": "new-move-context",
                },
            ],
        )
        base_manifest_items = [
            {"id": "unit-1.0-stable", "source": "Stable line"},
            {"id": "unit-1.0-moved", "source": "Moved line"},
        ]
        target_manifest_items = [
            {"id": "unit-2.0-stable", "source": "Stable line"},
            {"id": "unit-2.0-moved", "source": "Moved line"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_snapshot_dir = tmp_path / "base-snapshot"
            target_snapshot_dir = tmp_path / "target-snapshot"
            base_snapshot_paths = self.fixture.versioning.export_project_snapshot(
                base,
                base_snapshot_dir,
            )
            target_snapshot_paths = self.fixture.versioning.export_project_snapshot(
                target,
                target_snapshot_dir,
            )
            base_manifest_path = self._write_manifest(
                tmp_path / "base-package",
                version="1.0",
                items=base_manifest_items,
                with_results=True,
            )

            records_result = batch.run_translation_records_export(
                base_snapshot_paths.snapshot_path,
                str(base_manifest_path),
                output_dir=str(tmp_path / "records"),
            )
            self.assertEqual(records_result["record_count"], 2)
            records_manifest = records_result["paths"]["manifest"]

            reconciliation_dir = tmp_path / "reconciliation"
            report = self.fixture.versioning.reconcile_project_snapshots(
                base,
                target,
            )
            reconciliation_paths = (
                self.fixture.versioning.export_reconciliation_report(
                    report,
                    reconciliation_dir,
                )
            )

            candidates_result = batch.run_reuse_candidates_build(
                base_snapshot_paths.snapshot_path,
                target_snapshot_paths.snapshot_path,
                reconciliation_paths.report_path,
                records_manifest,
                output_dir=str(tmp_path / "reuse"),
            )
            self.assertEqual(candidates_result["candidate_count"], 2)
            reuse_report = candidates_result["paths"]["report"]

            template_rows = [
                json.loads(line)
                for line in Path(
                    candidates_result["paths"]["decisions_template"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            decisions_path = tmp_path / "decisions.jsonl"
            decisions_path.write_text(
                "".join(
                    json.dumps(
                        {
                            **row,
                            "action": "accept",
                            "reviewer": {"type": "human", "name": "reviewer"},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for row in template_rows
                ),
                encoding="utf-8",
            )
            decisions_result = batch.run_reuse_decisions_import(
                reuse_report,
                str(decisions_path),
                output_dir=str(tmp_path / "reuse-decided"),
            )
            self.assertEqual(decisions_result["summary"]["status_accepted"], 2)
            decided_report = decisions_result["paths"]["report"]

            target_manifest_path = self._write_manifest(
                tmp_path / "target-package",
                version="2.0",
                items=target_manifest_items,
                with_results=False,
            )
            with mock.patch.object(batch, "save_manifest") as save_manifest:
                export_result = batch.run_reuse_results_export(
                    decided_report,
                    str(target_manifest_path),
                )
            self.assertEqual(export_result["reused_items"], 2)
            self.assertEqual(export_result["parent_items_kept"], 0)
            merged_rows = [
                json.loads(line)
                for line in Path(
                    export_result["result_jsonl_path"]
                ).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(merged_rows), 1)
            items = merged_rows[0]["normalized_response"]["translations"]
            self.assertEqual(
                {item["id"] for item in items},
                {"unit-2.0-stable", "unit-2.0-moved"},
            )
            translations = {item["id"]: item["translation"] for item in items}
            self.assertEqual(translations["unit-2.0-stable"], "译-unit-1.0-stable")
            self.assertEqual(translations["unit-2.0-moved"], "译-unit-1.0-moved")
            updated_manifest = save_manifest.call_args[0][0]
            self.assertEqual(updated_manifest["job_state"], "RESULTS_MERGED")
            self.assertEqual(len(updated_manifest["reuse_export_history"]), 1)
            for key in ("last_check_at", "last_check_summary"):
                self.assertNotIn(key, updated_manifest)


if __name__ == "__main__":
    unittest.main()
