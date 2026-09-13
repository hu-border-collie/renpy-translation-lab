"""Doctor engine-status checks for adapter/schema/snapshot/writeback (#424 P6 S3b)."""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
import translation_plan
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


def _project_context(tmp, *, include_project_file=True):
    root = Path(tmp) / "Game" / "work"
    tl_dir = root / "game" / "tl" / "schinese"
    tl_dir.mkdir(parents=True)
    if include_project_file:
        (tl_dir / "script.rpy").write_text(
            'translate schinese start:\n'
            '    # "Hello there."\n'
            '    "Hello there."\n',
            encoding="utf-8",
        )
    return root, tl_dir


class DoctorEngineStatusTests(unittest.TestCase):
    maxDiff = None

    def _build_snapshot(self, root, tl_dir):
        return build_translation_snapshot(
            RenPyAdapter(legacy_module=batch.legacy),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
            include_occurrences=False,
            include_task_payloads=False,
        )

    @contextlib.contextmanager
    def _doctor_paths(self, tmp, root, tl_dir):
        jobs_dir = Path(tmp) / "logs" / "batch_jobs"
        snapshots_dir = Path(tmp) / "logs" / "project_snapshots"
        with (
            mock.patch.object(batch.legacy, "BASE_DIR", str(root)),
            mock.patch.object(batch.legacy, "TL_DIR", str(tl_dir)),
            mock.patch.object(batch.legacy, "INCLUDE_FILES", set()),
            mock.patch.object(batch.legacy, "INCLUDE_PREFIXES", set()),
            mock.patch.object(batch, "BATCH_JOBS_DIR", str(jobs_dir)),
            mock.patch.object(
                batch,
                "LATEST_MANIFEST_FILE",
                str(jobs_dir / "latest_manifest.txt"),
            ),
            mock.patch.object(batch, "PROJECT_SNAPSHOTS_DIR", str(snapshots_dir)),
        ):
            yield jobs_dir, snapshots_dir

    @staticmethod
    def _issue_codes(status):
        return [str(issue.get("code") or "") for issue in status.get("issues") or []]

    @staticmethod
    def _write_latest_manifest(jobs_dir, manifest):
        package_dir = jobs_dir / "pkg-1"
        package_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = package_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / "latest_manifest.txt").write_text(
            str(manifest_path),
            encoding="utf-8",
        )
        return manifest_path

    def test_capabilities_and_live_scan_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, snapshots_dir):
                snapshot = self._build_snapshot(root, tl_dir)
                status = batch.collect_doctor_engine_status(snapshot)

            self.assertEqual(status["status"], "ok")
            self.assertEqual(status["engine"], "renpy")
            capabilities = status["capabilities"]["capabilities"]
            self.assertIn("text_span_replace", capabilities["declarative_writeback"])
            self.assertEqual(capabilities["selected_localization_mode"], "hybrid")
            self.assertEqual(status["live"]["locator_schema_versions"], {"1": 1})
            self.assertEqual(status["live"]["candidate_schema_versions"], {"1": 1})
            self.assertTrue(status["capabilities"]["behavior_digest"])
            self.assertFalse(jobs_dir.exists())
            # Catalog provenance is explicitly reported as an inferred boundary.
            self.assertIn("engine.catalog.freshness_unknown", self._issue_codes(status))
            self.assertIn("engine.catalog.provenance_inferred", self._issue_codes(status))
            self.assertEqual(status["issue_counts"]["error"], 0)

    def test_missing_native_catalog_is_writeback_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            with self._doctor_paths(tmp, root, tl_dir):
                status = batch.collect_doctor_engine_status(
                    None,
                    has_tl_files=False,
                )

            self.assertEqual(status["status"], "attention")
            self.assertIn("engine.writeback.catalog_missing", self._issue_codes(status))

    def test_existing_tl_catalog_is_not_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir):
                snapshot = self._build_snapshot(root, tl_dir)
                status = batch.collect_doctor_engine_status(
                    snapshot,
                    has_tl_files=True,
                )

            self.assertNotIn("engine.writeback.catalog_missing", self._issue_codes(status))

    def test_locator_schema_mismatch_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir):
                snapshot = self._build_snapshot(root, tl_dir)
                with mock.patch.object(RenPyAdapter, "locator_schema_version", 99):
                    status = batch.collect_doctor_engine_status(snapshot)

            self.assertEqual(status["status"], "blocked")
            codes = self._issue_codes(status)
            self.assertIn("engine.locator.schema_mismatch", codes)
            issue = next(
                item
                for item in status["issues"]
                if item["code"] == "engine.locator.schema_mismatch"
            )
            self.assertEqual(issue["details"]["expected"], 99)

    def test_snapshot_schema_mismatch_is_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            with self._doctor_paths(tmp, root, tl_dir) as (_jobs_dir, snapshots_dir):
                package_dir = snapshots_dir / "v1"
                package_dir.mkdir(parents=True)
                (package_dir / "project_snapshot.json").write_text(
                    json.dumps(
                        {
                            "kind": "project_snapshot",
                            "project_snapshot_schema_version": 999,
                            "project_snapshot_digest_schema_version": 1,
                            "engine": "renpy",
                            "adapter_version": "1.1.8",
                            "coverage": {"coverage_schema_version": 1},
                        }
                    ),
                    encoding="utf-8",
                )
                status = batch.collect_doctor_engine_status(None)

            self.assertEqual(status["status"], "attention")
            codes = self._issue_codes(status)
            self.assertIn("engine.snapshot.schema_unsupported", codes)
            self.assertEqual(
                status["snapshot"]["latest"]["project_snapshot_schema_version"], 999
            )

    def test_missing_snapshot_is_not_an_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            with self._doctor_paths(tmp, root, tl_dir):
                status = batch.collect_doctor_engine_status(None)

            self.assertFalse(status["snapshot"]["checked"])
            self.assertNotIn("engine.snapshot.invalid", self._issue_codes(status))

    def test_invalid_snapshot_manifest_is_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            with self._doctor_paths(tmp, root, tl_dir) as (_jobs_dir, snapshots_dir):
                package_dir = snapshots_dir / "v1"
                package_dir.mkdir(parents=True)
                (package_dir / "project_snapshot.json").write_text(
                    "{not json", encoding="utf-8"
                )
                status = batch.collect_doctor_engine_status(None)

            self.assertEqual(status["status"], "attention")
            self.assertIn("engine.snapshot.invalid", self._issue_codes(status))

    def test_malformed_snapshot_schema_is_stable_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            with self._doctor_paths(tmp, root, tl_dir) as (_jobs_dir, snapshots_dir):
                package_dir = snapshots_dir / "v1"
                package_dir.mkdir(parents=True)
                (package_dir / "project_snapshot.json").write_text(
                    json.dumps(
                        {
                            "kind": "project_snapshot",
                            "project_snapshot_schema_version": "many",
                            "project_snapshot_digest_schema_version": 1,
                            "engine": "renpy",
                            "coverage": {"coverage_schema_version": 1},
                        }
                    ),
                    encoding="utf-8",
                )
                status = batch.collect_doctor_engine_status(None)

            codes = self._issue_codes(status)
            self.assertIn("engine.snapshot.schema_unsupported", codes)
            self.assertNotIn("engine.snapshot.check_failed", codes)

    def test_malformed_manifest_version_is_stable_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, _snapshots):
                self._write_latest_manifest(jobs_dir, {"manifest_version": "two"})
                status = batch.collect_doctor_engine_status(None)

            self.assertEqual(status["writeback"]["status"], "attention")
            codes = self._issue_codes(status)
            self.assertIn("engine.writeback.manifest_invalid", codes)
            self.assertNotIn("engine.writeback.check_failed", codes)

    def test_legacy_manifest_is_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, _snapshots):
                self._write_latest_manifest(jobs_dir, {"manifest_version": 1})
                status = batch.collect_doctor_engine_status(None)

            self.assertEqual(status["writeback"]["status"], "attention")
            self.assertIn("engine.writeback.legacy_manifest", self._issue_codes(status))

    def test_valid_plan_passes_writeback_preconditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, _snapshots):
                snapshot = self._build_snapshot(root, tl_dir)
                identity = batch._batch_plan_source_identity(
                    batch.TranslationFileJobs([], adapter_snapshot=snapshot)
                ).to_dict()
                plan = translation_plan.refresh_plan_fingerprint(
                    {
                        "schema_version": translation_plan.PLAN_SCHEMA_VERSION,
                        "execution_strategy": translation_plan.STRATEGY_GEMINI_BATCH,
                        "source_identity": identity,
                    }
                )
                self._write_latest_manifest(
                    jobs_dir,
                    {"manifest_version": 2, "translation_plan": plan},
                )
                status = batch.collect_doctor_engine_status(snapshot)

            self.assertEqual(status["writeback"]["status"], "ok")
            self.assertTrue(status["writeback"]["source_identity_checked"])
            self.assertNotIn("engine.writeback.source_stale", self._issue_codes(status))
            self.assertEqual(status["status"], "ok")

    def test_plan_fingerprint_mismatch_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, _snapshots):
                snapshot = self._build_snapshot(root, tl_dir)
                identity = batch._batch_plan_source_identity(
                    batch.TranslationFileJobs([], adapter_snapshot=snapshot)
                ).to_dict()
                plan = {
                    "schema_version": translation_plan.PLAN_SCHEMA_VERSION,
                    "execution_strategy": translation_plan.STRATEGY_GEMINI_BATCH,
                    "source_identity": identity,
                }
                self._write_latest_manifest(
                    jobs_dir,
                    {"manifest_version": 2, "translation_plan": plan},
                )
                status = batch.collect_doctor_engine_status(snapshot)

            self.assertEqual(status["status"], "blocked")
            self.assertEqual(status["writeback"]["status"], "blocked")
            self.assertIn(
                "engine.writeback.plan_fingerprint_mismatch",
                self._issue_codes(status),
            )
            envelope = batch.build_machine_success_envelope(
                "doctor",
                {"engine_status": status, "warnings": []},
                mock.Mock(),
            )
            self.assertEqual(envelope["status"], "blocked")
            self.assertEqual(cli_contract.strict_exit_code(envelope), cli_contract.EXIT_BLOCKED)

    def test_source_change_after_build_is_stale_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir) as (jobs_dir, _snapshots):
                snapshot = self._build_snapshot(root, tl_dir)
                identity = batch._batch_plan_source_identity(
                    batch.TranslationFileJobs([], adapter_snapshot=snapshot)
                ).to_dict()
                plan = translation_plan.refresh_plan_fingerprint(
                    {
                        "schema_version": translation_plan.PLAN_SCHEMA_VERSION,
                        "execution_strategy": translation_plan.STRATEGY_GEMINI_BATCH,
                        "source_identity": identity,
                    }
                )
                self._write_latest_manifest(
                    jobs_dir,
                    {"manifest_version": 2, "translation_plan": plan},
                )
                (tl_dir / "script.rpy").write_text(
                    'translate schinese start:\n'
                    '    # "Hello there, friend."\n'
                    '    "Hello there, friend."\n',
                    encoding="utf-8",
                )
                changed_snapshot = self._build_snapshot(root, tl_dir)
                status = batch.collect_doctor_engine_status(changed_snapshot)

            self.assertEqual(status["status"], "attention")
            codes = self._issue_codes(status)
            self.assertIn("engine.writeback.source_stale", codes)
            issue = next(
                item
                for item in status["issues"]
                if item["code"] == "engine.writeback.source_stale"
            )
            self.assertIn("source_file_digests_changed", issue["details"]["reasons"])

    def test_translation_progress_helper_reuses_adapter_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp)
            with self._doctor_paths(tmp, root, tl_dir):
                progress, snapshot = batch._collect_doctor_translation_progress()

            self.assertIsNotNone(snapshot)
            self.assertEqual(progress["pending_task_count"], 1)
            self.assertTrue(snapshot.inventory.candidates)

    def test_blocked_engine_status_suppresses_workflow_state(self):
        report = {
            "layout_status": "ready",
            "mode": "existing_tl_only",
            "counts": {"rpy_files": 1},
            "pending_task_count": 2,
            "translated_task_count": 0,
            "total_task_count": 2,
            "engine_status": {"status": "blocked", "issues": []},
        }
        batch.finalize_doctor_actionable_signals(report)
        self.assertEqual(report["workflow_state"], "")

    def test_collect_doctor_report_includes_engine_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tl_dir = _project_context(tmp, include_project_file=False)
            sentinel = {
                "status": "ok",
                "engine": "renpy",
                "issues": [],
                "issue_counts": {"error": 0, "warning": 0, "info": 0},
            }
            config_path = Path(tmp) / "translator_config.json"
            config_path.write_text(
                json.dumps({"game_root": str(root)}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(batch.legacy, "BASE_DIR", str(root)),
                mock.patch.object(batch.legacy, "TL_DIR", str(tl_dir)),
                mock.patch.object(batch.legacy, "TRANSLATOR_CONFIG", str(config_path)),
                mock.patch.object(
                    batch,
                    "collect_tl_doctor_counts",
                    return_value={
                        "rpy_files": 0,
                        "translate_blocks": 0,
                        "string_sections": 0,
                        "old_lines": 0,
                        "new_lines": 0,
                        "commented_original_lines": 0,
                    },
                ),
                mock.patch.object(
                    batch.legacy,
                    "_guess_source_game_dir",
                    return_value="",
                ),
                mock.patch.object(
                    batch.legacy,
                    "get_prepare_template_command_info",
                    return_value={"available": False, "kind": "", "reason": ""},
                ),
                mock.patch.object(batch.legacy, "resolve_original_game_dir", return_value=""),
                mock.patch.object(
                    batch.legacy,
                    "work_dir_bootstrap_allowed",
                    return_value=(False, str(root), ""),
                ),
                mock.patch.object(
                    batch,
                    "collect_doctor_context_status",
                    return_value={"rag": {"enabled": False}, "source_index": {"enabled": False}},
                ),
                mock.patch.object(batch, "collect_pending_file_jobs", return_value=[]),
                mock.patch.object(
                    batch,
                    "collect_doctor_engine_status",
                    return_value=sentinel,
                ),
                mock.patch.object(batch.legacy, "is_work_dir_empty", return_value=False),
                mock.patch("os.path.isdir", return_value=False),
            ):
                report = batch.collect_doctor_report()

            self.assertIs(report["engine_status"], sentinel)


if __name__ == "__main__":
    unittest.main()
