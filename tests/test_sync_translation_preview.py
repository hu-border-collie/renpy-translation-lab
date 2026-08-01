import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sync_translation_preview as preview
import translator_runtime as runtime
from atomic_io import file_sha256


class SyncTranslationPreviewTests(unittest.TestCase):
    def _create_preview(self, root: Path, names=("a.rpy",)):
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        rows = []
        for index, name in enumerate(names, start=1):
            target = tl_dir / name
            source = f'    "Hello {index}"\n'
            proposed = f'    "你好 {index}"\n'
            target.write_text(source, encoding="utf-8")
            rows.append(
                {
                    "relative_path": name,
                    "source_text": source,
                    "source_sha256": file_sha256(target),
                    "preview_text": proposed,
                    "progress_entries": [f"id:{index}"],
                }
            )
        manifest_path, manifest = preview.create_sync_preview(
            log_dir=root / "logs",
            project_root=root,
            tl_dir=tl_dir,
            files=rows,
        )
        return tl_dir, Path(manifest_path), manifest

    def test_create_preview_does_not_modify_project_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, manifest = self._create_preview(root)

            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "Hello 1"\n')
            self.assertEqual(manifest["state"], "preview_ready")
            self.assertEqual(manifest["summary"]["files_changed"], 1)
            self.assertTrue(manifest_path.is_file())
            report = (manifest_path.parent / "preview.diff").read_text(encoding="utf-8")
            self.assertIn('-    "Hello 1"', report)
            self.assertIn('+    "你好 1"', report)
    def test_load_accepts_legacy_preview_without_failures_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _tl_dir, manifest_path, manifest = self._create_preview(root)
            manifest.pop("failures")
            manifest["preview_fingerprint"] = preview._fingerprint(manifest)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            loaded = preview.load_sync_preview(manifest_path)

            self.assertNotIn("failures", loaded)


    def test_create_preview_records_partial_adapter_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)

            manifest_path, manifest = preview.create_sync_preview(
                log_dir=root / "logs",
                project_root=root,
                tl_dir=tl_dir,
                files=(),
                failures=(
                    {
                        "relative_path": "broken.rpy",
                        "reason_code": "common.locator.unresolved",
                        "message": "ambiguous occurrence",
                    },
                ),
            )

            self.assertTrue(Path(manifest_path).is_file())
            self.assertEqual(manifest["summary"]["failure_files"], 1)
            self.assertEqual(
                manifest["summary"]["adapter_writeback_status"],
                "partial",
            )
            self.assertEqual(
                manifest["failures"][0]["reason_code"],
                "common.locator.unresolved",
            )

            persisted = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            persisted["failures"][0]["message"] = "edited audit detail"
            Path(manifest_path).write_text(
                json.dumps(persisted, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                preview.load_sync_preview(manifest_path)

    def test_build_sync_adapter_preview_isolates_plan_failure(self):
        error = runtime.WritebackPlanError("common.locator.unresolved", "ambiguous")
        with mock.patch.object(runtime, "build_sync_adapter_writeback_plan", side_effect=error):
            plan, rendered, failure = runtime.build_sync_adapter_preview(
                object(), object(), "broken.rpy", (), {0: [(0, 1, "你好", "", '"')]}
            )
        self.assertIsNone(plan)
        self.assertIsNone(rendered)
        self.assertEqual(failure["reason_code"], "common.locator.unresolved")

    def test_sync_validation_rejects_changed_tags_and_placeholders(self):
        valid, message = runtime.validate_translation(
            "Hello [player] {i}%s{/i}",
            "你好 {i}{/i}",
        )

        self.assertFalse(valid)
        self.assertIn("placeholders/tags changed", message)

        valid, message = runtime.validate_translation(
            "Hello [player] {i}%s{/i}",
            "你好 [player] {i}%s{/i}",
        )
        self.assertTrue(valid, message)

    def test_sync_validation_ignores_literal_percent_text(self):
        samples = (
            ("Save 50% off", "五折优惠"),
            ("Get a 20%discount", "享受八折优惠"),
            ("The price is 50% lower", "价格降低一半"),
        )

        for original, translated in samples:
            with self.subTest(original=original):
                valid, message = runtime.validate_translation(original, translated)
                self.assertTrue(valid, message)

        valid, message = runtime.validate_translation(
            "Hello %s, you have %(count)d items",
            "你好 %s，你有 %(count)d 件物品",
        )
        self.assertTrue(valid, message)

    def test_runtime_default_generates_preview_without_source_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)
            target = tl_dir / "script.rpy"
            # Adapter scan drives pending tasks; keep a simple string unit.
            source = '    "Hello"\n'
            target.write_text(source, encoding="utf-8")

            def translate_batch(batch, replacements, usage_run_id="", **_kwargs):
                task = batch[0]
                replacements.setdefault(task["line"], []).append(
                    (
                        task["start"],
                        task["end"],
                        "你好",
                        task.get("prefix") or "",
                        task["quote"],
                    )
                )
                return [task.get("progress_entry") or f"id:{task['line']}"]

            with (
                mock.patch.object(runtime, "BASE_DIR", str(root)),
                mock.patch.object(runtime, "TL_DIR", str(tl_dir)),
                mock.patch.object(runtime, "LOG_DIR", str(root / "logs")),
                mock.patch.object(runtime, "SYNC_BACKEND", "litellm"),
                mock.patch.object(runtime, "PREP_ENABLED", False),
                mock.patch.object(runtime, "INCLUDE_FILES", []),
                mock.patch.object(runtime, "INCLUDE_PREFIXES", []),
                mock.patch.object(runtime, "load_config"),
                mock.patch.object(runtime, "load_translator_settings"),
                mock.patch.object(runtime, "load_glossary"),
                mock.patch.object(runtime, "load_progress", return_value={}),
                mock.patch.object(runtime, "process_batch_with_retry", side_effect=translate_batch),
            ):
                manifest_path = runtime.run_translation()

            self.assertEqual(target.read_text(encoding="utf-8"), source)
            manifest = preview.load_sync_preview(manifest_path)
            proposed = Path(manifest_path).parent / manifest["files"][0]["preview_path"]
            self.assertEqual(proposed.read_text(encoding="utf-8"), '    "你好"\n')
            coverage_dir = Path(manifest_path).parent / "coverage"
            self.assertEqual(
                {path.name for path in coverage_dir.iterdir()},
                {
                    "coverage_candidates.jsonl",
                    "coverage_report.json",
                    "coverage_review.md",
                    "coverage_review_template.json",
                },
            )

    def test_apply_revalidates_then_writes_and_marks_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, _manifest = self._create_preview(root)
            progress = []

            applied = preview.apply_sync_preview(
                manifest_path,
                active_project_root=root,
                active_tl_dir=tl_dir,
                on_file_applied=lambda entry: progress.extend(entry["progress_entries"]),
            )

            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "你好 1"\n')
            self.assertEqual(applied["state"], "applied")
            self.assertEqual(progress, ["id:1"])
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["state"], "applied")

    def test_apply_blocks_all_writes_when_any_source_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, _manifest = self._create_preview(root, ("a.rpy", "b.rpy"))
            (tl_dir / "b.rpy").write_text('    "Changed"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Source changed after sync preview: b.rpy"):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                )

            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "Hello 1"\n')

    def test_apply_rejects_different_project_and_modified_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, manifest = self._create_preview(root)
            with self.assertRaisesRegex(ValueError, "different project"):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root / "other",
                    active_tl_dir=tl_dir,
                )

            proposed = manifest_path.parent / manifest["files"][0]["preview_path"]
            proposed.write_text('    "篡改"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "proposed file changed"):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                )

    def test_apply_rolls_back_transaction_failure_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, _manifest = self._create_preview(root, ("a.rpy", "b.rpy"))
            with mock.patch.object(
                preview,
                "atomic_write_many_lines",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    preview.apply_sync_preview(
                        manifest_path,
                        active_project_root=root,
                        active_tl_dir=tl_dir,
                    )

            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "Hello 1"\n')
            self.assertEqual((tl_dir / "b.rpy").read_text(encoding="utf-8"), '    "Hello 2"\n')

            applied = preview.apply_sync_preview(
                manifest_path,
                active_project_root=root,
                active_tl_dir=tl_dir,
            )
            self.assertEqual(applied["state"], "applied")
            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "你好 1"\n')
            self.assertEqual((tl_dir / "b.rpy").read_text(encoding="utf-8"), '    "你好 2"\n')


if __name__ == "__main__":
    unittest.main()
