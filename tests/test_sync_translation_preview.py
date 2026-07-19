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

    def test_runtime_default_generates_preview_without_source_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.mkdir(parents=True)
            target = tl_dir / "script.rpy"
            source = '    "Hello"\n'
            target.write_text(source, encoding="utf-8")

            task = {
                "line": 0,
                "start": 4,
                "end": 11,
                "text": "Hello",
                "quote": '"',
                "prefix": "",
            }

            def translate_batch(batch, replacements):
                replacements.setdefault(0, []).append((4, 11, "你好", "", '"'))
                return [batch[0]["progress_entry"]]

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
                mock.patch.object(runtime, "collect_tasks", return_value=[task]),
                mock.patch.object(runtime, "process_batch_with_retry", side_effect=translate_batch),
            ):
                manifest_path = runtime.run_translation()

            self.assertEqual(target.read_text(encoding="utf-8"), source)
            manifest = preview.load_sync_preview(manifest_path)
            proposed = Path(manifest_path).parent / manifest["files"][0]["preview_path"]
            self.assertEqual(proposed.read_text(encoding="utf-8"), '    "你好"\n')

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

    def test_apply_can_resume_after_later_atomic_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir, manifest_path, _manifest = self._create_preview(root, ("a.rpy", "b.rpy"))
            real_write = preview.atomic_write_text

            def fail_second_target(path, text, **kwargs):
                if Path(path) == tl_dir / "b.rpy":
                    raise OSError("disk full")
                return real_write(path, text, **kwargs)

            with mock.patch.object(preview, "atomic_write_text", side_effect=fail_second_target):
                with self.assertRaisesRegex(OSError, "disk full"):
                    preview.apply_sync_preview(
                        manifest_path,
                        active_project_root=root,
                        active_tl_dir=tl_dir,
                    )

            self.assertEqual((tl_dir / "a.rpy").read_text(encoding="utf-8"), '    "你好 1"\n')
            self.assertEqual((tl_dir / "b.rpy").read_text(encoding="utf-8"), '    "Hello 2"\n')

            applied = preview.apply_sync_preview(
                manifest_path,
                active_project_root=root,
                active_tl_dir=tl_dir,
            )
            self.assertEqual(applied["state"], "applied")
            self.assertEqual((tl_dir / "b.rpy").read_text(encoding="utf-8"), '    "你好 2"\n')


if __name__ == "__main__":
    unittest.main()
