"""Diagnostics refresh avoids redundant I/O while observing preview changes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.support import gui_test_support

try:
    from gui_qt.app import MainWindow
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiDiagnosticsRefreshTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "preview.json"
        self.path.write_text('{"report_path":"one.diff"}', encoding="utf-8")
        self.window = MainWindow.__new__(MainWindow)
        self.window._current_work_mode = lambda: WorkMode.SYNC_TRANSLATION
        self.window.kill_btn = SimpleNamespace(isEnabled=lambda: False)
        self.window.state = SimpleNamespace(
            get_game_root=lambda: None,
            get_sync_script_path=lambda: Path("sync.py"),
            get_batch_script_path=lambda: Path("batch.py"),
        )
        self.window.sync_translation_page = SimpleNamespace(
            preview_manifest_path=lambda: str(self.path),
        )
        self.window._context_library_config_snapshot = {}
        self.window._diagnostics_refresh_input_key = None
        for name in (
            "_set_diagnostics_context", "_update_probe_btn_enabled",
            "_update_compare_variants_btn_enabled", "_update_keyword_merge_btn_enabled",
            "_update_split_btn_enabled",
        ):
            setattr(self.window, name, mock.Mock())
        builder = mock.patch("gui_qt.app.sync_diagnostics_context")
        self.builder = builder.start()
        self.addCleanup(builder.stop)

    def test_unchanged_preview_is_not_read_again_until_forced(self):
        original = Path.read_text
        with mock.patch.object(Path, "read_text", autospec=True, side_effect=original) as read:
            self.window._refresh_diagnostics_context()
            self.window._refresh_diagnostics_context()
            self.window._refresh_diagnostics_context()
            self.assertEqual(read.call_count, 1)
            self.assertEqual(self.builder.call_count, 1)
            self.window._refresh_diagnostics_context(force=True)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(self.builder.call_count, 2)

    def test_changed_size_refreshes_even_when_mtime_is_preserved(self):
        self.window._refresh_diagnostics_context()
        original = self.path.stat()
        payload = {"report_path": "another-report.diff"}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.window._refresh_diagnostics_context()
        self.assertEqual(self.builder.call_count, 2)
        self.assertEqual(self.builder.call_args.kwargs["preview_manifest"], payload)

    def test_atomic_replacement_refreshes_with_same_size_and_mtime(self):
        self.window._refresh_diagnostics_context()
        original = self.path.stat()
        replacement = self.path.with_suffix(".replacement")
        replacement.write_text('{"report_path":"two.diff"}', encoding="utf-8")
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        replacement.replace(self.path)
        self.window._refresh_diagnostics_context()
        self.assertEqual(self.builder.call_count, 2)
        self.assertEqual(
            self.builder.call_args.kwargs["preview_manifest"], {"report_path": "two.diff"},
        )

    def test_invalid_and_removed_preview_clear_previous_metadata(self):
        self.window._refresh_diagnostics_context()
        self.path.write_text("{invalid", encoding="utf-8")
        self.window._refresh_diagnostics_context()
        self.assertIsNone(self.builder.call_args.kwargs["preview_manifest"])
        self.path.unlink()
        self.window._refresh_diagnostics_context()
        self.assertEqual(self.builder.call_count, 3)
        self.assertIsNone(self.builder.call_args.kwargs["preview_manifest"])

    def test_known_task_does_not_look_up_unrelated_latest_manifest(self):
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._load_diagnostics_manifest = mock.Mock(return_value={"loaded": True})
        self.window.state.get_game_root = mock.Mock(side_effect=AssertionError("unneeded lookup"))
        self.window._workflow = SimpleNamespace(manifest_path="active.json")
        self.window._writeback_manifest_path = "remembered.json"
        self.assertEqual(
            self.window._current_diagnostics_manifest(), ("active.json", {"loaded": True}),
        )
        self.window._workflow = None
        self.assertEqual(
            self.window._current_diagnostics_manifest(), ("remembered.json", {"loaded": True}),
        )
        self.window.state.get_game_root.assert_not_called()

    def test_unbound_task_still_looks_up_latest_for_current_project(self):
        self.window._current_work_mode = lambda: WorkMode.BATCH_TRANSLATION
        self.window._workflow = None
        self.window._writeback_manifest_path = ""
        self.window.state.get_game_root = lambda: Path("project")
        self.window.state.get_latest_manifest_path_for_mode = mock.Mock(
            return_value=Path("latest.json"),
        )
        self.window._load_diagnostics_manifest = mock.Mock(return_value={"loaded": True})
        self.assertEqual(
            self.window._current_diagnostics_manifest(), ("latest.json", {"loaded": True}),
        )
        self.window.state.get_latest_manifest_path_for_mode.assert_called_once_with(
            Path("project"), WorkMode.BATCH_TRANSLATION,
        )
