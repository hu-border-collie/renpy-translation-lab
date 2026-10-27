"""MV preparation gathers material choices without adopting a scope or review."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.support import gui_test_support
from tests.support.mv_project import make_mv_project

try:
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_qt.dialogs.mv_project_setup_dialog import MvProjectSetupDialog
except ImportError as exc:
    MvProjectSetupDialog, IMPORT_ERROR = None, exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(MvProjectSetupDialog is None, IMPORT_ERROR)
class MvProjectSetupDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_save_keeps_full_game_scope_and_only_returns_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            file, config, _ = make_mv_project(Path(temporary))
            before = file.read_bytes()
            dialog = MvProjectSetupDialog(
                config=config, config_file=file, target_root=config["game_root"]
            )
            self.addCleanup(dialog.deleteLater)
            self.assertTrue(dialog.target_edit.isReadOnly())
            self.assertEqual(dialog.setup["scope"], "full_project")
            dialog._on_save()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
            result = dialog.result_config()
            self.assertEqual(result.get("include_files"), config.get("include_files"))
            self.assertEqual(
                result["rpgmaker_mv"].get("occurrence_scope_file"),
                config["rpgmaker_mv"].get("occurrence_scope_file"),
            )
            self.assertFalse(result["prepare"]["enabled"])
            self.assertEqual(file.read_bytes(), before)

    def test_missing_window_and_same_source_cannot_finish_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            file, config, _ = make_mv_project(Path(temporary))
            dialog = MvProjectSetupDialog(
                config=config, config_file=file, target_root=config["game_root"]
            )
            self.addCleanup(dialog.deleteLater)
            dialog.window_edit.setText("missing-window.json")
            dialog._on_save()
            self.assertIsNone(dialog.result_config())
            self.assertTrue(dialog.error_label.text())
            dialog.window_edit.setText(config["rpgmaker_mv"]["support_window_file"])
            dialog.source_edit.setText(config["game_root"])
            dialog._on_save()
            self.assertIsNone(dialog.result_config())
            self.assertIn("independent", dialog.error_label.text())
            self.assertEqual(json.loads(file.read_text(encoding="utf-8")), config)
