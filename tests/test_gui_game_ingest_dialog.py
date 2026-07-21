"""Tests for game ingest naming dialog preview."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.game_ingest_dialog import GameIngestDialog
except ImportError as exc:
    GameIngestDialog = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(GameIngestDialog is None, IMPORT_ERROR)
class GuiGameIngestDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_name_change_updates_folder_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            real_src = workspace / "src"
            real_src.mkdir()
            (workspace / "Game_Taken").mkdir()

            dialog = GameIngestDialog(None, workspace_root=workspace)
            dialog._set_source(real_src)
            dialog._name_edit.setText("Glory Hounds")
            self._app.processEvents()
            self.assertEqual(dialog._folder_preview.text(), "Game_GloryHounds")
            # 将创建到 = workspace destination (native path), not the source path.
            expected_dest = str(workspace.resolve() / "Game_GloryHounds")
            self.assertEqual(dialog._path_preview.text(), expected_dest)
            self.assertTrue(
                dialog._path_preview.text().endswith("Game_GloryHounds")
            )
            # Must not clip to bare drive letter only (the screenshot bug: "C:").
            self.assertNotEqual(dialog._path_preview.text().rstrip("/\\"), "C:")
            self.assertGreater(len(dialog._path_preview.text()), 3)
            self.assertNotEqual(
                dialog._path_preview.text(),
                str(real_src.resolve()),
            )
            self.assertEqual(dialog._source_edit.text(), str(real_src.resolve()))
            self.assertEqual(dialog._error_label.text(), "")
            self.assertTrue(dialog._ok_button.isEnabled())

            dialog._name_edit.setText("Taken")
            self._app.processEvents()
            self.assertEqual(dialog._folder_preview.text(), "Game_Taken")
            self.assertTrue(dialog._error_label.text())
            self.assertFalse(dialog._ok_button.isEnabled())

            dialog._name_edit.setText("Fresh Name")
            self._app.processEvents()
            self.assertEqual(dialog._folder_preview.text(), "Game_FreshName")
            self.assertEqual(
                dialog._path_preview.text(),
                str(workspace.resolve() / "Game_FreshName"),
            )
            self.assertEqual(dialog._error_label.text(), "")
            self.assertTrue(dialog._ok_button.isEnabled())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
