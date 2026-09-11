"""Tests for GUI export/apply-export destination dialog."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.export_writeback_dialog import (
        APPLY_EXPORT_MODE,
        EXPORT_ONLY_MODE,
        ExportPreviewFacts,
        ExportWritebackDialog,
    )
except ImportError as exc:
    ExportWritebackDialog = None  # type: ignore[assignment,misc]
    APPLY_EXPORT_MODE = "apply-export"
    EXPORT_ONLY_MODE = "export-only"
    ExportPreviewFacts = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(ExportWritebackDialog is None, IMPORT_ERROR)
class GuiExportWritebackDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _dialog(self, root: Path) -> ExportWritebackDialog:
        game_root = root / "project"
        package_dir = root / "package"
        game_root.mkdir()
        package_dir.mkdir()
        facts = ExportPreviewFacts(
            game_root=str(game_root),
            tl_dir=str(game_root / "game" / "tl" / "schinese"),
            package_dir=str(package_dir),
            pending_files=2,
            pending_lines=6,
            gate_label="可以写回翻译",
            files=(("chapter01/dialogue.rpy", 3), ("chapter02/strings.rpy", 3)),
        )
        return ExportWritebackDialog(None, facts=facts, start_dir=root)

    def test_path_validation_and_mode_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialog = self._dialog(root)
            self._app.processEvents()
            self.assertEqual(dialog.selected_mode(), EXPORT_ONLY_MODE)
            self.assertFalse(dialog._ok_button.isEnabled())
            self.assertEqual(dialog._tree.topLevelItemCount(), 2)

            export_root = root / "exports"
            dialog._path_edit.setText(str(export_root))
            self._app.processEvents()
            self.assertTrue(dialog._ok_button.isEnabled())
            self.assertEqual(dialog._target_root_label.text(), str(export_root.resolve()))

            dialog._mode_combo.setCurrentIndex(1)
            self._app.processEvents()
            self.assertEqual(dialog.selected_mode(), APPLY_EXPORT_MODE)
            dialog._accept()
            self._app.processEvents()
            choice = dialog.choice()
            self.assertIsNotNone(choice)
            assert choice is not None
            self.assertEqual(choice.mode, APPLY_EXPORT_MODE)
            self.assertEqual(choice.export_root, str(export_root.resolve()))
            dialog.close()

    def test_non_empty_destination_without_receipt_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialog = self._dialog(root)
            export_root = root / "exports"
            export_root.mkdir()
            (export_root / "unrelated.txt").write_bytes(b"keep")
            dialog._path_edit.setText(str(export_root))
            self._app.processEvents()
            self.assertFalse(dialog._ok_button.isEnabled())
            self.assertIn("非空", dialog._notice_label.text())
            dialog.close()

    def test_overlap_with_game_root_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dialog = self._dialog(root)
            dialog._path_edit.setText(str(root / "project"))
            self._app.processEvents()
            self.assertFalse(dialog._ok_button.isEnabled())
            self.assertIn("overlap", dialog._notice_label.text().lower())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
