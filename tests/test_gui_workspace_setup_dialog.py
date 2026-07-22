"""Tests for workspace create/attach dialog."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry

try:
    from PySide6.QtWidgets import QApplication, QDialog

    from gui_qt.workspace_setup_dialog import WorkspaceSetupDialog
except ImportError as exc:
    WorkspaceSetupDialog = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    WorkspaceSetupDialog is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class GuiWorkspaceSetupDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_empty_dir_plan_enables_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            self.assertTrue(dialog._ok_button.isEnabled())
            self.assertEqual(dialog._ok_button.text(), "创建工作区")
            self.assertIn("空目录", dialog._scene_label.text())

    def test_corrupt_registry_disables_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / registry.REGISTRY_FILENAME).write_text("{bad", encoding="utf-8")
            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            self.assertFalse(dialog._ok_button.isEnabled())
            self.assertIn("损坏", dialog._error_label.text() + dialog._scene_label.text())

    def test_accept_applies_and_returns_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            dialog._on_accept()
            self.assertEqual(dialog.result(), int(QDialog.DialogCode.Accepted))
            payload = dialog.result_payload()
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload.workspace.resolve(), workspace.resolve())
            self.assertTrue(payload.created_registry)
            self.assertTrue((workspace / registry.REGISTRY_FILENAME).is_file())
            data = json.loads(
                (workspace / registry.REGISTRY_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(data["projects"], [])

    def test_browse_sets_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            dialog = WorkspaceSetupDialog(None)
            with mock.patch(
                "gui_qt.workspace_setup_dialog.QFileDialog.getExistingDirectory",
                return_value=str(workspace),
            ):
                dialog._browse()
            self.assertIsNotNone(dialog._selected)
            assert dialog._selected is not None
            self.assertEqual(dialog._selected.resolve(), workspace.resolve())
            self.assertTrue(dialog._ok_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
