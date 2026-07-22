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

    def test_accept_applies_workspace_then_skip_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            dialog._apply_workspace()
            self.assertEqual(dialog._stack.currentIndex(), 1)
            self.assertTrue((workspace / registry.REGISTRY_FILENAME).is_file())
            dialog._sdk_skip.setChecked(True)
            dialog._finish_sdk()
            self.assertEqual(dialog.result(), int(QDialog.DialogCode.Accepted))
            payload = dialog.result_payload()
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload.workspace.resolve(), workspace.resolve())
            self.assertTrue(payload.created_registry)
            self.assertIsNone(payload.sdk_dir)
            self.assertIn("跳过", payload.sdk_message)
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

    def test_sdk_page_can_persist_browsed_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            sdk_dir = Path(tmp) / "renpy-fake-sdk"
            sdk_dir.mkdir()
            (sdk_dir / "renpy.py").write_text("# renpy\n", encoding="utf-8")
            config = Path(tmp) / "translator_config.json"

            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            dialog._apply_workspace()
            dialog._browse_path_edit.setText(str(sdk_dir))
            dialog._sdk_browse.setChecked(True)
            with mock.patch(
                "gui_qt.workspace_setup_dialog.save_renpy_sdk_dir",
                side_effect=lambda path, config_path=None: Path(path),
            ) as save_mock:
                dialog._finish_sdk()
            save_mock.assert_called()
            payload = dialog.result_payload()
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload.sdk_dir.resolve(), sdk_dir.resolve())


if __name__ == "__main__":
    unittest.main()
