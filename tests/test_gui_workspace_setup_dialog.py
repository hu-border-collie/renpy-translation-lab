"""Tests for workspace create/attach dialog."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtGui import QCloseEvent
    from PySide6.QtWidgets import QApplication, QDialog

    from gui_qt.workspace_setup_dialog import (
        WorkspaceSetupDialog,
        WorkspaceSetupDialogResult,
    )
except ImportError as exc:
    WorkspaceSetupDialog = None  # type: ignore[assignment,misc]
    QObject = None  # type: ignore[assignment,misc]
    QCloseEvent = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


if WorkspaceSetupDialog is not None:

    class _FakeSdkWorker(QObject):
        """Deterministic SdkInstallWorker stand-in: no thread, no blocking wait."""

        finished = Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.cancel_requested = False
            self.interruption_requested = False
            self._running = True

        def isRunning(self):
            return self._running

        def request_cancel(self):
            self.cancel_requested = True

        def requestInterruption(self):
            self.interruption_requested = True

        def finish(self):
            """Deliver the real terminal signal a QThread would emit."""
            self._running = False
            self.finished.emit()


def _process(rounds: int = 5) -> None:
    app = QApplication.instance()
    for _ in range(rounds):
        app.processEvents()


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
            # Must not auto-scan until user clicks 查找.
            self.assertFalse(dialog._sdk_found.isEnabled())
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

    def test_skip_after_sdk_failure_preserves_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            dialog._apply_workspace()
            dialog._sdk_status_message = "SHA-256 校验失败：示例"
            dialog._sdk_skip.setChecked(True)
            dialog._finish_sdk()
            payload = dialog.result_payload()
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertIn("SHA-256", payload.sdk_message)
            self.assertIn("未配置", payload.sdk_message)

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

    def test_download_option_reuses_existing_sdk_without_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "ws"
            workspace.mkdir()
            sdk_dir = workspace / "renpy-8.5.3-sdk"
            sdk_dir.mkdir()
            (sdk_dir / "renpy.py").write_text("# renpy\n", encoding="utf-8")

            dialog = WorkspaceSetupDialog(None, initial_path=workspace)
            dialog._apply_workspace()
            dialog._download_target_edit.setText(str(sdk_dir))
            dialog._sdk_download.setChecked(True)
            with mock.patch(
                "gui_qt.workspace_setup_dialog.save_renpy_sdk_dir",
                side_effect=lambda path, config_path=None: Path(path),
            ) as save_mock:
                with mock.patch.object(dialog, "_start_download") as start_dl:
                    dialog._finish_sdk()
            start_dl.assert_not_called()
            save_mock.assert_called()
            payload = dialog.result_payload()
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload.sdk_dir.resolve(), sdk_dir.resolve())
            self.assertIn("跳过下载", payload.sdk_message)


    def test_reject_with_active_sdk_worker_defers_without_blocking(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker

        dialog._on_reject()

        self.assertTrue(worker.cancel_requested)
        self.assertTrue(worker.interruption_requested)
        self.assertEqual(dialog._sdk_stop_pending, "reject")
        self.assertIs(dialog._sdk_worker, worker)
        self.assertFalse(dialog._ok_button.isEnabled())
        self.assertIn("正在取消", dialog._sdk_status.text())
        self.assertEqual(dialog.result(), 0)

        # A second reject during the deferred stop only refreshes the hint.
        dialog._on_reject()
        self.assertEqual(dialog._sdk_stop_pending, "reject")
        self.assertIn("仍在结束", dialog._sdk_status.text())

    def test_deferred_reject_settles_on_real_finished(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        dialog._on_reject()

        worker.finish()

        self.assertIsNone(dialog._sdk_worker)
        self.assertEqual(dialog._sdk_stop_pending, "")
        self.assertTrue(dialog._ok_button.isEnabled())
        self.assertIn("已取消", dialog._sdk_status.text())
        # Contract: dialog stays open so the user can skip or retry.
        self.assertEqual(dialog.result(), 0)

    def test_sdk_completed_suppressed_during_deferred_stop(self):
        from renpy_sdk_install import SdkInstallResult

        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        dialog._on_reject()

        dialog._on_sdk_completed(SdkInstallResult(ok=True, message="成功"))

        self.assertFalse(dialog._ok_button.isEnabled())
        self.assertIn("正在取消", dialog._sdk_status.text())
        # Ownership is still held until the real finished signal.
        self.assertIs(dialog._sdk_worker, worker)

        worker.finish()
        self.assertIsNone(dialog._sdk_worker)

    def test_worker_reference_held_between_completed_and_finished(self):
        from renpy_sdk_install import SdkInstallResult

        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        # Same terminal wiring the real creation path installs.
        dialog._wire_sdk_worker_terminal(worker)

        dialog._on_sdk_completed(SdkInstallResult(ok=False, message="失败示例"))
        self.assertIs(dialog._sdk_worker, worker)

        worker.finish()
        self.assertIsNone(dialog._sdk_worker)

    def test_close_with_active_sdk_worker_waits_for_real_finished(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker

        event = QCloseEvent()
        dialog.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertEqual(dialog._sdk_stop_pending, "close")
        self.assertTrue(worker.cancel_requested)
        self.assertIn("自动关闭", dialog._sdk_status.text())

        worker.finish()
        _process()
        # No workspace applied: the deferred close settles as a reject.
        self.assertEqual(dialog._sdk_stop_pending, "")
        self.assertIsNone(dialog._sdk_worker)
        self.assertEqual(dialog.result(), int(QDialog.DialogCode.Rejected))

    def test_close_during_pending_reject_upgrades_to_deferred_close(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        dialog._on_reject()
        self.assertEqual(dialog._sdk_stop_pending, "reject")

        event = QCloseEvent()
        dialog.closeEvent(event)

        self.assertFalse(event.isAccepted())
        self.assertEqual(dialog._sdk_stop_pending, "close")
        self.assertIn("自动关闭", dialog._sdk_status.text())

        worker.finish()
        _process()
        self.assertEqual(dialog._sdk_stop_pending, "")
        self.assertEqual(dialog.result(), int(QDialog.DialogCode.Rejected))

    def test_escape_reject_with_active_worker_defers_without_closing(self):
        dialog = WorkspaceSetupDialog(None)
        dialog.show()
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker

        dialog.reject()

        self.assertEqual(dialog.result(), 0)
        self.assertFalse(dialog.isHidden())
        self.assertTrue(worker.cancel_requested)
        self.assertEqual(dialog._sdk_stop_pending, "reject")

    def test_progress_ticks_do_not_overwrite_stop_copy(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        dialog._on_reject()

        dialog._on_sdk_progress("download", 3, 10)

        self.assertIn("正在取消", dialog._sdk_status.text())

    def test_deferred_close_keeps_applied_workspace(self):
        dialog = WorkspaceSetupDialog(None)
        worker = _FakeSdkWorker(dialog)
        dialog._sdk_worker = worker
        dialog._workspace_result = WorkspaceSetupDialogResult(
            workspace=Path("X:/ws"),
            message="已接入",
            project_count=0,
            created_registry=True,
        )
        dialog._stack.setCurrentIndex(1)

        event = QCloseEvent()
        dialog.closeEvent(event)
        self.assertEqual(dialog._sdk_stop_pending, "close")

        worker.finish()
        _process()

        self.assertEqual(dialog.result(), int(QDialog.DialogCode.Accepted))
        payload = dialog.result_payload()
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("跳过", payload.sdk_message)


if __name__ == "__main__":
    unittest.main()
