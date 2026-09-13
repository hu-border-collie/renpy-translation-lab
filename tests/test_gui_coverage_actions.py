"""GUI coverage/review actions and import flow tests (#424 P6 S3c)."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication

    from gui_qt import app as app_module
    from gui_qt.app import MainWindow
    from gui_qt.coverage_actions import CoverageReviewImportResult
    from gui_qt.user_copy import COVERAGE_REVIEW_COPY
except ImportError as exc:
    QObject = None  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    CoverageReviewImportResult = None  # type: ignore[assignment,misc]
    COVERAGE_REVIEW_COPY = {}
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


if QObject is not None:

    class _FakeImportWorker(QObject):
        completed = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.deleted = False

        def start(self) -> None:  # pragma: no cover - never started in tests
            pass

        def deleteLater(self) -> None:  # noqa: N802 - Qt spelling
            self.deleted = True


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiCoverageActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_doctor_tab_exposes_coverage_actions(self) -> None:
        self.assertEqual(
            self.window.coverage_rescan_btn.text(),
            COVERAGE_REVIEW_COPY["rescan"],
        )
        self.assertEqual(
            self.window.coverage_import_btn.text(),
            COVERAGE_REVIEW_COPY["import_review"],
        )
        self.assertTrue(self.window.coverage_rescan_btn.toolTip())

    def test_coverage_actions_follow_project_and_running_state(self) -> None:
        with mock.patch.object(
            self.window.state,
            "get_game_root",
            return_value="C:/Game/work",
        ):
            self.window._sync_doctor_coverage_buttons(running=False)
            self.assertTrue(self.window.coverage_rescan_btn.isEnabled())
            self.assertTrue(self.window.coverage_import_btn.isEnabled())

            self.window._sync_doctor_coverage_buttons(running=True)
            self.assertFalse(self.window.coverage_rescan_btn.isEnabled())
            self.assertFalse(self.window.coverage_import_btn.isEnabled())

        with mock.patch.object(self.window.state, "get_game_root", return_value=""):
            self.window._sync_doctor_coverage_buttons(running=False)
            self.assertFalse(self.window.coverage_rescan_btn.isEnabled())
            self.assertFalse(self.window.coverage_import_btn.isEnabled())

    def test_rescan_button_reruns_doctor(self) -> None:
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch.object(self.window, "_on_run_doctor") as run_doctor,
        ):
            self.window._sync_doctor_coverage_buttons(running=False)
            self.window.coverage_rescan_btn.click()
        run_doctor.assert_called_once_with()

    def test_import_button_opens_json_dialog_for_current_project(self) -> None:
        with (
            mock.patch.object(
                self.window,
                "_confirm_unsaved_config_before_workflow",
                return_value=True,
            ),
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch.object(
                app_module.QFileDialog,
                "getOpenFileName",
                return_value=("", ""),
            ) as picker,
        ):
            self.window._sync_doctor_coverage_buttons(running=False)
            self.window.coverage_import_btn.click()

        picker.assert_called_once()
        self.assertIsNone(self.window._coverage_import_worker)

    def test_import_completion_success_reruns_doctor(self) -> None:
        worker = _FakeImportWorker()
        self.window._coverage_import_worker = worker
        self.window._coverage_import_game_root = "C:/Game/work"
        result = CoverageReviewImportResult(
            ok=True,
            payload={
                "status": "imported",
                "review_path": "C:/Game/work/translation_context/coverage_review.json",
                "review_status": "human_reviewed",
                "review_policy": "human_required",
                "unresolved_findings": 0,
            },
        )
        worker.completed.connect(self.window._on_coverage_review_import_completed)
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch.object(app_module, "message_box_information") as info,
            mock.patch.object(self.window, "_on_run_doctor") as run_doctor,
        ):
            worker.completed.emit(result)

        self.assertIsNone(self.window._coverage_import_worker)
        self.assertTrue(worker.deleted)
        info.assert_called_once()
        run_doctor.assert_called_once_with()

    def test_import_completion_refusal_reports_structured_code(self) -> None:
        worker = _FakeImportWorker()
        self.window._coverage_import_worker = worker
        self.window._coverage_import_game_root = "C:/Game/work"
        result = CoverageReviewImportResult(
            ok=False,
            error="coverage review is stale",
            error_code="COVERAGE_REVIEW_STALE",
            suggested_action="reexport_review",
        )
        worker.completed.connect(self.window._on_coverage_review_import_completed)
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch.object(app_module, "message_box_warning") as warning,
            mock.patch.object(self.window, "_on_run_doctor") as run_doctor,
        ):
            worker.completed.emit(result)

        self.assertIn("COVERAGE_REVIEW_STALE", warning.call_args.args[2])
        run_doctor.assert_not_called()

    def test_import_completion_ignores_project_switch(self) -> None:
        worker = _FakeImportWorker()
        self.window._coverage_import_worker = worker
        self.window._coverage_import_game_root = "C:/Game/old"
        result = CoverageReviewImportResult(
            ok=True,
            payload={"review_path": "C:/Game/old/review.json"},
        )
        worker.completed.connect(self.window._on_coverage_review_import_completed)
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/new",
            ),
            mock.patch.object(app_module, "message_box_warning") as warning,
            mock.patch.object(self.window, "_on_run_doctor") as run_doctor,
        ):
            worker.completed.emit(result)

        self.assertIn("项目已切换", warning.call_args.args[2])
        run_doctor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
