import unittest
from pathlib import Path
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiWorkflowGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_confirm_unsaved_config_before_workflow_cancel_returns_false(self):
        window = MainWindow.__new__(MainWindow)
        window._loading_config_to_ui = False
        window._config_ui_saved_snapshot = {"api_key_count": 1}
        window._current_config_ui_snapshot = lambda: {"api_key_count": 2}

        message_box = mock.Mock()
        save_btn, discard_btn, cancel_btn = object(), object(), object()
        message_box.addButton.side_effect = [save_btn, discard_btn, cancel_btn]
        message_box.clickedButton.return_value = cancel_btn
        with (
            mock.patch.object(MainWindow, "_config_tab_has_unsaved_changes", return_value=True),
            mock.patch("gui_qt.app.QMessageBox", return_value=message_box),
        ):
            allowed = window._confirm_unsaved_config_before_workflow()

        self.assertFalse(allowed)

    def test_update_keyword_merge_btn_disabled_when_not_ready(self):
        window = MainWindow.__new__(MainWindow)
        window.keyword_merge_btn = mock.Mock()
        window.kill_btn = mock.Mock()
        window.kill_btn.isEnabled.return_value = False
        window._resolve_keyword_merge_candidates_path = mock.Mock(return_value="")
        window._resolve_keyword_merge_glossary_path = mock.Mock(return_value="")

        window._update_keyword_merge_btn_enabled()

        window.keyword_merge_btn.setEnabled.assert_called_once_with(False)

    def test_switch_game_root_invalidates_manifest_caches(self):
        window = MainWindow.__new__(MainWindow)
        window.state = mock.Mock()
        window.state.set_game_root.return_value = (Path("C:/Games/Example/work"), False)
        window.runner = mock.Mock()
        window.runner.is_running.return_value = False
        window._is_doctor_running = mock.Mock(return_value=False)
        window._invalidate_doctor_worker = mock.Mock()
        window._refresh_project_label = mock.Mock()
        window._load_config_to_ui = mock.Mock()
        window._current_work_mode = mock.Mock()
        window._set_doctor_summary = mock.Mock()
        window._set_workflow_summary = mock.Mock()
        window._set_writeback_summary = mock.Mock()
        window._apply_work_mode_ui = mock.Mock()
        window._refresh_diagnostics_context = mock.Mock()
        window._append_log = mock.Mock()
        window._clear_game_root_redirect_notice = mock.Mock()
        window._doctor_output_lines = []
        window._workflow = None
        window._workflow_step_output_lines = []
        window._clear_completed_manifest_snapshot = mock.Mock()
        window._writeback_manifest_path = ""
        window._keyword_merge_candidates_path = ""
        window._doctor_check_completed = False
        window._doctor_summary_status = ""
        window._last_doctor_report = None
        window._last_doctor_report_game_root = ""

        invalidate_calls: list[Path | None] = []

        def capture_invalidate(manifest_path=None):
            invalidate_calls.append(manifest_path)

        window._invalidate_manifest_caches = capture_invalidate

        with mock.patch("gui_qt.app.work_mode_spec") as work_mode_spec_mock:
            spec = mock.Mock()
            spec.is_bootstrap = False
            spec.supports_translation_writeback = True
            spec.mode = mock.Mock()
            work_mode_spec_mock.return_value = spec
            self.assertTrue(window._switch_game_root("C:/Games/Example/work"))

        self.assertEqual(invalidate_calls, [None])


if __name__ == "__main__":
    unittest.main()