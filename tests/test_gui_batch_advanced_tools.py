"""Batch execute advanced tools: probe + split (GUI IA P2a / #164)."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from PySide6.QtWidgets import QApplication, QPushButton

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    QPushButton = object  # type: ignore[misc,assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import WorkMode
from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiBatchAdvancedToolsTests(unittest.TestCase):
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
        self.window.close()
        self.window.deleteLater()

    def test_probe_and_split_live_on_workbench_not_diagnostics_panel(self) -> None:
        self.assertTrue(hasattr(self.window, "batch_advanced_frame"))
        self.assertIs(self.window.probe_btn.parentWidget(), self.window.batch_advanced_bar)
        self.assertIs(self.window.split_btn.parentWidget(), self.window.batch_advanced_bar)
        # Diagnostics panel still has A/B and keyword merge, not probe/split as children.
        diag = self.window.diagnostics_action_panel
        diag_buttons = {b.text() for b in diag.findChildren(QPushButton)}
        self.assertNotIn("试跑样本请求", diag_buttons)
        self.assertNotIn("拆分翻译包", diag_buttons)
        self.assertIn("翻译 A/B 对比", diag_buttons)

    def test_advanced_bar_stays_on_batch_across_status_tabs(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        for tab in (0, 1, 2):
            self.window._focus_workbench_status_tab(tab)
            self.assertFalse(
                self.window.batch_advanced_frame.isHidden(),
                msg=f"advanced tools should stay on batch (status tab {tab})",
            )

        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(self.window.batch_advanced_frame.isHidden())

    def test_split_entrypoint_uses_workbench_drawer(self) -> None:
        drawer_calls: list[bool] = []
        expand_calls: list[bool] = []
        self.window._show_workbench_log_drawer = lambda: drawer_calls.append(True)  # type: ignore[method-assign]
        self.window._expand_diagnostics_log = lambda **_k: expand_calls.append(True)  # type: ignore[method-assign]
        self.window._current_diagnostics_manifest = lambda: (  # type: ignore[method-assign]
            "C:/tmp/manifest.json",
            {
                "version": 1,
                "mode": "translation",
                "chunks": [{"id": 1}, {"id": 2}],
            },
        )
        self.window._prompt_split_options = lambda: {  # type: ignore[method-assign]
            "max_chunks": 1,
            "max_items": 0,
            "display_name_prefix": "",
        }
        self.window._set_diagnostics_context = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._append_log = lambda _text: None  # type: ignore[method-assign]
        self.window._set_task_running = lambda _running: None  # type: ignore[method-assign]
        self.window._clear_log_view = lambda: None  # type: ignore[method-assign]
        self.window.runner = MagicMock()
        self.window.state.get_batch_script_path = lambda: Path("C:/tool/gemini_translate_batch.py")  # type: ignore[method-assign]
        self.window.state.get_logs_dir = lambda: Path("C:/tool/logs")  # type: ignore[method-assign]
        self.window._submit_max_cost_from_config = lambda: 0.0  # type: ignore[method-assign]

        with patch(
            "gui_qt.app.translation_split_ready",
            return_value=(True, "ok"),
        ):
            self.window._on_run_split()

        self.assertEqual(drawer_calls, [True])
        self.assertEqual(expand_calls, [])
        self.window.runner.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
