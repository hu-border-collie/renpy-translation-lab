"""Tests for the diagnostics-only log surface."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiLogFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        cls._app = app or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_workbench_has_no_log_drawer(self) -> None:
        self.assertFalse(hasattr(self.window, "workbench_log_drawer"))
        self.assertFalse(hasattr(self.window, "workbench_log_view"))
        self.assertTrue(hasattr(self.window, "log_view"))

    def test_clear_log_clears_diagnostics_document(self) -> None:
        self.window.log_view.setPlainText("to-clear")
        self.window._clear_log_view()
        self.assertEqual(self.window.log_view.toPlainText(), "")

    def test_expand_diagnostics_log_switches_to_diagnostics_tab(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window.resize(1280, 900)
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(6):
            self._app.processEvents()
        self.window.diagnostics_splitter.setSizes([420, 180])
        for _ in range(4):
            self._app.processEvents()
        idle_sizes = list(self.window.diagnostics_splitter.sizes())
        idle_context, idle_log = idle_sizes
        self.assertGreater(idle_context, 0)
        self.assertGreater(idle_log, 0)

        self.window.tab_widget.setCurrentWidget(workbench)
        for _ in range(2):
            self._app.processEvents()
        self.window.diagnostics_splitter.setSizes([idle_context, idle_log])
        for _ in range(2):
            self._app.processEvents()
        idle_context = self.window.diagnostics_splitter.sizes()[0]
        idle_log = self.window.diagnostics_splitter.sizes()[1]

        self.window._expand_diagnostics_log()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        anim = getattr(self.window, "_splitter_anim", None)
        self.assertIsNotNone(anim)
        assert anim is not None
        anim.setCurrentTime(anim.duration())
        for _ in range(4):
            self._app.processEvents()

        sizes = self.window.diagnostics_splitter.sizes()
        self.assertEqual(len(sizes), 2)
        self.assertGreater(sizes[1], 0)
        total = max(sum(sizes), 1)
        running_context_target = int(total * 0.32)
        self.assertLessEqual(
            sizes[0],
            max(idle_context, running_context_target + 1),
        )
        self.assertGreaterEqual(sizes[1], idle_log)

    def test_expand_diagnostics_log_without_switch_keeps_workbench(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window._expand_diagnostics_log(switch_tab=False)

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)

    def test_reveal_log_from_workbench_opens_diagnostics(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window._reveal_log_for_active_context()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )

    def test_reveal_log_on_diagnostics_stays_on_diagnostics(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)

        self.window._reveal_log_for_active_context()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )

    def test_deprecated_focus_log_tab_opens_diagnostics(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window._focus_log_tab()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )

    def test_probe_entrypoint_does_not_add_workbench_log_surface(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._current_diagnostics_manifest = lambda: (  # type: ignore[method-assign]
            "C:/tmp/manifest.json",
            {
                "version": 1,
                "mode": "translation",
                "input_jsonl_path": "C:/tmp/requests.jsonl",
            },
        )
        self.window._prompt_probe_options = lambda: {  # type: ignore[method-assign]
            "limit": 1,
            "offset": 0,
            "api_key_index": None,
        }
        self.window._set_diagnostics_context = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._append_log = lambda _text: None  # type: ignore[method-assign]
        self.window._set_task_running = lambda _running: None  # type: ignore[method-assign]
        self.window.runner = MagicMock()
        self.window.state.get_batch_script_path = lambda: Path(  # type: ignore[method-assign]
            "C:/tool/gemini_translate_batch.py"
        )
        self.window.state.get_logs_dir = lambda: Path("C:/tool/logs")  # type: ignore[method-assign]
        self.window._submit_max_cost_from_config = lambda: 0.0  # type: ignore[method-assign]

        self.window._on_run_probe()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertFalse(hasattr(self.window, "workbench_log_drawer"))
        self.window.runner.run.assert_called_once()

    def test_start_translation_keeps_current_task_page(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window.state.get_game_root = lambda: "C:/game/work"  # type: ignore[method-assign]
        self.window._doctor_allows_translate_action = lambda: True  # type: ignore[method-assign]
        self.window._confirm_unsaved_config_before_workflow = lambda: True  # type: ignore[method-assign]
        self.window._begin_translation_workflow = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._clear_completed_manifest_snapshot = lambda: None  # type: ignore[method-assign]
        self.window._set_writeback_summary = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._clear_log_view = lambda: None  # type: ignore[method-assign]

        self.window._on_start_translation()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertFalse(hasattr(self.window, "workbench_log_drawer"))


if __name__ == "__main__":
    unittest.main()
