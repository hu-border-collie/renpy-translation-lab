"""Tests for workbench log drawer and dual log-focus APIs (GUI IA P0a / #158)."""
from __future__ import annotations

import unittest
from pathlib import Path
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
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def _install_log_spies(self) -> tuple[list[object], list[object]]:
        drawer_calls: list[object] = []
        expand_calls: list[object] = []

        def show_drawer() -> None:
            drawer_calls.append(True)

        def expand_log(*, switch_tab: bool = True) -> None:
            expand_calls.append({"switch_tab": switch_tab})

        self.window._show_workbench_log_drawer = show_drawer  # type: ignore[method-assign]
        self.window._expand_diagnostics_log = expand_log  # type: ignore[method-assign]
        return drawer_calls, expand_calls

    def test_log_views_share_document(self) -> None:
        self.assertIs(
            self.window.workbench_log_view.document(),
            self.window.log_view.document(),
        )
        self.window.log_view.setPlainText("shared-line")
        self.assertIn("shared-line", self.window.workbench_log_view.toPlainText())

    def test_clear_log_clears_shared_document(self) -> None:
        self.window.log_view.setPlainText("to-clear")
        self.window._clear_log_view()
        self.assertEqual(self.window.log_view.toPlainText(), "")
        self.assertEqual(self.window.workbench_log_view.toPlainText(), "")

    def test_show_workbench_log_drawer_does_not_switch_main_tab(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._show_workbench_log_drawer()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)
        # Window may not be shown in offscreen tests; isHidden tracks setVisible.
        self.assertFalse(self.window.workbench_log_body.isHidden())
        self.assertEqual(self.window.workbench_log_toggle_btn.text(), "折叠")

    def test_expand_diagnostics_log_switches_to_diagnostics_tab(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        # Splitter sizes only stick after the diagnostics page is laid out.
        self.window.resize(1280, 900)
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(6):
            self._app.processEvents()
        self.window.diagnostics_splitter.setSizes([420, 180])
        for _ in range(4):
            self._app.processEvents()
        idle_sizes = list(self.window.diagnostics_splitter.sizes())
        idle_context = idle_sizes[0]
        idle_log = idle_sizes[1]
        self.assertGreater(idle_context, 0)
        self.assertGreater(idle_log, 0)

        # Start from workbench so switch_tab is exercised.
        self.window.tab_widget.setCurrentWidget(workbench)
        for _ in range(2):
            self._app.processEvents()
        # Re-apply idle split after tab change (some platforms rebalance on hide).
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
        # Drive the running animation to its end so we assert real splitter output.
        anim = getattr(self.window, "_splitter_anim", None)
        self.assertIsNotNone(anim)
        assert anim is not None
        anim.setCurrentTime(anim.duration())
        for _ in range(4):
            self._app.processEvents()

        sizes = self.window.diagnostics_splitter.sizes()
        self.assertEqual(len(sizes), 2)
        self.assertGreater(sizes[1], 0)
        # Running layout shrinks the context pane and grows the log pane
        # (or at least does not shrink the log relative to idle).
        total = max(sum(sizes), 1)
        running_context_target = int(total * 0.32)
        self.assertLessEqual(sizes[0], max(idle_context, running_context_target + 1))
        self.assertGreaterEqual(sizes[1], idle_log)

    def test_expand_diagnostics_log_without_switch_keeps_workbench(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window._expand_diagnostics_log(switch_tab=False)

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)

    def test_reveal_log_for_active_context_prefers_drawer_off_diagnostics(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._reveal_log_for_active_context()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)

    def test_reveal_log_for_active_context_on_diagnostics_expands_log(self) -> None:
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._reveal_log_for_active_context()

        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        # Drawer should not be required when already on diagnostics.
        self.assertFalse(self.window._workbench_log_drawer_expanded)

    def test_deprecated_focus_log_tab_aliases_workbench_drawer(self) -> None:
        workbench = self.window.tab_widget.widget(0)
        self.window.tab_widget.setCurrentWidget(workbench)
        self.window._set_workbench_log_drawer_expanded(False)

        self.window._focus_log_tab()

        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertTrue(self.window._workbench_log_drawer_expanded)

    def test_probe_entrypoint_uses_workbench_log_drawer(self) -> None:
        """P2a: probe starts from batch execute; expand workbench drawer, not diagnostics."""
        drawer_calls, expand_calls = self._install_log_spies()
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
        self.window.state.get_batch_script_path = lambda: Path("C:/tool/gemini_translate_batch.py")  # type: ignore[method-assign]
        self.window.state.get_logs_dir = lambda: Path("C:/tool/logs")  # type: ignore[method-assign]
        self.window._submit_max_cost_from_config = lambda: 0.0  # type: ignore[method-assign]

        self.window._on_run_probe()

        self.assertEqual(len(drawer_calls), 1)
        self.assertEqual(len(expand_calls), 0)
        self.window.runner.run.assert_called_once()

    def test_start_translation_entrypoint_uses_workbench_drawer(self) -> None:
        """Runtime guard: starting translation expands the workbench drawer only."""
        drawer_calls, expand_calls = self._install_log_spies()
        self.window.state.get_game_root = lambda: "C:/game/work"  # type: ignore[method-assign]
        self.window._doctor_allows_translate_action = lambda: True  # type: ignore[method-assign]
        self.window._confirm_unsaved_config_before_workflow = lambda: True  # type: ignore[method-assign]
        self.window._begin_translation_workflow = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._clear_completed_manifest_snapshot = lambda: None  # type: ignore[method-assign]
        self.window._set_writeback_summary = lambda *_a, **_k: None  # type: ignore[method-assign]
        self.window._clear_log_view = lambda: None  # type: ignore[method-assign]

        self.window._on_start_translation()

        self.assertEqual(len(drawer_calls), 1)
        self.assertEqual(len(expand_calls), 0)


if __name__ == "__main__":
    unittest.main()
