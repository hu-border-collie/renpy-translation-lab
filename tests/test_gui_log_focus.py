"""Tests for the diagnostics-only log surface."""
from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock
from unittest.mock import MagicMock

try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (
        QApplication,
        QLineEdit,
        QPushButton,
        QScrollArea,
    )

    from gui_qt.app import MainWindow
    from gui_qt.diagnostics_context import DiagnosticsCommand, DiagnosticsContext
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    DiagnosticsCommand = None  # type: ignore[assignment,misc]
    DiagnosticsContext = None  # type: ignore[assignment,misc]
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
        gui_test_support.close_main_window(self.window)
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
        for _ in range(6):
            self._app.processEvents()

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
        self.assertLessEqual(sizes[0], running_context_target + 8)

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

    def test_empty_log_is_collapsed_at_small_regular_and_wide_sizes(self) -> None:
        fixture_command = (
            "python gemini_translate_batch.py status --manifest fixture.json "
            + "--fixture-option "
            + "value-" * 24
        )
        commands = [
            DiagnosticsCommand(f"fixture-{index}", f"python tool.py status --n {index}")
            for index in range(5)
        ] + [DiagnosticsCommand("long-fixture", fixture_command)]
        context = DiagnosticsContext(
            status="ready",
            heading="原创布局 fixture",
            message="无项目路径、配置或凭据。",
            facts=["仅用于诊断布局回归。"],
            paths=[],
            commands=commands,
            manifest_json_preview="{}",
        )
        self.window.resize(960, 640)
        self.window.show()
        for _ in range(4):
            self._app.processEvents()
        self.window._on_header_log_clicked()
        self.window._set_diagnostics_context(context)

        for size in ((960, 640), (1280, 900), (1920, 1080)):
            with self.subTest(size=size):
                self.window.resize(*size)
                self.window.show()
                for _ in range(4):
                    self._app.processEvents()
                self.window.diagnostics_inner_tabs.setCurrentIndex(1)
                for _ in range(4):
                    self._app.processEvents()

                sizes = self.window.diagnostics_splitter.sizes()
                self.assertEqual(len(sizes), 2)
                self.assertGreater(sizes[0], 0)
                self.assertEqual(sizes[1], 0)
                self.assertTrue(self.window.diagnostics_log_panel.isHidden())
                self.assertFalse(self.window._diagnostics_splitter_user_adjusted)
                self.assertEqual(
                    self.window.diagnostics_log_toggle_btn.text(),
                    "显示运行日志",
                )
                scroll = self.window.findChild(
                    QScrollArea,
                    "diagnostics_commands_scroll",
                )
                self.assertIsNotNone(scroll)
                assert scroll is not None
                self.assertEqual(scroll.verticalScrollBar().maximum(), 0)

        edits = self.window.diagnostics_commands_host.findChildren(
            QLineEdit,
            "diagnostics_command_edit",
        )
        self.assertEqual(edits[-1].text(), fixture_command)
        copy_button = next(
            button
            for button in edits[-1].parentWidget().findChildren(QPushButton)
            if button.text() == "复制"
        )
        copy_button.click()
        self.assertEqual(QGuiApplication.clipboard().text(), fixture_command)

    def test_task_output_is_visible_and_does_not_keep_resizing(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        for _ in range(4):
            self._app.processEvents()
        workbench = self.window._workbench_tab
        self.window.tab_widget.setCurrentWidget(workbench)

        self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
        started = self.window._start_cli_command(
            "original_fixture",
            Path("fixture_cli.py"),
            ["--no-provider"],
        )
        self.assertTrue(started)
        self.assertTrue(self.window._diagnostics_log_panel_visible)
        self.assertIs(self.window.tab_widget.currentWidget(), workbench)
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())

        self.window._on_header_log_clicked()
        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        self.assertGreater(self.window.diagnostics_splitter.sizes()[1], 0)
        self.assertFalse(self.window._diagnostics_splitter_user_adjusted)
        self.assertTrue(self.window.diagnostics_log_empty_label.isVisible())
        self._app.processEvents()
        first_sizes = list(self.window.diagnostics_splitter.sizes())

        self.window.runner.line_ready.emit("fixture: task started")
        self.window._flush_pending_log_lines()
        self.assertFalse(self.window.diagnostics_log_empty_label.isVisible())
        second_sizes = list(self.window.diagnostics_splitter.sizes())
        for index in range(20):
            self.window.runner.line_ready.emit(
                f"fixture: ordinary output {index}"
            )
        self.window._flush_pending_log_lines()
        self.assertEqual(self.window.log_view.toPlainText().count("fixture:"), 21)
        self.assertEqual(self.window.diagnostics_splitter.sizes(), second_sizes)
        self.assertNotEqual(first_sizes[1], 0)

        scrollbar = self.window.log_view.verticalScrollBar()
        scrollbar.setValue(0)
        self.window.runner.line_ready.emit(
            "fixture: later line while viewing history"
        )
        self.window._flush_pending_log_lines()
        self.assertEqual(scrollbar.value(), 0)

        self.window._set_task_running(False)
        self.assertTrue(self.window._diagnostics_log_panel_visible)
        self.assertIn("fixture: task started", self.window.log_view.toPlainText())
        self.window._on_clear_log()
        self.assertEqual(self.window.log_view.toPlainText(), "")
        self.assertTrue(self.window.diagnostics_log_panel.isHidden())

    def test_runner_output_does_not_rescan_history_per_line(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        for _ in range(4):
            self._app.processEvents()
        history = "\n".join(
            f"original fixture history {index}: {'x' * 96}"
            for index in range(100)
        )
        self.window.log_view.setPlainText(history)
        self.assertTrue(self.window._diagnostics_log_has_content())
        self.window.tab_widget.setCurrentWidget(self.window._workbench_tab)
        self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.assertTrue(
            self.window._start_cli_command(
                "original_fixture",
                Path("fixture_cli.py"),
                ["--no-provider"],
            )
        )
        self.window._on_header_log_clicked()

        with mock.patch.object(
            self.window.log_view,
            "toPlainText",
            wraps=self.window.log_view.toPlainText,
        ) as read_document:
            for index in range(300):
                self.window.runner.line_ready.emit(
                    f"fixture: queued output {index}"
                )
            self.assertEqual(read_document.call_count, 0)
            self.assertEqual(len(self.window._pending_log_lines), 300)

            self.window._flush_pending_log_lines()
            self.assertEqual(read_document.call_count, 0)

        text = self.window.log_view.toPlainText()
        self.assertIn("original fixture history 99:", text)
        self.assertTrue(text.endswith("fixture: queued output 299"))
        self.assertTrue(self.window._diagnostics_log_has_content())
        self.assertTrue(self.window.log_view.isVisible())
        self.assertFalse(self.window.diagnostics_log_empty_label.isVisible())

        self.window._clear_log_view()
        self.assertFalse(self.window._diagnostics_log_has_content())
        self.assertTrue(self.window.log_view.isHidden())
        self.assertTrue(self.window.diagnostics_log_empty_label.isVisible())

        self.window.runner.line_ready.emit("   \n")
        self.assertFalse(self.window._diagnostics_log_has_content())
        self.window._flush_pending_log_lines()
        self.assertFalse(self.window._diagnostics_log_has_content())
        self.assertTrue(self.window.log_view.isHidden())
        self.assertTrue(self.window.diagnostics_log_empty_label.isVisible())

        self.window.runner.line_ready.emit("fixture: output after clear")
        self.assertTrue(self.window._diagnostics_log_has_content())
        self.assertFalse(self.window.log_view.isHidden())
        self.window._flush_pending_log_lines()
        self.assertEqual(
            self.window.log_view.toPlainText(),
            "   \nfixture: output after clear",
        )
        self.assertFalse(self.window.diagnostics_log_empty_label.isVisible())
        self.window._set_task_running(False)

    def test_explicit_collapse_survives_task_start_but_error_reopens_log(self) -> None:
        self.window.show()
        self.window._on_header_log_clicked()
        toggle = self.window.diagnostics_log_toggle_btn

        toggle.click()
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())
        self.assertTrue(self.window.diagnostics_log_empty_label.isVisible())
        self.assertEqual(toggle.text(), "隐藏运行日志")

        toggle.click()
        self.assertTrue(self.window.diagnostics_log_panel.isHidden())
        self.assertEqual(toggle.text(), "显示运行日志")
        self.assertTrue(self.window._diagnostics_splitter_user_adjusted)
        hidden_ratio = self.window._diagnostics_splitter_manual_context_ratio
        self.assertIsNotNone(hidden_ratio)
        assert hidden_ratio is not None

        self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.window._start_cli_command(
            "original_fixture",
            Path("fixture_cli.py"),
            ["--no-provider"],
        )
        # A task start is automatic: an explicit collapse still wins.
        self.assertTrue(self.window.diagnostics_log_panel.isHidden())

        # The status bar points at the log, so an error re-opens it with the
        # ratio the user had when they hid the panel.
        self.window.runner.error.emit("fixture: error after explicit collapse")
        self.window._flush_pending_log_lines()
        anim = getattr(self.window, "_splitter_anim", None)
        if anim is not None:
            anim.setCurrentTime(anim.duration())
        for _ in range(4):
            self._app.processEvents()
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())
        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        self.assertIn(
            "fixture: error after explicit collapse",
            self.window.log_view.toPlainText(),
        )
        sizes = self.window.diagnostics_splitter.sizes()
        total = max(sum(sizes), 1)
        self.assertAlmostEqual(sizes[0] / total, hidden_ratio, delta=0.06)
        self.window._set_task_running(False)

    def test_show_click_keeps_running_balance_for_the_next_task(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(6):
            self._app.processEvents()

        # Peeking at the empty log is a visibility choice, not a split choice.
        self.window.diagnostics_log_toggle_btn.click()
        for _ in range(4):
            self._app.processEvents()
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())
        self.assertFalse(self.window._diagnostics_splitter_user_adjusted)
        idle_log_size = self.window.diagnostics_splitter.sizes()[1]

        self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.assertTrue(
            self.window._start_cli_command(
                "original_fixture",
                Path("fixture_cli.py"),
                ["--no-provider"],
            )
        )
        anim = getattr(self.window, "_splitter_anim", None)
        if anim is not None:
            anim.setCurrentTime(anim.duration())
        for _ in range(4):
            self._app.processEvents()

        sizes = self.window.diagnostics_splitter.sizes()
        total = max(sum(sizes), 1)
        self.assertLessEqual(sizes[0], int(total * 0.32) + 8)
        self.assertGreater(sizes[1], idle_log_size)
        self.window._set_task_running(False)

    def test_clear_after_show_click_returns_to_empty_state(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(6):
            self._app.processEvents()
        self.window.diagnostics_log_toggle_btn.click()
        for _ in range(4):
            self._app.processEvents()

        self.window.runner.line_ready.emit("fixture: output before clear")
        self.window._flush_pending_log_lines()
        for _ in range(4):
            self._app.processEvents()
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())

        self.window._on_clear_log()
        for _ in range(6):
            self._app.processEvents()
        self.assertTrue(self.window.diagnostics_log_panel.isHidden())
        self.assertTrue(self.window.diagnostics_log_empty_label.isHidden())
        self.assertEqual(
            self.window.diagnostics_log_toggle_btn.text(),
            "显示运行日志",
        )

    def test_header_log_click_reopens_a_hidden_log_with_content(self) -> None:
        self.window.show()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        for _ in range(4):
            self._app.processEvents()
        self.window.diagnostics_log_toggle_btn.click()
        for _ in range(4):
            self._app.processEvents()
        self.window.runner.line_ready.emit("fixture: retained output")
        self.window._flush_pending_log_lines()
        for _ in range(4):
            self._app.processEvents()
        self.window.diagnostics_log_toggle_btn.click()
        for _ in range(4):
            self._app.processEvents()
        self.assertTrue(self.window.diagnostics_log_panel.isHidden())

        self.window.tab_widget.setCurrentWidget(self.window._workbench_tab)
        self.window._on_header_log_clicked()
        anim = getattr(self.window, "_splitter_anim", None)
        if anim is not None:
            anim.setCurrentTime(anim.duration())
        for _ in range(4):
            self._app.processEvents()
        self.assertIs(
            self.window.tab_widget.currentWidget(),
            self.window._diagnostics_tab,
        )
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())
        self.assertIn(
            "fixture: retained output",
            self.window.log_view.toPlainText(),
        )

    def test_runner_error_reveals_original_fixture_output(self) -> None:
        workbench = self.window._workbench_tab
        for size in ((960, 640), (1280, 900), (1920, 1080)):
            with self.subTest(size=size):
                self.window.resize(*size)
                self.window.show()
                self.window.tab_widget.setCurrentWidget(workbench)
                for _ in range(4):
                    self._app.processEvents()

                self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
                self.assertTrue(
                    self.window._start_cli_command(
                        "original_fixture",
                        Path("fixture_cli.py"),
                        ["--no-provider"],
                    )
                )
                self.assertIs(self.window.tab_widget.currentWidget(), workbench)
                self.window.runner.line_ready.emit("fixture: ordinary output")
                self.window._flush_pending_log_lines()
                active_sizes = self.window.diagnostics_splitter.sizes()
                self.assertGreater(active_sizes[1], 0)
                self.assertFalse(self.window.diagnostics_log_panel.isHidden())

                error_line = f"fixture: simulated local error at {size[0]}"
                self.window.runner.error.emit(error_line)
                self.window._flush_pending_log_lines()
                self.assertIs(
                    self.window.tab_widget.currentWidget(),
                    self.window._diagnostics_tab,
                )
                self.assertTrue(self.window._diagnostics_log_panel_visible)
                self.assertIn(error_line, self.window.log_view.toPlainText())

                self.window._set_task_running(False)
                self.assertFalse(self.window.diagnostics_log_panel.isHidden())
                self.window._on_clear_log()
                self.assertTrue(self.window.diagnostics_log_panel.isHidden())

    def test_actual_splitter_drag_survives_task_page_resize_and_clear(self) -> None:
        self.window.resize(1280, 900)
        self.window.show()
        for _ in range(6):
            self._app.processEvents()
        self.window._on_header_log_clicked()
        self.window.runner.run = MagicMock(return_value=True)  # type: ignore[method-assign]
        self.window._start_cli_command(
            "original_fixture",
            Path("fixture_cli.py"),
            ["--no-provider"],
        )
        for _ in range(4):
            self._app.processEvents()
        self.assertTrue(self.window.diagnostics_log_empty_label.isVisible())

        splitter = self.window.diagnostics_splitter
        handle = splitter.handle(1)
        original_sizes = list(splitter.sizes())
        start_global = handle.mapToGlobal(handle.rect().center())
        end_global = QPoint(start_global.x(), start_global.y() + 70)
        QTest.mousePress(
            handle,
            Qt.MouseButton.LeftButton,
            pos=handle.rect().center(),
        )
        QTest.mouseMove(handle, handle.mapFromGlobal(end_global), delay=80)
        QTest.mouseRelease(
            handle,
            Qt.MouseButton.LeftButton,
            pos=handle.mapFromGlobal(end_global),
        )
        for _ in range(4):
            self._app.processEvents()

        self.assertTrue(self.window._diagnostics_splitter_user_adjusted)
        dragged_sizes = list(splitter.sizes())
        self.assertNotEqual(dragged_sizes, original_sizes)
        self.window.runner.line_ready.emit("fixture: output after drag")
        self.window._flush_pending_log_lines()
        self.window._set_task_running(False)
        self.window.tab_widget.setCurrentWidget(self.window._workbench_tab)
        self.window.resize(960, 640)
        for _ in range(6):
            self._app.processEvents()
        self.window.tab_widget.setCurrentWidget(self.window._diagnostics_tab)
        current_sizes = splitter.sizes()
        before_ratio = dragged_sizes[0] / sum(dragged_sizes)
        after_ratio = current_sizes[0] / max(sum(current_sizes), 1)
        self.assertAlmostEqual(after_ratio, before_ratio, delta=0.08)

        self.window._on_clear_log()
        self.assertFalse(self.window.diagnostics_log_panel.isHidden())
        self.assertEqual(
            self.window.diagnostics_log_toggle_btn.text(),
            "隐藏运行日志",
        )


if __name__ == "__main__":
    unittest.main()
