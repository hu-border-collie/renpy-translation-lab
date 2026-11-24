"""#541 A: offline service results, page wiring, freshness and layout evidence."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.support import gui_test_support
from foundation import cli_contract
from foundation import doctor_recommendations as rec

try:
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from gui_qt.app import MainWindow
    from gui_qt.reports.doctor_report import stale_summary
    from gui_qt.project_state import ProjectState
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def report_fixture(*, blocked=False, warning=False, empty=False):
    """Original result shapes; no credentials or game excerpts."""
    return {
        "mode": "blocked_missing_template" if blocked else "existing_tl_only",
        "layout_status": "failed" if blocked else "ready",
        "is_work_root": True,
        "tl_exists": not empty,
        "counts": {"rpy_files": 0 if empty else 1, "translate_blocks": 0},
        "pending_task_count": 0 if empty else 2,
        "pending_file_count": 0 if empty else 1,
        "warnings": ["Original fixture warning: inspect local structure."] if warning else [],
        "recommendations": [{"code": rec.ENABLE_PREPARE}] if blocked else [],
    }


def run_doctor_cli(state):
    """Read an original project's real CLI envelope and diagnostic stream."""
    result = subprocess.run(
        [sys.executable, "-B", str(state.get_batch_script_path()), "doctor",
         "--output", "json", "--non-interactive", "--config-file", str(state.config_path)],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=35,
    )
    return cli_contract.parse_result_envelope(result.stdout), result.stderr, result.returncode


class HeldDoctorCli:
    """Control process results while exercising the window's real runner signal wiring."""

    def __init__(self, window):
        self.runner = window.runner
        self.run = self.runner.run
        self.is_active = self.runner.is_active
        self.active = False
        self.args = None
        self.stop_calls = 0

    def install(self, case):
        for name, replacement in (
            ("run", self.start),
            ("is_active", lambda: self.active),
            ("kill", self.stop),
            ("request_stop", self.stop),
        ):
            patch = mock.patch.object(self.runner, name, replacement)
            patch.start()
            case.addCleanup(patch.stop)
        case.addCleanup(self.settle)

    def start(self, script, args):
        self.active = True
        self.args = (script, args)
        return True

    def stop(self):
        self.stop_calls += 1
        return True

    def settle(self):
        if self.active:
            self.finish(exit_code=-1)

    def finish(self, report=None, *, error="", exit_code=0, stdout=None, stderr=""):
        if stdout is None and report is not None:
            stdout = json.dumps(cli_contract.success_envelope("doctor", status="ready", result=report))
        if stdout is None and error:
            stdout = json.dumps(cli_contract.error_envelope("doctor", code="CONFIG_INVALID", message=error))
            exit_code = 1
        if stdout:
            self.runner.stdout_line_ready.emit(stdout)
            self.runner.line_ready.emit(stdout)
        if stderr:
            self.runner.stderr_line_ready.emit(stderr)
            self.runner.line_ready.emit(stderr)
        self.active = False
        self.runner.finished.emit(exit_code)


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class ReadinessSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "work"
        self.tl = self.root / "game" / "tl" / "schinese"
        self.tl.mkdir(parents=True)
        (self.tl / "sample.rpy").write_text(
            'translate schinese strings:\n    old "Follow the paper boat."\n    new ""\n'
            '    old "The lantern is still warm."\n    new ""\n',
            encoding="utf-8",
        )
        state = ProjectState()
        state.config_path = Path(self.tmp.name) / "translator_config.json"
        state.api_keys_path = Path(self.tmp.name) / "empty_keys.json"
        state._game_root = self.root
        self.config = {"game_root": str(self.root), "prepare": {"enabled": False}}
        state.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.window = MainWindow(project_state=state, bootstrap_config=self.config)
        self.addCleanup(self.close_window)
        # Fixed credential availability, never load an actual key.
        self.window.state.get_api_key_status = lambda **kwargs: (1, "fixture")
        self.window._confirm_unsaved_config_before_workflow = lambda: True
        provider_patch = mock.patch(
            "translator_runtime.create_genai_client",
            side_effect=AssertionError("Provider forbidden"),
        )
        self.provider = provider_patch.start()
        self.addCleanup(provider_patch.stop)
        self.doctor = HeldDoctorCli(self.window)
        self.doctor.install(self)

    def close_window(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def page_texts(self):
        return [s.readiness_label.text() for s in self.window._translation_target_sections.values()]

    def assert_pages(self, text):
        for page in self.page_texts():
            self.assertIn(text, page)

    def start_doctor(self):
        self.window._on_run_doctor()
        return self.doctor

    def finish_doctor(self, report=None, *, error=""):
        self.start_doctor().finish(report, error=error)

    def preflight(self, *, status="ready", confirmed=False):
        self.window.runner.run = mock.Mock(return_value=True)
        self.window._on_start_translation()
        payload = {
            "status": status,
            "strategy": "gemini_batch",
            "profile": {"model": "fixture"},
            "project": {"root": str(self.root), "file_count": 1},
            "counts": {"pending_items": 2, "files_with_pending": 1, "chunks": 1},
            "risks": [],
        }
        self.window._translate_preflight_output_lines = [
            json.dumps(
                cli_contract.success_envelope("translate-preflight", status=status, result=payload)
            )
        ]
        with (
            mock.patch(
                "gui_qt.app.message_box_question", return_value="yes" if confirmed else "no"
            ),
            mock.patch("gui_qt.app.message_box_information"),
        ):
            self.window._on_finished(0)

    def test_original_scan_service_reaches_both_pages_without_provider(self):
        runner = self.window.runner
        with (
            mock.patch.object(runner, "run", self.doctor.run),
            mock.patch.object(runner, "is_active", self.doctor.is_active),
        ):
            self.window._on_run_doctor()
            deadline_ms = 30_000
            while self.window._is_doctor_active() and deadline_ms > 0:
                QTest.qWait(25)
                deadline_ms -= 25
            if self.window._is_doctor_active():
                runner._proc.kill()
                QTest.qWait(100)
                self.fail("doctor CLI did not settle")
        self.assertTrue(self.window._doctor_check_completed, self.window._readiness_doctor_error)
        self.assert_pages("待译：2 条")
        self.assertIn("翻译文件：1 个", self.window.doctor_facts_label.text())
        self.provider.assert_not_called()

    def test_doctor_start_uses_saved_config_without_loading_in_process_runtime(self):
        with mock.patch(
            "translator_runtime.load_translator_settings",
            side_effect=AssertionError("GUI must not load the translator runtime for doctor"),
        ) as load:
            worker = self.start_doctor()
        self.assertEqual(worker.args, (
            self.window.state.get_batch_script_path(),
            ["doctor", "--output", "json", "--non-interactive", "--config-file", str(self.window.state.config_path)],
        ))
        load.assert_not_called()
        worker.finish(report_fixture())
        self.provider.assert_not_called()

    def test_doctor_uses_only_stdout_to_publish_readiness(self):
        stdout = json.dumps(cli_contract.success_envelope("doctor", status="ready", result=report_fixture()))
        worker = self.start_doctor()
        worker.finish(stdout=stdout, stderr="Original diagnostic log")
        self.assertTrue(self.window._doctor_check_completed)
        self.window._flush_pending_log_lines()
        self.assertIn("Original diagnostic log", self.window.log_view.toPlainText())
        self.assertNotIn('"schema_version"', self.window.log_view.toPlainText())
        self.start_doctor().finish(stderr=stdout)
        self.assertFalse(self.window._doctor_check_completed)
        self.assert_pages("环境检查失败")

    def test_doctor_displays_structured_cli_configuration_failure(self):
        self.window.state.config_path.write_text(
            json.dumps(dict(self.config, tl_subdir="../outside-project")), encoding="utf-8",
        )
        worker = self.start_doctor()
        envelope, logs, exit_code = run_doctor_cli(self.window.state)
        self.assertFalse(envelope["ok"])
        self.assertNotEqual(exit_code, 0)
        worker.finish(stdout=json.dumps(envelope), stderr=logs, exit_code=exit_code)
        self.assert_pages("环境检查失败")
        self.assertIn("Invalid tl_subdir configuration", self.window.doctor_details_label.text())
        self.assertFalse(self.window._doctor_allows_translate_action())

    def test_cancel_and_timeout_win_over_a_success_envelope(self):
        for stop, message in (
            (self.window._on_kill, "环境检查已取消"),
            (self.window._doctor_timeout_timer.timeout.emit, "环境检查超过 5 分钟"),
        ):
            with self.subTest(message=message):
                self.finish_doctor(report_fixture())
                worker = self.start_doctor()
                stop()
                self.assertTrue(self.window._task_running)
                worker.finish(report_fixture())
                self.assertFalse(self.window._task_running)
                self.assertFalse(self.window._doctor_check_completed)
                self.assertFalse(self.window._doctor_timeout_timer.isActive())
                self.window._flush_pending_log_lines()
                self.assertIn(message, self.window.log_view.toPlainText())
        self.assertEqual(self.doctor.stop_calls, 2)

    def test_empty_unchecked_running_block_warning_ready_failure_stale(self):
        self.window.state._game_root = None
        self.window._sync_readiness_display()
        self.assert_pages("未选择项目")
        self.window.state._game_root = self.root
        self.window._sync_readiness_display()
        self.assert_pages("环境检查未完成")
        worker = self.start_doctor()
        self.assert_pages("环境检查中")
        self.assertNotIn("待译：2", self.window.doctor_facts_label.text())
        worker.finish(report_fixture(blocked=True, empty=True))
        self.assert_pages("有阻塞项")
        self.assertIn("待译：0 条", self.window.doctor_facts_label.text())
        self.assertFalse(self.window.translate_btn.isEnabled())
        self.finish_doctor(report_fixture(warning=True))
        self.assert_pages("有注意事项")
        self.assertIn("fixture warning", self.window.doctor_details_label.text())
        self.finish_doctor(report_fixture())
        self.assert_pages("项目检查通过")
        self.assert_pages("启动预检未运行")
        self.finish_doctor(error="Original fixture exception: scan unavailable")
        self.assert_pages("环境检查失败")
        self.assertIn("scan unavailable", self.window.doctor_details_label.text())
        self.assertNotIn("待译：2", self.window.doctor_facts_label.text())
        self.window.state.config_path.write_text(
            json.dumps(dict(self.config, batch={"model": "fixture-after-failure"})),
            encoding="utf-8",
        )
        self.window._sync_readiness_display()
        self.assert_pages("结果已过期")
        self.assertNotIn("scan unavailable", self.window.doctor_details_label.text())
        self.window._set_doctor_summary(stale_summary())
        self.assert_pages("结果已过期")

    def test_unknown_counts_and_secondary_facts_are_preserved(self):
        partial = {
            "mode": "existing_tl_only",
            "layout_status": "ready",
            "counts": {},
            "recommendations": [{"code": rec.ENABLE_RAG_FOR_CONSISTENCY}],
        }
        self.finish_doctor(partial)
        self.assertIn("待译：未知", self.window.doctor_facts_label.text())
        self.assertNotIn("可选优化", self.window.doctor_facts_label.text())
        self.assertIn("可选优化", self.window.doctor_details_label.text())
        self.assertFalse(self.window.doctor_details_label.isVisible())
        self.window._on_doctor_details_clicked()
        self.assertFalse(self.window.doctor_details_label.isHidden())

    def test_failed_doctor_keeps_visible_summary_without_enabling_translation(self):
        self.finish_doctor(error="Original offline scan failure")
        self.assertFalse(self.window._doctor_check_completed)
        self.assertFalse(self.window.translate_btn.isEnabled())
        self.assertEqual(self.window.doctor_page_stack.currentIndex(), 0)
        self.assertIn("Original offline scan failure", self.window.doctor_details_label.text())
        self.assertIn("未知", self.window.doctor_facts_label.text())
        self.provider.assert_not_called()

    def test_late_doctor_keeps_stale_summary_visible_after_page_refresh(self):
        worker = self.start_doctor()
        self.window.state.config_path.write_text(
            json.dumps(dict(self.config, batch={"model": "original-new-settings"})),
            encoding="utf-8",
        )
        worker.finish(report_fixture())
        self.window._sync_workbench_empty_states()
        self.assertFalse(self.window._doctor_check_completed)
        self.assertEqual(self.window.doctor_page_stack.currentIndex(), 0)
        self.assertIn("已过期", self.window.doctor_message_label.text())
        self.assertNotIn("待译：2", self.window.doctor_facts_label.text())
        self.provider.assert_not_called()

    def test_config_reload_invalidates_both_pages_and_revert_does_not_revive(self):
        self.finish_doctor(report_fixture())
        before = self.window.translate_btn.isEnabled()
        updated = dict(self.config, batch={"model": "fixture-other"})
        self.window.state.config_path.write_text(json.dumps(updated), encoding="utf-8")
        self.window._on_reload_config()
        self.assert_pages("结果已过期")
        self.assertIn("未知", self.window.doctor_facts_label.text())
        self.window.state.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.window._sync_readiness_display()
        self.assert_pages("结果已过期")
        self.assertEqual(self.window.translate_btn.isEnabled(), before)

    def test_project_context_override_change_invalidates_evidence(self):
        from foundation.project_context_settings import save_project_context_settings

        self.finish_doctor(report_fixture())
        save_project_context_settings(self.root, {"rag_enabled": True})
        self.window._sync_readiness_display()
        self.assert_pages("结果已过期")

    def test_late_doctor_after_config_change_cannot_publish_ready(self):
        worker = self.start_doctor()
        self.window.state.config_path.write_text(
            json.dumps(dict(self.config, batch={"model": "other"})), encoding="utf-8"
        )
        worker.finish(report_fixture())
        self.assert_pages("结果已过期")
        self.assertFalse(self.window._doctor_check_completed)

    def test_old_doctor_owns_runner_until_settled_but_cannot_update_new_project(self):
        old = self.start_doctor()
        self.window._switch_game_root(str(self.root.parent / "another" / "work"))
        self.assertEqual(old.stop_calls, 1)
        self.assertFalse(self.window._doctor_timeout_timer.isActive())
        self.window.runner.run = mock.Mock(wraps=old.start)
        self.window._on_run_doctor()
        self.window.runner.run.assert_not_called()
        old.finish(report_fixture(), stderr="Original stale project diagnostic")
        self.assert_pages("结果已过期")
        self.assertIsNone(self.window._readiness_doctor_report)
        self.window._flush_pending_log_lines()
        self.assertNotIn("Original stale project diagnostic", self.window.log_view.toPlainText())
        self.start_doctor().finish(error="new fixture failure")
        self.assert_pages("环境检查失败")

    def test_task_preflight_running_ready_blocked_failed_and_switch(self):
        self.finish_doctor(report_fixture())
        self.window.runner.run = mock.Mock(return_value=True)
        self.window._on_start_translation()
        self.assert_pages("正在启动预检")
        self.window._active_command = ""
        self.window._set_task_running(False)
        self.preflight()
        self.assert_pages("启动预检通过")
        self.assert_pages("不代表可写回")
        self.window._set_work_mode(WorkMode.SYNC_TRANSLATION, refresh_manifest_writeback=False)
        self.assert_pages("预检结果已过期")
        self.window._set_work_mode(WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False)
        self.assert_pages("预检结果已过期")
        self.preflight(status="blocked")
        self.assert_pages("启动预检阻断")
        self.window._on_start_translation()
        self.window._translate_preflight_output_lines = ["Original fixture malformed result"]
        with mock.patch("gui_qt.app.message_box_information"):
            self.window._on_finished(1)
        self.assert_pages("启动预检失败")
        self.provider.assert_not_called()

    def test_preflight_late_result_and_deferred_start_reject_changed_task(self):
        self.finish_doctor(report_fixture())
        self.window.runner.run = mock.Mock(return_value=True)
        self.window._on_start_translation()
        self.window._on_translation_target_selected("fixture-other", "gemini_batch")
        self.window._translate_preflight_output_lines = ["{}"]
        with mock.patch("gui_qt.app.message_box_question") as confirm:
            self.window._on_finished(0)
        confirm.assert_not_called()
        self.assertIsNone(self.window._pending_translation_start)
        self.assert_pages("预检结果已过期")
        self.preflight(confirmed=True)
        self.window._set_work_mode(WorkMode.SYNC_TRANSLATION, refresh_manifest_writeback=False)
        with mock.patch("gui_qt.app.SyncTranslationWorkflow.start_new") as create:
            QApplication.processEvents()
        create.assert_not_called()
        self.assertEqual(self.window.runner.run.call_count, 1)

    def test_old_confirmation_callback_cannot_consume_a_new_preflight(self):
        self.finish_doctor(report_fixture())
        self.preflight(confirmed=True)
        old_identity = self.window._pending_translation_start["readiness_identity"]
        self.window._on_start_translation()
        pending = self.window._pending_translation_start
        self.assertNotEqual(old_identity, pending["readiness_identity"])
        QApplication.processEvents()
        self.assertIs(self.window._pending_translation_start, pending)
        self.assertEqual(self.window._active_command, "translate_preflight")

    def test_real_preflight_service_result_uses_existing_cli_signal_path(self):
        import gemini_translate_batch as batch
        import translator_runtime as runtime
        from tests.cli.test_translate_preflight import fake_context, routing_section

        self.finish_doctor(report_fixture())
        self.window._set_work_mode(WorkMode.SYNC_TRANSLATION, refresh_manifest_writeback=False)
        self.window._sync_work_modes_requiring_api_key = lambda: frozenset()
        section = routing_section()
        self.window.runner.run = mock.Mock(return_value=True)
        self.window._on_start_translation()
        cfg = runtime.default_runtime_config()
        cfg.base_dir = str(self.root)
        cfg.tl_dir = str(self.tl)
        cfg.log_dir = str(self.root.parent / "logs")
        cfg.model_routing_config = section
        with (
            redirect_stdout(io.StringIO()),
            runtime.runtime_config_scope(cfg),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=fake_context(section),
            ),
            mock.patch(
                "providers.model_capability_probe.default_credential_loader", return_value="offline-fixture"
            ),
        ):
            payload = batch.run_translate_preflight(
                batch.build_arg_parser().parse_args(["translate-preflight", "--strategy", "sync"])
            )
        self.window.runner.stdout_line_ready.emit(
            json.dumps(
                cli_contract.success_envelope(
                    "translate-preflight", status=payload["status"], result=payload
                )
            )
        )
        with (
            mock.patch("gui_qt.app.message_box_question", return_value="no"),
            mock.patch("gui_qt.app.message_box_information"),
        ):
            self.window._on_finished(0)
        self.assert_pages("启动预检通过")
        self.assert_pages("待译：3 条")
        self.assertEqual(self.window.runner.run.call_count, 1)
        self.provider.assert_not_called()

    def test_settings_destination_uses_existing_navigation_and_preserves_edits(self):
        self.finish_doctor(report_fixture(blocked=True))
        self.window._focus_settings_section("context")
        page = self.window._context_page()
        page.rag_enabled_cb.setChecked(True)
        self.window._doctor_settings_buttons["project"].click()
        self.assertEqual(self.window.settings_nav.currentRow(), self.window._settings_nav_rows["project"])
        self.window._focus_settings_section("context")
        self.assertTrue(page.rag_enabled_cb.isChecked())

    def test_summary_refresh_has_no_extra_scan_or_provider_requests(self):
        self.finish_doctor(report_fixture())
        with (
            mock.patch("gemini_translate_batch.collect_doctor_report") as scan,
            mock.patch.object(self.window.runner, "run") as cli,
        ):
            for _ in range(5):
                self.window._sync_readiness_display()
                self.window._set_work_mode(
                    WorkMode.SYNC_TRANSLATION, refresh_manifest_writeback=False
                )
                self.window._set_work_mode(
                    WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False
                )
            scan.assert_not_called()
            cli.assert_not_called()
        self.provider.assert_not_called()

    def test_preprocessing_uses_saved_settings_and_refreshes_both_executions(self):
        config = dict(self.config, batch={"source_index": {"enabled": True}}, sync={"rag": {"enabled": True}})
        self.window.state.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.window._refresh_translation_target_choices()
        for section in self.window._translation_target_sections.values():
            self.assertEqual(section.preprocessing_label.text().splitlines()[0], "已开启：原文索引")
        self.window._set_work_mode(WorkMode.SYNC_TRANSLATION, refresh_manifest_writeback=False)
        for section in self.window._translation_target_sections.values():
            self.assertEqual(section.preprocessing_label.text().splitlines()[0], "已开启：历史译文检索（RAG）")
        config["sync"]["rag"]["enabled"] = False
        self.window.state.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.window._load_config_to_ui(pages=set())
        for section in self.window._translation_target_sections.values():
            self.assertIn("已开启：无可选增强", section.preprocessing_label.text())
        self.provider.assert_not_called()

    def test_preprocessing_settings_navigation_preserves_unsaved_context_edits(self):
        self.window._focus_settings_section("context")
        page = self.window._context_page()
        page.rag_enabled_cb.setChecked(True)
        self.window._sync_readiness_display()
        for section in self.window._translation_target_sections.values():
            self.assertIn("未开启：历史译文检索（RAG）", section.preprocessing_label.text())
        self.window._translation_target_sections[WorkMode.BATCH_TRANSLATION].context_settings_btn.click()
        self.assertEqual(self.window.settings_nav.currentRow(), self.window._settings_nav_rows["context"])
        self.assertTrue(page.rag_enabled_cb.isChecked())
        self.assertEqual(json.loads(self.window.state.config_path.read_text(encoding="utf-8")), self.config)
        self.provider.assert_not_called()

    def test_layout_at_small_regular_and_wide_sizes(self):
        from PySide6.QtGui import QFont
        from gui_qt.appearance.theme import apply_theme

        old_stylesheet = self.app.styleSheet()
        old_font = self.app.font()
        self.addCleanup(self.app.setStyleSheet, old_stylesheet)
        self.addCleanup(self.app.setFont, old_font)
        self.app.setFont(QFont("Microsoft YaHei UI", 10))
        apply_theme(self.app, Path(__file__).resolve().parents[3] / "gui_qt" / "resources", "light")
        evidence_dir = os.environ.get("RTL_READINESS_LAYOUT_DIR")
        for width, height in ((960, 640), (1280, 800), (1600, 900)):
            self.window.resize(width, height)
            self.window.show()
            for name, report in (
                ("blocked", report_fixture(blocked=True)),
                ("warning", report_fixture(warning=True)),
                ("ready", report_fixture()),
            ):
                self.finish_doctor(report)
                for _ in range(5):
                    QApplication.processEvents()
                QTest.qWait(150)
                self.assertEqual((self.window.width(), self.window.height()), (width, height))
                self.assertGreater(self.window.doctor_message_label.width(), 250)
                if evidence_dir:
                    path = Path(evidence_dir)
                    path.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(
                        self.window.grab().save(str(path / f"doctor-{name}-{width}x{height}.png"))
                    )
            for mode in (WorkMode.BATCH_TRANSLATION, WorkMode.SYNC_TRANSLATION):
                self.window._set_work_mode(mode, refresh_manifest_writeback=False)
                for _ in range(5):
                    QApplication.processEvents()
                QTest.qWait(150)
                section = self.window._translation_target_sections[mode]
                for label in (section.readiness_label, section.preprocessing_label):
                    self.assertTrue(label.isVisible())
                    self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
                    self.assertTrue(section.rect().contains(label.geometry()))
                if evidence_dir:
                    self.assertTrue(
                        self.window.grab().save(
                            str(
                                Path(evidence_dir)
                                / f"translation-{mode.value}-{width}x{height}.png"
                            )
                        )
                    )


if __name__ == "__main__":
    unittest.main()
