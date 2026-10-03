"""#541 A: offline service results, page wiring, freshness and layout evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests import gui_test_support
import cli_contract
import doctor_recommendations as rec

try:
    from PySide6.QtCore import QObject, Signal, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from gui_qt.app import MainWindow
    from gui_qt.doctor_report import stale_summary
    from gui_qt.doctor_worker import DoctorWorkerResult, run_doctor_check
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


if MainWindow is not None:

    class HeldDoctor(QObject):
        """Control signal delivery independently of a real offline service call."""

        completed = Signal(object)
        finished = Signal()

        def __init__(self, config=None, parent=None):
            super().__init__(parent)
            self.config = config
            self.active = False

        def start(self):
            self.active = True

        def isRunning(self):
            return self.active

        def requestInterruption(self):
            self.active = False

        def finish(self, result):
            self.active = False
            self.completed.emit(result)
            self.finished.emit()


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
        import translator_runtime as runtime

        doctor_config = runtime.default_runtime_config()
        doctor_config.base_dir = str(self.root)
        doctor_config.work_game_dir = str(self.root / "game")
        doctor_config.tl_dir = str(self.tl)
        doctor_config.glossary_file = str(self.root / "glossary.json")
        # Snapshot seam stays offline: do not load developer credentials.
        self.window._snapshot_runtime_config_for_job = lambda **kwargs: doctor_config
        provider_patch = mock.patch(
            "translator_runtime.create_genai_client",
            side_effect=AssertionError("Provider forbidden"),
        )
        self.provider = provider_patch.start()
        self.addCleanup(provider_patch.stop)
        worker_patch = mock.patch("gui_qt.app.DoctorWorker", HeldDoctor)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)

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
        return self.window._doctor_worker

    def finish_doctor(self, report=None, *, error=""):
        worker = self.start_doctor()
        worker.finish(DoctorWorkerResult(not error, report, "", error))

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
        import translator_runtime as runtime

        cfg = runtime.default_runtime_config()
        cfg.base_dir = str(self.root)
        cfg.work_game_dir = str(self.root / "game")
        cfg.tl_dir = str(self.tl)
        cfg.tl_subdir = "game/tl/schinese"
        cfg.glossary_file = str(self.root / "glossary.json")
        worker = self.start_doctor()
        with mock.patch.object(runtime, "TRANSLATOR_CONFIG", str(self.window.state.config_path)):
            result = run_doctor_check(cfg)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.report["pending_task_count"], 2)
        worker.finish(result)
        self.assert_pages("待译：2 条")
        self.assertIn("翻译文件：1 个", self.window.doctor_facts_label.text())
        self.provider.assert_not_called()

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
        worker.finish(DoctorWorkerResult(True, report_fixture(blocked=True, empty=True), ""))
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
        from project_context_settings import save_project_context_settings

        self.finish_doctor(report_fixture())
        save_project_context_settings(self.root, {"rag_enabled": True})
        self.window._sync_readiness_display()
        self.assert_pages("结果已过期")

    def test_late_doctor_after_config_change_cannot_publish_ready(self):
        worker = self.start_doctor()
        self.window.state.config_path.write_text(
            json.dumps(dict(self.config, batch={"model": "other"})), encoding="utf-8"
        )
        worker.finish(DoctorWorkerResult(True, report_fixture(), ""))
        self.assert_pages("结果已过期")
        self.assertFalse(self.window._doctor_check_completed)

    def test_queued_old_doctor_after_project_switch_and_rerun_is_rejected(self):
        old = self.start_doctor()
        # Queue before disconnect, reproducing a real late Qt delivery.
        old.completed.disconnect(self.window._on_doctor_completed)
        old.completed.connect(self.window._on_doctor_completed, Qt.ConnectionType.QueuedConnection)
        old.completed.emit(DoctorWorkerResult(True, report_fixture(), ""))
        self.window._switch_game_root(str(self.root.parent / "another" / "work"))
        current = self.start_doctor()
        QApplication.processEvents()
        self.assert_pages("环境检查中")
        self.assertIs(self.window._doctor_worker, current)
        current.finish(DoctorWorkerResult(False, None, "", "new fixture failure"))
        old.finished.emit()
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
        from tests.test_translate_preflight import fake_context, routing_section

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
            runtime.runtime_config_scope(cfg),
            mock.patch.object(
                runtime,
                "prepare_sync_translation_execution_context",
                return_value=fake_context(section),
            ),
            mock.patch(
                "model_capability_probe.default_credential_loader", return_value="offline-fixture"
            ),
        ):
            payload = batch.run_translate_preflight(
                batch.build_arg_parser().parse_args(["translate-preflight", "--strategy", "sync"])
            )
        self.window._on_cli_line_ready(
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
        self.assertEqual(self.window.settings_category_combo.currentData(), "project")
        self.window._focus_settings_section("context")
        self.assertTrue(page.rag_enabled_cb.isChecked())

    def test_summary_refresh_has_no_extra_scan_or_provider_requests(self):
        self.finish_doctor(report_fixture())
        with (
            mock.patch("gemini_translate_batch.collect_doctor_report") as scan,
            mock.patch("gui_qt.app.DoctorWorker") as worker,
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
            worker.assert_not_called()
            cli.assert_not_called()
        self.provider.assert_not_called()

    def test_layout_at_small_regular_and_wide_sizes(self):
        from PySide6.QtGui import QFont
        from gui_qt.theme import apply_theme

        old_stylesheet = self.app.styleSheet()
        old_font = self.app.font()
        self.addCleanup(self.app.setStyleSheet, old_stylesheet)
        self.addCleanup(self.app.setFont, old_font)
        self.app.setFont(QFont("Microsoft YaHei UI", 10))
        apply_theme(self.app, Path(__file__).resolve().parents[1] / "gui_qt" / "resources", "light")
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
                label = section.readiness_label
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
