"""GUI-facing revision corpus contract and page-state regression tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli_contract

from gui_qt.operation_identity import revision_corpus_export_identity
from gui_qt.revision_corpus_report import (
    RevisionCorpusExportResult,
    summarize_revision_corpus_output,
)
from gui_qt.revision_corpus_workflow import RevisionCorpusExportWorkflow
from gui_qt.workbench import WorkbenchPageActions
from gui_qt.workbench_session import WorkbenchModeSession
from gui_qt.work_modes import WorkMode
from tests import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.workbench.revision_page import RevisionPage
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    RevisionPage = None  # type: ignore[assignment,misc]
    GUI_IMPORT_ERROR = exc
else:
    GUI_IMPORT_ERROR = None


def _envelope(*, item_count: int = 2, file_count: int = 1, manifest: str) -> str:
    root = str(Path(manifest).parent)
    return json.dumps(
        cli_contract.success_envelope(
            "export-revision-corpus",
            status="completed",
            result={
                "output_dir": root,
                "corpus_jsonl": str(Path(root) / "revision_corpus.jsonl"),
                "corpus_markdown": str(Path(root) / "revision_corpus.md"),
                "corpus_manifest": manifest,
                "item_count": item_count,
                "file_count": file_count,
                "source_changed_during_scan": False,
            },
            artifacts={
                "corpus_jsonl": str(Path(root) / "revision_corpus.jsonl"),
                "corpus_markdown": str(Path(root) / "revision_corpus.md"),
                "corpus_manifest": manifest,
            },
        ),
        ensure_ascii=False,
    )


class RevisionCorpusReportTests(unittest.TestCase):
    def test_success_consumes_envelope_and_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "revision_corpus_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kind": "revision_corpus",
                        "created_at": "2026-08-23T10:20:30+00:00",
                        "source": {"source_changed_during_scan": False},
                    }
                ),
                encoding="utf-8",
            )

            update, result = summarize_revision_corpus_output(
                _envelope(manifest=str(manifest)),
                0,
            )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(update.status, "done")
        self.assertEqual(result.item_count, 2)
        self.assertEqual(result.file_count, 1)
        self.assertEqual(result.created_at, "2026-08-23T10:20:30+00:00")
        self.assertIn("JSONL：", "\n".join(update.facts))
        self.assertIn("生成时间：2026-08-23", "\n".join(update.facts))

    def test_no_work_is_structured_without_being_presented_as_successful_work(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "revision_corpus_manifest.json"
            manifest.write_text(
                json.dumps({"created_at": "2026-08-23T10:20:30+00:00"}),
                encoding="utf-8",
            )
            update, result = summarize_revision_corpus_output(
                _envelope(item_count=0, file_count=0, manifest=str(manifest)),
                0,
            )

        self.assertEqual(update.status, "done")
        self.assertIn("没有可导出的", update.heading)
        self.assertIsNotNone(result)
        self.assertEqual(result.item_count, 0)

    def test_failed_machine_envelope_has_actionable_chinese_summary(self):
        envelope = cli_contract.error_envelope(
            "export-revision-corpus",
            code="PRECONDITION_FAILED",
            message="TL directory is missing",
            suggested_action="inspect_configuration_and_artifacts",
        )
        update, result = summarize_revision_corpus_output(
            json.dumps(envelope),
            5,
        )

        self.assertIsNone(result)
        self.assertEqual(update.status, "failed")
        self.assertIn("请先按提示", update.message)
        self.assertIn("错误码：PRECONDITION_FAILED", update.facts)

    def test_malformed_stdout_is_rejected_instead_of_parsed_as_free_text(self):
        update, result = summarize_revision_corpus_output(
            "Exported revision corpus: C:/not-a-contract\n",
            0,
        )

        self.assertIsNone(result)
        self.assertEqual(update.status, "failed")
        self.assertIn("机器结果", update.message)


class RevisionCorpusWorkflowTests(unittest.TestCase):
    def test_operation_identity_changes_with_project(self):
        self.assertNotEqual(
            revision_corpus_export_identity(game_root="C:/Games/A/work"),
            revision_corpus_export_identity(game_root="C:/Games/B/work"),
        )

    def test_workflow_requests_existing_machine_contract(self):
        workflow = RevisionCorpusExportWorkflow(
            operation_identity=revision_corpus_export_identity(game_root="C:/Games/A/work")
        )
        step = workflow.current_step()

        self.assertIsNotNone(step)
        assert step is not None
        self.assertEqual(
            step.args,
            [
                "export-revision-corpus",
                "--strict-exit-codes",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_stale_update_discards_result(self):
        workflow = RevisionCorpusExportWorkflow(operation_identity="old")
        update = workflow.stale_update()

        self.assertEqual(update.status, "stale")
        self.assertIsNone(workflow.result)
        self.assertIsNone(workflow.current_step())


@gui_test_support.skip_unless_gui(RevisionPage is None, GUI_IMPORT_ERROR)
class RevisionCorpusPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = RevisionPage()
        self.page.activate(WorkMode.REVISION, WorkbenchModeSession())
        self.page.set_project_ready(True)

    def tearDown(self) -> None:
        self.page.deleteLater()

    def _enable_export(self) -> None:
        self.page.set_controls(
            start_enabled=True,
            resume_enabled=False,
            resume_visible=True,
            resume_label="继续订正",
            writeback_enabled=False,
            result_message="可开始",
            export_enabled=True,
            export_tooltip="可导出",
        )

    def test_export_entry_and_result_card_fit_960_by_640(self) -> None:
        actions: list[str] = []
        self.page.set_action_callbacks(
            WorkbenchPageActions(action=actions.append)
        )
        self._enable_export()
        self.assertTrue(self.page.export_corpus_btn.isEnabled())
        self.page.export_corpus_btn.click()
        self.assertEqual(actions, ["export_revision_corpus"])

        result = RevisionCorpusExportResult(
            status="completed",
            output_dir="C:/exports/revision-corpus",
            jsonl_path="C:/exports/revision-corpus/revision_corpus.jsonl",
            markdown_path="C:/exports/revision-corpus/revision_corpus.md",
            manifest_path="C:/exports/revision-corpus/revision_corpus_manifest.json",
            item_count=12,
            file_count=3,
            created_at="2026-08-23T10:20:30+00:00",
        )
        self.page.set_corpus_export_result(result)
        self.page.resize(960, 640)
        self.page.show()
        self._app.processEvents()

        self.assertTrue(self.page.corpus_result.isVisible())
        self.assertIn("条目数：12", self.page.corpus_result_summary.text())
        self.assertIn("生成时间：2026-08-23", self.page.corpus_result_created_at.text())
        self.assertLessEqual(self.page.preferred_height(960), 640)

    def test_running_and_project_reset_gate_export_result(self) -> None:
        self._enable_export()
        self.assertTrue(self.page.export_corpus_btn.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.export_corpus_btn.isEnabled())
        self.page.set_task_running(False)
        self._enable_export()
        self.page.set_corpus_export_result(
            RevisionCorpusExportResult(
                status="completed",
                output_dir="C:/exports/revision-corpus",
                item_count=1,
                file_count=1,
            )
        )
        self.page.reset_project()
        self.assertFalse(self.page.export_corpus_btn.isEnabled())
        self.assertFalse(self.page.corpus_result.isVisible())
        self.assertIsNone(self.page.corpus_export_result())


@gui_test_support.skip_unless_gui(RevisionPage is None, GUI_IMPORT_ERROR)
class RevisionCorpusPreflightTests(unittest.TestCase):
    def _window(self):
        from gui_qt.app import MainWindow

        window = MainWindow.__new__(MainWindow)
        window._current_work_mode = lambda: WorkMode.REVISION
        window._task_running = False
        window._cli_runner_is_active = lambda: False
        window._doctor_check_completed = True
        window._doctor_allows_translate_action = lambda: True
        window._last_doctor_report_game_root = "C:/Games/Current/work"
        window.state = mock.Mock()
        window.state.get_game_root.return_value = Path("C:/Games/Current/work")
        return window

    def test_no_project_is_disabled_before_click(self) -> None:
        window = self._window()
        window.state.get_game_root.return_value = None

        allowed, message = window._revision_corpus_export_preflight()

        self.assertFalse(allowed)
        self.assertIn("选择项目", message)

    def test_environment_not_ready_is_disabled_before_click(self) -> None:
        window = self._window()
        window._doctor_check_completed = False

        allowed, message = window._revision_corpus_export_preflight()

        self.assertFalse(allowed)
        self.assertIn("环境检查", message)

    def test_running_and_no_work_are_disabled_before_click(self) -> None:
        window = self._window()
        window._task_running = True
        allowed, running_message = window._revision_corpus_export_preflight()
        self.assertFalse(allowed)
        self.assertIn("运行", running_message)

        window._task_running = False
        window._last_doctor_report = {
            "counts": {"rpy_files": 1, "old_lines": 0, "new_lines": 0},
            "translated_task_count": 0,
        }
        allowed, no_work_message = window._revision_corpus_export_preflight()
        self.assertFalse(allowed)
        self.assertIn("没有可导出", no_work_message)

    def test_ready_report_enables_export(self) -> None:
        window = self._window()
        window._last_doctor_report = {
            "counts": {"rpy_files": 1, "old_lines": 2, "new_lines": 2},
            "translated_task_count": 2,
        }

        allowed, message = window._revision_corpus_export_preflight()

        self.assertTrue(allowed)
        self.assertEqual(message, "")

    def test_stale_doctor_count_requires_recheck_after_translation_write(self) -> None:
        window = self._window()
        window._last_doctor_report = {"translated_task_count": 0}
        window._revision_corpus_doctor_report_stale = True

        allowed, message = window._revision_corpus_export_preflight()

        self.assertFalse(allowed)
        self.assertIn("重新运行环境检查", message)

    def test_blank_template_counts_do_not_enable_export(self) -> None:
        window = self._window()
        window._last_doctor_report = {
            "counts": {
                "rpy_files": 1,
                "old_lines": 20,
                "new_lines": 20,
                "commented_original_lines": 20,
            },
        }

        allowed, message = window._revision_corpus_export_preflight()

        self.assertFalse(allowed)
        self.assertIn("没有可导出", message)


if __name__ == "__main__":
    unittest.main()
