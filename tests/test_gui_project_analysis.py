import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from tests import gui_test_support

import project_analysis as pa
from gui_qt.check_report import idle_writeback_summary_for_work_mode
from gui_qt.project_analysis_workflow import (
    ProjectAnalysisWorkflow,
    generation_facts_from_output,
    discover_keyword_summary_path,
)
from gui_qt.translation_workflow import WorkflowUpdate
from gui_qt.work_modes import (
    WorkMode,
    WorkbenchNavItem,
    work_mode_spec,
    workbench_nav_for_work_mode,
)
from gui_qt.workflow_factory import create_workflow

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.project_analysis_review_dialog import (
        ProjectAnalysisReviewDialog,
        build_project_analysis_review_data,
    )
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    build_project_analysis_review_data = None  # type: ignore[assignment]
    REVIEW_IMPORT_ERROR = exc
else:
    REVIEW_IMPORT_ERROR = None


class ProjectAnalysisWorkflowTests(unittest.TestCase):
    def test_full_sequence_is_ingest_build_generate(self):
        workflow = ProjectAnalysisWorkflow.start_new(
            keyword_summary_path="C:/reports/keyword_chunk_summaries.jsonl"
        )

        self.assertEqual(
            workflow.current_step().args,
            [
                "project-analysis-ingest-keywords",
                "--summary-jsonl",
                "C:/reports/keyword_chunk_summaries.jsonl",
            ],
        )
        self.assertTrue(workflow.complete_current_step(0, "{}").should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["project-analysis-build-structure"],
        )
        self.assertTrue(workflow.complete_current_step(0, "{}").should_continue)
        self.assertEqual(workflow.current_step().args, ["project-analysis-generate"])
        update = workflow.complete_current_step(0, "{}")
        self.assertEqual(update.status, "done")
        self.assertFalse(update.should_continue)
        self.assertIn("翻译使用预览", update.message)
        self.assertNotIn("实际注入预览", update.message)

    def test_generation_progress_is_rendered_as_gui_facts(self):
        output = (
            'PROJECT_ANALYSIS_PROGRESS '
            '{"stage":"complete","completed":1,"total":1,'
            '"usage":{"requests":4,"input_tokens":40,"output_tokens":16,'
            '"estimated_cost":0.000104,"currency":"USD"}}\n'
        )
        facts = generation_facts_from_output(output)

        self.assertIn("全部完成 1/1", facts[0])
        self.assertIn("模型请求：4", facts[1])
        self.assertIn("0.000104 USD", facts[2])

        unknown = generation_facts_from_output(
            'PROJECT_ANALYSIS_PROGRESS {"stage":"future","completed":0,"total":1}'
        )
        self.assertIn("生成中 0/1", unknown[0])

    def test_step_keys_match_concrete_requested_sequence(self):
        self.assertEqual(
            ProjectAnalysisWorkflow.start_new(build=False, generate=True).step_keys(),
            ("project-analysis-generate",),
        )
        self.assertEqual(
            ProjectAnalysisWorkflow.start_new(build=True, generate=True).step_keys(),
            ("project-analysis-build-structure", "project-analysis-generate"),
        )

    def test_failed_stage_stops_and_can_restart_from_artifacts(self):
        workflow = ProjectAnalysisWorkflow.start_new(build=True, generate=True)
        update = workflow.complete_current_step(1, "failed")

        self.assertEqual(update.status, "failed")
        self.assertIsNone(workflow.current_step())
        self.assertIn("重新开始", update.message)

    def test_factory_and_navigation_recognize_project_analysis(self):
        workflow = create_workflow(WorkMode.PROJECT_ANALYSIS)

        self.assertIsInstance(workflow, ProjectAnalysisWorkflow)
        self.assertEqual(
            workbench_nav_for_work_mode(WorkMode.PROJECT_ANALYSIS),
            WorkbenchNavItem.CONTEXT,
        )

    def test_result_surface_copy_points_to_real_review_entry(self):
        spec = work_mode_spec(WorkMode.PROJECT_ANALYSIS)
        summary = idle_writeback_summary_for_work_mode(WorkMode.PROJECT_ANALYSIS)

        self.assertEqual(spec.writeback_tab_label, "结果说明")
        self.assertFalse(spec.supports_resume)
        self.assertEqual(spec.resume_button_label, "")
        self.assertIn("上下文库", summary.message)
        self.assertIn("审查内容", summary.message)
        self.assertIn("启用到翻译", summary.message)

    def test_discovery_prefers_newest_current_project_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "game" / "work"
            root.mkdir(parents=True)
            package = Path(tmp) / "package"
            package.mkdir()
            manifest_path = package / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            old = package / "keyword_chunk_summaries.jsonl"
            old.write_text("{}\n", encoding="utf-8")
            copied = root.parent / "extracted_keywords" / "keyword_chunk_summaries.jsonl"
            copied.parent.mkdir()
            copied.write_text("{}\n", encoding="utf-8")
            os.utime(old, (1, 1))
            os.utime(copied, (2, 2))

            result = discover_keyword_summary_path(
                game_root=str(root),
                manifest_path=str(manifest_path),
                manifest={},
            )

            self.assertEqual(Path(result), copied)


@unittest.skipIf(
    build_project_analysis_review_data is None,
    f"GUI dependencies unavailable: {REVIEW_IMPORT_ERROR}",
)
class ProjectAnalysisReviewTests(unittest.TestCase):
    def test_review_timestamp_preserves_freshness_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = pa.ProjectAnalysisStore(tmp)
            store.save_brief_text("draft", published=False)
            manifest = pa.empty_manifest(store_dir=tmp)
            manifest["artifacts"][pa.KIND_PROJECT_BRIEF] = {
                "id": "project_brief",
                "status": pa.STATUS_REVIEW_REQUIRED,
                "draft_present": True,
                "published_present": False,
                "lineage": pa.empty_lineage(source_fingerprint="fresh-fp"),
            }
            store.save_manifest(manifest)

            result = pa.mark_project_brief_reviewed(
                tmp,
                reviewed_at="2026-07-24T00:00:00Z",
            )
            saved = store.load_manifest()
            brief = saved["artifacts"][pa.KIND_PROJECT_BRIEF]

            self.assertEqual(result["reviewed_at"], "2026-07-24T00:00:00Z")
            self.assertEqual(brief["status"], pa.STATUS_REVIEW_REQUIRED)
            self.assertEqual(brief["lineage"]["source_fingerprint"], "fresh-fp")
            self.assertEqual(
                brief["lineage"]["reviewed_at"],
                "2026-07-24T00:00:00Z",
            )

    def test_review_data_keeps_full_diff_and_actual_injection_result(self):
        class FakeStore:
            store_dir = "C:/store"

            def load_brief_text(self, *, published):
                return "published\n" if published else "draft\n" + ("x" * 900)

            def load_manifest(self):
                return {
                    "artifacts": {
                        pa.KIND_PROJECT_BRIEF: {
                            "lineage": pa.empty_lineage(reviewed_at="2026-07-24T00:00:00Z")
                        }
                    }
                }

            def load_summaries(self, kind):
                if kind == pa.KIND_LABEL:
                    return [
                        {
                            "id": "label:start",
                            "kind": kind,
                            "status": "draft",
                            "summary": "evidence summary",
                            "source_files": ["script.rpy"],
                            "line_span": [3, 8],
                            "evidence_item_ids": ["item-1"],
                        }
                    ]
                return []

            def load_routes(self):
                return []

        injection = {
            "text": "published",
            "injectable": True,
            "reason": "",
            "diagnostics": "fingerprint=fresh",
            "status": {},
        }
        with (
            mock.patch(
                "gui_qt.project_analysis_review_dialog.resolve_project_analysis_store",
                return_value=FakeStore(),
            ),
            mock.patch(
                "gui_qt.project_analysis_review_dialog.load_injectable_project_brief",
                return_value=injection,
            ) as load_preview,
        ):
            data = build_project_analysis_review_data(
                base_dir="C:/game/work",
                live_fingerprint="fresh",
                inject_enabled=True,
                max_brief_chars=321,
            )

        self.assertGreater(len(data["draft"]), 500)
        self.assertIn("+draft", data["diff"])
        self.assertEqual(data["records"][0]["line_span"], [3, 8])
        self.assertEqual(data["injection"], injection)
        load_preview.assert_called_once_with(
            "C:/store",
            expected_source_fingerprint="fresh",
            max_chars=321,
            enabled=True,
        )


@unittest.skipIf(
    MainWindow is None,
    f"GUI dependencies unavailable: {REVIEW_IMPORT_ERROR}",
)
class ProjectAnalysisAppGlueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_keyword_summary_choice_handles_yes_no_and_cancel(self):
        self.window.state.get_game_root = mock.Mock(return_value="C:/Games/Demo/work")
        self.window._latest_keyword_extraction_manifest = mock.Mock(
            return_value=("C:/batch/manifest.json", {})
        )
        latest = "C:/batch/keyword_chunk_summaries.jsonl"

        with mock.patch(
            "gui_qt.app.discover_keyword_summary_path",
            return_value=latest,
        ):
            for answer, expected in (
                ("yes", (True, latest)),
                ("no", (True, "")),
                ("cancel", (False, "")),
            ):
                with (
                    self.subTest(answer=answer),
                    mock.patch(
                        "gui_qt.app.message_box_question",
                        return_value=answer,
                    ),
                ):
                    self.assertEqual(
                        self.window._choose_project_analysis_keyword_summary(),
                        expected,
                    )

    def test_cancel_keyword_choice_aborts_workflow_start(self):
        self.window._choose_project_analysis_keyword_summary = mock.Mock(
            return_value=(False, "")
        )
        self.window._begin_translation_workflow = mock.Mock()
        self.window._update_project_analysis_timeline = mock.Mock()

        self.window._start_project_analysis_workflow(
            build=True,
            generate=True,
            offer_keywords=True,
        )

        self.window._begin_translation_workflow.assert_not_called()
        self.window._update_project_analysis_timeline.assert_not_called()

    def test_generate_only_start_uses_concrete_timeline(self):
        self.window._set_work_mode(
            WorkMode.PROJECT_ANALYSIS,
            refresh_manifest_writeback=False,
        )
        self.window._clear_log_view = mock.Mock()
        self.window._show_workbench_log_drawer = mock.Mock()
        self.window._begin_translation_workflow = mock.Mock()

        self.window._start_project_analysis_workflow(
            build=False,
            generate=True,
            offer_keywords=False,
        )

        workflow = self.window._begin_translation_workflow.call_args.args[0]
        self.assertEqual(workflow.step_keys(), ("project-analysis-generate",))
        self.assertEqual(
            self.window.timeline.steps,
            [("project-analysis-generate", "生成摘要")],
        )
        self.assertEqual(self.window._active_command, "project_analysis_workflow")
        self.assertEqual(self.window._task_stop_button_label(), "停止分析")

    def test_completion_auto_opens_review_only_for_done(self):
        scheduled: list[tuple[int, object]] = []
        self.window._set_work_mode(
            WorkMode.PROJECT_ANALYSIS,
            refresh_manifest_writeback=False,
        )
        self.window._refresh_context_library_panel = mock.Mock()
        self.window._refresh_diagnostics_context = mock.Mock()
        self.window._set_workflow_update = mock.Mock()
        self.window._set_writeback_summary = mock.Mock()
        self.window._clear_workflow_progress_ui = mock.Mock()
        self.window._set_task_running = mock.Mock()
        self.window.statusBar().showMessage = mock.Mock()

        with mock.patch.object(
            QTimer,
            "singleShot",
            side_effect=lambda delay, callback: scheduled.append((delay, callback)),
        ):
            for status in ("failed", "done"):
                workflow = ProjectAnalysisWorkflow.start_new(
                    build=False,
                    generate=True,
                )
                workflow.complete_current_step = mock.Mock(
                    return_value=WorkflowUpdate(
                        status=status,
                        heading=status,
                        message=status,
                        facts=[],
                    )
                )
                self.window._workflow = workflow
                self.window._active_command = "project_analysis_workflow"
                self.window._workflow_step_output_lines = []
                self.window._on_workflow_step_finished(0)

        review_callbacks = [
            callback
            for delay, callback in scheduled
            if delay == 0
            and getattr(callback, "__name__", "")
            == "_show_project_analysis_review_dialog"
        ]
        self.assertEqual(len(review_callbacks), 1)

    def test_direct_command_completion_refreshes_context_status(self):
        self.window._append_log = mock.Mock()
        self.window._set_task_running = mock.Mock()
        self.window._refresh_context_library_panel = mock.Mock()
        self.window.statusBar().showMessage = mock.Mock()
        self.window._active_command = "project_analysis_publish"

        self.window._on_finished(0)

        self.window._set_task_running.assert_called_once_with(False)
        self.window._refresh_context_library_panel.assert_called_once_with(running=False)
        self.assertEqual(self.window._active_command, "")

    def test_publish_action_rechecks_gate_and_blocks_run(self):
        class FakeDialog:
            requested_action = "publish"

            def __init__(self, **_kwargs):
                pass

            def exec(self):
                return 0

        self.window.state.get_game_root = mock.Mock(return_value="C:/Games/Demo/work")
        self.window._saved_project_analysis_flags = mock.Mock(
            return_value={"inject_enabled": True}
        )
        self.window._project_analysis_live_fingerprint = mock.Mock(return_value="fp")
        self.window._project_analysis_max_brief_chars = mock.Mock(return_value=4000)
        self.window._project_analysis_publish_gate = mock.Mock(
            return_value=(False, "fp", "stale draft")
        )
        self.window._run_project_analysis_command = mock.Mock()

        with (
            mock.patch("gui_qt.app.ProjectAnalysisReviewDialog", FakeDialog),
            mock.patch("gui_qt.app.message_box_warning") as warning,
        ):
            self.window._show_project_analysis_review_dialog()

        self.window._project_analysis_publish_gate.assert_called_once_with()
        self.window._run_project_analysis_command.assert_not_called()
        warning.assert_called_once()

    def test_unpublish_confirmation_short_circuits_page_and_dialog(self):
        self.window._task_running = False
        self.window.runner.is_running = mock.Mock(return_value=False)
        self.window._confirm_unsaved_config_before_workflow = mock.Mock(return_value=True)
        self.window.state.get_game_root = mock.Mock(return_value="C:/Games/Demo/work")
        self.window._run_project_analysis_command = mock.Mock()

        with mock.patch(
            "gui_qt.app.message_box_question",
            return_value="no",
        ):
            self.window._on_context_library_action("project_analysis_unpublish")
        self.window._run_project_analysis_command.assert_not_called()

        dialog_host = SimpleNamespace(
            requested_action="",
            accept=mock.Mock(),
        )
        with mock.patch(
            "gui_qt.project_analysis_review_dialog.message_box_question",
            return_value="no",
        ):
            ProjectAnalysisReviewDialog._request_unpublish(dialog_host)
        self.assertEqual(dialog_host.requested_action, "")
        dialog_host.accept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
