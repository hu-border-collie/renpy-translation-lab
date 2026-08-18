"""Tests for non-batch workbench pages + round-trip sessions (GUI IA P1c / #162)."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.check_report import WritebackSummary
    from gui_qt.doctor_report import DoctorSummary
    from gui_qt.final_review_dialog import FinalReviewFindingsDialog
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    WritebackSummary = None  # type: ignore[misc,assignment]
    DoctorSummary = None  # type: ignore[misc,assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import (
    WorkMode,
    WorkbenchNavItem,
    work_mode_submode_label,
    workbench_nav_spec,
)
from gui_qt.user_copy import PROJECT_ANALYSIS_COPY, REVISION_PROPOSAL_COPY
from gui_qt.workbench import WorkbenchPageActions
from gui_qt.workbench_session import WorkbenchModeSession
from tests import gui_test_support


class TaskPageMetaTests(unittest.TestCase):
    def test_submode_labels_are_short(self) -> None:
        self.assertEqual(work_mode_submode_label(WorkMode.KEYWORD_EXTRACTION), "批量")
        self.assertEqual(work_mode_submode_label(WorkMode.SYNC_KEYWORD_EXTRACTION), "同步")
        self.assertEqual(work_mode_submode_label(WorkMode.REVISION), "批量")
        self.assertEqual(work_mode_submode_label(WorkMode.FINAL_REVIEW), "终审")
        self.assertEqual(work_mode_submode_label(WorkMode.BOOTSTRAP_RAG), "记忆库")

    def test_context_nav_hides_submode_combo(self) -> None:
        self.assertFalse(workbench_nav_spec(WorkbenchNavItem.CONTEXT).show_submode)
        self.assertTrue(workbench_nav_spec(WorkbenchNavItem.KEYWORDS).show_submode)

    def test_session_tracks_ui_snapshots(self) -> None:
        session = WorkbenchModeSession(
            workflow_status="ready",
            workflow_heading="完成",
            workflow_message="done",
        )
        self.assertFalse(session.is_empty())
        self.assertTrue(session.has_workflow_ui())
        facts_only = WorkbenchModeSession(workflow_facts=["project: demo"])
        self.assertFalse(facts_only.is_empty())
        self.assertTrue(facts_only.has_workflow_ui())


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiTaskPageTests(unittest.TestCase):
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

    def test_sync_page_shows_warning_and_start_label(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertFalse(self.window.workbench_stack.isHidden())
        # Project gate (#298/#316): until doctor passes, the page shows the
        # environment-check CTA instead of the task controls.
        self.assertIs(page.page_stack.currentWidget(), page.empty_state)
        self.assertTrue(page.content_page.isHidden())
        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="可以开始同步翻译。",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.assertIs(page.page_stack.currentWidget(), page.content_page)
        self.assertTrue(self.window.workflow_empty_state.isHidden())
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertFalse(page.risk_warning.isHidden())
        self.assertIn("不会修改", page.risk_warning.text())
        self.assertEqual(page.start_btn.text(), "开始同步翻译")
        self.assertEqual(page.start_btn.objectName(), "sync_translation_start_btn")
        self.assertEqual(page.stop_btn.objectName(), "sync_translation_stop_btn")
        self.assertEqual(page.apply_btn.objectName(), "sync_translation_apply_btn")
        self.assertFalse(page.apply_btn.isEnabled())
        self.assertTrue(self.window.sync_mode_warning.isHidden())
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertTrue(hasattr(page, "status_section"))
        self.assertFalse(hasattr(self.window, "workbench_log_drawer"))
        self.assertTrue(self.window.context_library_panel.isHidden())
        self.assertGreaterEqual(
            page.preferred_height(320),
            page.preferred_height(900),
        )

    def test_sync_page_uses_start_stop_callbacks(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        starts: list[bool] = []
        stops: list[bool] = []
        writebacks: list[bool] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                start=lambda: starts.append(True),
                stop=lambda: stops.append(True),
                writeback=lambda: writebacks.append(True),
            )
        )
        page.set_start_enabled(True)
        page.start_btn.click()
        page.set_task_running(True)
        page.stop_btn.click()
        page.set_task_running(False)
        page.set_preview_ready("C:/run/manifest.json")
        page.apply_btn.click()

        self.assertEqual(starts, [True])
        self.assertEqual(stops, [True])
        self.assertEqual(writebacks, [True])
        self.assertFalse(page.start_btn.isEnabled())
        self.assertFalse(page.stop_btn.isEnabled())

    def test_sync_page_start_enabled_after_doctor_summary(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        self.assertFalse(page.start_btn.isEnabled())

        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="可以开始同步翻译。",
                facts=["game_root: demo"],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.assertTrue(self.window.translate_btn.isEnabled())
        self.assertTrue(page.start_btn.isEnabled())
        self.assertEqual(page.start_btn.text(), "开始同步翻译")

        self.window._set_doctor_summary(
            DoctorSummary(
                status="warning",
                heading="可生成模板",
                message="请先生成翻译模板。",
                facts=[],
                findings=[],
                mode="can_generate_template",
            )
        )
        self.assertTrue(page.start_btn.isEnabled())
        self.assertEqual(page.start_btn.text(), "生成翻译模板")

    def test_sync_page_embeds_its_progress_state(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        self.assertFalse(hasattr(page, "render_summary"))
        self.window._set_workflow_summary(
            "running",
            "正在同步翻译",
            "处理中…",
            ["files: 2/10"],
        )
        # Status lives inside the page, not the shared card (#298).
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertIn("正在同步翻译", page.status_section.status_badge.text())
        self.assertIn("files: 2/10", page.status_section.facts_label.text())
        self.assertFalse(page.status_section.isHidden())

    def test_project_analysis_owns_workflow_status_and_progress(self) -> None:
        """#298 review: context workflows never write into the hidden shared card."""
        self.window._set_work_mode(
            WorkMode.PROJECT_ANALYSIS,
            refresh_manifest_writeback=False,
        )
        page = self.window.context_library_page
        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Demo/work",
            project_analysis_enabled=True,
            project_analysis_inject_enabled=False,
        )
        self.window._set_workflow_summary(
            "running",
            "正在构建项目结构",
            "正在读取剧情节点。",
            ["阶段：结构"],
        )
        self.window._workflow_progress_base_facts = ["阶段：结构"]
        self.window._workflow_progress = SimpleNamespace(
            visible=True,
            indeterminate=False,
            total=4,
            current=2,
            label="生成 2/4",
            facts=("当前：路线摘要",),
        )
        self.window._apply_workflow_progress_ui()

        section = page.bootstrap_status_section
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertFalse(section.isHidden())
        self.assertIn("正在构建项目结构", section.status_badge.text())
        self.assertFalse(section.progress_bar.isHidden())
        self.assertEqual(section.progress_bar.value(), 2)
        self.assertIn("阶段：结构", section.facts_label.text())
        self.assertIn("当前：路线摘要", section.facts_label.text())

    def test_task_pages_gate_on_project_prep(self) -> None:
        """#298/#316: non-batch task pages show the doctor CTA until ready."""
        cases = (
            (WorkMode.SYNC_TRANSLATION, "sync_translation_page"),
            (WorkMode.KEYWORD_EXTRACTION, "keywords_page"),
            (WorkMode.REVISION, "revision_page"),
        )
        for mode, attr in cases:
            self.window._set_work_mode(mode, refresh_manifest_writeback=False)
            # Reset doctor state so the gate is exercised for every mode.
            self.window._doctor_check_completed = False
            self.window._doctor_summary_status = ""
            self.window._doctor_summary_mode = ""
            self.window._sync_workbench_empty_states()
            page = getattr(self.window, attr)
            self.assertIs(page.page_stack.currentWidget(), page.empty_state)
            btn = page.empty_state._action_btn
            self.assertIsNotNone(btn)
            assert btn is not None
            self.assertEqual(btn.text(), "去环境检查")
            self.assertEqual(btn.objectName(), "primary_btn")

            self.window._doctor_check_completed = True
            self.window._set_doctor_summary(
                DoctorSummary(
                    status="ready",
                    heading="项目检查通过",
                    message="ok",
                    facts=[],
                    findings=[],
                    mode="existing_tl_only",
                )
            )
            self.assertIs(page.page_stack.currentWidget(), page.content_page)

    def test_task_page_gate_cta_opens_project_prepare(self) -> None:
        """#316: the gate CTA routes to 项目与环境."""
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        self.assertIs(page.page_stack.currentWidget(), page.empty_state)
        # Route switch is synchronous; avoid processEvents so queued layout
        # timers from earlier windows cannot touch stale widgets.
        page.empty_state.action_clicked.emit()
        self.assertEqual(self.window._current_shell_route(), "project_prepare")

    def test_task_empty_state_copy_is_mode_specific(self) -> None:
        """#315: per-page empty copy instead of shared batch wording."""
        cases = (
            (
                WorkMode.SYNC_TRANSLATION,
                "sync_translation_empty_state",
                ("同步翻译", "差异预览", "确认后才写回"),
            ),
            (
                WorkMode.KEYWORD_EXTRACTION,
                "keywords_empty_state",
                ("提取关键词", "候选报告", "合并到 glossary.json"),
            ),
            (
                WorkMode.REVISION,
                "revision_empty_state",
                ("订正预览", "确认预览后才可写回"),
            ),
        )
        for mode, object_name, expected_copy in cases:
            self.window._set_work_mode(mode, refresh_manifest_writeback=False)
            page = self.window.workbench_stack.currentWidget()
            self.assertEqual(page.empty_state.objectName(), object_name)
            self.assertIn("环境检查", page.empty_state._title_label.text())
            description = page.empty_state._desc_label.text()
            for keyword in expected_copy:
                self.assertIn(keyword, description)

    def test_context_rows_do_not_print_unselected_project(self) -> None:
        """#298: no '项目 未选择项目' copy without a project."""
        self.window._set_work_mode(
            WorkMode.BOOTSTRAP_RAG,
            refresh_manifest_writeback=False,
        )
        page = self.window.context_library_page
        page.set_context_status(
            rag_enabled=True,
            source_index_enabled=False,
            game_root="",
            project_analysis_label="未生成",
        )
        self.assertNotIn("未选择项目", page.rag_status_label.text())
        self.assertIn("请先选择项目", page.rag_status_label.text())
        page.set_context_status(
            rag_enabled=True,
            source_index_enabled=False,
            game_root="C:/Games/Demo/work",
            project_analysis_label="未生成",
        )
        self.assertIn("C:/Games/Demo/work", page.rag_status_label.text())
        self.assertNotIn("请先选择项目", page.rag_status_label.text())

    def test_batch_hides_sync_warning(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.assertTrue(self.window.sync_mode_warning.isHidden())

    def test_keywords_page_shows_merge_not_revision(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        # Force writeback buttons for keyword-only path.
        summary = WritebackSummary(
            status="ready",
            heading="关键词完成",
            message="可合并",
            facts=[],
            findings=[],
            can_apply=False,
            manifest_path="C:/kw/manifest.json",
        )
        with mock.patch.object(
            self.window,
            "_resolve_keyword_merge_candidates_path",
            return_value="C:/kw/candidates.json",
        ), mock.patch(
            "gui_qt.app.keyword_merge_ready",
            return_value=(True, "ok"),
        ), mock.patch.object(
            self.window,
            "_resolve_keyword_merge_glossary_path",
            return_value="C:/kw/glossary.json",
        ):
            self.window._set_writeback_summary(summary)

        page = self.window.keywords_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertFalse(self.window.workbench_stack.isHidden())
        self.assertTrue(self.window._mode_frame.isHidden())
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertTrue(hasattr(page, "status_section"))
        self.assertTrue(self.window.keyword_merge_writeback_btn.isHidden())
        self.assertTrue(page.merge_btn.isEnabled())
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.apply_btn.isHidden())
        self.assertFalse(hasattr(self.window, "work_submode_combo"))

    def test_gate_keeps_finished_results_visible(self) -> None:
        """#298 review: a regressed doctor state must not hide finished results."""
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="ok",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.window._set_workflow_summary(
            "done",
            "同步完成",
            "已写入 3 条译文",
            ["files: 3"],
        )
        page = self.window.sync_translation_page
        self.assertIs(page.page_stack.currentWidget(), page.content_page)

        # Doctor state regresses (re-check failed): the gate must not hide
        # the finished result view.
        self.window._doctor_check_completed = False
        self.window._doctor_summary_status = "block"
        self.window._sync_workbench_empty_states()
        self.assertIs(page.page_stack.currentWidget(), page.content_page)

    def test_gate_hides_nonterminal_status_when_doctor_regresses(self) -> None:
        """#298 review: waiting/running snapshots are not finished results."""
        cases = (
            (WorkMode.SYNC_TRANSLATION, "sync_translation_page"),
            (WorkMode.KEYWORD_EXTRACTION, "keywords_page"),
            (WorkMode.REVISION, "revision_page"),
        )
        for mode, attr in cases:
            with self.subTest(mode=mode.value):
                self.window._set_work_mode(mode, refresh_manifest_writeback=False)
                page = getattr(self.window, attr)
                for status in ("waiting", "running"):
                    page.set_project_ready(True)
                    page.set_workflow_status(
                        status,
                        "任务尚未完成",
                        "仍在处理中。",
                        [],
                    )
                    page.set_project_ready(False)
                    self.assertIs(
                        page.page_stack.currentWidget(),
                        page.empty_state,
                    )

    def test_batch_gate_keeps_results_when_doctor_regresses(self) -> None:
        """#298 review: batch results stay visible when doctor state regresses."""
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="ok",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.window._set_workflow_summary(
            "done",
            "翻译完成",
            "结果已下载。",
            [],
        )
        page = self.window.batch_translation_page
        self.assertIs(page.page_stack.currentWidget(), page.content_page)

        self.window._doctor_check_completed = False
        self.window._doctor_summary_status = "block"
        self.window._sync_workbench_empty_states()
        self.assertIs(page.page_stack.currentWidget(), page.content_page)

    def test_batch_gate_keeps_archived_result_when_doctor_regresses(self) -> None:
        """#327 review: an idle completed snapshot remains reachable."""
        from gui_qt.manifest_resume_summary import ManifestWorkflowDisplay

        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="ok",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        manifest_path = "C:/Games/Demo/work/logs/done/manifest.json"
        self.window._completed_manifest_snapshot = {
            "manifest_path": manifest_path,
            "display": ManifestWorkflowDisplay(
                status="done",
                heading="翻译完成",
                message="结果已下载。",
                facts=(),
                timeline_step_key=None,
                workflow=None,
                selected_manifest_path=manifest_path,
                archive_when_idle=True,
            ),
            "split_entries": [],
        }
        self.window._viewing_completed_manifest = False
        self.window._refresh_workflow_idle_summary()
        self.assertEqual(
            self.window.workflow_status_label.property("status"),
            "idle",
        )

        self.window._doctor_check_completed = False
        self.window._doctor_summary_status = "block"
        self.window._sync_workbench_empty_states()

        page = self.window.batch_translation_page
        self.assertIs(page.page_stack.currentWidget(), page.content_page)
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertFalse(self.window.view_last_completed_btn.isHidden())

    def test_batch_gate_hides_nonterminal_status_when_doctor_regresses(self) -> None:
        """#298 review: batch uses the same finished-result test as task pages."""
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window.state.get_game_root = lambda: "C:/Games/Demo/work"  # type: ignore[method-assign]
        self.window._workflow = None
        self.window._writeback_manifest_path = ""
        self.window._doctor_check_completed = False
        self.window._doctor_summary_status = "block"
        page = self.window.batch_translation_page

        for status in ("waiting", "running"):
            with self.subTest(status=status):
                self.window.workflow_status_label.set_status(
                    status,
                    "任务尚未完成",
                )
                self.window._sync_workbench_empty_states(
                    resume_available=(False, ""),
                )
                self.assertIs(
                    page.page_stack.currentWidget(),
                    page.empty_state,
                )

    def test_batch_page_gate_owns_first_use_cta(self) -> None:
        """#316 review: batch page hides the disabled action stack without a project."""
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.batch_translation_page
        self.assertIs(page.page_stack.currentWidget(), page.empty_state)
        self.assertTrue(page.content_page.isHidden())
        btn = page.empty_state._action_btn
        self.assertIsNotNone(btn)
        assert btn is not None
        self.assertEqual(btn.objectName(), "primary_btn")

        self.window._doctor_check_completed = True
        self.window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="ok",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )
        self.assertIs(page.page_stack.currentWidget(), page.content_page)

    def test_keywords_and_revision_route_writeback_into_page(self) -> None:
        """#298 review: writeback results render inside keywords/revision pages."""
        for mode, attr in (
            (WorkMode.KEYWORD_EXTRACTION, "keywords_page"),
            (WorkMode.REVISION, "revision_page"),
        ):
            self.window._set_work_mode(mode, refresh_manifest_writeback=False)
            summary = WritebackSummary(
                status="ready",
                heading="写回可执行",
                message="检查通过，可以写回。",
                facts=["files: 3"],
                findings=["[需处理] 有 1 处问题待确认"],
                can_apply=True,
                manifest_path="C:/m.json",
            )
            self.window._set_writeback_summary(summary)
            page = getattr(self.window, attr)
            self.assertEqual(
                page.status_section.status_badge.property("status"),
                "ready",
            )
            self.assertIn("写回可执行", page.status_section.status_badge.text())
            self.assertIn("检查通过", page.status_section.message_label.text())
            self.assertIn("files: 3", page.status_section.facts_label.text())
            self.assertIn(
                "需处理",
                page.status_section.details_label.text(),
            )

    def test_keywords_page_uses_callbacks_and_local_mode_selector(self) -> None:
        page = self.window.keywords_page
        starts: list[bool] = []
        resumes: list[bool] = []
        stops: list[bool] = []
        merges: list[bool] = []
        selected: list[WorkMode] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                start=lambda: starts.append(True),
                resume=lambda: resumes.append(True),
                stop=lambda: stops.append(True),
                writeback=lambda: merges.append(True),
                select_mode=selected.append,
            )
        )
        page.activate(WorkMode.KEYWORD_EXTRACTION, WorkbenchModeSession())
        page.set_controls(
            start_enabled=True,
            resume_enabled=True,
            resume_visible=True,
            resume_label="继续提取",
            merge_enabled=True,
            merge_message="关键词候选已就绪。",
        )
        page.start_btn.click()
        page.resume_btn.click()
        page.merge_btn.click()
        page.set_task_running(True)
        page.stop_btn.click()
        page.set_task_running(False)
        page.mode_combo.setCurrentIndex(
            page.mode_combo.findData(WorkMode.SYNC_KEYWORD_EXTRACTION.value)
        )

        self.assertEqual(starts, [True])
        self.assertEqual(resumes, [True])
        self.assertEqual(merges, [True])
        self.assertEqual(stops, [True])
        self.assertEqual(selected, [WorkMode.SYNC_KEYWORD_EXTRACTION])

    def test_keywords_page_mode_selector_switches_main_window(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        page = self.window.keywords_page
        page.mode_combo.setCurrentIndex(
            page.mode_combo.findData(WorkMode.SYNC_KEYWORD_EXTRACTION.value)
        )

        self.assertEqual(self.window._work_mode, WorkMode.SYNC_KEYWORD_EXTRACTION)
        self.assertFalse(hasattr(self.window, "work_submode_combo"))
        self.assertFalse(page.mode_combo.isHidden())

    def test_keywords_page_mirrors_waiting_resume_and_running_lock(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        workflow = mock.Mock()
        step = mock.Mock()
        step.key = "status"
        workflow.current_step.return_value = step
        workflow.manifest_path = ""
        self.window._workflow = workflow
        self.window._set_workflow_summary(
            "waiting",
            "正在等待云端结果",
            "可查询状态。",
        )
        page = self.window.keywords_page

        self.assertEqual(page.resume_btn.text(), "查询云端状态")
        self.window._set_task_running(False)
        self.assertFalse(self.window.translate_btn.isEnabled())
        self.assertFalse(page.start_btn.isEnabled())
        self.assertTrue(page.resume_btn.isEnabled())
        self.window._set_task_running(True)
        self.assertFalse(page.mode_combo.isEnabled())
        self.assertFalse(page.start_btn.isEnabled())
        self.assertFalse(page.resume_btn.isEnabled())
        self.assertFalse(page.merge_btn.isEnabled())
        self.assertTrue(page.stop_btn.isEnabled())

    def test_revision_page_shows_apply_revision_not_translation_apply(self) -> None:
        self.window._set_work_mode(
            WorkMode.REVISION,
            refresh_manifest_writeback=False,
        )
        summary = WritebackSummary(
            status="ready",
            heading="订正可写回",
            message="预览通过",
            facts=[],
            findings=[],
            can_apply=True,
            manifest_path="C:/rev/manifest.json",
        )
        self.window._set_writeback_summary(summary)
        page = self.window.revision_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertFalse(self.window.workbench_stack.isHidden())
        self.assertTrue(self.window._mode_frame.isHidden())
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertTrue(hasattr(page, "status_section"))
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertTrue(page.writeback_btn.isEnabled())
        self.assertTrue(self.window.apply_btn.isHidden())
        self.assertTrue(self.window.keyword_merge_writeback_btn.isHidden())

    def test_revision_page_uses_callbacks_and_local_mode_selector(self) -> None:
        page = self.window.revision_page
        starts: list[bool] = []
        resumes: list[bool] = []
        stops: list[bool] = []
        writebacks: list[bool] = []
        actions: list[str] = []
        selected: list[WorkMode] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                start=lambda: starts.append(True),
                resume=lambda: resumes.append(True),
                stop=lambda: stops.append(True),
                writeback=lambda: writebacks.append(True),
                select_mode=selected.append,
                action=actions.append,
            )
        )
        page.activate(WorkMode.REVISION, WorkbenchModeSession())
        page.set_controls(
            start_enabled=True,
            resume_enabled=True,
            resume_visible=True,
            resume_label="继续订正",
            writeback_enabled=True,
            result_message="订正预览已通过。",
        )
        page.start_btn.click()
        self.assertFalse(page.import_proposals_btn.isHidden())
        self.assertEqual(page.import_proposals_btn.text(), REVISION_PROPOSAL_COPY["action"])
        self.assertEqual(
            page.import_proposals_btn.toolTip(), REVISION_PROPOSAL_COPY["tooltip"]
        )
        page.import_proposals_btn.click()
        page.resume_btn.click()
        page.writeback_btn.click()
        page.set_task_running(True)
        page.stop_btn.click()
        page.set_task_running(False)
        page.mode_combo.setCurrentIndex(
            page.mode_combo.findData(WorkMode.SYNC_REVISION.value)
        )

        self.assertEqual(starts, [True])
        self.assertEqual(resumes, [True])
        self.assertEqual(writebacks, [True])
        self.assertEqual(actions, ["import_revision_proposals"])
        self.assertEqual(stops, [True])
        self.assertEqual(selected, [WorkMode.SYNC_REVISION])

    def test_revision_proposal_action_starts_import_workflow(self) -> None:
        proposal_path = "C:/review/proposals.jsonl"
        corpus_manifest_path = "D:/exports/revision_corpus_manifest.json"
        with (
            mock.patch.object(
                self.window,
                "_confirm_unsaved_config_before_workflow",
                return_value=True,
            ),
            mock.patch(
                "gui_qt.app.QFileDialog.getOpenFileName",
                side_effect=[
                    (proposal_path, "JSON Lines (*.jsonl)"),
                    (corpus_manifest_path, "JSON (*.json)"),
                ],
            ),
            mock.patch.object(self.window, "_set_writeback_summary") as set_summary,
            mock.patch.object(self.window, "_clear_log_view") as clear_log,
            mock.patch.object(self.window, "_show_workbench_log_drawer") as show_log,
            mock.patch.object(
                self.window, "_begin_translation_workflow"
            ) as begin_workflow,
        ):
            self.window._on_final_review_page_action("import_revision_proposals")

        workflow = begin_workflow.call_args.args[0]
        self.assertEqual(workflow.proposal_path, proposal_path)
        self.assertEqual(workflow.corpus_manifest_path, corpus_manifest_path)
        self.assertEqual(
            begin_workflow.call_args.kwargs["log_heading"],
            REVISION_PROPOSAL_COPY["running"],
        )
        self.assertEqual(begin_workflow.call_args.kwargs["status_tab"], 1)
        set_summary.assert_called_once()
        clear_log.assert_called_once_with()
        show_log.assert_called_once_with()

    def test_revision_page_mode_selector_and_running_lock(self) -> None:
        self.window._set_work_mode(WorkMode.REVISION, refresh_manifest_writeback=False)
        page = self.window.revision_page
        page.mode_combo.setCurrentIndex(page.mode_combo.findData(WorkMode.SYNC_REVISION.value))
        self.assertEqual(self.window._work_mode, WorkMode.SYNC_REVISION)
        self.window._set_task_running(True)
        self.assertFalse(page.mode_combo.isEnabled())
        self.assertFalse(page.start_btn.isEnabled())
        self.assertFalse(page.resume_btn.isEnabled())
        self.assertFalse(page.writeback_btn.isEnabled())
        self.assertTrue(page.stop_btn.isEnabled())

    def test_final_review_is_integrated_into_revision_page(self) -> None:
        self.window._set_work_mode(WorkMode.FINAL_REVIEW, refresh_manifest_writeback=False)
        page = self.window.revision_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertEqual(self.window._workbench_nav_item, WorkbenchNavItem.REVISION)
        self.assertEqual(page.mode_combo.currentData(), WorkMode.FINAL_REVIEW.value)
        self.assertEqual(page.start_btn.text(), "开始最终审校")
        self.assertFalse(page.review_findings_btn.isHidden())
        self.assertTrue(page.import_proposals_btn.isHidden())
        self.assertEqual(page.writeback_btn.text(), "写回所选订正")
        self.assertIn("人工选择", self.window.work_mode_hint_label.text())

    def test_final_review_project_reset_preserves_resume_contract(self) -> None:
        page = self.window.revision_page
        page.activate(WorkMode.FINAL_REVIEW, WorkbenchModeSession())
        page.reset_project()

        self.assertFalse(page.resume_btn.isHidden())
        self.assertEqual(page.resume_btn.text(), "继续审查")
        self.assertFalse(page.resume_btn.isEnabled())

    def test_final_review_findings_button_requires_an_enabled_action(self) -> None:
        page = self.window.revision_page
        actions: list[str] = []
        page.set_action_callbacks(WorkbenchPageActions(action=actions.append))
        page.activate(WorkMode.FINAL_REVIEW, WorkbenchModeSession())
        page.review_findings_btn.setEnabled(False)
        page._trigger_review_findings()
        self.assertEqual(actions, [])

        page.review_findings_btn.setEnabled(True)
        page.review_findings_btn.click()
        self.assertEqual(actions, ["select_final_review_findings"])

    def test_final_review_dialog_disables_preview_until_selection(self) -> None:
        dialog = FinalReviewFindingsDialog([
            {
                "finding_id": "f1",
                "suggested_revision": "新译文",
                "selection_state": "none",
                "revision_state": "none",
            }
        ])
        self.assertFalse(dialog._ok_button.isEnabled())
        dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
        self.assertTrue(dialog._ok_button.isEnabled())
        dialog.close()
        empty_dialog = FinalReviewFindingsDialog([])
        self.assertFalse(empty_dialog._ok_button.isEnabled())
        empty_dialog.close()


    def test_final_review_context_cache_reloads_after_invalidation(self) -> None:
        self.window._work_mode = WorkMode.FINAL_REVIEW
        manifest_path = Path(__file__)
        package = {"manifest": {"status": "done"}, "findings": []}
        with mock.patch.object(
            self.window.state,
            "get_game_root",
            return_value=manifest_path.parent,
        ), mock.patch.object(
            self.window.state,
            "get_latest_manifest_path_for_mode",
            return_value=manifest_path,
        ), mock.patch(
            "final_review.load_campaign_package",
            return_value=package,
        ) as load:
            self.assertEqual(self.window._final_review_findings_context()[1], package)
            self.assertEqual(self.window._final_review_findings_context()[1], package)
            self.assertEqual(load.call_count, 1)
            self.window._invalidate_manifest_caches(manifest_path)
            self.assertEqual(self.window._final_review_findings_context()[1], package)
            self.assertEqual(load.call_count, 2)

    def test_final_review_action_rechecks_running_guards(self) -> None:
        with mock.patch.object(
            self.window,
            "_confirm_unsaved_config_before_workflow",
            return_value=True,
        ) as confirm, mock.patch.object(
            self.window,
            "_final_review_findings_context",
        ) as context:
            self.window._task_running = True
            self.window._on_final_review_page_action("select_final_review_findings")
            confirm.assert_not_called()
            context.assert_not_called()

            self.window._task_running = False
            self.window.runner.is_running = lambda: True
            self.window._on_final_review_page_action("select_final_review_findings")
            confirm.assert_not_called()
            context.assert_not_called()

            self.window.runner.is_running = lambda: False
            confirm.return_value = False
            self.window._on_final_review_page_action("select_final_review_findings")
            confirm.assert_called_once_with()
            context.assert_not_called()

    def test_batch_page_owns_actions_and_running_lock(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.batch_translation_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertNotIn("apply", page.buttons)
        self.assertFalse(hasattr(page, "writeback_row"))

        actions: list[str] = []
        page.set_action_callbacks(WorkbenchPageActions(action=actions.append))
        page.set_controls(
            {
                "start": (True, True, "开始翻译"),
                "resume": (True, True, "继续翻译"),
                "stop": (True, False, "停止"),
                "split_submit": (True, True, "提交剩余包"),
                "probe": (True, True, "试跑样本请求"),
                "split": (True, True, "拆分翻译包"),
            }
        )
        self.assertFalse(page.buttons["stop"].isHidden())
        self.assertFalse(page.buttons["stop"].isEnabled())
        self.assertFalse(page.split_frame.isHidden())
        self.assertIs(page.buttons["probe"].parentWidget(), page.main_bar)
        self.assertIs(page.buttons["split"].parentWidget(), page.main_bar)
        self.assertFalse(hasattr(page, "more_toggle_btn"))

        page.buttons["start"].click()
        page.buttons["resume"].click()
        page.buttons["split_submit"].click()
        page.buttons["probe"].click()
        page.buttons["split"].click()

        page.set_task_running(True)
        self.assertTrue(page.buttons["start"].isHidden())
        self.assertTrue(page.buttons["resume"].isHidden())
        self.assertFalse(page.buttons["stop"].isHidden())
        self.assertFalse(page.buttons["stop"].isEnabled())
        self.assertTrue(page.split_frame.isHidden())
        page.buttons["stop"].click()

        page.set_controls(
            {
                **page._state.controls,
                "stop": (True, True, page._labels["stop"]),
            }
        )
        self.assertTrue(page.buttons["stop"].isEnabled())
        page.buttons["stop"].click()

        self.assertEqual(
            actions,
            ["start", "resume", "split_submit", "probe", "split", "stop"],
        )
        for action in ("start", "resume", "split_submit", "probe", "split"):
            self.assertFalse(page.buttons[action].isEnabled())
        self.assertTrue(page.buttons["stop"].isEnabled())

    def test_batch_writeback_actions_remain_available_on_result_tab(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._set_writeback_summary(
            WritebackSummary(
                status="safe",
                heading="ready",
                message="ready",
                facts=[],
                findings=[],
                can_apply=True,
                manifest_path="C:/batch/manifest.json",
            )
        )
        self.window._focus_workbench_status_tab(2)

        self.assertFalse(self.window.writeback_primary_bar.isHidden())
        self.assertFalse(self.window.apply_btn.isHidden())
        self.assertTrue(self.window.apply_btn.isEnabled())
        self.assertFalse(self.window.writeback_issues_toggle_btn.isHidden())

    def test_batch_resume_reenables_immediately_when_task_stops(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        class WaitingWorkflow:
            manifest_path = ""

            @staticmethod
            def current_step():
                return object()

        self.window._workflow = WaitingWorkflow()
        self.window._set_task_running(True)
        self.assertFalse(self.window.batch_translation_page.buttons["resume"].isEnabled())

        self.window._set_task_running(False)

        self.assertFalse(self.window.kill_btn.isEnabled())
        self.assertTrue(self.window.resume_btn.isEnabled())
        self.assertTrue(self.window.batch_translation_page.buttons["resume"].isEnabled())

    def test_batch_tools_share_the_primary_responsive_bar(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.batch_translation_page
        page.set_controls(
            {
                "start": (True, True, "开始翻译"),
                "stop": (True, False, "停止"),
                "probe": (True, True, "试跑样本请求"),
                "split": (True, True, "拆分翻译包"),
            }
        )

        for action in ("start", "stop", "probe", "split"):
            self.assertIs(page.buttons[action].parentWidget(), page.main_bar)
            self.assertFalse(page.buttons[action].isHidden())
        self.assertFalse(page.buttons["stop"].isEnabled())
        self.assertFalse(hasattr(page, "more_toggle_btn"))
        self.assertGreaterEqual(
            page.preferred_height(320),
            page.preferred_height(900),
        )
    def test_context_page_shows_compact_status_rows(self) -> None:
        self.window._set_work_mode(
            WorkMode.BOOTSTRAP_RAG,
            refresh_manifest_writeback=False,
        )
        self.assertFalse(self.window.context_library_panel.isHidden())
        self.assertIs(
            self.window.workbench_stack.currentWidget(),
            self.window.context_library_page,
        )
        self.assertFalse(self.window.workbench_stack.isHidden())
        self.assertTrue(self.window._mode_frame.isHidden())
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertTrue(self.window.workbench_status_card.isHidden())
        self.assertFalse(hasattr(self.window, "work_submode_combo"))
        self.assertTrue(self.window.translate_btn.isHidden())
        # Task routes show the compact identity bar; prep actions stay hidden.
        self.assertFalse(self.window.global_project_bar.isHidden())
        self.assertTrue(self.window.doctor_btn.isHidden())
        page = self.window.context_library_page
        self.assertTrue(hasattr(page, "bootstrap_status_section"))
        self.assertEqual(page.rag_status_row.title_label.text(), "记忆库")
        self.assertEqual(page.source_index_status_row.title_label.text(), "原文索引")
        self.assertEqual(page.project_analysis_status_row.title_label.text(), "项目分析")
        self.assertIn("项目", self.window.context_rag_status_label.text())
        self.assertIn("项目", self.window.context_source_index_status_label.text())
        self.assertIn("项目", self.window.context_project_analysis_status_label.text())
        self.assertIs(
            page.bootstrap_rag_btn.parentWidget(),
            page.rag_status_row,
        )
        self.assertIs(
            page.bootstrap_source_index_btn.parentWidget(),
            page.source_index_status_row,
        )
        self.assertEqual(page.project_analysis_generate_btn.text(), "开始分析")
        self.assertEqual(page.project_analysis_review_btn.text(), "审查内容")
        self.assertEqual(page.project_analysis_publish_btn.text(), "启用到翻译")
        self.assertEqual(page.project_analysis_unpublish_btn.text(), "停止用于翻译")

    def test_context_page_uses_callbacks_and_owns_empty_state(self) -> None:
        page = self.window.context_library_page
        prebuilds: list[str] = []
        opens: list[bool] = []
        actions: list[str] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                prebuild=prebuilds.append,
                open_settings=lambda: opens.append(True),
                action=actions.append,
            )
        )

        # No project: the unified doctor gate owns the CTA (#298).
        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="",
            project_analysis_label="未生成",
        )
        self.assertIs(page.page_stack.currentWidget(), page.project_gate_state)
        page.project_gate_state.action_clicked.emit()
        self.assertEqual(actions, ["open_doctor"])

        # Project selected but nothing enabled: settings empty state.
        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_label="未生成",
        )
        self.assertIs(page.page_stack.currentWidget(), page.empty_state)
        page.empty_state.action_clicked.emit()
        self.assertEqual(opens, [True])

        page.set_context_status(
            rag_enabled=True,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_status={
                "overall_status": "missing",
                "store_exists": False,
            },
        )
        page.bootstrap_rag_btn.click()
        self.assertEqual(prebuilds, ["rag"])

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_status={
                "overall_status": "published",
                "store_exists": True,
                "chunk_count": 1,
                "label_count": 0,
                "route_count": 0,
                "brief_status": "published",
                "injectable": True,
            },
        )
        self.assertIs(page.page_stack.currentWidget(), page.status_page)
        self.assertIn("已启用", page.project_analysis_status_label.text())

    def test_project_analysis_actions_follow_product_lifecycle(self) -> None:
        page = self.window.context_library_page
        actions: list[str] = []
        page.set_action_callbacks(WorkbenchPageActions(action=actions.append))

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_enabled=True,
            project_analysis_status={
                "overall_status": "missing",
                "store_exists": False,
            },
        )
        self.assertIs(page.page_stack.currentWidget(), page.status_page)
        self.assertEqual(page.project_analysis_generate_btn.text(), "开始分析")
        self.assertTrue(page.project_analysis_generate_btn.isEnabled())
        self.assertFalse(page.project_analysis_publish_btn.isEnabled())
        page.project_analysis_generate_btn.click()
        self.assertEqual(actions, ["project_analysis_build_structure"])

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_enabled=True,
            project_analysis_status={
                "overall_status": "draft",
                "store_exists": True,
                "structure_present": True,
                "label_count": 0,
                "route_count": 0,
                "brief_draft_present": True,
            },
        )
        self.assertEqual(page.project_analysis_generate_btn.text(), "生成项目摘要")
        self.assertTrue(page.project_analysis_review_btn.isEnabled())
        self.assertTrue(page.project_analysis_publish_btn.isEnabled())
        self.assertFalse(page.project_analysis_unpublish_btn.isEnabled())

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_enabled=True,
            project_analysis_inject_enabled=True,
            project_analysis_status={
                "overall_status": "published",
                "store_exists": True,
                "label_count": 3,
                "route_count": 1,
                "brief_published_present": True,
                "injectable": True,
            },
        )
        self.assertEqual(page.project_analysis_generate_btn.text(), "更新项目摘要")
        self.assertFalse(page.project_analysis_publish_btn.isEnabled())
        self.assertTrue(page.project_analysis_unpublish_btn.isEnabled())
        self.assertIn("当前会用于翻译", page.project_analysis_status_label.text())

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_enabled=True,
            project_analysis_status={
                "overall_status": "stale",
                "store_exists": True,
                "label_count": 3,
            },
        )
        self.assertEqual(page.project_analysis_generate_btn.text(), "重新分析")
        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
            project_analysis_enabled=True,
            project_analysis_status={
                "overall_status": "failed",
                "store_exists": True,
                "structure_present": False,
            },
        )
        self.assertEqual(page.project_analysis_generate_btn.text(), "重新分析")

    def test_context_navigation_uses_background_status_refresh(self) -> None:
        self.window._context_library_status_cache = None
        self.window._context_library_status_job = None
        pool = mock.Mock()
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=Path("C:/Games/Demo/work"),
            ),
            mock.patch.object(
                self.window,
                "_refresh_context_library_panel",
            ) as sync_refresh,
            mock.patch.object(
                self.window.state,
                "load_translator_config",
            ) as load_config,
            mock.patch(
                "gui_qt.app.QThreadPool.globalInstance",
                return_value=pool,
            ),
        ):
            self.window._set_work_mode(
                WorkMode.BOOTSTRAP_RAG,
                refresh_manifest_writeback=False,
            )

        pool.start.assert_called_once()
        sync_refresh.assert_not_called()
        load_config.assert_not_called()
        self.window._context_library_status_job = None
        self.window._context_library_status_pending_base = ""

    def test_project_analysis_primary_action_has_an_icon(self) -> None:
        self.window._refresh_action_icons()

        self.assertFalse(
            self.window.context_library_page.project_analysis_generate_btn.icon().isNull()
        )

    def test_project_analysis_complete_user_journey_actions(self) -> None:
        page = self.window.context_library_page
        actions: list[str] = []
        page.set_action_callbacks(WorkbenchPageActions(action=actions.append))
        common = {
            "rag_enabled": False,
            "source_index_enabled": False,
            "game_root": "C:/Games/Example/work",
            "project_analysis_enabled": True,
        }

        page.set_context_status(
            **common,
            project_analysis_status={"overall_status": "missing", "store_exists": False},
        )
        page.project_analysis_generate_btn.click()

        page.set_context_status(
            **common,
            project_analysis_status={
                "overall_status": "draft",
                "store_exists": True,
                "structure_present": True,
                "brief_draft_present": True,
            },
        )
        page.project_analysis_review_btn.click()
        page.project_analysis_publish_btn.click()

        page.set_context_status(
            **common,
            project_analysis_inject_enabled=True,
            project_analysis_status={
                "overall_status": "published",
                "store_exists": True,
                "structure_present": True,
                "brief_published_present": True,
                "injectable": True,
            },
        )
        page.project_analysis_unpublish_btn.click()

        page.set_context_status(
            **common,
            project_analysis_status={
                "overall_status": "stale",
                "store_exists": True,
                "structure_present": True,
            },
        )
        page.project_analysis_generate_btn.click()

        self.assertEqual(
            actions,
            [
                "project_analysis_build_structure",
                "project_analysis_review",
                "project_analysis_publish",
                "project_analysis_unpublish",
                "project_analysis_build_structure",
            ],
        )
        visible_copy = " ".join(
            [
                page.project_analysis_generate_btn.text(),
                page.project_analysis_generate_btn.toolTip(),
                page.project_analysis_build_btn.toolTip(),
                page.project_analysis_review_btn.text(),
                page.project_analysis_publish_btn.text(),
                page.project_analysis_publish_btn.toolTip(),
                page.project_analysis_unpublish_btn.text(),
                page.project_analysis_unpublish_btn.toolTip(),
            ]
        ).lower()
        self.assertEqual(
            page.project_analysis_build_btn.toolTip(),
            PROJECT_ANALYSIS_COPY["rebuild_tip"],
        )
        for developer_term in (
            "brief",
            "draft",
            "published",
            "fingerprint",
            "label",
            "route",
        ):
            self.assertNotIn(developer_term, visible_copy)

    def test_global_prep_buttons_visible_only_on_project_route(self) -> None:
        """#298: task pages keep compact identity; prep actions stay on 项目与环境."""
        for mode in (
            WorkMode.BATCH_TRANSLATION,
            WorkMode.SYNC_TRANSLATION,
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.REVISION,
            WorkMode.BOOTSTRAP_RAG,
        ):
            with self.subTest(mode=mode):
                self.window._set_work_mode(mode, refresh_manifest_writeback=False)
                self.assertFalse(self.window.global_project_bar.isHidden())
                self.assertTrue(self.window.doctor_btn.isHidden())
                self.assertTrue(self.window.bootstrap_work_btn.isHidden())
                self.assertFalse(self.window.global_switch_project_btn.isHidden())

        self.window._activate_shell_route("project_prepare")
        self.assertFalse(self.window.doctor_btn.isHidden())
        self.assertFalse(self.window.bootstrap_work_btn.isHidden())

    def test_context_bootstrap_buttons_disabled_while_running(self) -> None:
        self.window._set_work_mode(
            WorkMode.BOOTSTRAP_RAG,
            refresh_manifest_writeback=False,
        )
        from gui_qt.context_library_worker import ContextLibraryStatusResult

        collected = ContextLibraryStatusResult(
            base_dir=str(self.window.state.get_game_root() or ""),
            live_fingerprint="fp",
            status={"overall_status": "missing", "store_exists": False},
            label="未生成",
            context_flags={
                "rag_enabled": True,
                "source_index_enabled": True,
                "project_analysis_enabled": False,
                "project_analysis_inject_enabled": False,
            },
        )
        with mock.patch(
            "gui_qt.app.collect_context_library_status",
            return_value=collected,
        ):
            self.window._refresh_context_library_panel()
            self.assertTrue(self.window.context_bootstrap_rag_btn.isEnabled())
            self.assertFalse(self.window.context_library_page.stop_btn.isHidden())
            self.assertFalse(self.window.context_library_page.stop_btn.isEnabled())
            self.window._set_task_running(True)
            self.assertFalse(self.window.context_bootstrap_rag_btn.isEnabled())
            self.assertFalse(self.window.context_bootstrap_source_index_btn.isEnabled())
            self.assertFalse(self.window.context_library_page.stop_btn.isHidden())
            self.assertTrue(self.window.context_library_page.stop_btn.isEnabled())
            self.assertEqual(
                self.window.context_library_page.stop_btn.objectName(),
                "context_library_stop_btn",
            )
            self.window.context_library_page.set_task_running(
                True, "bootstrap_rag"
            )
            self.assertEqual(
                self.window.context_library_page.stop_btn.text(),
                "停止预建",
            )
            # Overlapping start must no-op while a task is already running.
            self.assertFalse(self.window._start_bootstrap_task("source_index"))
            self.window._set_task_running(False)
            self.assertTrue(self.window.context_bootstrap_rag_btn.isEnabled())
            self.assertFalse(self.window.context_library_page.stop_btn.isHidden())
            self.assertFalse(self.window.context_library_page.stop_btn.isEnabled())

    def test_context_status_ready_ignores_stale_config_digest(self) -> None:
        from gui_qt.context_library_worker import ContextLibraryStatusResult
        from gui_qt.operation_identity import context_library_config_digest

        game_root = Path("C:/Games/Current/work")
        current_base = str(game_root)
        current_config = {"batch": {"rag": {"enabled": True}}}
        stale_config = {"batch": {"rag": {"enabled": False}}}
        current = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-now",
            status={"overall_status": "published", "store_exists": True},
            label="当前已启用",
            context_flags={"rag_enabled": True},
            config_digest=context_library_config_digest(current_config),
        )
        stale = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-old",
            status={"overall_status": "missing", "store_exists": False},
            label="过期扫描",
            context_flags={"rag_enabled": False},
            config_digest=context_library_config_digest(stale_config),
        )
        self.window._context_library_config_snapshot = current_config
        self.window._context_library_status_cache = current
        self.window._context_library_status_job = object()
        self.window._render_context_library_panel(
            flags=current.context_flags,
            analysis_flags={"enabled": False, "inject_enabled": False},
            game_root=current_base,
            result=current,
            running=False,
        )
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=game_root,
            ),
            mock.patch("gui_qt.app.QTimer.singleShot") as timer,
        ):
            self.window._on_context_library_status_ready(stale)

        self.assertIsNone(self.window._context_library_status_job)
        self.assertEqual(self.window._context_library_status_cache, current)
        status_text = self.window.context_library_page.project_analysis_status_label.text()
        self.assertIn("当前已启用", status_text)
        self.assertNotIn("过期扫描", status_text)
        timer.assert_called_once()

    def test_context_status_ready_applies_matching_identity(self) -> None:
        from gui_qt.context_library_worker import ContextLibraryStatusResult
        from gui_qt.operation_identity import context_library_config_digest

        game_root = Path("C:/Games/Current/work")
        current_base = str(game_root)
        current_config = {"batch": {"rag": {"enabled": True}}}
        result = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-now",
            status={"overall_status": "published", "store_exists": True},
            label="扫描完成",
            context_flags={"rag_enabled": True},
            config_digest=context_library_config_digest(current_config),
        )
        self.window._context_library_config_snapshot = current_config
        self.window._context_library_status_cache = None
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=game_root,
            ),
            mock.patch("gui_qt.app.QTimer.singleShot") as timer,
        ):
            self.window._on_context_library_status_ready(result)

        self.assertEqual(self.window._context_library_status_cache, result)
        self.assertIn(
            "扫描完成",
            self.window.context_library_page.project_analysis_status_label.text(),
        )
        timer.assert_not_called()

    def test_context_status_ready_ignores_stale_project_flags(self) -> None:
        from gui_qt.context_library_worker import ContextLibraryStatusResult
        from gui_qt.operation_identity import context_library_config_digest

        game_root = Path("C:/Games/Current/work")
        current_base = str(game_root)
        same_config = {"theme": "dark"}
        old_flags = {
            "rag_enabled": False,
            "source_index_enabled": False,
            "bootstrap_on_build": True,
            "project_analysis_enabled": False,
            "project_analysis_inject_enabled": False,
        }
        new_flags = dict(old_flags, rag_enabled=True)
        current = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-now",
            status={"overall_status": "published", "store_exists": True},
            label="当前已启用",
            context_flags=new_flags,
            config_digest=context_library_config_digest(
                same_config,
                context_flags=new_flags,
            ),
        )
        stale = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-old",
            status={"overall_status": "missing", "store_exists": False},
            label="过期扫描",
            context_flags=old_flags,
            config_digest=context_library_config_digest(
                same_config,
                context_flags=old_flags,
            ),
        )
        self.window._context_library_config_snapshot = same_config
        self.window._context_library_flags_cache = (current_base, dict(old_flags))
        self.window._context_library_status_cache = current
        self.window._context_library_status_job = object()
        self.window._render_context_library_panel(
            flags=current.context_flags,
            analysis_flags={"enabled": False, "inject_enabled": False},
            game_root=current_base,
            result=current,
            running=False,
        )
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=game_root,
            ),
            mock.patch(
                "gui_qt.app.read_batch_context_flags",
                return_value=dict(new_flags),
            ),
            mock.patch("gui_qt.app.QTimer.singleShot") as timer,
        ):
            self.window._on_context_library_status_ready(stale)

        self.assertIsNone(self.window._context_library_status_job)
        self.assertEqual(self.window._context_library_status_cache, current)
        self.assertEqual(
            self.window._context_library_flags_cache,
            (current_base, old_flags),
        )
        status_text = self.window.context_library_page.project_analysis_status_label.text()
        self.assertIn("当前已启用", status_text)
        self.assertNotIn("过期扫描", status_text)
        timer.assert_called_once()

    def test_context_status_ready_applies_live_flags_despite_stale_cache(self) -> None:
        from gui_qt.context_library_worker import ContextLibraryStatusResult
        from gui_qt.operation_identity import context_library_config_digest

        game_root = Path("C:/Games/Current/work")
        current_base = str(game_root)
        same_config = {"theme": "dark"}
        old_flags = {
            "rag_enabled": False,
            "source_index_enabled": False,
            "bootstrap_on_build": True,
            "project_analysis_enabled": False,
            "project_analysis_inject_enabled": False,
        }
        new_flags = dict(old_flags, rag_enabled=True)
        result = ContextLibraryStatusResult(
            base_dir=current_base,
            live_fingerprint="fp-now",
            status={"overall_status": "published", "store_exists": True},
            label="扫描完成",
            context_flags=new_flags,
            config_digest=context_library_config_digest(
                same_config,
                context_flags=new_flags,
            ),
        )
        self.window._context_library_config_snapshot = same_config
        self.window._context_library_flags_cache = (current_base, dict(old_flags))
        self.window._context_library_status_cache = None
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value=game_root,
            ),
            mock.patch(
                "gui_qt.app.read_batch_context_flags",
                return_value=dict(new_flags),
            ),
            mock.patch("gui_qt.app.QTimer.singleShot") as timer,
        ):
            self.window._on_context_library_status_ready(result)

        self.assertEqual(self.window._context_library_status_cache, result)
        self.assertEqual(
            self.window._context_library_flags_cache,
            (current_base, new_flags),
        )
        self.assertIn(
            "扫描完成",
            self.window.context_library_page.project_analysis_status_label.text(),
        )
        timer.assert_not_called()

    def test_roundtrip_keyword_candidates_and_merge_button(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.window._keyword_merge_candidates_path = "C:/kw/candidates.json"
        self.window._writeback_manifest_path = "C:/kw/manifest.json"
        summary = WritebackSummary(
            status="ready",
            heading="关键词完成",
            message="可合并",
            facts=["candidates: C:/kw/candidates.json"],
            findings=[],
            can_apply=False,
            manifest_path="C:/kw/manifest.json",
        )
        with mock.patch.object(
            self.window,
            "_resolve_keyword_merge_candidates_path",
            return_value="C:/kw/candidates.json",
        ), mock.patch(
            "gui_qt.app.keyword_merge_ready",
            return_value=(True, "ok"),
        ), mock.patch.object(
            self.window,
            "_resolve_keyword_merge_glossary_path",
            return_value="C:/kw/glossary.json",
        ):
            self.window._set_writeback_summary(summary)

            self.window._set_work_mode(
                WorkMode.BATCH_TRANSLATION,
                refresh_manifest_writeback=False,
            )
            self.assertEqual(self.window._keyword_merge_candidates_path, "")

            self.window._set_work_mode(
                WorkMode.KEYWORD_EXTRACTION,
                refresh_manifest_writeback=True,
            )
            self.assertEqual(
                self.window._keyword_merge_candidates_path,
                "C:/kw/candidates.json",
            )
            self.assertEqual(
                self.window._writeback_manifest_path,
                "C:/kw/manifest.json",
            )
            # Snapshot should restore merge readiness without needing the file on disk.
            self.assertTrue(self.window.keyword_merge_writeback_btn.isHidden())
            self.assertTrue(self.window.keywords_page.merge_btn.isEnabled())

    def test_roundtrip_revision_writeback_state(self) -> None:
        self.window._set_work_mode(
            WorkMode.REVISION,
            refresh_manifest_writeback=False,
        )
        summary = WritebackSummary(
            status="ready",
            heading="订正可写回",
            message="预览通过",
            facts=[],
            findings=[],
            can_apply=True,
            manifest_path="C:/rev/manifest.json",
        )
        self.window._set_writeback_summary(summary)

        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._set_work_mode(
            WorkMode.REVISION,
            refresh_manifest_writeback=True,
        )
        self.assertEqual(self.window._writeback_manifest_path, "C:/rev/manifest.json")
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.revision_page.writeback_btn.isEnabled())

    def test_roundtrip_preserves_workflow_progress_ui(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._set_workflow_summary(
            "ready",
            "同步完成",
            "已写入 3 条译文",
            ["project: demo"],
        )
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=True,
        )
        page = self.window.sync_translation_page
        self.assertEqual(
            page.status_section.status_badge.property("status"),
            "ready",
        )
        self.assertIn(
            "已写入 3 条译文",
            page.status_section.message_label.text(),
        )
        self.assertIn("demo", page.status_section.facts_label.text())

    def test_keywords_submode_uses_short_labels(self) -> None:
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        labels = [
            self.window.keywords_page.mode_combo.itemText(i)
            for i in range(self.window.keywords_page.mode_combo.count())
        ]
        self.assertEqual(labels, ["批量", "同步"])


if __name__ == "__main__":
    unittest.main()
