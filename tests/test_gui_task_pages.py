"""Tests for non-batch workbench pages + round-trip sessions (GUI IA P1c / #162)."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.check_report import WritebackSummary
    from gui_qt.doctor_report import DoctorSummary
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
from gui_qt.workbench import WorkbenchPageActions
from gui_qt.workbench_session import WorkbenchModeSession
from tests import gui_test_support


class TaskPageMetaTests(unittest.TestCase):
    def test_submode_labels_are_short(self) -> None:
        self.assertEqual(work_mode_submode_label(WorkMode.KEYWORD_EXTRACTION), "批量")
        self.assertEqual(work_mode_submode_label(WorkMode.SYNC_KEYWORD_EXTRACTION), "同步")
        self.assertEqual(work_mode_submode_label(WorkMode.REVISION), "批量")
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
        self.window.close()
        self.window.deleteLater()

    def test_sync_page_shows_warning_and_start_label(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        page = self.window.sync_translation_page
        self.assertIs(self.window.workbench_stack.currentWidget(), page)
        self.assertFalse(self.window.workbench_stack.isHidden())
        self.assertFalse(page.risk_warning.isHidden())
        self.assertIn("备份", page.risk_warning.text())
        self.assertEqual(page.start_btn.text(), "开始同步翻译")
        self.assertEqual(page.start_btn.objectName(), "sync_translation_start_btn")
        self.assertEqual(page.stop_btn.objectName(), "sync_translation_stop_btn")
        self.assertTrue(self.window.sync_mode_warning.isHidden())
        self.assertTrue(self.window._workbench_actions_column.isHidden())
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(1))
        self.assertFalse(self.window.workbench_status_tabs.tabBar().isTabVisible(2))
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
        page.set_action_callbacks(
            WorkbenchPageActions(
                start=lambda: starts.append(True),
                stop=lambda: stops.append(True),
            )
        )
        page.set_start_enabled(True)
        page.start_btn.click()
        page.set_task_running(True)
        page.stop_btn.click()

        self.assertEqual(starts, [True])
        self.assertEqual(stops, [True])
        self.assertFalse(page.start_btn.isEnabled())
        self.assertTrue(page.stop_btn.isEnabled())

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
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertEqual(self.window.workbench_status_tabs.currentIndex(), 1)
        self.assertIn("正在同步翻译", self.window.workflow_status_label.text())
        self.assertIn("files: 2/10", self.window.workflow_facts_label.text())

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
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(1))
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(2))
        self.assertTrue(self.window.keyword_merge_writeback_btn.isHidden())
        self.assertTrue(page.merge_btn.isEnabled())
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.apply_btn.isHidden())
        self.assertFalse(hasattr(self.window, "work_submode_combo"))

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
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(1))
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(2))
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
        selected: list[WorkMode] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                start=lambda: starts.append(True),
                resume=lambda: resumes.append(True),
                stop=lambda: stops.append(True),
                writeback=lambda: writebacks.append(True),
                select_mode=selected.append,
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
        self.assertEqual(stops, [True])
        self.assertEqual(selected, [WorkMode.SYNC_REVISION])

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
        self.assertFalse(self.window.workbench_status_card.isHidden())
        self.assertTrue(self.window.workbench_status_tabs.tabBar().isTabVisible(1))
        self.assertFalse(self.window.workbench_status_tabs.tabBar().isTabVisible(2))
        self.assertFalse(hasattr(self.window, "work_submode_combo"))
        self.assertTrue(self.window.translate_btn.isHidden())
        # Project-level prep actions stay on the hidden project-only bar.
        self.assertTrue(self.window.global_project_bar.isHidden())
        page = self.window.context_library_page
        self.assertEqual(page.rag_status_row.title_label.text(), "记忆库")
        self.assertEqual(page.source_index_status_row.title_label.text(), "原文索引")
        self.assertIn("项目", self.window.context_rag_status_label.text())
        self.assertIn("项目", self.window.context_source_index_status_label.text())
        self.assertIs(
            page.bootstrap_rag_btn.parentWidget(),
            page.rag_status_row,
        )
        self.assertIs(
            page.bootstrap_source_index_btn.parentWidget(),
            page.source_index_status_row,
        )

    def test_context_page_uses_callbacks_and_owns_empty_state(self) -> None:
        page = self.window.context_library_page
        prebuilds: list[str] = []
        opens: list[bool] = []
        page.set_action_callbacks(
            WorkbenchPageActions(
                prebuild=prebuilds.append,
                open_settings=lambda: opens.append(True),
            )
        )

        page.set_context_status(
            rag_enabled=False,
            source_index_enabled=False,
            game_root="",
        )
        self.assertIs(page.page_stack.currentWidget(), page.empty_state)
        page.empty_state.action_clicked.emit()
        self.assertEqual(opens, [True])

        page.set_context_status(
            rag_enabled=True,
            source_index_enabled=False,
            game_root="C:/Games/Example/work",
        )
        page.bootstrap_rag_btn.click()
        self.assertEqual(prebuilds, ["rag"])

    def test_global_prep_buttons_visible_on_all_task_pages(self) -> None:
        for mode in (
            WorkMode.BATCH_TRANSLATION,
            WorkMode.SYNC_TRANSLATION,
            WorkMode.KEYWORD_EXTRACTION,
            WorkMode.REVISION,
            WorkMode.BOOTSTRAP_RAG,
        ):
            with self.subTest(mode=mode):
                self.window._set_work_mode(mode, refresh_manifest_writeback=False)
                self.assertFalse(self.window.doctor_btn.isHidden())
                self.assertFalse(self.window.bootstrap_work_btn.isHidden())

    def test_context_bootstrap_buttons_disabled_while_running(self) -> None:
        self.window._set_work_mode(
            WorkMode.BOOTSTRAP_RAG,
            refresh_manifest_writeback=False,
        )
        with mock.patch.object(
            self.window,
            "_saved_batch_context_flags",
            return_value={"rag_enabled": True, "source_index_enabled": True},
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
            # Overlapping start must no-op while a task is already running.
            self.assertFalse(self.window._start_bootstrap_task("source_index"))
            self.window._set_task_running(False)
            self.assertTrue(self.window.context_bootstrap_rag_btn.isEnabled())
            self.assertFalse(self.window.context_library_page.stop_btn.isHidden())
            self.assertFalse(self.window.context_library_page.stop_btn.isEnabled())

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
        self.assertEqual(
            self.window.workflow_status_label.property("status"),
            "ready",
        )
        self.assertIn("已写入 3 条译文", self.window.workflow_message_label.text())
        self.assertIn("demo", self.window.workflow_facts_label.text())

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
