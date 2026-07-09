"""Tests for non-batch workbench pages + round-trip sessions (GUI IA P1c / #162)."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.check_report import WritebackSummary
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    WritebackSummary = None  # type: ignore[misc,assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from gui_qt.work_modes import (
    WorkMode,
    WorkbenchNavItem,
    work_mode_submode_label,
    workbench_nav_spec,
)
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
        self.assertFalse(self.window.sync_mode_warning.isHidden())
        self.assertIn("备份", self.window.sync_mode_warning.text())
        self.assertEqual(self.window.translate_btn.text(), "开始同步翻译")
        self.assertTrue(self.window.context_library_panel.isHidden())

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

        self.assertFalse(self.window.keyword_merge_writeback_btn.isHidden())
        self.assertTrue(self.window.keyword_merge_writeback_btn.isEnabled())
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.apply_btn.isHidden())
        self.assertFalse(self.window.work_submode_combo.isHidden())

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
        self.assertFalse(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.apply_revision_btn.isEnabled())
        self.assertTrue(self.window.apply_btn.isHidden())
        self.assertTrue(self.window.keyword_merge_writeback_btn.isHidden())

    def test_batch_page_hides_revision_apply(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        summary = WritebackSummary(
            status="ready",
            heading="可写回",
            message="ok",
            facts=[],
            findings=[],
            can_apply=True,
            manifest_path="C:/batch/manifest.json",
        )
        self.window._set_writeback_summary(summary)
        self.assertTrue(self.window.apply_revision_btn.isHidden())
        self.assertFalse(self.window.apply_btn.isHidden())

    def test_context_page_shows_status_cards(self) -> None:
        self.window._set_work_mode(
            WorkMode.BOOTSTRAP_RAG,
            refresh_manifest_writeback=False,
        )
        self.assertFalse(self.window.context_library_panel.isHidden())
        self.assertTrue(self.window.work_submode_combo.isHidden())
        self.assertTrue(self.window.translate_btn.isHidden())
        self.assertTrue(self.window.doctor_btn.isHidden())
        self.assertIn("记忆库", self.window.context_rag_status_label.text())
        self.assertIn("原文索引", self.window.context_source_index_status_label.text())

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
            self.window._set_task_running(True)
            self.assertFalse(self.window.context_bootstrap_rag_btn.isEnabled())
            self.assertFalse(self.window.context_bootstrap_source_index_btn.isEnabled())
            # Overlapping start must no-op while a task is already running.
            self.assertFalse(self.window._start_bootstrap_task("source_index"))
            self.window._set_task_running(False)
            self.assertTrue(self.window.context_bootstrap_rag_btn.isEnabled())

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
            self.assertFalse(self.window.keyword_merge_writeback_btn.isHidden())
            self.assertTrue(self.window.keyword_merge_writeback_btn.isEnabled())

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
        self.assertFalse(self.window.apply_revision_btn.isHidden())
        self.assertTrue(self.window.apply_revision_btn.isEnabled())

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
            self.window.work_submode_combo.itemText(i)
            for i in range(self.window.work_submode_combo.count())
        ]
        self.assertEqual(labels, ["批量", "同步"])


if __name__ == "__main__":
    unittest.main()
