"""Tests for shared workbench task-page layout primitives."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QWidget

    from gui_qt.workbench.keywords_page import KeywordsPage
    from gui_qt.workbench.revision_page import RevisionPage
    from gui_qt.workbench.task_controls import (
        TaskControlSection,
        TaskPageLayout,
        TaskStatusActionRow,
    )
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    QComboBox = None  # type: ignore[assignment,misc]
    QPushButton = None  # type: ignore[assignment,misc]
    QWidget = None  # type: ignore[assignment,misc]
    KeywordsPage = None  # type: ignore[assignment,misc]
    RevisionPage = None  # type: ignore[assignment,misc]
    TaskControlSection = None  # type: ignore[assignment,misc]
    TaskPageLayout = None  # type: ignore[assignment,misc]
    TaskStatusActionRow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(TaskPageLayout is None, IMPORT_ERROR)
class TaskControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_task_page_layout_uses_shared_spacing_and_notice(self) -> None:
        page = QWidget()
        task_layout = TaskPageLayout(page)
        notice = task_layout.add_notice("请先备份")

        margins = task_layout.root.contentsMargins()
        self.assertEqual(
            (margins.left(), margins.top(), margins.right(), margins.bottom()),
            (0, 0, 0, 0),
        )
        self.assertEqual(task_layout.root.spacing(), 8)
        self.assertEqual(notice.objectName(), "task_page_notice")
        self.assertEqual(notice.property("tone"), "warning")
        self.assertTrue(notice.wordWrap())

    def test_task_section_owns_title_and_responsive_action_bar(self) -> None:
        section = TaskControlSection("翻译任务", role="translation")
        start = QPushButton("开始翻译")
        stop = QPushButton("停止")
        section.add_action(start, min_width=108)
        section.add_action(stop, min_width=80)
        section.finish_setup()

        self.assertEqual(section.objectName(), "task_control_section")
        self.assertEqual(section.property("taskRole"), "translation")
        self.assertEqual(section.property("sectionLevel"), "primary")
        self.assertEqual(section.title_label.text(), "翻译任务")
        self.assertIs(start.parentWidget(), section.action_bar)
        self.assertIs(stop.parentWidget(), section.action_bar)

    def test_status_action_row_expands_detail_and_right_aligns_action(self) -> None:
        action = QPushButton("预建记忆库")
        row = TaskStatusActionRow("记忆库", action)
        row.set_status("已启用 · 项目 C:/Games/Demo/work")

        self.assertEqual(row.objectName(), "task_status_action_row")
        self.assertEqual(row.title_label.text(), "记忆库")
        self.assertIn("已启用", row.status_label.text())
        self.assertIs(action.parentWidget(), row)
        self.assertEqual(action.minimumWidth(), 116)

        row.resize(900, 120)
        row.show()
        self._app.processEvents()
        self.assertGreater(row.status_label.width(), 500)
        self.assertGreaterEqual(action.geometry().right(), row.width() - 13)

        row.resize(340, 120)
        self._app.processEvents()
        self.assertFalse(
            row.status_label.geometry().intersects(action.geometry()),
        )
        row.close()

    def test_mode_selector_is_compact_and_result_hint_wraps(self) -> None:
        page = QWidget()
        task_layout = TaskPageLayout(page)
        combo = QComboBox()
        mode_label = task_layout.add_mode_selector("任务模式：", combo)
        result_hint = task_layout.add_result_hint("完成后可继续处理结果。")

        self.assertEqual(mode_label.objectName(), "task_mode_label")
        self.assertEqual(combo.minimumWidth(), 160)
        self.assertEqual(combo.maximumWidth(), 240)
        self.assertEqual(result_hint.objectName(), "task_result_hint")
        self.assertTrue(result_hint.wordWrap())

    def test_keywords_and_revision_pages_use_shared_action_sections(self) -> None:
        keywords = KeywordsPage()
        revision = RevisionPage()

        self.assertIsInstance(keywords.actions, TaskControlSection)
        self.assertEqual(keywords.actions.property("taskRole"), "keywords")
        self.assertIs(keywords.start_btn.parentWidget(), keywords.actions.action_bar)
        self.assertEqual(keywords.mode_combo.maximumWidth(), 240)
        self.assertEqual(keywords.result_hint.objectName(), "task_result_hint")

        self.assertIsInstance(revision.actions, TaskControlSection)
        self.assertEqual(revision.actions.property("taskRole"), "revision")
        self.assertIs(revision.writeback_btn.parentWidget(), revision.actions.action_bar)
        self.assertEqual(revision.mode_combo.maximumWidth(), 240)
        self.assertEqual(revision.result_hint.objectName(), "task_result_hint")

    def test_keyword_and_revision_actions_wrap_at_narrow_width(self) -> None:
        for page in (KeywordsPage(), RevisionPage()):
            with self.subTest(page=page.objectName()):
                page.resize(360, 180)
                page.show()
                self._app.processEvents()
                page.actions.reflow()
                self._app.processEvents()
                narrow_rows = page.actions.action_bar._row_count

                page.resize(900, 180)
                self._app.processEvents()
                page.actions.reflow()
                self._app.processEvents()
                wide_rows = page.actions.action_bar._row_count

                self.assertGreaterEqual(narrow_rows, 2)
                self.assertEqual(wide_rows, 1)
                page.close()

    def test_secondary_section_exposes_theme_level(self) -> None:
        section = TaskControlSection(
            "拆分任务",
            role="batch_split",
            secondary=True,
        )
        self.assertEqual(section.property("sectionLevel"), "secondary")


if __name__ == "__main__":
    unittest.main()
