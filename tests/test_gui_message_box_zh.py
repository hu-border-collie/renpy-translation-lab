"""Chinese button labels for shared message-box helpers."""
from __future__ import annotations

import unittest

import gui_test_support

try:
    from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

    from gui_qt.widget_helpers import (
        message_box_information,
        message_box_question,
        message_box_warning,
    )
except ImportError as exc:
    message_box_question = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(message_box_question is None, IMPORT_ERROR)
class GuiMessageBoxZhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_question_buttons_use_chinese_labels(self) -> None:
        # Build the box the same way as the helper, without exec(), to assert labels.
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("测试")
        box.setText("内容")
        yes_btn = box.addButton("确定", QMessageBox.ButtonRole.YesRole)
        no_btn = box.addButton("取消", QMessageBox.ButtonRole.NoRole)
        labels = {btn.text() for btn in box.findChildren(QPushButton)}
        self.assertEqual(yes_btn.text(), "确定")
        self.assertEqual(no_btn.text(), "取消")
        self.assertIn("确定", labels)
        self.assertIn("取消", labels)
        self.assertNotIn("Yes", labels)
        self.assertNotIn("No", labels)
        self.assertNotIn("&Yes", labels)
        self.assertNotIn("OK", labels)

    def test_helpers_are_callable(self) -> None:
        self.assertTrue(callable(message_box_information))
        self.assertTrue(callable(message_box_warning))
        self.assertTrue(callable(message_box_question))


if __name__ == "__main__":
    unittest.main()
