"""Chinese button labels for shared message-box helpers."""
from __future__ import annotations

import unittest
from unittest import mock

import gui_test_support

try:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from gui_qt.widget_helpers import (
        build_information_box,
        build_question_box,
        build_warning_box,
        message_box_information,
        message_box_question,
        message_box_warning,
    )
    from gui_qt.app import MainWindow
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

    def test_build_question_buttons_use_chinese_labels_and_roles(self) -> None:
        box = build_question_box(
            None,
            "测试",
            "内容",
            yes_text="是",
            no_text="否",
            cancel_text="取消",
        )
        labels = {btn.text() for btn in box.buttons()}
        self.assertEqual(labels, {"是", "否", "取消"})
        roles = {box.buttonRole(btn) for btn in box.buttons()}
        self.assertEqual(
            roles,
            {
                QMessageBox.ButtonRole.YesRole,
                QMessageBox.ButtonRole.NoRole,
                QMessageBox.ButtonRole.RejectRole,
            },
        )
        self.assertEqual(box.defaultButton().text(), "是")

    def test_build_question_default_button_follows_default_argument(self) -> None:
        box = build_question_box(
            None,
            "测试",
            "内容",
            yes_text="继续",
            no_text="取消",
            default="no",
        )
        self.assertEqual(box.defaultButton().text(), "取消")

    def test_information_and_warning_builders_use_chinese_ok(self) -> None:
        for builder in (build_information_box, build_warning_box):
            with self.subTest(builder=builder.__name__):
                box = builder(None, "测试", "内容")
                self.assertEqual([btn.text() for btn in box.buttons()], ["确定"])
                self.assertEqual(
                    box.buttonRole(box.buttons()[0]),
                    QMessageBox.ButtonRole.AcceptRole,
                )
                self.assertEqual(box.defaultButton().text(), "确定")

    def test_main_window_font_download_confirm_uses_chinese_buttons(self) -> None:
        window = MainWindow()
        try:
            captured: list[QMessageBox] = []

            def fake_exec(box: QMessageBox) -> int:
                captured.append(box)
                return 0

            with mock.patch.object(QMessageBox, "exec", fake_exec):
                window._on_download_recommended_fonts()
            self.assertEqual(len(captured), 1)
            box = captured[0]
            labels = {btn.text() for btn in box.buttons()}
            self.assertEqual(labels, {"下载", "取消"})
            self.assertIn(
                QMessageBox.ButtonRole.YesRole,
                {box.buttonRole(btn) for btn in box.buttons()},
            )
            self.assertEqual(box.defaultButton().text(), "下载")
        finally:
            gui_test_support.close_main_window(window)
            window.deleteLater()

    def test_helpers_are_callable(self) -> None:
        self.assertTrue(callable(message_box_information))
        self.assertTrue(callable(message_box_warning))
        self.assertTrue(callable(message_box_question))

    def test_message_box_question_maps_clicked_button_to_reply(self) -> None:
        box = build_question_box(
            None,
            "测试",
            "内容",
            yes_text="是",
            no_text="否",
            cancel_text="取消",
        )
        roles = {
            box.buttonRole(btn): btn
            for btn in box.buttons()
        }
        yes_btn = roles[QMessageBox.ButtonRole.YesRole]
        no_btn = roles[QMessageBox.ButtonRole.NoRole]
        cancel_btn = roles[QMessageBox.ButtonRole.RejectRole]

        with mock.patch(
            "gui_qt.widget_helpers.build_question_box",
            return_value=box,
        ):
            with mock.patch.object(QMessageBox, "exec", lambda self: None):
                for btn, expected in (
                    (yes_btn, "yes"),
                    (no_btn, "no"),
                    (cancel_btn, "cancel"),
                ):
                    with self.subTest(button=btn.text()):
                        with mock.patch.object(
                            QMessageBox,
                            "clickedButton",
                            lambda self, target=btn: target,
                        ):
                            self.assertEqual(
                                message_box_question(
                                    None,
                                    "测试",
                                    "内容",
                                    cancel_text="取消",
                                ),
                                expected,
                            )


if __name__ == "__main__":
    unittest.main()
