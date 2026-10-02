import unittest

from tests import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.doctor_report import DoctorSummary
    from gui_qt.template_generation_report import (
        summarize_template_generation_output,
        template_generation_to_doctor_summary,
    )
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiTranslateButtonRuleTests(unittest.TestCase):
    def test_label_and_template_action_follow_mode_and_readiness(self):
        # These rules only read two fields; no widgets or QApplication needed.
        window = MainWindow.__new__(MainWindow)
        cases = (
            (WorkMode.BATCH_TRANSLATION, "can_generate_template", "生成翻译模板", True),
            (WorkMode.BATCH_TRANSLATION, "existing_tl_only", "开始翻译", False),
            (WorkMode.KEYWORD_EXTRACTION, "can_generate_template", "提取关键词", False),
        )
        for mode, doctor_mode, label, generate_template in cases:
            with self.subTest(mode=mode, doctor_mode=doctor_mode):
                window._work_mode = mode
                window._doctor_summary_mode = doctor_mode
                self.assertEqual(window._translate_button_label(), label)
                self.assertEqual(window._should_generate_template_only(), generate_template)


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiTranslateButtonLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()
        self._app.processEvents()

    def test_doctor_summary_updates_button_label_and_enabled_state(self):
        self.window._work_mode = WorkMode.BATCH_TRANSLATION
        self.window._doctor_check_completed = True
        unknown_output = """
Template generation summary:
- status: pending
- tl_dir: C:\\Games\\Example\\work\\game\\tl\\schinese
- tl_exists: False
- rpy_files: 0
- language: schinese
- message:
"""
        unknown_summary = template_generation_to_doctor_summary(
            summarize_template_generation_output(unknown_output, exit_code=0)
        )
        cases = (
            (
                "template_missing",
                DoctorSummary(
                    status="warning", heading="检查完成", message="模板尚未生成。",
                    facts=[], findings=[], mode="can_generate_template",
                ),
                "生成翻译模板", True,
            ),
            (
                "ready",
                DoctorSummary(
                    status="ready", heading="项目检查通过", message="可以开始翻译。",
                    facts=[], findings=[], mode="existing_tl_only",
                ),
                "开始翻译", True,
            ),
            (
                "blocked",
                DoctorSummary(
                    status="blocked", heading="项目检查失败", message="环境检查失败。",
                    facts=[], findings=[], mode="existing_tl_only",
                ),
                "开始翻译", False,
            ),
            (
                "warning",
                DoctorSummary(
                    status="warning", heading="检查完成", message="记忆库尚未建立。",
                    facts=[], findings=[], mode="existing_tl_only",
                ),
                "开始翻译", True,
            ),
            ("unknown_cli_status", unknown_summary, "生成翻译模板", True),
            (
                "template_generation_failed",
                DoctorSummary(
                    status="blocked", heading="翻译模板生成失败", message="模板生成失败。",
                    facts=[], findings=[], mode="can_generate_template",
                ),
                "生成翻译模板", True,
            ),
            (
                "template_generated",
                DoctorSummary(
                    status="ready", heading="翻译模板已生成", message="可以开始翻译。",
                    facts=["翻译文件：12 个"], findings=[], mode="existing_tl_only",
                ),
                "开始翻译", True,
            ),
        )
        # Exercise real summary-to-button updates on one idle window, including
        # blocked -> warning and template generation -> translation transitions.
        for name, summary, label, enabled in cases:
            with self.subTest(case=name):
                self.window._set_doctor_summary(summary)
                self.assertEqual(self.window.translate_btn.text(), label)
                self.assertEqual(self.window.translate_btn.isEnabled(), enabled)

    def test_batch_translation_button_disabled_without_doctor_check(self):
        self.window._work_mode = WorkMode.BATCH_TRANSLATION
        self.window._doctor_check_completed = False

        self.assertFalse(self.window.translate_btn.isEnabled())

    def test_keyword_mode_does_not_require_doctor_check(self):
        self.window._work_mode = WorkMode.KEYWORD_EXTRACTION
        self.window._doctor_check_completed = False
        self.window._apply_work_mode_ui()

        self.assertTrue(self.window.translate_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
