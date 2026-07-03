import unittest

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.doctor_report import DoctorSummary
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(MainWindow is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
class GuiTranslateButtonLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
    def test_batch_mode_shows_generate_template_when_doctor_mode_requires_it(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_summary_mode = "can_generate_template"

        self.assertEqual(window._translate_button_label(), "生成翻译模板")
        self.assertTrue(window._should_generate_template_only())

    def test_batch_mode_shows_start_translation_when_template_exists(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_summary_mode = "existing_tl_only"

        self.assertEqual(window._translate_button_label(), "开始翻译")
        self.assertFalse(window._should_generate_template_only())

    def test_keyword_mode_keeps_extract_label_even_when_template_missing(self):
        window = MainWindow()
        window._work_mode = WorkMode.KEYWORD_EXTRACTION
        window._doctor_summary_mode = "can_generate_template"

        self.assertEqual(window._translate_button_label(), "提取关键词")
        self.assertFalse(window._should_generate_template_only())

    def test_set_doctor_summary_updates_button_label(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_check_completed = True
        window._set_doctor_summary(
            DoctorSummary(
                status="warning",
                heading="检查完成，但有需要处理的事项",
                message="Ren'Py 模板生成环境可用；翻译模板尚未生成。",
                facts=[],
                findings=[],
                mode="can_generate_template",
            )
        )

        self.assertEqual(window.translate_btn.text(), "生成翻译模板")

    def test_batch_translation_button_disabled_without_doctor_check(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_check_completed = False

        self.assertFalse(window.translate_btn.isEnabled())

    def test_batch_translation_button_enabled_after_doctor_check(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_check_completed = True
        window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="项目检查通过",
                message="work 目录已就绪，可以开始翻译流程。",
                facts=[],
                findings=[],
                mode="existing_tl_only",
            )
        )

        self.assertTrue(window.translate_btn.isEnabled())

    def test_button_switches_to_start_translation_after_template_generated(self):
        window = MainWindow()
        window._work_mode = WorkMode.BATCH_TRANSLATION
        window._doctor_check_completed = True
        window._set_doctor_summary(
            DoctorSummary(
                status="ready",
                heading="翻译模板已生成",
                message="翻译模板已就绪，可以开始翻译流程。",
                facts=["翻译文件：12 个"],
                findings=[],
                mode="existing_tl_only",
            )
        )

        self.assertEqual(window.translate_btn.text(), "开始翻译")
        self.assertTrue(window.translate_btn.isEnabled())

    def test_keyword_mode_does_not_require_doctor_check(self):
        window = MainWindow()
        window._work_mode = WorkMode.KEYWORD_EXTRACTION
        window._doctor_check_completed = False
        window._apply_work_mode_ui()

        self.assertTrue(window.translate_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()