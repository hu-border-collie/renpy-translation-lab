import unittest

from gui_qt.template_generation_report import (
    summarize_template_generation_output,
    template_generation_to_doctor_summary,
)


TEMPLATE_OUTPUT = """
Template generation summary:
- status: ready
- tl_dir: C:\\Games\\Example\\work\\game\\tl\\schinese
- tl_exists: True
- rpy_files: 12
- language: schinese
- message: Translation template ready with 12 TL file(s).
"""


class GuiTemplateGenerationReportTests(unittest.TestCase):
    def test_summarize_ready_template_generation_output(self):
        summary = summarize_template_generation_output(TEMPLATE_OUTPUT, exit_code=0)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.rpy_files, 12)
        self.assertTrue(any("翻译文件：12 个" in fact for fact in summary.facts))
        self.assertTrue(any("目标语言：schinese" in fact for fact in summary.facts))

    def test_summarize_failed_template_generation_output(self):
        output = TEMPLATE_OUTPUT.replace("status: ready", "status: failed")

        summary = summarize_template_generation_output(output, exit_code=1)

        self.assertEqual(summary.status, "blocked")
        self.assertEqual(summary.heading, "翻译模板生成失败")

    def test_summarize_unknown_status_is_blocked(self):
        output = """
Template generation summary:
- status: pending
- tl_dir: C:\\Games\\Example\\work\\game\\tl\\schinese
- tl_exists: True
- rpy_files: 0
- language: schinese
- message:
"""

        summary = summarize_template_generation_output(output, exit_code=0)

        self.assertEqual(summary.status, "blocked")
        self.assertIn("未知状态", summary.message)

    def test_template_generation_to_doctor_summary_keeps_generate_mode_on_failure(self):
        output = TEMPLATE_OUTPUT.replace("status: ready", "status: failed")
        summary = summarize_template_generation_output(output, exit_code=1)
        doctor_summary = template_generation_to_doctor_summary(summary)

        self.assertEqual(doctor_summary.mode, "can_generate_template")

    def test_template_generation_to_doctor_summary_switches_to_existing_tl_mode(self):
        summary = summarize_template_generation_output(TEMPLATE_OUTPUT, exit_code=0)
        doctor_summary = template_generation_to_doctor_summary(summary)

        self.assertEqual(doctor_summary.mode, "existing_tl_only")
        self.assertTrue(
            any("重新运行「环境检查」" in fact for fact in doctor_summary.facts)
        )


if __name__ == "__main__":
    unittest.main()