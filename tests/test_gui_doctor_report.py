import unittest

from gui_qt.doctor_report import parse_doctor_output, summarize_doctor_output


DOCTOR_OUTPUT = """
============================================================
Gemini Batch Translator (Ren'Py)
============================================================
Doctor report:
- Base dir: C:\\Games\\Example\\work
- TL dir: C:\\Games\\Example\\work\\game\\tl\\schinese (exists: True)
- Language: schinese
- Template generation: unavailable (no command resolved)
- Mode: existing_tl_only
- TL scan: rpy_files=3, translate_blocks=2, string_sections=1, old_lines=10, new_lines=10, commented_original_lines=2
"""


class GuiDoctorReportTests(unittest.TestCase):
    def test_parse_doctor_output_extracts_mode_and_counts(self):
        parsed = parse_doctor_output(DOCTOR_OUTPUT)

        self.assertEqual(parsed["mode"], "existing_tl_only")
        self.assertEqual(parsed["tl_exists"], True)
        self.assertEqual(parsed["counts"]["rpy_files"], 3)
        self.assertEqual(parsed["counts"]["old_lines"], 10)

    def test_successful_existing_tl_without_api_key_is_warning(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=0)

        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.heading, "检查完成，但有需要处理的事项")
        self.assertIn("已有翻译文件", summary.message)
        self.assertTrue(any("API Key" in finding for finding in summary.findings))

    def test_successful_clean_report_is_ready(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=2)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertTrue(any("API Key：已配置 2 个" in fact for fact in summary.facts))
        self.assertEqual(summary.findings, [])

    def test_environment_api_key_source_is_reported(self):
        summary = summarize_doctor_output(
            DOCTOR_OUTPUT,
            exit_code=0,
            api_key_count=1,
            api_key_source="environment",
        )

        self.assertTrue(any("环境变量" in fact for fact in summary.facts))

    def test_blocked_missing_template_is_blocked(self):
        output = DOCTOR_OUTPUT.replace("existing_tl_only", "blocked_missing_template")

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "blocked")
        self.assertEqual(summary.heading, "需要先准备翻译模板")

    def test_nonzero_exit_is_blocked(self):
        summary = summarize_doctor_output("startup failed", exit_code=1, api_key_count=1)

        self.assertEqual(summary.status, "blocked")
        self.assertEqual(summary.heading, "项目检查失败")
        self.assertIn("命令行检查没有正常完成", summary.message)

    def test_doctor_warnings_are_preserved(self):
        output = DOCTOR_OUTPUT + "\nWarnings:\n- old/new line counts differ; string translation blocks may be malformed.\n"

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "warning")
        self.assertTrue(any("old/new line counts differ" in finding for finding in summary.findings))


if __name__ == "__main__":
    unittest.main()
