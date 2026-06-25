import unittest

from gui_qt.doctor_report import (
    format_tl_scan_facts,
    parse_doctor_output,
    stale_summary,
    summarize_doctor_output,
)
from gui_qt.user_copy import format_doctor_recommendation_fact


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
- Pending translation: task_count=5, file_count=2
"""

GENERATE_TEMPLATE_OUTPUT = """
============================================================
Gemini Batch Translator (Ren'Py)
============================================================
Doctor report:
- Base dir: C:\\Games\\Example\\work
- TL dir: C:\\Games\\Example\\work\\game\\tl\\schinese (exists: False)
- Language: schinese
- Template generation: available (renpy-sdk)
- Template command: C:\\RenPy\\renpy.exe C:\\Games\\Example\\work\\game translate schinese
- Mode: can_generate_template
- Is work root: True
- Layout status: attention
- TL scan: rpy_files=0, translate_blocks=0, string_sections=0, old_lines=0, new_lines=0, commented_original_lines=0
"""

READY_OUTPUT = """
Doctor report:
- Base dir: C:\\Games\\Example\\work
- TL dir: C:\\Games\\Example\\work\\game\\tl\\schinese (exists: True)
- Language: schinese
- Template generation: unavailable (no command resolved)
- Mode: existing_tl_only
- Is work root: True
- Layout status: ready
- TL scan: rpy_files=3, translate_blocks=2, string_sections=1, old_lines=10, new_lines=10, commented_original_lines=2
"""

FAILED_OUTPUT = """
Doctor report:
- Base dir: C:\\Games\\Example
- TL dir: C:\\Games\\Example\\game\\tl\\schinese (exists: False)
- Language: schinese
- Template generation: unavailable (no command resolved)
- Mode: blocked_missing_template
- Is work root: False
- Work dir: C:\\Games\\Example\\work (exists: False, empty: True)
- Original game dir: (not found)
- Layout status: failed
- TL scan: rpy_files=0, translate_blocks=0, string_sections=0, old_lines=0, new_lines=0, commented_original_lines=0
"""

FAILED_ON_WORK_OUTPUT = """
Doctor report:
- Base dir: C:\\Games\\Example\\work
- TL dir: C:\\Games\\Example\\work\\game\\tl\\schinese (exists: False)
- Language: schinese
- Template generation: unavailable (no command resolved)
- Mode: blocked_missing_template
- Is work root: True
- Work dir: C:\\Games\\Example\\work (exists: True, empty: True)
- Original game dir: (not found)
- Layout status: failed
- TL scan: rpy_files=0, translate_blocks=0, string_sections=0, old_lines=0, new_lines=0, commented_original_lines=0
"""

SWITCH_TO_WORK_OUTPUT = """
Doctor report:
- Base dir: C:\\Games\\Example
- TL dir: C:\\Games\\Example\\game\\tl\\schinese (exists: False)
- Language: schinese
- Template generation: available (sdk)
- Mode: can_generate_template
- Is work root: False
- Work dir: C:\\Games\\Example\\work (exists: False, empty: True)
- Original game dir: C:\\Games\\Example\\original\\game
- Layout status: switch_to_work
- TL scan: rpy_files=0, translate_blocks=0, string_sections=0, old_lines=0, new_lines=0, commented_original_lines=0
Recommendations:
- game_root should use work directory; switch to C:\\Games\\Example\\work
"""


class GuiDoctorReportTests(unittest.TestCase):
    def test_parse_doctor_output_extracts_mode_and_counts(self):
        parsed = parse_doctor_output(DOCTOR_OUTPUT)

        self.assertEqual(parsed["mode"], "existing_tl_only")
        self.assertEqual(parsed["tl_exists"], True)
        self.assertEqual(parsed["counts"]["rpy_files"], 3)
        self.assertEqual(parsed["counts"]["old_lines"], 10)
        self.assertEqual(parsed["pending"]["task_count"], 5)
        self.assertEqual(parsed["pending"]["file_count"], 2)

    def test_successful_existing_tl_without_api_key_is_warning(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=0)

        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.heading, "检查完成，但有需要处理的事项")
        self.assertIn("已有翻译文件", summary.message)
        self.assertTrue(any("API 密钥" in fact or fact.startswith("建议：") for fact in summary.facts))
        self.assertEqual(summary.findings, [])

    def test_successful_clean_report_is_ready(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=2)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertTrue(any("API 密钥：已配置 2 个" in fact for fact in summary.facts))
        self.assertTrue(any("翻译文件：3 个" in fact for fact in summary.facts))
        self.assertTrue(any("待翻译条目：约 5 条" in fact for fact in summary.facts))
        self.assertTrue(any("剧情对话：2 条" in fact for fact in summary.facts))
        self.assertTrue(any("界面字符串：10 条" in fact for fact in summary.facts))
        self.assertFalse(any("old/new 行数" in fact for fact in summary.facts))
        self.assertEqual(summary.findings, [])

    def test_can_generate_template_without_tl_files_is_warning(self):
        summary = summarize_doctor_output(
            GENERATE_TEMPLATE_OUTPUT,
            exit_code=0,
            api_key_count=1,
        )

        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.heading, "检查完成，但有需要处理的事项")
        self.assertEqual(summary.findings, [])
        self.assertTrue(any("翻译模板尚未生成" in summary.message for _ in [0]))
        self.assertTrue(any("翻译文件：0 个" in fact for fact in summary.facts))

    def test_can_generate_template_with_empty_tl_dir_is_warning(self):
        output = GENERATE_TEMPLATE_OUTPUT.replace("(exists: False)", "(exists: True)")

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "warning")
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
        summary = summarize_doctor_output(FAILED_OUTPUT, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "blocked")
        self.assertEqual(summary.heading, "项目检查失败")
        self.assertIn("original/game", summary.message)
        self.assertTrue(any("work 目录：不存在" in fact for fact in summary.facts))
        self.assertTrue(any("original/game：不存在" in fact for fact in summary.facts))

    def test_failed_on_work_root_uses_work_specific_message(self):
        summary = summarize_doctor_output(FAILED_ON_WORK_OUTPUT, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "blocked")
        self.assertIn("work 目录为空", summary.message)
        self.assertNotIn("当前项目目录下没有 work 目录", summary.message)

    def test_ready_layout_status_is_green(self):
        summary = summarize_doctor_output(READY_OUTPUT, exit_code=0, api_key_count=2)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertIn("就绪", summary.message)

    def test_switch_to_work_layout_status_is_warning(self):
        summary = summarize_doctor_output(SWITCH_TO_WORK_OUTPUT, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.heading, "建议使用 work 目录")
        self.assertTrue(any("切换到" in fact for fact in summary.facts))
        self.assertTrue(any("original/game：存在" in fact for fact in summary.facts))

    def test_nonzero_exit_is_blocked(self):
        summary = summarize_doctor_output("startup failed", exit_code=1, api_key_count=1)

        self.assertEqual(summary.status, "blocked")
        self.assertEqual(summary.heading, "项目检查失败")
        self.assertIn("环境检查没有正常完成", summary.message)

    def test_doctor_warnings_are_preserved(self):
        output = DOCTOR_OUTPUT + "\nWarnings:\n- old/new line counts differ; string translation blocks may be malformed.\n"

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)

        self.assertEqual(summary.status, "warning")
        self.assertTrue(any("界面字符串块" in finding for finding in summary.findings))

    def test_doctor_recommendations_are_rendered_as_facts(self):
        output = DOCTOR_OUTPUT + (
            "\nRecommendations:\n"
            "- work directory is missing or empty and original/game exists; "
            "run: python gemini_translate_batch.py bootstrap-work "
            "(copies original/game into work/game without generating TL).\n"
        )

        parsed = parse_doctor_output(output)
        self.assertEqual(len(parsed["recommendations"]), 1)

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)
        self.assertEqual(summary.findings, [])
        self.assertTrue(any(fact.startswith("建议：") for fact in summary.facts))
        self.assertTrue(any("准备工作目录" in fact for fact in summary.facts))

    def test_format_doctor_recommendation_fact_uses_fact_style(self):
        fact = format_doctor_recommendation_fact(
            "work directory is missing or empty and original/game exists; "
            "run: python gemini_translate_batch.py bootstrap-work "
            "(copies original/game into work/game without generating TL)."
        )

        self.assertEqual(fact, "建议：点击「准备工作目录」")

    def test_stale_summary_marks_previous_result_invalid(self):
        summary = stale_summary()

        self.assertEqual(summary.status, "stale")
        self.assertIn("重新运行", summary.heading)

    def test_format_tl_scan_facts_separates_dialogue_and_strings(self):
        facts = format_tl_scan_facts(
            {
                "rpy_files": 49,
                "translate_blocks": 49901,
                "commented_original_lines": 49863,
                "old_lines": 637,
                "new_lines": 637,
            },
            pending={"task_count": 48394, "file_count": 49},
        )

        self.assertIn("翻译文件：49 个", facts)
        self.assertTrue(any("待翻译条目：约 48394 条" in fact for fact in facts))
        self.assertTrue(any("剧情对话：49863 条" in fact for fact in facts))
        self.assertTrue(any("界面字符串：637 条" in fact for fact in facts))


if __name__ == "__main__":
    unittest.main()