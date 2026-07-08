import unittest

from gui_qt.doctor_report import (
    doctor_report_to_parsed,
    format_project_assets_facts,
    format_tl_scan_facts,
    parse_doctor_output,
    stale_summary,
    summarize_doctor_output,
    summarize_doctor_report,
)
import doctor_recommendations as doctor_rec

from gui_qt.user_copy import format_doctor_recommendation_fact, primary_recommendation_message


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

CONTEXT_OUTPUT = DOCTOR_OUTPUT + """
- RAG context: enabled=True, store_dir=C:/logs/rag_store/example, store_exists=True, history_records=12, bootstrap_on_build=True, updated_at=2026-06-29T12:00:00, error=
- Source index context: enabled=True, store_dir=C:/logs/source_index_store/example, store_exists=False, source_segments=0, schema_version=, updated_at=, error=
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


def _report_from_parsed(parsed: dict) -> dict:
    counts = parsed.get("counts") if isinstance(parsed.get("counts"), dict) else {}
    context_status: dict = {}
    rag_context = parsed.get("rag_context")
    if isinstance(rag_context, dict):
        context_status["rag"] = rag_context
    source_index_context = parsed.get("source_index_context")
    if isinstance(source_index_context, dict):
        context_status["source_index"] = source_index_context
    pending = parsed.get("pending") if isinstance(parsed.get("pending"), dict) else {}
    report = {
        "base_dir": parsed.get("base_dir", ""),
        "tl_dir": parsed.get("tl_dir", ""),
        "tl_exists": parsed.get("tl_exists"),
        "language": parsed.get("language", ""),
        "mode": parsed.get("mode", ""),
        "layout_status": parsed.get("layout_status", ""),
        "is_work_root": parsed.get("is_work_root"),
        "work_dir": parsed.get("work_dir", ""),
        "work_exists": parsed.get("work_exists"),
        "work_empty": parsed.get("work_empty"),
        "counts": counts,
        "warnings": parsed.get("warnings", []),
        "recommendations": parsed.get("recommendations", []),
        "context_status": context_status,
        "project_assets": parsed.get("project_assets"),
    }
    if pending:
        report["pending_task_count"] = pending.get("task_count", 0)
        report["pending_file_count"] = pending.get("file_count", 0)
    if "original_game_dir" in parsed:
        report["original_game_dir"] = parsed.get("original_game_dir", "")
    return report


class GuiDoctorReportTests(unittest.TestCase):
    def test_summarize_doctor_report_matches_stdout_parser(self):
        for output, api_key_count, api_key_source in (
            (DOCTOR_OUTPUT, 2, ""),
            (CONTEXT_OUTPUT, 1, ""),
            (READY_OUTPUT, 2, ""),
            (FAILED_OUTPUT, 1, ""),
        ):
            parsed = parse_doctor_output(output)
            report = _report_from_parsed(parsed)
            from_output = summarize_doctor_output(
                output,
                exit_code=0,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            from_report = summarize_doctor_report(
                report,
                exit_code=0,
                api_key_count=api_key_count,
                api_key_source=api_key_source,
            )
            self.assertEqual(from_report, from_output, msg=output[:40])

    def test_doctor_report_to_parsed_preserves_pending_counts(self):
        parsed = parse_doctor_output(DOCTOR_OUTPUT)
        report = doctor_report_to_parsed(_report_from_parsed(parsed))
        pending = report.get("pending")
        self.assertIsInstance(pending, dict)
        self.assertEqual(pending.get("task_count"), 5)
        self.assertEqual(pending.get("file_count"), 2)

    def test_parse_doctor_output_extracts_mode_and_counts(self):
        parsed = parse_doctor_output(DOCTOR_OUTPUT)

        self.assertEqual(parsed["mode"], "existing_tl_only")
        self.assertEqual(parsed["tl_exists"], True)
        self.assertEqual(parsed["counts"]["rpy_files"], 3)
        self.assertEqual(parsed["counts"]["old_lines"], 10)
        self.assertEqual(parsed["pending"]["task_count"], 5)
        self.assertEqual(parsed["pending"]["file_count"], 2)

    def test_format_project_assets_facts_reports_missing_files(self):
        facts = format_project_assets_facts(
            {
                "glossary_exists": False,
                "macro_exists": False,
                "glossary_file": "C:/Games/Example/work/glossary.json",
                "macro_setting_file": "C:/Games/Example/work/macro_setting.md",
            }
        )

        self.assertTrue(any("术语表：当前项目缺少 glossary.json" in fact for fact in facts))
        self.assertTrue(any("风格设定：当前项目缺少 macro_setting.md" in fact for fact in facts))

    def test_format_project_assets_facts_reports_mismatched_paths(self):
        facts = format_project_assets_facts(
            {
                "glossary_exists": True,
                "glossary_matches_project": False,
                "macro_exists": True,
                "macro_matches_project": False,
                "glossary_file": "C:/Games/Other/work/glossary.json",
                "macro_setting_file": "C:/Games/Other/work/macro_setting.md",
            }
        )

        self.assertTrue(any("术语表：路径与当前项目不匹配" in fact for fact in facts))
        self.assertTrue(any("风格设定：路径与当前项目不匹配" in fact for fact in facts))

    def test_primary_recommendation_message_ignores_template_prep_suggestions(self):
        message = primary_recommendation_message(
            ["建议：点击「生成翻译模板」"],
        )

        self.assertEqual(message, "")

    def test_parse_doctor_output_extracts_context_status(self):
        parsed = parse_doctor_output(CONTEXT_OUTPUT)

        self.assertEqual(parsed["rag_context"]["enabled"], True)
        self.assertEqual(parsed["rag_context"]["history_records"], 12)
        self.assertEqual(parsed["rag_context"]["bootstrap_on_build"], True)
        self.assertEqual(parsed["source_index_context"]["enabled"], True)
        self.assertEqual(parsed["source_index_context"]["store_exists"], False)
        self.assertEqual(parsed["source_index_context"]["source_segments"], 0)

    def test_successful_report_shows_context_status(self):
        summary = summarize_doctor_output(CONTEXT_OUTPUT, exit_code=0, api_key_count=1)

        self.assertTrue(any("记忆库：已启用，记录数 12" in fact for fact in summary.facts))
        self.assertTrue(any("记忆库路径：C:/logs/rag_store/example" in fact for fact in summary.facts))
        self.assertTrue(any("原文索引：已启用，尚未创建" in fact for fact in summary.facts))
        self.assertTrue(any("原文索引路径：C:/logs/source_index_store/example" in fact for fact in summary.facts))

    def test_successful_existing_tl_without_api_key_is_warning(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=0)

        self.assertEqual(summary.status, "warning")
        self.assertEqual(summary.heading, "检查完成，但有需要处理的事项")
        self.assertIn("翻译模板已就绪", summary.message)
        self.assertTrue(any("API 密钥" in fact or fact.startswith("建议：") for fact in summary.facts))
        self.assertEqual(summary.findings, [])

    def test_successful_clean_report_is_ready(self):
        summary = summarize_doctor_output(DOCTOR_OUTPUT, exit_code=0, api_key_count=2)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertTrue(any("API 密钥：已配置 2 个" in fact for fact in summary.facts))
        self.assertTrue(any("翻译文件：3 个" in fact for fact in summary.facts))
        self.assertTrue(any("待翻译条目：约 5 条" in fact for fact in summary.facts))
        self.assertTrue(any("不代表批量翻译漏翻" in fact for fact in summary.facts))
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
        self.assertEqual(summary.mode, "can_generate_template")
        self.assertIn("翻译模板尚未生成", summary.message)
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

    def test_switch_to_work_shows_resolved_work_path(self):
        output = SWITCH_TO_WORK_OUTPUT.replace(
            "Work dir: C:\\Games\\Example\\work (exists: False, empty: True)",
            "Work dir: C:\\Games\\Example\\work (exists: True, empty: False)",
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)

        self.assertTrue(any("work 目录：C:\\Games\\Example\\work" in fact for fact in summary.facts))
        self.assertFalse(any("work 目录：不存在" in fact for fact in summary.facts))

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
        self.assertTrue(
            any(fact.startswith("注意：") and "界面字符串块" in fact for fact in summary.facts)
        )
        self.assertFalse(any(fact.startswith("- ") for fact in summary.facts))

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

    def test_api_key_facts_appear_before_doctor_recommendations(self):
        output = DOCTOR_OUTPUT + (
            "\nRecommendations:\n"
            "- game_root should use work directory; switch to C:\\Games\\Example\\work\n"
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=2)
        api_index = next(
            i for i, fact in enumerate(summary.facts) if fact.startswith("API 密钥：")
        )
        recommendation_index = next(
            i for i, fact in enumerate(summary.facts) if fact.startswith("建议：将项目路径")
        )
        self.assertLess(api_index, recommendation_index)

    def test_format_doctor_recommendation_fact_uses_fact_style(self):
        fact = format_doctor_recommendation_fact(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_WORK)
        )

        self.assertEqual(fact, "建议：点击「准备工作目录」")

    def test_format_doctor_recommendation_fact_for_ready_pending_lines(self):
        fact = format_doctor_recommendation_fact(
            doctor_rec.make_doctor_recommendation(doctor_rec.START_PENDING_BATCH)
        )

        self.assertEqual(
            fact,
            "建议：切换到「翻译 · 批量翻译」，点击「开始翻译」打包并提交云端任务",
        )

    def test_format_doctor_recommendation_fact_for_empty_rag(self):
        fact = format_doctor_recommendation_fact(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG)
        )

        self.assertEqual(
            fact,
            "建议：先在「分析与准备」运行「预建记忆库」，再开始批量翻译",
        )

    def test_format_doctor_recommendation_fact_supports_legacy_english_strings(self):
        fact = format_doctor_recommendation_fact(
            "Pending translation lines are ready; start batch translation when API keys are configured."
        )

        self.assertEqual(
            fact,
            "建议：切换到「翻译 · 批量翻译」，点击「开始翻译」打包并提交云端任务",
        )

    def test_primary_recommendation_message_for_incremental_translation(self):
        message = primary_recommendation_message([doctor_rec.START_INCREMENTAL_BATCH])

        self.assertEqual(message, "补译环境已就绪，可以开始批量翻译。")

    def test_summarize_doctor_output_parses_json_cli_recommendations(self):
        cli_line = doctor_rec.format_doctor_recommendation_cli_line(
            doctor_rec.make_doctor_recommendation(doctor_rec.BOOTSTRAP_RAG)
        )
        output = DOCTOR_OUTPUT + f"\nRecommendations:\n- {cli_line}\n"

        parsed = parse_doctor_output(output)
        self.assertEqual(len(parsed["recommendations"]), 1)
        self.assertEqual(parsed["recommendations"][0]["code"], doctor_rec.BOOTSTRAP_RAG)

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=3)
        self.assertEqual(summary.message, "记忆库尚未建立，建议先预建记忆库再开始翻译。")
        self.assertEqual(summary.status, "warning")
        self.assertTrue(any("预建记忆库" in fact for fact in summary.facts))

    def test_summarize_doctor_output_parses_json_cli_switch_to_work(self):
        cli_line = doctor_rec.format_doctor_recommendation_cli_line(
            doctor_rec.make_doctor_recommendation(
                doctor_rec.SWITCH_TO_WORK,
                work_dir="C:\\Games\\Example\\work",
            )
        )
        output = SWITCH_TO_WORK_OUTPUT.replace(
            "- game_root should use work directory; switch to C:\\Games\\Example\\work",
            f"- {cli_line}",
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=1)
        self.assertTrue(
            any(fact == "建议：将项目路径切换到C:\\Games\\Example\\work" for fact in summary.facts)
        )

    def test_format_doctor_recommendation_fact_for_unknown_code(self):
        fact = format_doctor_recommendation_fact(
            {
                "code": doctor_rec.UNKNOWN,
                "params": {},
                "detail": "Some future recommendation text that has no mapping yet.",
            }
        )

        self.assertEqual(fact, "建议：收到未识别的诊断建议，请查看诊断日志了解详情。")
        self.assertNotIn("future recommendation", fact)

    def test_summarize_doctor_output_uses_primary_recommendation_message(self):
        output = DOCTOR_OUTPUT + (
            "\nRecommendations:\n"
            "- RAG store is enabled but empty; run bootstrap-rag before batch translation.\n"
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=3)

        self.assertEqual(summary.message, "记忆库尚未建立，建议先预建记忆库再开始翻译。")
        self.assertEqual(summary.status, "warning")
        self.assertTrue(
            any("预建记忆库" in fact for fact in summary.facts),
        )

    def test_substantially_complete_stays_green_despite_optional_rag_wording(self):
        output = DOCTOR_OUTPUT.replace(
            "- Pending translation: task_count=5, file_count=2",
            "- Pending translation: task_count=45, file_count=12",
        ).replace(
            "commented_original_lines=2",
            "commented_original_lines=85000",
        ) + (
            "\nRecommendations:\n"
            "- Project is substantially complete; remaining pending lines are minor. "
            "Batch translation and RAG bootstrap are optional.\n"
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=3)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertEqual(
            summary.message,
            "项目已基本译完；剩余待译行很少，可忽略或按需补译。",
        )

    def test_incremental_ready_stays_green_with_informational_rag_warning(self):
        output = DOCTOR_OUTPUT + (
            "\nWarnings:\n"
            "- RAG store contains legacy ID format keys. They will be seamlessly "
            "migrated on the next successful writeback.\n"
            "\nRecommendations:\n"
            "- Incremental translation is ready; start batch translation when API keys are configured.\n"
        )

        summary = summarize_doctor_output(output, exit_code=0, api_key_count=3)

        self.assertEqual(summary.status, "ready")
        self.assertEqual(summary.heading, "项目检查通过")
        self.assertEqual(summary.message, "补译环境已就绪，可以开始批量翻译。")
        self.assertTrue(any("记忆库含有旧版键格式" in fact for fact in summary.facts))

    def test_partial_source_index_shows_progress_in_facts(self):
        output = """
Doctor report:
- Base dir: C:/Games/Example/work
- TL dir: C:/Games/Example/work/game/tl/schinese (exists: True)
- Mode: existing_tl_only
- Layout status: ready
- TL scan: rpy_files=3, translate_blocks=10, string_sections=0, old_lines=0, new_lines=0, commented_original_lines=100
- Pending translation: task_count=8500, file_count=12
- Source index context: enabled=True, store_dir=C:/logs/source_index_store/demo, store_exists=True, source_segments=4200, expected_segments=12000, schema_version=1, updated_at=, error=
Recommendations:
- Source index bootstrap is incomplete; run bootstrap-source-index.
"""
        summary = summarize_doctor_output(output, exit_code=0, api_key_count=3)

        self.assertTrue(
            any("原文索引：已启用，片段数 4200/12000" in fact for fact in summary.facts)
        )
        self.assertTrue(
            any("继续运行「预建原文索引」补全索引库" in fact for fact in summary.facts)
        )
        self.assertFalse(
            any("提交约 8500" in fact for fact in summary.facts)
        )

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
        self.assertTrue(any("不代表批量翻译漏翻" in fact for fact in facts))
        self.assertTrue(any("剧情对话：49863 条" in fact for fact in facts))
        self.assertTrue(any("界面字符串：637 条" in fact for fact in facts))


if __name__ == "__main__":
    unittest.main()