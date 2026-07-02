import unittest

from gui_qt.check_report import (
    idle_writeback_summary_for_work_mode,
    parse_check_output,
    summarize_apply_output,
    summarize_check_output,
    summarize_manifest_writeback,
)
from gui_qt.work_modes import WorkMode


CHECK_OUTPUT_SAFE = """
Pending files: 2
Pending lines: 18
Failure items: 0
Recoverable valid items: 18
Safety status: safe
Check failure report: C:\\pkg\\check_failures.jsonl
"""

CHECK_OUTPUT_WARN = """
Pending files: 1
Pending lines: 4
Failure items: 2
Safety status: warn
Warn reasons:
- source_mismatch: 2
Check failure report: C:\\pkg\\check_failures.jsonl
"""

APPLY_OUTPUT = """
Safety status: safe
Pending files: 2
Pending lines: 18
Applied files: 2
Applied lines: 18
Failures logged: 0
"""


class GuiCheckReportTests(unittest.TestCase):
    def test_parse_check_output_extracts_counts_and_reasons(self):
        parsed = parse_check_output(CHECK_OUTPUT_WARN)

        self.assertEqual(parsed["safety_status"], "warn")
        self.assertEqual(parsed["pending_files"], 1)
        self.assertEqual(parsed["pending_lines"], 4)
        self.assertEqual(parsed["failure_items"], 2)
        self.assertTrue(any("source_mismatch" in finding for finding in parsed["findings"]))

    def test_summarize_check_output_nonzero_exit_blocks_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=1,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "failed")
        self.assertFalse(summary.can_apply)

    def test_summarize_check_output_already_applied_blocks_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
            already_applied=True,
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)

    def test_summarize_safe_check_enables_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)
        self.assertIn("2 个文件", "\n".join(summary.facts))

    def test_summarize_warn_check_blocks_apply(self):
        summary = summarize_check_output(CHECK_OUTPUT_WARN, exit_code=0)

        self.assertEqual(summary.status, "warn")
        self.assertFalse(summary.can_apply)
        self.assertTrue(summary.findings)
        self.assertTrue(
            any(fact.startswith("注意：") and "source_mismatch" in fact for fact in summary.facts)
        )
        self.assertFalse(any(fact.startswith("- ") for fact in summary.facts))

    def test_summarize_warn_check_points_to_remediation_commands(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_WARN,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "warn")
        self.assertIn("查看问题清单", summary.message)
        self.assertEqual(summary.message.count("补译"), 1)
        self.assertIn("重新检查", summary.message)
        self.assertIn("可写回", summary.message)

    def test_summarize_apply_output_marks_completed(self):
        summary = summarize_apply_output(
            APPLY_OUTPUT,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 2 个文件", "\n".join(summary.facts))

    def test_summarize_apply_output_shows_next_split_manifest(self):
        summary = summarize_apply_output(
            APPLY_OUTPUT + "Next split manifest: C:\\pkg\\part02\\manifest.json\n",
            exit_code=0,
            manifest_path="C:\\pkg\\part01\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertIn("下一拆分包", "\n".join(summary.facts))
        self.assertIn("继续提交", summary.message)

    def test_summarize_manifest_writeback_shows_next_split_manifest(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\part01\\manifest.json",
                "applied_at": "2026-06-30T12:00:00",
                "apply_summary": {"applied_files": 1, "applied_lines": 5},
                "next_split_manifest_path": "C:\\pkg\\part02\\manifest.json",
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertIn("下一拆分包", "\n".join(summary.facts))
        self.assertIn("下一拆分包", summary.message)

    def test_summarize_manifest_warn_points_to_remediation_commands(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "last_check_summary": {
                    "safety_level": "warn",
                    "pending_files": 1,
                    "pending_lines": 4,
                    "failure_items": 2,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "warn")
        self.assertFalse(summary.can_apply)
        self.assertIn("查看问题清单", summary.message)
        self.assertEqual(summary.message.count("补译"), 1)
        self.assertIn("重新检查", summary.message)
        self.assertIn("可写回", summary.message)

    def test_summarize_manifest_writeback_from_last_check(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "last_check_summary": {
                    "safety_level": "safe",
                    "pending_files": 3,
                    "pending_lines": 12,
                    "failure_items": 0,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)

    def test_summarize_manifest_writeback_after_apply_failure(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": r"C:\pkg\manifest.json",
                "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
            }
        )

        self.assertEqual(summary.status, "failed")
        self.assertFalse(summary.can_apply)
        self.assertIn("查看写回失败报告", summary.message)

    def test_summarize_manifest_writeback_after_apply_failure_and_recheck(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": r"C:\pkg\manifest.json",
                "last_check_summary": {
                    "safety_level": "safe",
                    "pending_files": 2,
                    "pending_lines": 10,
                },
            }
        )

        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)

    def test_summarize_manifest_writeback_after_apply(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "applied_at": "2026-06-18T12:00:00",
                "apply_summary": {"applied_files": 1, "applied_lines": 5},
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)

    def test_idle_writeback_summary_for_keyword_mode_disables_apply(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.KEYWORD_EXTRACTION)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("关键词", summary.message)

    def test_idle_writeback_summary_for_bootstrap_mode_disables_apply(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.BOOTSTRAP_RAG)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("预建", summary.message)

    def test_idle_writeback_summary_for_sync_mode_explains_direct_writeback(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.SYNC_TRANSLATION)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("同步翻译", summary.message)
        self.assertNotIn("重新检查", summary.message)


if __name__ == "__main__":
    unittest.main()