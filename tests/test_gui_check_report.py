import unittest

from gui_qt.check_report import (
    parse_check_output,
    summarize_apply_output,
    summarize_check_output,
    summarize_manifest_writeback,
)


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

    def test_summarize_warn_check_points_to_remediation_commands(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_WARN,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "warn")
        self.assertIn("补救命令", summary.message)
        self.assertIn("retry", summary.message)
        self.assertIn("safe", summary.message)

    def test_summarize_apply_output_marks_completed(self):
        summary = summarize_apply_output(
            APPLY_OUTPUT,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 2 个文件", "\n".join(summary.facts))

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
        self.assertIn("补救命令", summary.message)
        self.assertIn("retry", summary.message)
        self.assertIn("safe", summary.message)

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


if __name__ == "__main__":
    unittest.main()