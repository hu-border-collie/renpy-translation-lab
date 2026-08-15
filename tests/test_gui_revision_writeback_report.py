import unittest

from gui_qt.revision_writeback_report import (
    summarize_revision_writeback_from_manifest,
    summarize_revision_writeback_from_preview_output,
)


PREVIEW_OUTPUT = """
Recoverable revision items: 2
Pending files: 1
Pending lines: 2
Failure items: 0
Preview JSONL: C:\\package\\revision_preview.jsonl
Preview Markdown: C:\\package\\revision_preview.md
"""


class GuiRevisionWritebackReportTests(unittest.TestCase):
    def test_preview_with_recoverable_items_enables_apply(self):
        summary = summarize_revision_writeback_from_preview_output(
            PREVIEW_OUTPUT,
            0,
            manifest_path="C:\\package\\manifest.json",
        )

        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)
        self.assertIn("可写回订正项：2", "\n".join(summary.facts))

    def test_preview_without_recoverable_items_blocks_apply(self):
        summary = summarize_revision_writeback_from_preview_output(
            "Recoverable revision items: 0\nFailure items: 1\n",
            0,
            manifest_path="C:\\package\\manifest.json",
        )

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)

    def test_preview_already_applied_blocks_apply(self):
        summary = summarize_revision_writeback_from_preview_output(
            PREVIEW_OUTPUT,
            0,
            manifest_path="C:\\package\\manifest.json",
            already_applied=True,
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)

    def test_preview_nonzero_exit_blocks_apply(self):
        summary = summarize_revision_writeback_from_preview_output(
            PREVIEW_OUTPUT,
            1,
            manifest_path="C:\\package\\manifest.json",
        )

        self.assertEqual(summary.status, "failed")
        self.assertFalse(summary.can_apply)

    def test_manifest_after_preview_enables_apply(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": "C:\\package\\manifest.json",
                "last_revision_preview": {
                    "jsonl_path": "C:\\package\\revision_preview.jsonl",
                    "markdown_path": "C:\\package\\revision_preview.md",
                    "summary": {
                        "valid_items": 2,
                        "pending_files": 1,
                        "pending_lines": 2,
                        "failure_items": 0,
                    },
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)

    def test_partial_proposal_import_never_enables_apply(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": r"C:\package\manifest.json",
                "proposal_import": {
                    "status": "partial",
                    "writeback_eligible": False,
                    "report_path": r"C:\package\proposal_import_report.json",
                },
                "last_revision_preview": {
                    "summary": {"valid_items": 1, "failure_items": 1}
                },
            }
        )
        self.assertIsNotNone(summary)
        self.assertFalse(summary.can_apply)
        self.assertEqual(summary.status, "failed")
        self.assertIn("partial", summary.message)

    def test_manifest_after_apply_blocks_apply(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": "C:\\package\\manifest.json",
                "revision_applied_at": "2026-06-25T20:00:00",
                "revision_apply_summary": {
                    "applied_files": 1,
                    "applied_lines": 2,
                },
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 1 个文件", "\n".join(summary.facts))

    def test_manifest_without_preview_returns_none(self):
        summary = summarize_revision_writeback_from_manifest(
            {"_manifest_path": "C:\\package\\manifest.json"}
        )

        self.assertIsNone(summary)

    def test_manifest_blocked_apply_reports_blocked(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": "C:\\package\\manifest.json",
                "revision_apply_state": "blocked",
                "revision_apply_blocked_reason": "results_changed",
                "revision_apply_message": "result JSONL changed since preview.",
                "revision_apply_summary": {
                    "applied_files": 0,
                    "applied_lines": 0,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "failed")
        self.assertEqual(summary.heading, "订正写回被阻止")
        self.assertIn("results_changed", summary.message)
        self.assertFalse(summary.can_apply)

    def test_manifest_no_op_apply_reports_idle(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": "C:\\package\\manifest.json",
                "revision_apply_state": "no_op",
                "revision_apply_summary": {
                    "applied_files": 0,
                    "applied_lines": 0,
                    "unchanged_items": 3,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "idle")
        self.assertIn("no-op", summary.message)
        self.assertFalse(summary.can_apply)
        self.assertIn("无需修改项：3", "\n".join(summary.facts))

    def test_manifest_partial_apply_reports_partial(self):
        summary = summarize_revision_writeback_from_manifest(
            {
                "_manifest_path": "C:\\package\\manifest.json",
                "revision_apply_state": "partial",
                "revision_apply_summary": {
                    "applied_files": 1,
                    "applied_lines": 1,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "applied")
        self.assertEqual(summary.heading, "订正部分写回")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 1 个文件", "\n".join(summary.facts))


if __name__ == "__main__":
    unittest.main()
