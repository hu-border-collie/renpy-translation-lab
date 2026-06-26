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

        self.assertEqual(summary.status, "warn")
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


if __name__ == "__main__":
    unittest.main()