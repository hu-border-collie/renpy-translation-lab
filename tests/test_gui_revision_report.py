import unittest

from gui_qt.revision_report import (
    parse_revision_summary,
    summarize_revision_apply_output,
    summarize_revision_preview_output,
    summarize_sync_revision_output,
)


PREVIEW_OUTPUT = """
Recoverable revision items: 2
Pending files: 1
Pending lines: 2
Failure items: 0
Preview JSONL: C:\\package\\revision_preview.jsonl
Preview Markdown: C:\\package\\revision_preview.md
"""

APPLY_OUTPUT = """
Recoverable revision items: 2
Applied files: 1
Applied lines: 2
Failures logged: 0
"""

SYNC_OUTPUT = """
Sync revision run: C:\\package\\sync_revisions
Recoverable revision items: 1
Pending files: 1
Pending lines: 1
Failure items: 0
Preview JSONL: C:\\package\\revision_preview.jsonl
Preview Markdown: C:\\package\\revision_preview.md
"""


class GuiRevisionReportTests(unittest.TestCase):
    def test_parse_revision_summary_extracts_preview_paths_and_counts(self):
        parsed = parse_revision_summary(PREVIEW_OUTPUT)

        self.assertEqual(parsed["valid_items"], 2)
        self.assertEqual(parsed["pending_files"], 1)
        self.assertEqual(parsed["pending_lines"], 2)
        self.assertEqual(parsed["preview_jsonl"], "C:\\package\\revision_preview.jsonl")
        self.assertEqual(parsed["preview_markdown"], "C:\\package\\revision_preview.md")

    def test_summarize_preview_success_marks_done(self):
        update = summarize_revision_preview_output(PREVIEW_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("订正预览完成", update.heading)
        self.assertTrue(any("预览 JSONL" in fact for fact in update.facts))
        self.assertTrue(any("预览 Markdown" in fact for fact in update.facts))

    def test_summarize_preview_zero_valid_items_still_done(self):
        update = summarize_revision_preview_output(
            "Recoverable revision items: 0\nFailure items: 1\n",
            0,
        )

        self.assertEqual(update.status, "done")
        self.assertIn("没有可写回的订正项", update.message)

    def test_summarize_preview_nonzero_exit_fails(self):
        update = summarize_revision_preview_output("boom", 1)

        self.assertEqual(update.status, "failed")
        self.assertIn("预览中断", update.heading)

    def test_summarize_sync_success_marks_done(self):
        update = summarize_sync_revision_output(SYNC_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("同步订正预览完成", update.heading)
        self.assertTrue(any("同步输出目录" in fact for fact in update.facts))

    def test_summarize_sync_no_work_marks_done(self):
        update = summarize_sync_revision_output("No revision source lines found.\n", 0)

        self.assertEqual(update.status, "done")
        self.assertIn("没有可订正的源行", update.heading)

    def test_summarize_sync_nonzero_exit_fails(self):
        update = summarize_sync_revision_output("boom", 1)

        self.assertEqual(update.status, "failed")
        self.assertIn("中断", update.heading)

    def test_summarize_apply_output_marks_completed(self):
        summary = summarize_revision_apply_output(
            APPLY_OUTPUT,
            0,
            manifest_path="C:\\package\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 1 个文件", "\n".join(summary.facts))


if __name__ == "__main__":
    unittest.main()