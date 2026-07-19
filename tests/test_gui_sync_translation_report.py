import unittest

from gui_qt.sync_translation_report import summarize_sync_translation_output


SUCCESS_OUTPUT = """
Found 2 files.
Processing: script.rpy
  Found 3 lines to translate.
  Translated 3/3 items. (Received 42 chars of translation)
  Previewed script.rpy.
Progress log: logs/sync_progress.json
Sync preview manifest: logs/sync_runs/demo/manifest.json
Sync preview report: logs/sync_runs/demo/preview.diff
Preview files: 1
Preview translations: 3
Preview status: safe
"""


class GuiSyncTranslationReportTests(unittest.TestCase):
    def test_summarize_success_marks_done(self):
        update = summarize_sync_translation_output(SUCCESS_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("预览", update.heading)
        self.assertTrue(any("待处理文件" in fact for fact in update.facts))

    def test_summarize_no_work_marks_done_without_files(self):
        update = summarize_sync_translation_output(
            "Found 1 files.\nProcessing: empty.rpy\n  No new lines to translate.\n",
            0,
        )

        self.assertEqual(update.status, "done")
        self.assertIn("没有待翻译内容", update.heading)

    def test_summarize_missing_api_keys_fails(self):
        update = summarize_sync_translation_output(
            "ERROR: No valid API keys found!\n",
            0,
        )

        self.assertEqual(update.status, "failed")
        self.assertIn("API Key", update.heading)

    def test_summarize_nonzero_exit_fails(self):
        update = summarize_sync_translation_output("boom", 1)

        self.assertEqual(update.status, "failed")

    def test_summarize_missing_tl_dir_fails_even_with_zero_files(self):
        update = summarize_sync_translation_output(
            "Found 0 files.\nWARNING: TL_DIR does not exist after prepare step.\n",
            0,
        )

        self.assertEqual(update.status, "failed")
        self.assertIn("翻译目录不存在", update.heading)

    def test_summarize_retry_exhausted_without_translation_fails(self):
        update = summarize_sync_translation_output(
            "Found 1 files.\n"
            "Processing: script.rpy\n"
            "  Found 3 lines to translate.\n"
            "  Translated 0/3 items. (Received 0 chars of translation)\n"
            "  Previewed script.rpy.\n",
            0,
        )

        self.assertEqual(update.status, "failed")
        self.assertIn("未完成", update.heading)

    def test_summarize_partial_translation_warns(self):
        update = summarize_sync_translation_output(
            "Found 1 files.\n"
            "Processing: script.rpy\n"
            "  Found 3 lines to translate.\n"
            "  Translated 1/3 items. (Received 12 chars of translation)\n"
            "  Previewed script.rpy.\n"
            "Sync preview manifest: logs/sync_runs/demo/manifest.json\n"
            "Sync preview report: logs/sync_runs/demo/preview.diff\n"
            "Preview files: 1\n"
            "Preview translations: 1\n"
            "Preview status: safe\n",
            0,
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("部分完成", update.message)


if __name__ == "__main__":
    unittest.main()
