import unittest

from gui_qt.sync_translation_report import summarize_sync_translation_output


SUCCESS_OUTPUT = """
Found 2 files.
Processing: script.rpy
  Found 3 lines to translate.
  Done with script.rpy.
Progress log: logs/sync_progress.json
"""


class GuiSyncTranslationReportTests(unittest.TestCase):
    def test_summarize_success_marks_done(self):
        update = summarize_sync_translation_output(SUCCESS_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("同步翻译完成", update.heading)
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


if __name__ == "__main__":
    unittest.main()