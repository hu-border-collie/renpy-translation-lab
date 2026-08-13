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
Sync local context: before=30, after=10, batches=2, truncated=1
Sync macro setting: file=macro_setting.md, applied=True, fingerprint=abc123
Preview status: safe
"""


class GuiSyncTranslationReportTests(unittest.TestCase):
    def test_summarize_success_marks_done(self):
        update = summarize_sync_translation_output(SUCCESS_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("预览", update.heading)
        self.assertTrue(any("待处理文件" in fact for fact in update.facts))
        self.assertTrue(any("局部上下文" in fact for fact in update.facts))
        self.assertTrue(
            any("前文 30 / 后文 10，2 个批次，截断 1 个" in fact for fact in update.facts)
        )
        self.assertTrue(
            any("风格设定：macro_setting.md（已应用）" in fact for fact in update.facts)
        )

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

    def test_summarize_all_unresolved_partial_preview_warns(self):
        update = summarize_sync_translation_output(
            "Found 1 files.\n"
            "Processing: script.rpy\n"
            "  Found 3 lines to translate.\n"
            "  Translated 0/3 items. (Received 0 chars of translation)\n"
            "Sync preview manifest: logs/sync_runs/demo/manifest.json\n"
            "Sync preview report: logs/sync_runs/demo/preview.diff\n"
            "Model contract completeness: 0/3\n"
            "Unresolved contract items: 3\n"
            "Preview status: partial\n",
            0,
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("部分完成", update.heading)
        self.assertIn("结果完整率：0/3", update.facts)
        self.assertIn("未解决结果：3 个", update.facts)

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

    def test_summarize_partial_adapter_preview_warns_and_remains_applyable(self):
        update = summarize_sync_translation_output(
            "Found 2 files.\n"
            "Processing: first.rpy\n"
            "  Found 1 lines to translate.\n"
            "  Translated 1/1 items. (Received 8 chars of translation)\n"
            "  Previewed first.rpy.\n"
            "Sync preview manifest: logs/sync_runs/demo/manifest.json\n"
            "Sync preview report: logs/sync_runs/demo/preview.diff\n"
            "Preview files: 1\n"
            "Preview translations: 1\n"
            "Preview failures: 1\n"
            "Preview status: partial\n",
            0,
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("部分完成", update.heading)
        self.assertIn("其余安全预览", update.message)
        self.assertTrue(any("预览失败文件" in fact for fact in update.facts))

    def test_summarize_partial_contract_explains_completeness_and_retry(self):
        update = summarize_sync_translation_output(
            "Found 1 files.\n"
            "Processing: script.rpy\n"
            "  Found 3 lines to translate.\n"
            "  Translated 2/3 items. (Received 12 chars of translation)\n"
            "  Previewed script.rpy.\n"
            "Sync preview manifest: logs/sync_runs/demo/manifest.json\n"
            "Sync preview report: logs/sync_runs/demo/preview.diff\n"
            "Model contract completeness: 2/3\n"
            "Targeted retries: 1 requests / 1 items\n"
            "Unresolved contract items: 1\n"
            "Preview status: partial\n",
            0,
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("完整性合同", update.message)
        self.assertIn("结果完整率：2/3", update.facts)
        self.assertIn("定点重试：1 次请求 / 1 项", update.facts)
        self.assertIn("未解决结果：1 个", update.facts)

    def test_summarize_surfaces_reasoning_and_text_output_diagnostics(self):
        update = summarize_sync_translation_output(
            SUCCESS_OUTPUT
            + "Sync output tokens: completion=23 reasoning=17 text=6\n"
            + "Reasoning budget warnings: 1\n"
            + "Truncated sync responses: 2\n",
            0,
        )

        self.assertIn(
            "输出 Token：completion 23 / reasoning 17 / 正文 6",
            update.facts,
        )
        self.assertIn("Reasoning 预算告警：1 次", update.facts)
        self.assertIn("输出截断：2 次", update.facts)


if __name__ == "__main__":
    unittest.main()
