import unittest

from gui_qt.keyword_report import (
    summarize_keyword_export_output,
    summarize_keyword_result_from_manifest,
    summarize_sync_keyword_output,
)


EXPORT_OUTPUT = """
Keyword candidates: 3 deduped from 5 raw
Chunk summaries: 2
JSONL: C:\\package\\keyword_candidates.jsonl
Markdown: C:\\package\\keyword_candidates.md
Summary JSONL: C:\\package\\keyword_chunk_summaries.jsonl
Summary Markdown: C:\\package\\keyword_chunk_summaries.md
"""


SYNC_OUTPUT = """
Sync keyword run: C:\\package\\sync_keywords
Keyword candidates: 2 deduped from 4 raw
Chunk summaries: 1
JSONL: C:\\package\\keyword_candidates.jsonl
Markdown: C:\\package\\keyword_candidates.md
Summary JSONL: C:\\package\\keyword_chunk_summaries.jsonl
Summary Markdown: C:\\package\\keyword_chunk_summaries.md
"""


class GuiKeywordReportTests(unittest.TestCase):
    def test_summarize_export_success_marks_done(self):
        update = summarize_keyword_export_output(EXPORT_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("关键词提取完成", update.heading)
        self.assertTrue(any("候选 JSONL" in fact for fact in update.facts))
        self.assertTrue(any("概要 Markdown" in fact for fact in update.facts))

    def test_summarize_export_nonzero_exit_fails(self):
        update = summarize_keyword_export_output("boom", 1)

        self.assertEqual(update.status, "failed")
        self.assertIn("导出中断", update.heading)

    def test_summarize_export_missing_candidate_stats_fails(self):
        update = summarize_keyword_export_output("JSONL: C:\\package\\out.jsonl\n", 0)

        self.assertEqual(update.status, "failed")
        self.assertIn("结果异常", update.heading)

    def test_summarize_sync_success_marks_done(self):
        update = summarize_sync_keyword_output(SYNC_OUTPUT, 0)

        self.assertEqual(update.status, "done")
        self.assertIn("同步关键词提取完成", update.heading)
        self.assertTrue(any("同步输出目录" in fact for fact in update.facts))

    def test_summarize_sync_no_work_marks_done(self):
        update = summarize_sync_keyword_output("No keyword source lines found.\n", 0)

        self.assertEqual(update.status, "done")
        self.assertIn("没有可提取的关键词源行", update.heading)

    def test_summarize_sync_nonzero_exit_fails(self):
        update = summarize_sync_keyword_output("boom", 1)

        self.assertEqual(update.status, "failed")
        self.assertIn("中断", update.heading)

    def test_summarize_keyword_result_from_manifest_shows_completed_report_paths(self):
        summary = summarize_keyword_result_from_manifest({
            "_manifest_path": "C:\\package\\manifest.json",
            "keyword_export": {
                "markdown_path": "C:\\package\\keyword_candidates.md",
                "jsonl_path": "C:\\package\\keyword_candidates.jsonl",
                "summary_markdown_path": "C:\\package\\keyword_chunk_summaries.md",
                "summary_jsonl_path": "C:\\package\\keyword_chunk_summaries.jsonl",
                "summary": {
                    "candidate_count_deduped": 3,
                    "candidate_count_raw": 5,
                    "chunk_summary_count": 2,
                },
            },
        })

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "safe")
        self.assertFalse(summary.can_apply)
        self.assertIn("关键词报告已生成", summary.heading)
        self.assertTrue(any("关键词候选：3 个去重 / 5 个原始" in fact for fact in summary.facts))
        self.assertTrue(any("候选 Markdown" in fact for fact in summary.facts))
        self.assertTrue(any("候选 Markdown：\n  C:\\package" in fact for fact in summary.facts))
        self.assertTrue(any("概要 JSONL" in fact for fact in summary.facts))

if __name__ == "__main__":
    unittest.main()
