import unittest

from gui_qt.keyword_workflow import (
    KeywordBatchWorkflow,
    extract_created_keyword_package_path,
)


BUILD_OUTPUT = """
Created keyword package: C:\\Games\\Example\\work\\logs\\batch_jobs\\kw1
Source files: 2
Chunks: 3
Source lines: 10
Mode: keyword_extraction
"""


class GuiKeywordWorkflowTests(unittest.TestCase):
    def test_extracts_created_keyword_package_path(self):
        self.assertEqual(
            extract_created_keyword_package_path(BUILD_OUTPUT),
            "C:\\Games\\Example\\work\\logs\\batch_jobs\\kw1",
        )

    def test_start_workflow_builds_then_submits_created_manifest(self):
        workflow = KeywordBatchWorkflow.start_new()

        self.assertEqual(workflow.current_step().args, ["build-keywords"])
        update = workflow.complete_current_step(0, BUILD_OUTPUT)

        self.assertTrue(update.should_continue)
        self.assertEqual(update.status, "running")
        self.assertEqual(
            workflow.current_step().args,
            ["submit", "C:\\Games\\Example\\work\\logs\\batch_jobs\\kw1\\manifest.json"],
        )

    def test_build_without_source_lines_finishes_without_submitting(self):
        workflow = KeywordBatchWorkflow.start_new()

        update = workflow.complete_current_step(0, "No keyword source lines found.\n")

        self.assertEqual(update.status, "done")
        self.assertIn("没有可提取的关键词源行", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_build_without_chunks_finishes_without_submitting(self):
        workflow = KeywordBatchWorkflow.start_new()

        update = workflow.complete_current_step(0, "No keyword chunks built.\n")

        self.assertEqual(update.status, "done")
        self.assertIn("没有可提取的关键词源行", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_status_waiting_stops_before_download(self):
        workflow = KeywordBatchWorkflow.resume_latest("C:\\package\\manifest.json")

        update = workflow.complete_current_step(0, "State: JOB_STATE_RUNNING\n")

        self.assertEqual(update.status, "waiting")
        self.assertTrue(any("任务状态：处理中" in fact for fact in update.facts))
        self.assertIsNone(workflow.current_step())

    def test_resume_unsubmitted_manifest_starts_from_submit(self):
        workflow = KeywordBatchWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": ""},
        )

        self.assertEqual(workflow.current_step().args, ["submit", "C:\\package\\manifest.json"])

    def test_resume_submitted_manifest_starts_from_status(self):
        workflow = KeywordBatchWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertEqual(workflow.current_step().args, ["status", "C:\\package\\manifest.json"])

    def test_status_succeeded_continues_to_download_and_export(self):
        workflow = KeywordBatchWorkflow.resume_latest("C:\\package\\manifest.json")

        status_update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
        self.assertTrue(status_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["download", "C:\\package\\manifest.json"])

        download_update = workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")
        self.assertTrue(download_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["export-keywords", "C:\\package\\manifest.json"])

        export_output = (
            "Keyword candidates: 3 deduped from 5 raw\n"
            "Chunk summaries: 2\n"
            "JSONL: C:\\package\\keyword_candidates.jsonl\n"
            "Markdown: C:\\package\\keyword_candidates.md\n"
            "Summary JSONL: C:\\package\\keyword_chunk_summaries.jsonl\n"
            "Summary Markdown: C:\\package\\keyword_chunk_summaries.md\n"
        )
        export_update = workflow.complete_current_step(0, export_output)
        self.assertEqual(export_update.status, "done")
        self.assertIn("关键词提取完成", export_update.heading)
        self.assertTrue(any("候选 JSONL" in fact for fact in export_update.facts))
        self.assertIsNone(workflow.current_step())

    def test_nonzero_exit_fails_current_step(self):
        workflow = KeywordBatchWorkflow.start_new()

        update = workflow.complete_current_step(1, "boom")

        self.assertEqual(update.status, "failed")
        self.assertFalse(update.should_continue)
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()