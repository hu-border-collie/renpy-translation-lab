import unittest

from gui_qt.sync_keyword_workflow import SyncKeywordWorkflow


class GuiSyncKeywordWorkflowTests(unittest.TestCase):
    def test_start_workflow_runs_sync_keywords_command(self):
        workflow = SyncKeywordWorkflow.start_new()

        step = workflow.current_step()

        self.assertEqual(step.script_basename, "gemini_translate_batch.py")
        self.assertEqual(step.args, ["sync-keywords"])

    def test_successful_run_finishes(self):
        workflow = SyncKeywordWorkflow.start_new()
        output = (
            "Sync keyword run: C:\\package\\sync_keywords\n"
            "Keyword candidates: 2 deduped from 3 raw\n"
            "Chunk summaries: 1\n"
            "JSONL: C:\\package\\keyword_candidates.jsonl\n"
            "Markdown: C:\\package\\keyword_candidates.md\n"
            "Summary JSONL: C:\\package\\keyword_chunk_summaries.jsonl\n"
            "Summary Markdown: C:\\package\\keyword_chunk_summaries.md\n"
        )

        update = workflow.complete_current_step(0, output)

        self.assertEqual(update.status, "done")
        self.assertIsNone(workflow.current_step())

    def test_empty_pending_steps_means_no_current_step(self):
        workflow = SyncKeywordWorkflow(pending_steps=[])

        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()