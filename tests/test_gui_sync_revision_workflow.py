import unittest

from gui_qt.sync_revision_workflow import SyncRevisionWorkflow


class GuiSyncRevisionWorkflowTests(unittest.TestCase):
    def test_start_workflow_runs_sync_revisions_command(self):
        workflow = SyncRevisionWorkflow.start_new()

        step = workflow.current_step()

        self.assertEqual(step.script_basename, "gemini_translate_batch.py")
        self.assertEqual(step.args, ["sync-revisions"])

    def test_successful_run_finishes(self):
        workflow = SyncRevisionWorkflow.start_new()
        output = (
            "Sync revision run: C:\\package\\sync_revisions\n"
            "Recoverable revision items: 1\n"
            "Pending files: 1\n"
            "Pending lines: 1\n"
            "Failure items: 0\n"
            "Preview JSONL: C:\\package\\revision_preview.jsonl\n"
            "Preview Markdown: C:\\package\\revision_preview.md\n"
        )

        update = workflow.complete_current_step(0, output)

        self.assertEqual(update.status, "done")
        self.assertIsNone(workflow.current_step())

    def test_empty_pending_steps_means_no_current_step(self):
        workflow = SyncRevisionWorkflow(pending_steps=[])

        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()