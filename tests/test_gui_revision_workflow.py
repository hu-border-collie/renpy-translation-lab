import unittest

from gui_qt.revision_workflow import (
    RevisionBatchWorkflow,
    extract_created_revision_package_path,
)


BUILD_OUTPUT = """
Created revision package: C:\\Games\\Example\\work\\logs\\batch_jobs\\rev1
Source files: 2
Chunks: 3
Revision items: 10
Mode: revision
"""

PREVIEW_OUTPUT = """
Recoverable revision items: 2
Pending files: 1
Pending lines: 2
Failure items: 0
Preview JSONL: C:\\package\\revision_preview.jsonl
Preview Markdown: C:\\package\\revision_preview.md
"""


class GuiRevisionWorkflowTests(unittest.TestCase):
    def test_extracts_created_revision_package_path(self):
        self.assertEqual(
            extract_created_revision_package_path(BUILD_OUTPUT),
            "C:\\Games\\Example\\work\\logs\\batch_jobs\\rev1",
        )

    def test_start_workflow_builds_then_submits_created_manifest(self):
        workflow = RevisionBatchWorkflow.start_new()

        self.assertEqual(workflow.current_step().args, ["build-revisions"])
        update = workflow.complete_current_step(0, BUILD_OUTPUT)

        self.assertTrue(update.should_continue)
        self.assertEqual(update.status, "running")
        self.assertEqual(
            workflow.current_step().args,
            ["submit", "C:\\Games\\Example\\work\\logs\\batch_jobs\\rev1\\manifest.json"],
        )

    def test_build_without_source_lines_finishes_without_submitting(self):
        workflow = RevisionBatchWorkflow.start_new()

        update = workflow.complete_current_step(0, "No revision source lines found.\n")

        self.assertEqual(update.status, "done")
        self.assertIn("没有可订正的源行", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_build_without_chunks_finishes_without_submitting(self):
        workflow = RevisionBatchWorkflow.start_new()

        update = workflow.complete_current_step(0, "No revision chunks built.\n")

        self.assertEqual(update.status, "done")
        self.assertIn("没有可订正的源行", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_status_waiting_stops_before_download(self):
        workflow = RevisionBatchWorkflow.resume_latest("C:\\package\\manifest.json")

        update = workflow.complete_current_step(0, "State: JOB_STATE_RUNNING\n")

        self.assertEqual(update.status, "waiting")
        self.assertTrue(any("任务状态：处理中" in fact for fact in update.facts))
        self.assertIsNone(workflow.current_step())

    def test_resume_unsubmitted_manifest_starts_from_submit(self):
        workflow = RevisionBatchWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": ""},
        )

        self.assertEqual(workflow.current_step().args, ["submit", "C:\\package\\manifest.json"])

    def test_resume_submitted_manifest_starts_from_status(self):
        workflow = RevisionBatchWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertEqual(workflow.current_step().args, ["status", "C:\\package\\manifest.json"])

    def test_status_succeeded_continues_to_download_and_preview(self):
        workflow = RevisionBatchWorkflow.resume_latest("C:\\package\\manifest.json")

        status_update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
        self.assertTrue(status_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["download", "C:\\package\\manifest.json"])

        download_update = workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")
        self.assertTrue(download_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["preview-revisions", "C:\\package\\manifest.json"])

        preview_update = workflow.complete_current_step(0, PREVIEW_OUTPUT)
        self.assertEqual(preview_update.status, "done")
        self.assertIn("订正预览完成", preview_update.heading)
        self.assertTrue(any("预览 JSONL" in fact for fact in preview_update.facts))
        self.assertIsNone(workflow.current_step())

    def test_nonzero_exit_fails_current_step(self):
        workflow = RevisionBatchWorkflow.start_new()

        update = workflow.complete_current_step(1, "boom")

        self.assertEqual(update.status, "failed")
        self.assertFalse(update.should_continue)
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()