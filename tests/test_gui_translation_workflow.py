import unittest

from gui_qt.translation_workflow import (
    TranslationWorkflow,
    extract_created_package_path,
    extract_job_state,
    extract_safety_status,
)


BUILD_OUTPUT = """
Created batch package: C:\\Games\\Example\\work\\logs\\batch_jobs\\job1
Pending files: 2
Chunks: 3
Items: 10
"""


class GuiTranslationWorkflowTests(unittest.TestCase):
    def test_extracts_cli_output_fields(self):
        self.assertEqual(
            extract_created_package_path(BUILD_OUTPUT),
            "C:\\Games\\Example\\work\\logs\\batch_jobs\\job1",
        )
        self.assertEqual(extract_job_state("State: JOB_STATE_SUCCEEDED\n"), "JOB_STATE_SUCCEEDED")
        self.assertEqual(extract_safety_status("Safety status: safe\n"), "safe")

    def test_start_workflow_builds_then_submits_created_manifest(self):
        workflow = TranslationWorkflow.start_new()

        self.assertEqual(workflow.current_step().args, ["build"])
        update = workflow.complete_current_step(0, BUILD_OUTPUT)

        self.assertTrue(update.should_continue)
        self.assertEqual(update.status, "running")
        self.assertEqual(
            workflow.current_step().args,
            ["submit", "C:\\Games\\Example\\work\\logs\\batch_jobs\\job1\\manifest.json"],
        )

    def test_build_without_pending_lines_finishes_without_submitting_stale_manifest(self):
        workflow = TranslationWorkflow.start_new()

        update = workflow.complete_current_step(0, "No pending lines to translate.\n")

        self.assertEqual(update.status, "done")
        self.assertIn("没有待翻译内容", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_status_waiting_stops_before_download(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")

        update = workflow.complete_current_step(0, "State: JOB_STATE_RUNNING\n")

        self.assertEqual(update.status, "waiting")
        self.assertTrue(any("JOB_STATE_RUNNING" in fact for fact in update.facts))
        self.assertIsNone(workflow.current_step())

    def test_status_succeeded_continues_to_download_and_check(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")

        status_update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
        self.assertTrue(status_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["download", "C:\\package\\manifest.json"])

        download_update = workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")
        self.assertTrue(download_update.should_continue)
        self.assertEqual(workflow.current_step().args, ["check", "C:\\package\\manifest.json"])

        check_update = workflow.complete_current_step(0, "Safety status: safe\n")
        self.assertEqual(check_update.status, "done")
        self.assertIn("safe", check_update.facts[-1])
        self.assertIsNone(workflow.current_step())

    def test_check_warn_is_done_but_not_writeback_ready(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")
        workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
        workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")

        update = workflow.complete_current_step(0, "Safety status: warn\n")

        self.assertEqual(update.status, "done")
        self.assertIn("不应写回", update.message)

    def test_nonzero_exit_fails_current_step(self):
        workflow = TranslationWorkflow.start_new()

        update = workflow.complete_current_step(1, "boom")

        self.assertEqual(update.status, "failed")
        self.assertFalse(update.should_continue)
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()
