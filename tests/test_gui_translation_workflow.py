import unittest

from gui_qt.translation_workflow import (
    TranslationWorkflow,
    extract_created_package_path,
    extract_job_state,
    extract_safety_status,
    manifest_path_for_package,
)


BUILD_OUTPUT = """
Created batch package: C:\\Games\\Example\\work\\logs\\batch_jobs\\job1
Pending files: 2
Chunks: 3
Items: 10
"""


class GuiTranslationWorkflowTests(unittest.TestCase):
    def test_manifest_path_for_package_preserves_input_path_style(self):
        self.assertEqual(
            manifest_path_for_package("C:\\Games\\Example\\work\\logs\\batch_jobs\\job1"),
            "C:\\Games\\Example\\work\\logs\\batch_jobs\\job1\\manifest.json",
        )
        self.assertEqual(
            manifest_path_for_package("/tmp/jobs/job1"),
            "/tmp/jobs/job1/manifest.json",
        )

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
        self.assertTrue(any("任务状态：处理中" in fact for fact in update.facts))
        self.assertIsNone(workflow.current_step())

    def test_resume_unsubmitted_manifest_starts_from_submit(self):
        workflow = TranslationWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": ""},
        )

        self.assertEqual(workflow.current_step().args, ["submit", "C:\\package\\manifest.json"])

    def test_resume_submitted_manifest_starts_from_status(self):
        workflow = TranslationWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertEqual(workflow.current_step().args, ["status", "C:\\package\\manifest.json"])

    def test_resume_safe_retry_manifest_merges_then_checks_parent(self):
        retry_path = r"C:\package\retry_parts\retry1\manifest.json"
        parent_path = r"C:\package\manifest.json"
        workflow = TranslationWorkflow.resume_manifest(
            retry_path,
            {
                "retry_of_manifest": parent_path,
                "job_name": "batches/retry",
                "job_state": "JOB_STATE_SUCCEEDED",
                "last_check_summary": {"safety_level": "safe"},
            },
        )

        self.assertEqual(workflow.current_step().args, ["merge-retry", parent_path, retry_path])
        update = workflow.complete_current_step(0, f"Merged retry results into: {parent_path}\n")
        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().args, ["check", parent_path])
        final = workflow.complete_current_step(0, "Safety status: safe\n")
        self.assertEqual(final.status, "done")
        self.assertEqual(workflow.manifest_path, parent_path)

    def test_retry_check_safe_continues_to_merge(self):
        retry_path = r"C:\package\retry_parts\retry1\manifest.json"
        parent_path = r"C:\package\manifest.json"
        workflow = TranslationWorkflow.resume_manifest(
            retry_path,
            {
                "retry_of_manifest": parent_path,
                "job_name": "batches/retry",
                "job_state": "JOB_STATE_SUCCEEDED",
            },
        )

        self.assertEqual(workflow.current_step().args, ["download", retry_path])
        workflow.complete_current_step(0, r"Saved results to: C:\package\retry_parts\retry1\results.jsonl" + "\n")
        self.assertEqual(workflow.current_step().args, ["check", retry_path])
        update = workflow.complete_current_step(0, "Safety status: safe\n")

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().args, ["merge-retry", parent_path, retry_path])

    def test_resume_checked_manifest_uses_last_check_summary_as_complete(self):
        workflow = TranslationWorkflow.resume_manifest(
            r"C:\package\manifest.json",
            {
                "job_name": "batches/example",
                "job_state": "JOB_STATE_SUCCEEDED",
                "last_check_summary": {"safety_level": "safe"},
            },
        )

        self.assertIsNone(workflow.current_step())

    def test_resume_succeeded_manifest_starts_from_download_and_check(self):
        manifest_path = r"C:\package\manifest.json"
        workflow = TranslationWorkflow.resume_manifest(
            manifest_path,
            {"job_name": "batches/example", "job_state": "JOB_STATE_SUCCEEDED"},
        )

        self.assertEqual(workflow.current_step().args, ["download", manifest_path])
        workflow.complete_current_step(0, "Saved results to: " + r"C:\package\results.jsonl" + "\n")
        self.assertEqual(workflow.current_step().args, ["check", manifest_path])

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
        self.assertIn("可写回", check_update.facts[-1])
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

    def test_submit_quota_failure_recommends_split(self):
        workflow = TranslationWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": ""},
        )

        output = (
            "RESOURCE_EXHAUSTED\n"
            "Suggested split command: python gemini_translate_batch.py split C:\\package\\manifest.json --max-chunks 400\n"
        )
        update = workflow.complete_current_step(1, output)

        self.assertEqual(update.status, "failed")
        self.assertIn("配额", update.message)
        self.assertIn("拆包", update.message)
        self.assertTrue(any("拆包命令" in fact for fact in update.facts))
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()
