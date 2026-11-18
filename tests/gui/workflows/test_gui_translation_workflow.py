import json
import unittest

from foundation import cli_contract

from gui_qt.workflows.translation_workflow import (
    TranslationWorkflow,
    extract_check_status,
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


def _status_output(state, next_action, *, retry_parent=""):
    return json.dumps(cli_contract.success_envelope(
        "status", status=state, result={
            "next_action": next_action,
            "retry_parent_manifest_path": retry_parent,
        },
    ))


def _check_output(status, decision=None, *, next_action="", warnings=0):
    check = {"check_status": status}
    if decision is not None:
        check["writeback_gate"] = {"decision": decision}
    if warnings:
        check["quality_gate"] = {"decision": "needs_review", "warning_count": warnings}
    return json.dumps(cli_contract.success_envelope(
        "check", status=status, result={"check": check, "next_action": next_action},
    ))


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

    def test_extracts_shared_json_envelope_fields(self):
        manifest_path = r"C:\pkg\manifest.json"
        status_output = json.dumps(
            cli_contract.success_envelope(
                "status",
                status="JOB_STATE_SUCCEEDED",
                artifacts={"manifest": manifest_path},
            )
        )
        check_output = json.dumps(
            cli_contract.success_envelope(
                "check",
                status="safe",
                artifacts={"manifest": manifest_path},
            )
        )

        self.assertEqual(extract_job_state(status_output), "JOB_STATE_SUCCEEDED")
        self.assertEqual(extract_safety_status(check_output), "safe")

    def test_json_build_envelope_sets_manifest_without_parsing_console_copy(self):
        manifest_path = r"C:\pkg\manifest.json"
        workflow = TranslationWorkflow.start_new()
        output = json.dumps(
            cli_contract.success_envelope(
                "build",
                status="LOCAL_ONLY",
                artifacts={"manifest": manifest_path},
            )
        )

        update = workflow.complete_current_step(0, output)

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.manifest_path, manifest_path)

    def test_start_workflow_builds_then_submits_created_manifest(self):
        workflow = TranslationWorkflow.start_new()

        self.assertEqual(
            workflow.current_step().args, ["build", "--output", "json", "--non-interactive"]
        )
        update = workflow.complete_current_step(0, BUILD_OUTPUT)

        self.assertTrue(update.should_continue)
        self.assertEqual(update.status, "running")
        self.assertEqual(
            workflow.current_step().args,
            [
                "submit",
                "C:\\Games\\Example\\work\\logs\\batch_jobs\\job1\\manifest.json",
                "--output",
                "json",
                "--non-interactive",
            ],
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

    def test_resume_runs_local_status_before_executing_cli_action(self):
        path = r"C:\package\manifest.json"
        for action, expected in (
            ("submit", ["submit", path, "--output", "json", "--non-interactive"]),
            ("resume-submit", ["submit", path, "--resume", "--output", "json", "--non-interactive"]),
            ("recover-submit", ["recover-submit", path]),
            ("status", ["status", path, "--output", "json", "--non-interactive"]),
        ):
            with self.subTest(action=action):
                workflow = TranslationWorkflow.resume_manifest(path)
                self.assertEqual(
                    workflow.current_step().args,
                    ["status", path, "--local", "--output", "json", "--non-interactive"],
                )
                update = workflow.complete_current_step(0, _status_output("LOCAL_ONLY", action))
                self.assertTrue(update.should_continue)
                self.assertEqual(workflow.current_step().args, expected)

    def test_resume_legacy_safe_retry_manifest_rechecks_before_merge(self):
        retry_path = r"C:\package\retry_parts\retry1\manifest.json"
        parent_path = r"C:\package\manifest.json"
        workflow = TranslationWorkflow.resume_manifest(retry_path)
        workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "check", retry_parent=parent_path))

        self.assertEqual(workflow.current_step().args, ["check", retry_path, "--output", "json", "--non-interactive"])
        update = workflow.complete_current_step(0, _check_output("ready", "allow", next_action="merge-retry"))
        self.assertTrue(update.should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["merge-retry", parent_path, retry_path, "--require-checked-retry", "--output", "json", "--non-interactive"],
        )
        update = workflow.complete_current_step(0, f"Merged retry results into: {parent_path}\n")
        self.assertTrue(update.should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["check", parent_path, "--output", "json", "--non-interactive"],
        )
        final = workflow.complete_current_step(0, _check_output("ready", "allow"))
        self.assertEqual(final.status, "done")
        self.assertEqual(workflow.manifest_path, parent_path)
        self.assertIn("可写回", final.message)

    def test_retry_check_with_only_warnings_continues_to_merge(self):
        retry_path = r"C:\package\retry_parts\retry1\manifest.json"
        parent_path = r"C:\package\manifest.json"
        workflow = TranslationWorkflow.resume_manifest(retry_path)
        workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "download", retry_parent=parent_path))

        self.assertEqual(
            workflow.current_step().args,
            ["download", retry_path, "--output", "json", "--non-interactive"],
        )
        workflow.complete_current_step(
            0, r"Saved results to: C:\package\retry_parts\retry1\results.jsonl" + "\n"
        )
        self.assertEqual(
            workflow.current_step().args,
            ["check", retry_path, "--output", "json", "--non-interactive"],
        )
        update = workflow.complete_current_step(0, _check_output(
            "ready_with_warnings", "allow", next_action="merge-retry", warnings=2,
        ))

        self.assertTrue(update.should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["merge-retry", parent_path, retry_path, "--require-checked-retry", "--output", "json", "--non-interactive"],
        )
        self.assertTrue(any("质量检查" in fact for fact in update.facts))

    def test_retry_does_not_merge_without_explicit_cli_authorization(self):
        for output in (
            "Safety status: safe\nWriteback gate: allow\n",
            _check_output("ready", next_action="merge-retry"),
            _check_output("ready", "deny", next_action="merge-retry"),
            _check_output("ready", "allow"),
        ):
            with self.subTest(output=output):
                workflow = TranslationWorkflow(
                    ["check"], manifest_path="retry.json", retry_parent_manifest_path="parent.json",
                )
                update = workflow.complete_current_step(0, output)
                self.assertFalse(update.should_continue)
                self.assertIsNone(workflow.current_step())
                self.assertNotIn("可以进入写回前确认", update.message)

    def test_resume_stops_when_cli_has_no_action(self):
        for parent in ("", "parent.json"):
            with self.subTest(parent=parent):
                workflow = TranslationWorkflow.resume_manifest("manifest.json")
                update = workflow.complete_current_step(
                    0, _status_output("JOB_STATE_SUCCEEDED", "", retry_parent=parent),
                )
                self.assertFalse(update.should_continue)
                self.assertIsNone(workflow.current_step())

    def test_status_does_not_infer_download_from_job_success(self):
        workflow = TranslationWorkflow.resume_latest("manifest.json")
        update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
        self.assertEqual(update.status, "failed")
        self.assertIsNone(workflow.current_step())

    def test_status_query_never_queues_download(self):
        workflow = TranslationWorkflow.resume_latest("manifest.json")
        workflow.only_query = True
        update = workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "download"))
        self.assertEqual(update.status, "ready")
        self.assertIsNone(workflow.current_step())

    def test_status_succeeded_continues_to_download_and_check(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")

        status_update = workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "download"))
        self.assertTrue(status_update.should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["download", "C:\\package\\manifest.json", "--output", "json", "--non-interactive"],
        )

        download_update = workflow.complete_current_step(
            0, "Saved results to: C:\\package\\results.jsonl\n"
        )
        self.assertTrue(download_update.should_continue)
        self.assertEqual(
            workflow.current_step().args,
            ["check", "C:\\package\\manifest.json", "--output", "json", "--non-interactive"],
        )

        check_update = workflow.complete_current_step(0, _check_output("ready", "allow"))
        self.assertEqual(check_update.status, "done")
        self.assertIn("可写回", check_update.facts[-1])
        self.assertIsNone(workflow.current_step())

    def test_check_warn_is_done_but_not_writeback_ready(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")
        workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "download"))
        workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")

        update = workflow.complete_current_step(0, "Safety status: warn\n")

        self.assertEqual(update.status, "done")
        self.assertIn("不应写回", update.message)

    def test_check_ready_with_warnings_is_writeback_ready(self):
        workflow = TranslationWorkflow.resume_latest("C:\\package\\manifest.json")
        workflow.complete_current_step(0, _status_output("JOB_STATE_SUCCEEDED", "download"))
        workflow.complete_current_step(0, "Saved results to: C:\\package\\results.jsonl\n")
        output = _check_output("ready_with_warnings", "allow", warnings=4)

        update = workflow.complete_current_step(0, output)

        self.assertEqual(update.status, "done")
        self.assertIn("可写回", update.message)
        self.assertIn("质量报警", update.heading)
        self.assertTrue(any("质量检查" in fact for fact in update.facts))

    def test_extract_legacy_check_status_for_display(self):
        output = (
            "Safety status: safe\n"
            "Writeback gate: allow\n"
            "Check status: ready_with_warnings\n"
        )

        self.assertEqual(extract_check_status(output), "ready_with_warnings")

    def test_nonzero_exit_fails_current_step(self):
        workflow = TranslationWorkflow.start_new()

        update = workflow.complete_current_step(1, "boom")

        self.assertEqual(update.status, "failed")
        self.assertFalse(update.should_continue)
        self.assertIsNone(workflow.current_step())

    def test_submit_quota_failure_recommends_split(self):
        workflow = TranslationWorkflow.resume_manifest(r"C:\package\manifest.json")
        workflow.complete_current_step(0, _status_output("LOCAL_ONLY", "submit"))

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

    def test_output_has_quota_error_ignores_counts_with_429_substring(self):
        from gui_qt.workflows.translation_workflow import output_has_quota_error

        self.assertFalse(output_has_quota_error("Processed 4290 items in 12 chunks"))
        self.assertTrue(output_has_quota_error("HTTP 429 Too Many Requests"))
        envelope_non_quota = json.dumps(
            cli_contract.error_envelope(
                "submit",
                code="COMMAND_REFUSED",
                message="Invalid chunk count 4290 in package",
            )
        )
        self.assertFalse(output_has_quota_error(envelope_non_quota))
        envelope_quota = json.dumps(
            cli_contract.error_envelope(
                "submit",
                code="REMOTE_RETRYABLE",
                message="Provider returned HTTP 429: rate limit exceeded",
                retryable=True,
            )
        )
        self.assertTrue(output_has_quota_error(envelope_quota))



class ProfilePinnedBuildTests(unittest.TestCase):
    def test_build_passes_profile_to_cli(self) -> None:
        from gui_qt.workflows.translation_workflow import TranslationWorkflow

        workflow = TranslationWorkflow.start_new(profile_id="legacy-batch")

        self.assertEqual(
            workflow.current_step().args,
            [
                "build",
                "--profile",
                "legacy-batch",
                "--output",
                "json",
                "--non-interactive",
            ],
        )


if __name__ == "__main__":
    unittest.main()
