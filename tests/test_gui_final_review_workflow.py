"""GUI final-review workflow tests (#255 PR C)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from gui_qt.final_review_workflow import FinalReviewWorkflow
from gui_qt.workflow_factory import create_workflow, resume_workflow
from gui_qt.work_modes import WorkMode


class FinalReviewWorkflowTests(unittest.TestCase):
    def test_factory_starts_final_review_in_maintenance_workflow(self):
        workflow = create_workflow(WorkMode.FINAL_REVIEW)
        self.assertIsInstance(workflow, FinalReviewWorkflow)
        self.assertEqual(workflow.current_step().args, ["final-review-build"])
        update = workflow.complete_current_step(0, "Created final-review campaign: C:/tmp/review")
        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "submit")

    def test_failed_step_stops_the_workflow(self):
        workflow = FinalReviewWorkflow.start_new()
        update = workflow.complete_current_step(1, "failed")
        self.assertEqual(update.status, "failed")
        self.assertEqual(update.heading, "最终审校流程中断")
        self.assertIsNone(workflow.current_step())

    def test_build_without_created_campaign_stops_before_submit(self):
        workflow = FinalReviewWorkflow.start_new()
        update = workflow.complete_current_step(0, "build completed without package")
        self.assertEqual(update.status, "failed")
        self.assertEqual(update.heading, "无法准备最终审校")
        self.assertIsNone(workflow.current_step())

    def test_succeeded_status_downloads_and_ingests_report(self):
        workflow = FinalReviewWorkflow(["status"], "C:/tmp/review/manifest.json")
        update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED")
        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "download")
        workflow.complete_current_step(0, "downloaded")
        self.assertEqual(workflow.current_step().key, "final-review-ingest-results")
        done = workflow.complete_current_step(0, "Findings: 2")
        self.assertEqual(done.status, "done")
        self.assertIn("选择", done.message)

    def test_query_only_stops_before_download(self):
        workflow = FinalReviewWorkflow(["status"], "C:/tmp/review/manifest.json")
        workflow.only_query = True
        update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED")
        self.assertEqual(update.status, "ready")
        self.assertFalse(update.should_continue)
        self.assertIsNone(workflow.current_step())

    def test_selected_findings_use_one_local_preview_step(self):
        workflow = FinalReviewWorkflow.create_revisions("C:/tmp/review/manifest.json", ["f1", "f2"])
        self.assertEqual(
            workflow.current_step().args,
            [
                "final-review-create-revisions",
                "C:/tmp/review/manifest.json",
                "--finding-id",
                "f1",
                "--finding-id",
                "f2",
            ],
        )
        output = "\n".join(
            [
                "Created final-review revision package: C:/tmp/revisions",
                "Recoverable revision items: 2",
                "Failure items: 0",
                "Preview JSONL: C:/tmp/revisions/revision_preview.jsonl",
            ]
        )
        update = workflow.complete_current_step(0, output)
        self.assertEqual(update.heading, "订正预览已生成")
        self.assertTrue(
            workflow.manifest_path.replace("\\", "/").endswith("revisions/manifest.json")
        )

    def test_resume_final_review_manifest_uses_final_review_commands(self):
        workflow = resume_workflow(
            WorkMode.FINAL_REVIEW,
            "C:/tmp/review/manifest.json",
            {"mode": "final_review", "job_name": "jobs/1", "job_state": "JOB_STATE_RUNNING"},
        )
        self.assertIsInstance(workflow, FinalReviewWorkflow)
        self.assertEqual(
            workflow.current_step().args,
            ["status", "C:/tmp/review/manifest.json", "--output", "json", "--non-interactive"],
        )

    def test_resume_does_not_treat_missing_counts_as_complete(self):
        workflow = FinalReviewWorkflow.resume_manifest(
            "C:/tmp/review/manifest.json",
            {"mode": "final_review", "summary": {"status_counts": {}}},
        )
        self.assertEqual(workflow.current_step().key, "final-review-resume")

    def test_build_sync_campaign_switches_to_run_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = os.path.join(tmp, "campaign")
            os.makedirs(package_dir)
            with open(
                os.path.join(package_dir, "manifest.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"mode": "final_review", "execution_strategy": "sync"},
                    handle,
                )

            workflow = FinalReviewWorkflow.start_new()
            update = workflow.complete_current_step(
                0,
                f"Created final-review campaign: {package_dir}",
            )

            self.assertTrue(update.should_continue)
            self.assertEqual(workflow.current_step().key, "final-review-run-sync")
            self.assertEqual(
                workflow.current_step().args,
                [
                    "final-review-run-sync",
                    workflow.manifest_path,
                    "--output",
                    "json",
                    "--non-interactive",
                ],
            )
            done = workflow.complete_current_step(
                0,
                json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "status": "completed",
                            "done_delta": 2,
                            "failed_delta": 0,
                            "finding_count": 1,
                        },
                    }
                ),
            )
            self.assertEqual(done.status, "done")
            self.assertIsNone(workflow.current_step())

    def test_resume_sync_manifest_runs_sync_step(self):
        workflow = resume_workflow(
            WorkMode.FINAL_REVIEW,
            "C:/tmp/review/manifest.json",
            {
                "mode": "final_review",
                "execution_strategy": "sync",
                "summary": {"status_counts": {}, "unit_count": 2},
            },
        )
        self.assertIsInstance(workflow, FinalReviewWorkflow)
        self.assertEqual(workflow.current_step().key, "final-review-run-sync")

    def test_sync_run_with_failed_units_reports_retry(self):
        workflow = FinalReviewWorkflow(["final-review-run-sync"], "C:/tmp/review/manifest.json")
        update = workflow.complete_current_step(
            0,
            json.dumps(
                {
                    "ok": True,
                    "result": {
                        "status": "completed",
                        "done_delta": 1,
                        "failed_delta": 1,
                        "finding_count": 0,
                    },
                }
            ),
        )
        self.assertEqual(update.status, "ready")
        self.assertIn("重试", update.message)
        self.assertEqual(
            workflow.current_step().key,
            "final-review-run-sync",
        )

    def test_sync_run_error_envelope_stops_workflow(self):
        workflow = FinalReviewWorkflow(
            ["final-review-run-sync"],
            "C:/tmp/review/manifest.json",
        )
        update = workflow.complete_current_step(
            4,
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "FINAL_REVIEW_SYNC_ABORTED",
                        "message": "authentication failed",
                    },
                }
            ),
        )

        self.assertEqual(update.status, "failed")
        self.assertEqual(update.heading, "最终审校同步执行中断")
        self.assertIsNone(workflow.current_step())

    def test_sync_run_partial_failure_with_strict_exit_keeps_retry_step(self):
        workflow = FinalReviewWorkflow(
            ["final-review-run-sync"],
            "C:/tmp/review/manifest.json",
        )
        update = workflow.complete_current_step(
            4,
            json.dumps(
                {
                    "ok": True,
                    "status": "failed",
                    "result": {
                        "status": "failed",
                        "done_delta": 1,
                        "failed_delta": 1,
                        "finding_count": 0,
                    },
                }
            ),
        )

        self.assertEqual(update.status, "ready")
        self.assertEqual(workflow.current_step().key, "final-review-run-sync")


if __name__ == "__main__":
    unittest.main()
