"""GUI final-review workflow tests (#255 PR C)."""
from __future__ import annotations

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
        self.assertEqual(workflow.current_step().args, [
            "final-review-create-revisions", "C:/tmp/review/manifest.json",
            "--finding-id", "f1", "--finding-id", "f2",
        ])
        output = "\n".join([
            "Created final-review revision package: C:/tmp/revisions",
            "Recoverable revision items: 2",
            "Failure items: 0",
            "Preview JSONL: C:/tmp/revisions/revision_preview.jsonl",
        ])
        update = workflow.complete_current_step(0, output)
        self.assertEqual(update.heading, "订正预览已生成")
        self.assertTrue(workflow.manifest_path.replace("\\", "/").endswith("revisions/manifest.json"))

    def test_resume_final_review_manifest_uses_final_review_commands(self):
        workflow = resume_workflow(
            WorkMode.FINAL_REVIEW,
            "C:/tmp/review/manifest.json",
            {"mode": "final_review", "job_name": "jobs/1", "job_state": "JOB_STATE_RUNNING"},
        )
        self.assertIsInstance(workflow, FinalReviewWorkflow)
        self.assertEqual(workflow.current_step().args, ["status", "C:/tmp/review/manifest.json"])


if __name__ == "__main__":
    unittest.main()
