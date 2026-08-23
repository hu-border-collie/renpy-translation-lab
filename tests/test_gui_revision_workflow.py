import unittest
import json

from gui_qt.revision_workflow import (
    RevisionBatchWorkflow,
    RevisionProposalConfirmWorkflow,
    RevisionProposalImportWorkflow,
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
            [
                "submit",
                "C:\\Games\\Example\\work\\logs\\batch_jobs\\rev1\\manifest.json",
                "--output",
                "json",
                "--non-interactive",
            ],
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

        self.assertEqual(
            workflow.current_step().args,
            ["submit", "C:\\package\\manifest.json", "--output", "json", "--non-interactive"],
        )

    def test_resume_submitted_manifest_starts_from_status(self):
        workflow = RevisionBatchWorkflow.resume_manifest(
            "C:\\package\\manifest.json",
            {"job_name": "batches/example"},
        )

        self.assertEqual(
            workflow.current_step().args,
            ["status", "C:\\package\\manifest.json", "--output", "json", "--non-interactive"],
        )

    def test_resume_succeeded_manifest_starts_from_download_and_preview(self):
        manifest_path = r"C:\package\manifest.json"
        workflow = RevisionBatchWorkflow.resume_manifest(
            manifest_path,
            {"job_name": "batches/example", "job_state": "JOB_STATE_SUCCEEDED"},
        )

        self.assertEqual(
            workflow.current_step().args,
            ["download", manifest_path, "--output", "json", "--non-interactive"],
        )
        workflow.complete_current_step(0, "Saved results to: " + r"C:\package\results.jsonl" + "\n")
        self.assertEqual(workflow.current_step().args, ["preview-revisions", manifest_path])

    def test_status_succeeded_continues_to_download_and_preview(self):
        workflow = RevisionBatchWorkflow.resume_latest("C:\\package\\manifest.json")

        status_update = workflow.complete_current_step(0, "State: JOB_STATE_SUCCEEDED\n")
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
            workflow.current_step().args, ["preview-revisions", "C:\\package\\manifest.json"]
        )

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

    def test_proposal_import_runs_one_local_preview_step(self):
        workflow = RevisionProposalImportWorkflow(r"C:\review\proposals.jsonl")
        self.assertEqual(
            workflow.current_step().args,
            ["import-revision-proposals", r"C:\review\proposals.jsonl"],
        )
        update = workflow.complete_current_step(
            0,
            "Revision proposal import status: previewed\n"
            "Manifest: C:\\package\\manifest.json\n",
        )
        self.assertEqual(update.status, "done")
        self.assertIn("预览已生成", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_proposal_import_passes_explicit_corpus_manifest(self):
        workflow = RevisionProposalImportWorkflow(
            r"C:\review\proposals.jsonl",
            r"D:\exports\revision_corpus_manifest.json",
        )
        self.assertEqual(
            workflow.current_step().args,
            [
                "import-revision-proposals",
                r"C:\review\proposals.jsonl",
                "--corpus-manifest",
                r"D:\exports\revision_corpus_manifest.json",
            ],
        )

    def test_stale_proposal_import_is_not_presented_as_writable(self):
        workflow = RevisionProposalImportWorkflow("proposal.jsonl")
        update = workflow.complete_current_step(
            0, "Revision proposal import status: stale\n"
        )
        self.assertEqual(update.status, "failed")
        self.assertIn("未通过安全校验", update.heading)

    def test_staged_proposal_import_uses_machine_contract_and_exposes_candidates(self):
        workflow = RevisionProposalImportWorkflow(
            r"C:\review\proposals.jsonl",
            stage=True,
            operation_identity="operation-1",
        )
        self.assertEqual(
            workflow.current_step().args,
            [
                "import-revision-proposals",
                r"C:\review\proposals.jsonl",
                "--stage",
                "--operation-identity",
                "operation-1",
                "--strict-exit-codes",
                "--output",
                "json",
                "--non-interactive",
            ],
        )
        output = {
            "schema_version": 1,
            "command": "import-revision-proposals",
            "ok": True,
            "status": "staged",
            "result": {
                "candidate_count": 2,
                "selectable_count": 1,
                "selected_count": 0,
                "unselected_count": 1,
                "invalid_count": 1,
                "stale_count": 0,
                "conflict_count": 0,
                "candidates": [{"identity_v2": "occ-1", "status": "valid"}],
            },
            "artifacts": {"staged_selection": r"C:\jobs\staged_selection.json"},
            "warnings": [],
            "error": None,
        }
        update = workflow.complete_current_step(0, json.dumps(output))
        self.assertEqual(update.status, "done")
        self.assertEqual(workflow.stage_result["candidates"][0]["identity_v2"], "occ-1")
        self.assertIn("明确勾选", update.message)

    def test_confirm_workflow_requests_serialized_selection_and_reports_preview(self):
        workflow = RevisionProposalConfirmWorkflow(
            r"C:\jobs\staged_selection.json",
            r"C:\jobs\selection.json",
            operation_identity="operation-1",
        )
        self.assertEqual(
            workflow.current_step().args,
            [
                "confirm-revision-proposals",
                r"C:\jobs\staged_selection.json",
                "--selection-file",
                r"C:\jobs\selection.json",
                "--strict-exit-codes",
                "--output",
                "json",
                "--non-interactive",
            ],
        )
        output = {
            "schema_version": 1,
            "command": "confirm-revision-proposals",
            "ok": True,
            "status": "previewed",
            "result": {"selected_count": 1},
            "artifacts": {"manifest": r"C:\jobs\confirmed\manifest.json"},
            "warnings": [],
            "error": None,
        }
        update = workflow.complete_current_step(0, json.dumps(output))
        self.assertEqual(update.status, "done")
        self.assertEqual(workflow.manifest_path, r"C:\jobs\confirmed\manifest.json")
        self.assertIn("预览已生成", update.heading)

    def test_staged_import_discards_late_result_after_project_switch(self):
        workflow = RevisionProposalImportWorkflow(
            r"C:\review\proposals.jsonl",
            stage=True,
            operation_identity="operation-1",
        )
        workflow.stage_result = {"candidate_count": 1}
        update = workflow.stale_update()
        self.assertEqual(update.status, "stale")
        self.assertIsNone(workflow.stage_result)
        self.assertIsNone(workflow.current_step())


if __name__ == "__main__":
    unittest.main()
