import json
import os
import tempfile
import unittest

import batch_submit_recovery

from gui_qt.batch_workflow_support import (
    build_recover_submit_cli_args,
    build_submit_cli_args,
    format_cost_estimate_facts,
    get_uncertain_submit_kind,
    load_cost_estimate_facts_from_manifest,
    plan_unsubmitted_workflow_steps,
    resolve_submit_max_cost,
)
from gui_qt.doctor_report import doctor_report_to_parsed, summarize_doctor_report
from gui_qt.translation_workflow import TranslationWorkflow


class GuiBatchWorkflowSupportTests(unittest.TestCase):
    def test_resolve_submit_max_cost(self):
        self.assertIsNone(resolve_submit_max_cost({}))
        self.assertIsNone(resolve_submit_max_cost({"batch": {"submit_max_cost": 0}}))
        self.assertEqual(
            resolve_submit_max_cost({"batch": {"submit_max_cost": 12.5}}),
            12.5,
        )

    def test_build_submit_cli_args_appends_max_cost(self):
        self.assertEqual(
            build_submit_cli_args("pkg/manifest.json"),
            ["submit", "pkg/manifest.json"],
        )
        self.assertEqual(
            build_submit_cli_args("pkg/manifest.json", 5),
            ["submit", "pkg/manifest.json", "--max-cost", "5"],
        )

    def test_format_cost_estimate_facts(self):
        facts = format_cost_estimate_facts(
            {
                "model": "gemini-3.1-flash-lite",
                "estimated_input_tokens": 1000,
                "estimated_output_tokens_max": 2000,
                "estimated_cost_min": 0.1,
                "estimated_cost_max": 0.9,
                "currency": "USD",
            }
        )
        self.assertIn("估算模型：gemini-3.1-flash-lite", facts)
        self.assertIn("估算成本：0.1000 至 0.9000 USD", facts)

    def test_load_cost_estimate_facts_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cost_estimate": {
                            "estimated_input_tokens": 10,
                            "estimated_output_tokens_max": 20,
                            "estimated_cost_min": 0.01,
                            "estimated_cost_max": 0.02,
                            "currency": "USD",
                        }
                    },
                    handle,
                )
            facts = load_cost_estimate_facts_from_manifest(manifest_path)
            self.assertIn("估算输入 token：10", facts)

    def test_build_submit_cli_args_appends_resume_for_upload_pending_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = os.path.join(tmp_dir, "package")
            os.makedirs(package_dir)
            jsonl_path = os.path.join(package_dir, "requests.jsonl")
            manifest_path = os.path.join(package_dir, "manifest.json")
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "job_name": "",
                        "submit_state": batch_submit_recovery.SUBMIT_STATE_UPLOADED,
                        "uploaded_file_name": "files/uploaded-1",
                        "submit_attempt_id": "attempt-1",
                        "request_checksum": "abc",
                    },
                    handle,
                )
            batch_submit_recovery.append_submit_journal_entry(
                package_dir,
                {
                    "event": batch_submit_recovery.EVENT_UPLOAD_COMPLETED,
                    "submit_attempt_id": "attempt-1",
                    "uploaded_file_name": "files/uploaded-1",
                },
            )
            self.assertEqual(
                build_submit_cli_args(manifest_path),
                ["submit", manifest_path, "--resume"],
            )

    def test_plan_unsubmitted_workflow_steps_prefers_recover_submit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = os.path.join(tmp_dir, "package")
            os.makedirs(package_dir)
            manifest_path = os.path.join(package_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"job_name": ""}, handle)
            batch_submit_recovery.append_submit_journal_entry(
                package_dir,
                {
                    "event": batch_submit_recovery.EVENT_JOB_CREATED,
                    "submit_attempt_id": "attempt-2",
                    "job_name": "batches/job-2",
                },
            )
            self.assertEqual(
                plan_unsubmitted_workflow_steps(manifest_path),
                ["recover-submit", "status"],
            )
            self.assertEqual(
                get_uncertain_submit_kind(manifest_path),
                "job_created_uncommitted",
            )
            self.assertEqual(
                build_recover_submit_cli_args(manifest_path),
                ["recover-submit", manifest_path],
            )

    def test_translation_workflow_submit_uses_max_cost(self):
        workflow = TranslationWorkflow.start_new(submit_max_cost=3.5)
        workflow.manifest_path = r"C:\package\manifest.json"
        workflow._pending_steps[:] = ["submit", "status"]

        self.assertEqual(
            workflow.current_step().args,
            ["submit", r"C:\package\manifest.json", "--max-cost", "3.5"],
        )

    def test_doctor_report_shows_tl_subdir(self):
        parsed = doctor_report_to_parsed(
            {
                "base_dir": "C:/work",
                "tl_dir": "C:/work/game/tl/schinese",
                "tl_subdir": "game/tl/schinese",
                "tl_exists": True,
                "language": "schinese",
                "mode": "existing_tl_only",
                "counts": {},
                "warnings": [],
                "recommendations": [],
            }
        )
        summary = summarize_doctor_report(parsed, exit_code=0, api_key_count=1)
        self.assertTrue(any("TL 路径：game/tl/schinese" in fact for fact in summary.facts))


if __name__ == "__main__":
    unittest.main()