import json
import unittest

from foundation.cli_contract import success_envelope

from gui_qt.reports.check_report import WritebackSummary
from gui_qt.workflows.retry_workflow import (
    create_retry_followup_workflow,
    describe_retry_followup_button,
    retry_followup_next_step,
    retry_followup_workflow_ready,
)

PARENT_PATH = r"C:\pkg\manifest.json"
RETRY_PATH = r"C:\pkg\retry_parts\retry1\manifest.json"


def _retry_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "retry_of_manifest": PARENT_PATH,
        "job_name": "",
        "job_state": "",
    }
    manifest.update(overrides)
    return manifest


class GuiRetryWorkflowTests(unittest.TestCase):
    def test_create_retry_followup_workflow_restores_parent_pointer(self):
        workflow = create_retry_followup_workflow(
            RETRY_PATH,
            PARENT_PATH,
        )

        self.assertEqual(workflow.restore_latest_manifest_path, PARENT_PATH)
        self.assertEqual(
            workflow.current_step().args,
            ["status", RETRY_PATH, "--local", "--output", "json", "--non-interactive"],
        )

    def test_create_retry_followup_workflow_passes_submit_max_cost(self):
        workflow = create_retry_followup_workflow(
            RETRY_PATH,
            PARENT_PATH,
            submit_max_cost=4.5,
        )
        workflow.complete_current_step(0, json.dumps(success_envelope(
            "status", status="LOCAL_ONLY", result={
                "next_action": "submit", "retry_parent_manifest_path": PARENT_PATH,
            },
        )))

        self.assertEqual(
            workflow.current_step().args,
            ["submit", RETRY_PATH, "--max-cost", "4.5", "--output", "json", "--non-interactive"],
        )

    def test_create_retry_followup_refuses_a_different_parent_from_cli(self):
        workflow = create_retry_followup_workflow(RETRY_PATH, PARENT_PATH)
        update = workflow.complete_current_step(0, json.dumps(success_envelope(
            "status", status="LOCAL_ONLY", result={
                "next_action": "submit", "retry_parent_manifest_path": "other.json",
            },
        )))
        self.assertEqual(update.status, "failed")
        self.assertIsNone(workflow.current_step())

    def test_describe_retry_followup_button_for_recheck_step(self):
        label, tooltip = describe_retry_followup_button(
            RETRY_PATH,
            _retry_manifest(
                job_name="batches/retry",
                job_state="JOB_STATE_SUCCEEDED",
                last_check_summary={"safety_level": "safe"},
            ),
            PARENT_PATH,
        )

        self.assertEqual(label, "继续补译")
        self.assertIn("校验", tooltip)

    def test_retry_followup_workflow_ready_requires_confirmed_preview(self):
        summary = WritebackSummary(
            status="warn",
            heading="",
            message="",
            facts=[],
            findings=[],
            can_apply=False,
            manifest_path=PARENT_PATH,
        )
        parent_manifest = {
            "last_check_summary": {"safety_level": "warn"},
            "last_retry_manifest_path": RETRY_PATH,
        }
        retry_manifest = _retry_manifest()

        self.assertFalse(
            retry_followup_workflow_ready(
                summary,
                parent_manifest=parent_manifest,
                retry_manifest=retry_manifest,
                retry_manifest_path=RETRY_PATH,
                parent_manifest_path=PARENT_PATH,
                confirmed_parent_paths=set(),
                supports_translation_writeback=True,
            )
        )
        self.assertTrue(
            retry_followup_workflow_ready(
                summary,
                parent_manifest=parent_manifest,
                retry_manifest=retry_manifest,
                retry_manifest_path=RETRY_PATH,
                parent_manifest_path=PARENT_PATH,
                confirmed_parent_paths={PARENT_PATH},
                supports_translation_writeback=True,
            )
        )

    def test_retry_followup_next_step_after_submit_is_status(self):
        step = retry_followup_next_step(
            RETRY_PATH,
            _retry_manifest(job_name="batches/retry"),
            PARENT_PATH,
        )

        self.assertIsNotNone(step)
        self.assertEqual(step.key, "status")
        self.assertEqual(step.args, ["status", RETRY_PATH, "--output", "json", "--non-interactive"])


if __name__ == "__main__":
    unittest.main()
