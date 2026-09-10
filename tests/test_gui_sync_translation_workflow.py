"""Durable Sync GUI workflow tests (#348 P3)."""
from __future__ import annotations

import json
import unittest

import cli_contract
from gui_qt.sync_translation_workflow import (
    SyncTranslationWorkflow,
    durable_sync_facts,
    parse_machine_envelope,
)

RUN_ID = "sync-run-v1-20260910T120000Z"


def machine_args(*args: str) -> list[str]:
    return [*args, "--output", "json", "--non-interactive"]


def snapshot(
    *,
    run_id: str = RUN_ID,
    status: str = "completed",
    next_action: str = "check",
    requests: dict | None = None,
    items: dict | None = None,
    usage: dict | None = None,
    cancellation: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "run_status": status,
        "revision": 3,
        "plan": {"plan_id": "plan-1", "plan_fingerprint": "abc"},
        "cancellation": cancellation or {"requested": False},
        "progress": {
            "requests": requests
            or {
                "total": 4,
                "active_leaf_total": 4,
                "pending": 0,
                "in_flight": 0,
                "succeeded": 4,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "superseded": 0,
                "outcome_unknown": 0,
                "cancelled": 0,
            },
            "items": items or {"expected": 10, "accepted": 10, "unresolved": 0},
            "attempts": {"total": 4, "unknown": 0, "late_ignored": 0},
            "usage": usage or {"known_calls": 4, "total_tokens": 1200},
        },
        "next_action": next_action,
    }


def success_output(
    command: str,
    *,
    status: str = "completed",
    result: dict | None = None,
    artifacts: dict | None = None,
) -> str:
    envelope = cli_contract.success_envelope(
        command,
        status=status,
        result=result or {},
        artifacts=artifacts or {},
    )
    return json.dumps(envelope, ensure_ascii=False)


def error_output(command: str, code: str, message: str = "failed") -> str:
    envelope = cli_contract.error_envelope(
        command,
        code=code,
        message=message,
        suggested_action="inspect_durable_sync_run",
    )
    return json.dumps(envelope, ensure_ascii=False)


class DurableSyncWorkflowPlanningTests(unittest.TestCase):
    def test_start_plan_uses_durable_machine_interface(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()

        step = workflow.current_step()

        self.assertEqual(step.script_basename, "gemini_translate_batch.py")
        self.assertEqual(step.key, "sync-start")
        self.assertEqual(step.args, machine_args("sync-start"))

    def test_query_latest_never_resumes_by_itself(self) -> None:
        workflow = SyncTranslationWorkflow.query_latest()

        step = workflow.current_step()

        self.assertEqual(step.key, "sync-status")
        self.assertEqual(step.args, machine_args("sync-status", "--latest"))

    def test_cancel_and_apply_use_explicit_run_ids(self) -> None:
        cancel = SyncTranslationWorkflow.cancel_run(RUN_ID)
        apply_workflow = SyncTranslationWorkflow.apply_run(RUN_ID)

        self.assertEqual(cancel.current_step().args, machine_args("sync-cancel", RUN_ID))
        self.assertEqual(
            apply_workflow.current_step().args,
            machine_args("apply", RUN_ID),
        )

    def test_derive_unknown_flags_are_explicit(self) -> None:
        retry = SyncTranslationWorkflow.derive_run(RUN_ID, retry_unknown=True)
        exclude = SyncTranslationWorkflow.derive_run(RUN_ID, exclude_unknown=True)

        self.assertEqual(
            retry.current_step().args,
            machine_args(
                "sync-derive",
                RUN_ID,
                "--retry-unknown",
                "--ack-duplicate-billing-risk",
            ),
        )
        self.assertEqual(
            exclude.current_step().args,
            machine_args("sync-derive", RUN_ID, "--exclude-unknown"),
        )


class DurableSyncWorkflowRunTests(unittest.TestCase):
    def test_completed_run_chains_check_then_preview(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()
        start = success_output(
            "sync-start",
            result=snapshot(),
            artifacts={"run_dir": "C:/logs/sync_runs/run1"},
        )

        update = workflow.complete_current_step(0, start)

        self.assertEqual(update.status, "running")
        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.run_id, RUN_ID)
        self.assertEqual(workflow.run_dir, "C:/logs/sync_runs/run1")
        self.assertEqual(workflow.current_step().key, "check")
        self.assertEqual(workflow.current_step().args, machine_args("check", RUN_ID))

        check = success_output(
            "check",
            status="safe",
            result={
                "check": {
                    "check_status": "ready",
                    "safety_level": "safe",
                    "writeback_gate": {"decision": "allow"},
                    "quality_gate": {"decision": "pass", "warning_count": 0},
                }
            },
            artifacts={"manifest": "C:/runs/run1/check_manifest.json"},
        )
        update = workflow.complete_current_step(0, check)

        self.assertEqual(update.status, "done")
        self.assertTrue(workflow.preview_ready)
        self.assertIn("可写回", update.heading)
        self.assertEqual(
            workflow.check_manifest,
            "C:/runs/run1/check_manifest.json",
        )
        self.assertIn("检查清单", "\n".join(update.facts))
        self.assertIsNone(workflow.current_step())

    def test_completed_with_errors_still_reaches_check(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-start",
                status="completed_with_errors",
                result=snapshot(status="completed_with_errors", next_action="derive"),
            ),
        )

        self.assertEqual(update.status, "running")
        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "check")

    def test_failed_run_stops_without_check(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()

        update = workflow.complete_current_step(
            1,
            success_output(
                "sync-start",
                status="failed",
                result=snapshot(status="failed", next_action="derive"),
            ),
        )

        self.assertEqual(update.status, "failed")
        self.assertFalse(workflow.preview_ready)
        self.assertIsNone(workflow.current_step())

    def test_cancelled_run_reports_without_check(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()

        update = workflow.complete_current_step(
            1,
            success_output(
                "sync-start",
                status="cancelled",
                result=snapshot(status="cancelled", next_action="derive"),
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("已取消", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_blocked_check_never_offers_writeback(self) -> None:
        workflow = SyncTranslationWorkflow.check_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "check",
                status="blocked",
                result={
                    "check": {
                        "check_status": "blocked",
                        "safety_level": "block",
                        "writeback_gate": {"decision": "deny", "reason": "structure"},
                        "quality_gate": {"decision": "pass"},
                    }
                },
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertFalse(workflow.preview_ready)
        self.assertIn("不能写回", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_quality_warning_keeps_preview_ready(self) -> None:
        workflow = SyncTranslationWorkflow.check_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "check",
                status="ready_with_warnings",
                result={
                    "check": {
                        "check_status": "ready_with_warnings",
                        "safety_level": "safe",
                        "writeback_gate": {"decision": "allow"},
                        "quality_gate": {
                            "decision": "needs_review",
                            "warning_count": 2,
                            "acknowledged_count": 0,
                            "blocker_count": 0,
                        },
                    }
                },
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertTrue(workflow.preview_ready)


class DurableSyncRecoveryTests(unittest.TestCase):
    def test_status_derives_next_step_from_public_snapshot(self) -> None:
        workflow = SyncTranslationWorkflow.resume_latest()

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-status",
                status="running",
                result=snapshot(status="running", next_action="resume"),
            ),
        )

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "sync-resume")
        self.assertEqual(workflow.current_step().args, machine_args("sync-resume", RUN_ID))

        update = workflow.complete_current_step(
            0,
            success_output("sync-resume", result=snapshot()),
        )

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "check")

    def test_status_check_action_chains_check_for_terminal_run(self) -> None:
        workflow = SyncTranslationWorkflow.resume_latest()

        update = workflow.complete_current_step(
            0,
            success_output("sync-status", result=snapshot(next_action="check")),
        )

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.current_step().key, "check")

        update = workflow.complete_current_step(
            0,
            success_output(
                "check",
                status="safe",
                result={
                    "check": {
                        "check_status": "ready",
                        "safety_level": "safe",
                        "writeback_gate": {"decision": "allow"},
                    }
                },
            ),
        )
        self.assertEqual(update.status, "done")
        self.assertTrue(workflow.preview_ready)

    def test_status_derive_requires_manual_confirmation(self) -> None:
        workflow = SyncTranslationWorkflow.resume_latest()

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-status",
                status="failed",
                result=snapshot(status="failed", next_action="derive"),
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertFalse(update.should_continue)
        self.assertIn("派生", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_status_without_run_is_a_warning(self) -> None:
        workflow = SyncTranslationWorkflow.query_latest()

        update = workflow.complete_current_step(
            1, error_output("sync-status", "SYNC_RUN_NOT_FOUND")
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("SYNC_RUN_NOT_FOUND", " ".join(update.facts))
        self.assertIsNone(workflow.current_step())

    def test_worker_interruption_locates_run_without_resuming(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()

        update = workflow.complete_current_step(1, "Traceback disconnected\n")

        self.assertTrue(update.should_continue)
        self.assertIn("定位", update.heading)
        self.assertEqual(workflow.current_step().key, "sync-status")
        self.assertEqual(
            workflow.current_step().args,
            machine_args("sync-status", "--latest"),
        )

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-status",
                status="running",
                result=snapshot(status="running", next_action="resume"),
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("已定位", update.heading)
        self.assertEqual(workflow.run_id, RUN_ID)
        self.assertIsNone(workflow.current_step())

    def test_recover_derives_wording_from_next_action(self) -> None:
        workflow = SyncTranslationWorkflow.start_new()
        workflow.complete_current_step(1, "")

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-status",
                status="failed",
                result=snapshot(status="failed", next_action="derive"),
            ),
        )

        self.assertIn("派生", update.message)
        self.assertNotIn("检查预览", update.message)

    def test_interruption_after_known_run_keeps_terminal_message(self) -> None:
        workflow = SyncTranslationWorkflow.resume_run(RUN_ID)

        update = workflow.complete_current_step(1, "")

        self.assertEqual(update.status, "failed")
        self.assertIn("不会", update.message)
        self.assertIsNone(workflow.current_step())

    def test_cancel_requested_reports_waiting_not_cancelled(self) -> None:
        workflow = SyncTranslationWorkflow.cancel_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-cancel",
                status="cancel_requested",
                result=snapshot(status="cancel_requested", next_action="wait_cancel"),
            ),
        )

        self.assertEqual(update.status, "waiting")
        self.assertIn("等待", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_cancel_that_lost_the_race_reports_missed(self) -> None:
        workflow = SyncTranslationWorkflow.cancel_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-cancel",
                status="completed",
                result=snapshot(status="completed", next_action="check"),
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("取消未生效", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_cancel_reports_terminal_snapshot(self) -> None:
        workflow = SyncTranslationWorkflow.cancel_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-cancel",
                status="cancelled",
                result=snapshot(status="cancelled", next_action="derive"),
            ),
        )

        self.assertEqual(update.status, "warning")
        self.assertIn("取消", update.heading)
        self.assertIsNone(workflow.current_step())

    def test_apply_reports_idempotent_result(self) -> None:
        workflow = SyncTranslationWorkflow.apply_run(RUN_ID)

        update = workflow.complete_current_step(
            0,
            success_output(
                "apply",
                status="already_applied",
                result={
                    "apply": {
                        "state": "applied",
                        "last_apply_result": "already_applied",
                        "applied_files": ["a.rpy"],
                    }
                },
            ),
        )

        self.assertEqual(update.status, "done")
        self.assertIn("已写回过", update.heading)
        self.assertIn("1 个", " ".join(update.facts))

    def test_derive_completed_run_chains_check(self) -> None:
        workflow = SyncTranslationWorkflow.derive_run(RUN_ID, exclude_unknown=True)

        update = workflow.complete_current_step(
            0,
            success_output(
                "sync-derive",
                result=snapshot(run_id="sync-run-v1-new", status="completed", next_action="check"),
            ),
        )

        self.assertTrue(update.should_continue)
        self.assertEqual(workflow.run_id, "sync-run-v1-new")
        self.assertEqual(workflow.current_step().key, "check")
        self.assertEqual(
            workflow.current_step().args,
            machine_args("check", "sync-run-v1-new"),
        )


class DurableSyncFactTests(unittest.TestCase):
    def test_facts_show_progress_and_unknown_risk(self) -> None:
        payload = snapshot(
            requests={
                "total": 4,
                "pending": 1,
                "in_flight": 1,
                "succeeded": 1,
                "retryable_failed": 0,
                "terminal_failed": 0,
                "superseded": 0,
                "outcome_unknown": 1,
                "cancelled": 0,
            },
            items={"expected": 10, "accepted": 7, "unresolved": 3},
            usage={"known_calls": 2, "total_tokens": 900, "estimated_cost": 0.02},
        )

        facts = durable_sync_facts(payload, run_dir="C:/runs/run1")

        joined = "\n".join(facts)
        self.assertIn(RUN_ID, joined)
        self.assertIn("C:/runs/run1", joined)
        self.assertIn("已完成", joined)
        self.assertIn("未解决 3", joined)
        self.assertIn("结果未知", joined)
        self.assertIn("900", joined)

    def test_envelope_parser_rejects_human_output(self) -> None:
        self.assertIsNone(parse_machine_envelope("Sync preview manifest: C:/run/manifest.json\n"))
        self.assertIsNotNone(
            parse_machine_envelope(success_output("sync-status", result=snapshot()))
        )


if __name__ == "__main__":
    unittest.main()
