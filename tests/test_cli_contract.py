import io
import json
import unittest
from pathlib import Path

import cli_contract


class CliContractTests(unittest.TestCase):
    def test_success_envelope_has_stable_shape_and_json_safe_values(self):
        envelope = cli_contract.success_envelope(
            "check",
            status="safe",
            result={
                "path": Path("manifest.json"),
                "values": {2, 1},
                "frozen_values": frozenset({4, 3}),
            },
            artifacts={"manifest": Path("manifest.json")},
            warnings=["note"],
        )

        self.assertEqual(envelope["schema_version"], 1)
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "safe")
        self.assertEqual(envelope["result"]["path"], "manifest.json")
        self.assertEqual(envelope["result"]["frozen_values"], [3, 4])
        self.assertEqual(envelope["result"]["values"], [1, 2])
        self.assertIsNone(envelope["error"])

    def test_error_envelope_exposes_programmatic_error_fields(self):
        envelope = cli_contract.error_envelope(
            "apply",
            code="COMMAND_REFUSED",
            message="unsafe",
            suggested_action="run_check_again",
            details={"reason": "stale"},
        )

        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["status"], "failed")
        self.assertEqual(envelope["error"]["code"], "COMMAND_REFUSED")
        self.assertFalse(envelope["error"]["retryable"])
        self.assertEqual(envelope["error"]["suggested_action"], "run_check_again")

    def test_write_json_envelope_writes_one_parseable_document(self):
        stream = io.StringIO()
        envelope = cli_contract.success_envelope("doctor", status="ready")

        cli_contract.write_json_envelope(envelope, stream)

        self.assertEqual(json.loads(stream.getvalue()), envelope)
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_write_json_envelope_supports_compact_serialization(self):
        stream = io.StringIO()
        envelope = cli_contract.success_envelope("status", status="pending")

        cli_contract.write_json_envelope(envelope, stream, compact=True)

        self.assertEqual(json.loads(stream.getvalue()), envelope)
        self.assertNotIn("\n  ", stream.getvalue())
        self.assertNotIn(": ", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_project_fields_preserves_nested_shape_and_omits_missing_paths(self):
        envelope = cli_contract.success_envelope(
            "check",
            status="warn",
            result={"check": {"safety_level": "warn", "failure_items": 2}},
            artifacts={"manifest": "manifest.json"},
        )

        projected = cli_contract.project_fields(
            envelope,
            [
                "command",
                "status",
                "result.check.safety_level",
                "artifacts.missing",
            ],
        )

        self.assertEqual(
            projected,
            {
                "command": "check",
                "status": "warn",
                "result": {"check": {"safety_level": "warn"}},
            },
        )

    def test_error_classification_exposes_stable_machine_actions(self):
        stale = cli_contract.classify_error(
            "Manifest or results changed after the last check.",
            exception_type="SystemExit",
        )
        retryable = cli_contract.classify_error(
            "429 RESOURCE_EXHAUSTED",
            exception_type="RuntimeError",
        )

        self.assertEqual(stale["code"], "STALE_STATE")
        self.assertEqual(stale["suggested_action"], "run_check_again")
        self.assertEqual(stale["exit_code"], cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(retryable["code"], "REMOTE_RETRYABLE")
        self.assertTrue(retryable["retryable"])
        self.assertEqual(retryable["exit_code"], cli_contract.EXIT_RETRYABLE)

        for message in (
            "Manifest has no valid check summary. Run check before apply.",
            "Manifest check summary was produced by an older check contract. "
            "Run check again before apply.",
        ):
            with self.subTest(message=message):
                preflight = cli_contract.classify_error(
                    message,
                    exception_type="SystemExit",
                )
                self.assertEqual(preflight["code"], "STALE_STATE")
                self.assertEqual(
                    preflight["suggested_action"],
                    "run_check_again",
                )
                self.assertEqual(
                    preflight["exit_code"],
                    cli_contract.EXIT_INVALID_STATE,
                )

        quotation = cli_contract.classify_error(
            "quotation ready",
            exception_type="RuntimeError",
        )
        self.assertEqual(quotation["code"], "INTERNAL_ERROR")

    def test_strict_exit_code_maps_successful_workflow_states(self):
        warn = cli_contract.success_envelope("check", status="warn")
        blocked = cli_contract.success_envelope("doctor", status="blocked")
        pending = cli_contract.success_envelope(
            "status",
            status="JOB_STATE_PENDING",
        )
        reconciliation_ready = cli_contract.success_envelope(
            "reconcile-project-snapshots",
            status="ready",
        )
        reconciliation_attention = cli_contract.success_envelope(
            "reconcile-project-snapshots",
            status="attention",
        )
        proposal_partial = cli_contract.success_envelope(
            "import-revision-proposals", status="partial"
        )
        proposal_stale = cli_contract.success_envelope(
            "import-revision-proposals", status="stale"
        )
        preview_ready = cli_contract.success_envelope(
            "preview-revisions", status="ready"
        )
        preview_warn = cli_contract.success_envelope(
            "preview-revisions", status="ready_with_warnings"
        )
        preview_blocked = cli_contract.success_envelope(
            "preview-revisions", status="blocked"
        )
        sync_ready = cli_contract.success_envelope(
            "sync-revisions", status="previewed"
        )
        sync_no_work = cli_contract.success_envelope(
            "sync-revisions", status="no_work"
        )
        sync_applied = cli_contract.success_envelope(
            "sync-revisions", status="applied"
        )
        sync_partial = cli_contract.success_envelope(
            "sync-revisions", status="partial"
        )
        sync_blocked = cli_contract.success_envelope(
            "sync-revisions", status="blocked"
        )

        self.assertEqual(
            cli_contract.strict_exit_code(warn),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(blocked),
            cli_contract.EXIT_BLOCKED,
        )
        self.assertEqual(cli_contract.strict_exit_code(pending), 0)
        self.assertEqual(
            cli_contract.strict_exit_code(reconciliation_ready),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(reconciliation_attention),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(proposal_partial),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(proposal_stale),
            cli_contract.EXIT_BLOCKED,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(preview_ready),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(preview_warn),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(preview_blocked),
            cli_contract.EXIT_BLOCKED,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(sync_ready),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(sync_no_work),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(sync_applied),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(sync_partial),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(sync_blocked),
            cli_contract.EXIT_BLOCKED,
        )
        unknown = cli_contract.success_envelope("check", status="unknown")
        unclassified_error = cli_contract.error_envelope(
            "apply",
            code="COMMAND_REFUSED",
            message="stopped",
        )
        self.assertEqual(
            cli_contract.strict_exit_code(unknown),
            cli_contract.EXIT_INVALID_STATE,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(unclassified_error),
            cli_contract.EXIT_BLOCKED,
        )

    def test_parse_result_envelope_accepts_shared_contract(self):
        envelope = cli_contract.success_envelope(
            "status",
            status="JOB_STATE_RUNNING",
        )

        parsed = cli_contract.parse_result_envelope(json.dumps(envelope))

        self.assertEqual(parsed, envelope)

    def test_parse_result_envelope_rejects_unknown_or_incomplete_schema(self):
        with self.assertRaises(ValueError):
            cli_contract.parse_result_envelope('{"schema_version": 999}')
        with self.assertRaises(ValueError):
            cli_contract.parse_result_envelope("[]")


if __name__ == "__main__":
    unittest.main()
