import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gemini_translate_batch as batch


class BatchCliContractTests(unittest.TestCase):
    @staticmethod
    def _machine_command_argv(command):
        if command == "export-project-snapshot":
            return [command, "--version-id", "test-version"]
        if command == "reconcile-project-snapshots":
            return [command, "base-snapshot.json", "target-snapshot.json"]
        if command == "build-translation-records":
            return [command, "snapshot.json", "manifest.json"]
        if command == "build-reuse-candidates":
            return [
                command,
                "base-snapshot.json",
                "target-snapshot.json",
                "reconciliation.json",
                "records.json",
            ]
        if command == "import-reuse-decisions":
            return [command, "reuse.json", "decisions.jsonl"]
        if command == "export-reuse-results":
            return [command, "reuse.json", "manifest.json"]
        if command == "import-revision-proposals":
            return [command, "proposals.jsonl"]
        if command == "merge-keywords-to-glossary":
            return [command, "candidates.jsonl"]
        if command in {
            "final-review-status",
            "final-review-export",
            "final-review-resume",
            "final-review-ingest-results",
            "final-review-create-revisions",
        }:
            return [command, "campaign-manifest.json"]
        return [command]

    def test_core_commands_accept_json_output_after_subcommand(self):
        parser = batch.build_arg_parser()

        for command in sorted(batch.MACHINE_OUTPUT_COMMANDS):
            with self.subTest(command=command):
                argv = self._machine_command_argv(command)
                args = parser.parse_args([*argv, "--output", "json"])
                self.assertEqual(args.command, command)
                self.assertEqual(args.output, "json")
                self.assertFalse(args.strict_exit_codes)
                strict_args = parser.parse_args(
                    [*argv, "--output", "json", "--strict-exit-codes"]
                )
                strict_invocation = parser.parse_args(
                    [*argv, "--non-interactive", "--require-explicit-target"]
                )
                self.assertTrue(strict_invocation.non_interactive)
                self.assertTrue(strict_invocation.require_explicit_target)

                self.assertTrue(strict_args.strict_exit_codes)

    def test_core_commands_expose_output_trimming_arguments(self):
        parser = batch.build_arg_parser()

        args = parser.parse_args(
            [
                "status",
                "manifest.json",
                "--output",
                "json",
                "--compact",
                "--fields",
                "status",
                "result.job_state,artifacts.manifest",
                "--output-file",
                "result.json",
            ]
        )

        self.assertTrue(args.compact)
        self.assertEqual(
            batch._machine_field_paths(args),
            ["status", "result.job_state", "artifacts.manifest"],
        )
        self.assertEqual(args.output_file, "result.json")

    def test_compact_fields_project_machine_stdout(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "job_state": "JOB_STATE_PENDING",
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "status",
                    "manifest.json",
                    "--output",
                    "json",
                    "--fields",
                    "command,status,result.job_state,artifacts.manifest",
                    "--compact",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "command": "status",
                "status": "JOB_STATE_PENDING",
                "result": {"job_state": "JOB_STATE_PENDING"},
                "artifacts": {"manifest": "C:/jobs/demo/manifest.json"},
            },
        )
        self.assertNotIn(": ", stdout.getvalue())

    def test_invalid_field_path_returns_structured_usage_error(self):
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = batch.main(
                [
                    "status",
                    "manifest.json",
                    "--output",
                    "json",
                    "--fields",
                    "result..job_state",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "INVALID_FIELD_PATH")
        self.assertEqual(
            payload["error"]["details"]["semantic_exit_code"],
            batch.cli_contract.EXIT_USAGE,
        )

    def test_parser_error_with_json_output_returns_structured_usage_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                ["status", "--output", "json", "--unknown-flag"]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["schema_version"], batch.cli_contract.CLI_SCHEMA_VERSION)
        self.assertEqual(payload["command"], "status")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "ARGUMENT_PARSE_ERROR")
        self.assertIn("unrecognized arguments: --unknown-flag", payload["error"]["message"])
        self.assertEqual(
            payload["error"]["details"]["semantic_exit_code"],
            batch.cli_contract.EXIT_USAGE,
        )
        self.assertFalse(payload["error"]["details"]["workflow_started"])
        self.assertIn("usage:", stderr.getvalue())
        self.assertIn("unrecognized arguments: --unknown-flag", stderr.getvalue())

    def test_parser_error_supports_equals_json_form_and_compact_output(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                ["status", "--output=json", "--compact", "--unknown-flag"]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "ARGUMENT_PARSE_ERROR")
        self.assertNotIn("\n  ", stdout.getvalue())
        self.assertTrue(stdout.getvalue().endswith("\n"))

    def test_parser_error_with_json_output_reports_missing_option_value(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["status", "--output", "json", "--fields"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["command"], "status")
        self.assertEqual(payload["error"]["code"], "ARGUMENT_PARSE_ERROR")
        self.assertIn("argument --fields", payload["error"]["message"])
        self.assertIn("argument --fields", stderr.getvalue())

    def test_machine_option_detection_stops_at_double_dash(self):
        text_stdout = io.StringIO()
        text_stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(text_stdout),
            contextlib.redirect_stderr(text_stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            batch.main(["status", "--", "--output", "json"])

        self.assertEqual(raised.exception.code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(text_stdout.getvalue(), "")
        self.assertIn("usage:", text_stderr.getvalue())

        machine_stdout = io.StringIO()
        machine_stderr = io.StringIO()
        with (
            contextlib.redirect_stdout(machine_stdout),
            contextlib.redirect_stderr(machine_stderr),
        ):
            exit_code = batch.main(
                [
                    "status",
                    "--output",
                    "json",
                    "--",
                    "--compact",
                    "--unknown-flag",
                ]
            )

        payload = json.loads(machine_stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "ARGUMENT_PARSE_ERROR")
        self.assertIn("\n  ", machine_stdout.getvalue())

    def test_earliest_machine_parse_error_uses_generic_cli_command(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["command"], "cli")
        self.assertEqual(payload["error"]["code"], "ARGUMENT_PARSE_ERROR")

    def test_non_json_output_value_stays_on_text_argparse_boundary(self):
        stderr = io.StringIO()

        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            batch.main(["status", "--output", "xml"])

        self.assertEqual(raised.exception.code, batch.cli_contract.EXIT_USAGE)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_text_parser_error_preserves_argparse_system_exit(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            batch.main(["status", "--unknown-flag"])

        self.assertEqual(raised.exception.code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("usage:", stderr.getvalue())
        self.assertIn("unrecognized arguments: --unknown-flag", stderr.getvalue())

    def test_field_projection_does_not_change_strict_exit_code(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_check_summary": {"safety_level": "warn"},
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "check",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                    "--fields",
                    "status",
                ]
            )

        self.assertEqual(exit_code, batch.cli_contract.EXIT_NEEDS_ACTION)
        self.assertEqual(json.loads(stdout.getvalue()), {"status": "warn"})

    def test_output_file_is_atomic_and_leaves_stdout_empty(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "job_state": "JOB_STATE_SUCCEEDED",
        }
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "nested" / "status.json"
            with (
                mock.patch.object(batch, "dispatch_command", return_value=manifest),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = batch.main(
                    [
                        "status",
                        "manifest.json",
                        "--output",
                        "json",
                        "--output-file",
                        str(output_path),
                    ]
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(payload["status"], "JOB_STATE_SUCCEEDED")
        self.assertEqual(payload["artifacts"]["output_file"], str(output_path))


    def test_output_file_preflight_failure_is_structured_before_dispatch(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            blocking_file = Path(tmp) / "not-a-directory"
            blocking_file.write_text("occupied", encoding="utf-8")
            output_path = blocking_file / "status.json"
            with (
                mock.patch.object(batch, "dispatch_command") as dispatch,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = batch.main(
                    [
                        "status",
                        "manifest.json",
                        "--output",
                        "json",
                        "--strict-exit-codes",
                        "--output-file",
                        str(output_path),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertFalse(payload["error"]["details"]["workflow_started"])
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_WRITE_FAILED")
        self.assertFalse(payload["error"]["details"]["command_completed"])
        self.assertEqual(
            payload["error"]["details"]["output_file"],
            os.path.abspath(str(output_path)),
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        dispatch.assert_not_called()

    def test_output_file_write_race_reports_completed_command_on_stdout(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "job_state": "JOB_STATE_SUCCEEDED",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(batch, "_preflight_output_file"),
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            mock.patch.object(
                batch,
                "atomic_write",
                side_effect=PermissionError("target became read-only"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                [
                    "status",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                    "--output-file",
                    "status.json",
                ]
            )

        payload = json.loads(stdout.getvalue())
        details = payload["error"]["details"]
        self.assertTrue(details["workflow_started"])
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_WRITE_FAILED")
        self.assertTrue(details["command_completed"])
        self.assertTrue(details["original_ok"])
        self.assertEqual(details["original_status"], "JOB_STATE_SUCCEEDED")
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_discovery_output_file_failure_returns_nonzero_and_json_stdout(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(batch, "_preflight_output_file"),
            mock.patch.object(
                batch,
                "atomic_write",
                side_effect=PermissionError("target became read-only"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                ["capabilities", "--output-file", "capabilities.json"]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_WRITE_FAILED")
        self.assertTrue(payload["error"]["details"]["command_completed"])
        self.assertTrue(payload["error"]["details"]["workflow_started"])
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_non_interactive_requires_explicit_manifest_target(self):
        for command in sorted(batch.EXPLICIT_TARGET_COMMANDS):
            with self.subTest(command=command):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with (
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    exit_code = batch.main(
                        [
                            command,
                            "--output",
                            "json",
                            "--non-interactive",
                            "--strict-exit-codes",
                        ]
                    )

                payload = json.loads(stdout.getvalue())
                self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
                self.assertEqual(
                    payload["error"]["code"],
                    "EXPLICIT_TARGET_REQUIRED",
                )
                self.assertEqual(
                    payload["error"]["suggested_action"],
                    "pass_manifest_path",
                )
                self.assertEqual(
                    payload["error"]["details"]["required_argument"],
                    "target",
                )

    def test_explicit_target_guard_is_opt_in_and_skips_targetless_commands(self):
        parser = batch.build_arg_parser()
        accepted = (
            ["doctor", "--non-interactive"],
            ["build", "--non-interactive"],
            ["status"],
            ["status", "manifest.json", "--non-interactive"],
            ["apply", "manifest.json", "--require-explicit-target"],
        )

        for argv in accepted:
            with self.subTest(argv=argv):
                batch.validate_machine_invocation(parser.parse_args(argv))

    def test_explicit_target_error_is_structured_without_strict_exit_codes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = batch.main(
                [
                    "status",
                    "--output",
                    "json",
                    "--require-explicit-target",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "EXPLICIT_TARGET_REQUIRED")
        self.assertEqual(
            payload["error"]["details"]["semantic_exit_code"],
            batch.cli_contract.EXIT_INVALID_STATE,
        )

    def test_invalid_manifest_json_returns_invalid_state_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text("{", encoding="utf-8")
            with (
                mock.patch.object(batch, "initialize_batch_logging"),
                mock.patch.object(batch.legacy, "load_config"),
                mock.patch.object(batch.legacy, "load_translator_settings"),
                mock.patch.object(batch.legacy, "load_glossary"),
                mock.patch.object(batch, "load_batch_settings"),
                mock.patch.object(batch, "print_banner"),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = batch.main(
                    [
                        "status",
                        str(manifest_path),
                        "--output",
                        "json",
                        "--non-interactive",
                        "--strict-exit-codes",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(payload["error"]["code"], "INVALID_MANIFEST_JSON")
        self.assertEqual(
            payload["error"]["suggested_action"],
            "rebuild_or_repair_manifest",
        )
        self.assertNotIn("Traceback", stderr.getvalue())


    def test_core_preconditions_raise_typed_contract_errors(self):
        with self.assertRaises(batch.cli_contract.MachineContractError) as path_error:
            batch._canonical_manifest_dir("relative/tl", "tl_dir")
        self.assertEqual(path_error.exception.code_name, "INVALID_MANIFEST_PATH")

        with (
            mock.patch.object(
                batch,
                "manifest_project_identity",
                return_value={
                    "tl_dir": "C:/different/tl",
                    "base_dir": "",
                    "source": "manifest",
                },
            ),
            mock.patch.object(batch.legacy, "TL_DIR", "C:/active/tl"),
            self.assertRaises(batch.cli_contract.MachineContractError) as project_error,
        ):
            batch.require_manifest_project_match({}, "apply")
        self.assertEqual(
            project_error.exception.code_name,
            "MANIFEST_PROJECT_MISMATCH",
        )
        self.assertEqual(
            project_error.exception.semantic_exit_code,
            batch.cli_contract.EXIT_INVALID_STATE,
        )

        decode_error = json.JSONDecodeError("invalid", "{", 0)
        with (
            mock.patch.object(batch, "load_json_file", return_value={}),
            mock.patch.object(
                batch.batch_cost_estimate,
                "attach_cost_estimate_to_manifest",
                side_effect=decode_error,
            ),
            self.assertRaises(batch.cli_contract.MachineContractError) as jsonl_error,
        ):
            batch.ensure_manifest_cost_estimate({"input_jsonl_path": "input.jsonl"})
        self.assertEqual(jsonl_error.exception.code_name, "INVALID_BATCH_INPUT_JSON")

        with (
            mock.patch.object(
                batch,
                "write_apply_failure_report",
                return_value="C:/jobs/demo/apply_failure_report.json",
            ),
            mock.patch.object(batch, "save_manifest"),
            self.assertRaises(batch.cli_contract.MachineContractError) as apply_error,
        ):
            batch.fail_apply_preflight(
                {},
                "unsafe_check_status",
                "Last check is not safe.",
            )
        self.assertEqual(apply_error.exception.code_name, "UNSAFE_CHECK_STATUS")
        self.assertEqual(
            apply_error.exception.semantic_exit_code,
            batch.cli_contract.EXIT_BLOCKED,
        )

    def test_typed_core_error_bypasses_message_classifier_in_machine_mode(self):
        stdout = io.StringIO()

        def dispatch(_parser, _args):
            return batch._canonical_manifest_dir("relative/tl", "tl_dir")

        with (
            mock.patch.object(batch, "dispatch_command", side_effect=dispatch),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "status",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(payload["error"]["code"], "INVALID_MANIFEST_PATH")
        self.assertEqual(
            payload["error"]["suggested_action"],
            "rebuild_or_repair_manifest",
        )
    def test_machine_result_builder_covers_manifest_workflow(self):
        args = SimpleNamespace(target="")
        base_manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "mode": "translation",
            "job_state": "JOB_STATE_PENDING",
            "summary": {"item_count": 2},
            "last_check_summary": {"safety_level": "safe"},
            "apply_summary": {"applied_lines": 2},
            "next_split_manifest_path": "C:/jobs/part02/manifest.json",
            "applied_at": "2026-07-29T12:00:00",
        }
        expected_status = {
            "build": "JOB_STATE_PENDING",
            "submit": "JOB_STATE_PENDING",
            "status": "JOB_STATE_PENDING",
            "download": "downloaded",
            "check": "safe",
            "apply": "applied",
        }

        for command, status in expected_status.items():
            with self.subTest(command=command):
                envelope = batch.build_machine_success_envelope(
                    command,
                    dict(base_manifest),
                    args,
                )
                self.assertTrue(envelope["ok"])
                self.assertEqual(envelope["command"], command)
                self.assertEqual(envelope["status"], status)
                self.assertEqual(
                    envelope["artifacts"]["manifest"],
                    "C:/jobs/demo/manifest.json",
                )
                if command == "apply":
                    self.assertEqual(
                        envelope["result"]["apply"]["next_split_manifest"],
                        "C:/jobs/part02/manifest.json",
                    )

    def test_machine_result_builder_covers_revision_apply_states(self):
        args = SimpleNamespace(target="")
        revision_manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "mode": "revision",
            "revision_apply_state": "partial",
            "revision_apply_summary": {"applied_files": 1, "applied_lines": 1},
        }
        envelope = batch.build_machine_success_envelope(
            "apply-revisions",
            dict(revision_manifest),
            args,
        )
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "partial")
        self.assertEqual(envelope["result"]["revision_apply_state"], "partial")
        self.assertEqual(envelope["result"]["revision_apply"]["applied_files"], 1)

        blocked = dict(revision_manifest)
        blocked["revision_apply_state"] = "blocked"
        blocked["revision_apply_blocked_reason"] = "all_items_blocked"
        envelope = batch.build_machine_success_envelope(
            "apply-revisions",
            blocked,
            args,
        )
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "blocked")
        self.assertEqual(
            envelope["result"]["revision_apply_blocked_reason"],
            "all_items_blocked",
        )

    def test_apply_revisions_is_registered_for_machine_output(self):
        self.assertIn("apply-revisions", batch.MACHINE_OUTPUT_COMMANDS)
        args = batch.build_arg_parser().parse_args(
            ["apply-revisions", "C:/jobs/demo/manifest.json", "--output", "json"]
        )
        self.assertEqual(args.output, "json")

    def test_noncore_workflow_commands_are_registered_for_machine_output(self):
        expected = {
            "build-revisions",
            "preview-revisions",
            "sync-revisions",
            "build-keywords",
            "export-keywords",
            "sync-keywords",
            "merge-keywords-to-glossary",
            "final-review-build",
            "final-review-status",
            "final-review-export",
            "final-review-resume",
            "final-review-ingest-results",
            "final-review-create-revisions",
        }
        self.assertTrue(expected <= batch.MACHINE_OUTPUT_COMMANDS)
        self.assertTrue(
            {
                "preview-revisions",
                "export-keywords",
                "final-review-status",
                "final-review-export",
                "final-review-resume",
                "final-review-ingest-results",
                "final-review-create-revisions",
            }
            <= batch.EXPLICIT_TARGET_COMMANDS
        )
        parser = batch.build_arg_parser()
        for command in sorted(expected):
            with self.subTest(command=command):
                argv = self._machine_command_argv(command)
                args = parser.parse_args([*argv, "--output", "json"])
                self.assertEqual(args.output, "json")

    def test_builder_commands_translate_manifest_paths_into_envelopes(self):
        args = SimpleNamespace(target="")
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "mode": "revision",
            "job_state": "",
            "summary": {"item_count": 2},
            "input_jsonl_path": "C:/jobs/demo/input.jsonl",
        }
        with mock.patch.object(
            batch, "load_manifest", return_value=manifest
        ) as load_manifest_mock:
            for command in (
                "build-revisions",
                "build-keywords",
                "final-review-build",
            ):
                with self.subTest(command=command):
                    envelope = batch.build_machine_success_envelope(
                        command,
                        "C:/jobs/demo/manifest.json",
                        args,
                    )
                    self.assertTrue(envelope["ok"])
                    self.assertEqual(envelope["status"], "LOCAL_ONLY")
                    self.assertEqual(
                        envelope["artifacts"]["manifest"],
                        "C:/jobs/demo/manifest.json",
                    )
                    self.assertEqual(
                        envelope["artifacts"]["input_jsonl"],
                        "C:/jobs/demo/input.jsonl",
                    )
                with self.subTest(command=command, case="no_work"):
                    calls_before = load_manifest_mock.call_count
                    no_work = batch.build_machine_success_envelope(
                        command,
                        "",
                        args,
                    )
                    self.assertEqual(no_work["status"], "no_work")
                    self.assertEqual(
                        no_work["result"]["reason"],
                        "no_source_items",
                    )
                    self.assertEqual(
                        load_manifest_mock.call_count,
                        calls_before,
                    )

    def test_revision_preview_machine_envelope_exposes_gates_and_artifacts(self):
        args = SimpleNamespace(target="")
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_revision_preview": {
                "jsonl_path": "C:/jobs/demo/revision_preview.jsonl",
                "markdown_path": "C:/jobs/demo/revision_preview.md",
                "quality_findings_path": "C:/jobs/demo/quality_findings.jsonl",
                "quality_findings_count": 2,
                "check_status": "ready_with_warnings",
                "writeback_gate": {"can_apply": True},
                "quality_gate": {"has_warnings": True},
                "summary": {"preview_entry_count": 3},
            },
            "final_review_source": {"manifest_path": "C:/jobs/fr/manifest.json"},
        }
        for command in ("preview-revisions", "final-review-create-revisions"):
            with self.subTest(command=command):
                envelope = batch.build_machine_success_envelope(
                    command,
                    dict(manifest),
                    args,
                )
                self.assertTrue(envelope["ok"])
                self.assertEqual(envelope["status"], "ready_with_warnings")
                self.assertEqual(
                    envelope["result"]["check_status"],
                    "ready_with_warnings",
                )
                self.assertTrue(envelope["result"]["writeback_gate"]["can_apply"])
                self.assertEqual(envelope["result"]["quality_findings_count"], 2)
                self.assertEqual(
                    envelope["artifacts"]["revision_preview_jsonl"],
                    "C:/jobs/demo/revision_preview.jsonl",
                )
                self.assertEqual(
                    envelope["artifacts"]["quality_findings"],
                    "C:/jobs/demo/quality_findings.jsonl",
                )
        create = batch.build_machine_success_envelope(
            "final-review-create-revisions",
            dict(manifest),
            args,
        )
        self.assertEqual(
            create["result"]["final_review_source"]["manifest_path"],
            "C:/jobs/fr/manifest.json",
        )
        plain = batch.build_machine_success_envelope(
            "preview-revisions",
            dict(manifest),
            args,
        )
        self.assertNotIn("final_review_source", plain["result"])

    def test_sync_revisions_machine_envelope_reuses_preview_and_apply_shapes(self):
        preview_manifest = {
            "_manifest_path": "C:/jobs/sync-rev/manifest.json",
            "last_revision_preview": {
                "jsonl_path": "C:/jobs/sync-rev/revision_preview.jsonl",
                "markdown_path": "C:/jobs/sync-rev/revision_preview.md",
                "quality_findings_path": "C:/jobs/sync-rev/quality_findings.jsonl",
                "quality_findings_count": 1,
                "check_status": "ready",
                "writeback_gate": {"can_apply": True},
                "quality_gate": {"has_warnings": False},
                "summary": {"preview_entry_count": 2},
            },
        }
        preview = batch.build_machine_success_envelope(
            "sync-revisions",
            dict(preview_manifest),
            SimpleNamespace(apply=False),
        )
        self.assertTrue(preview["ok"])
        self.assertEqual(preview["status"], "ready")
        self.assertEqual(
            preview["result"]["manifest_path"],
            "C:/jobs/sync-rev/manifest.json",
        )
        self.assertTrue(preview["result"]["writeback_gate"]["can_apply"])
        self.assertEqual(
            preview["artifacts"]["revision_preview_jsonl"],
            "C:/jobs/sync-rev/revision_preview.jsonl",
        )
        self.assertNotIn("revision_apply_state", preview["result"])

        applied = {
            "_manifest_path": "C:/jobs/sync-rev/manifest.json",
            "revision_apply_state": "applied",
            "revision_applied_at": "2026-08-23T00:00:00",
            "revision_apply_summary": {"applied_files": 1},
            "last_revision_preview": preview_manifest["last_revision_preview"],
        }
        apply_envelope = batch.build_machine_success_envelope(
            "sync-revisions",
            dict(applied),
            SimpleNamespace(apply=True),
        )
        self.assertTrue(apply_envelope["ok"])
        self.assertEqual(apply_envelope["status"], "applied")
        self.assertEqual(apply_envelope["result"]["revision_apply_state"], "applied")
        self.assertEqual(
            apply_envelope["result"]["revision_apply"]["applied_files"],
            1,
        )

    def test_keyword_export_machine_envelope_lists_review_artifacts(self):
        args = SimpleNamespace(target="C:/jobs/kw/manifest.json")
        keyword_manifest = {"_manifest_path": "C:/jobs/kw/manifest.json"}
        export = {
            "jsonl_path": "C:/jobs/kw/keyword_candidates.jsonl",
            "markdown_path": "C:/jobs/kw/keyword_candidates.md",
            "summary_jsonl_path": "C:/jobs/kw/keyword_chunk_summaries.jsonl",
            "summary_markdown_path": "C:/jobs/kw/keyword_chunk_summaries.md",
            "summary": {"candidate_count_deduped": 5},
            "history_evidence": {"occurrence_count": 4},
        }
        with mock.patch.object(batch, "load_manifest", return_value=keyword_manifest):
            envelope = batch.build_machine_success_envelope(
                "export-keywords",
                dict(export),
                args,
            )
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "completed")
        self.assertEqual(
            envelope["result"]["manifest_path"],
            "C:/jobs/kw/manifest.json",
        )
        self.assertEqual(
            envelope["result"]["summary"]["candidate_count_deduped"],
            5,
        )
        self.assertEqual(
            envelope["artifacts"]["keyword_candidates"],
            "C:/jobs/kw/keyword_candidates.jsonl",
        )
        self.assertEqual(
            envelope["artifacts"]["keyword_chunk_summaries"],
            "C:/jobs/kw/keyword_chunk_summaries.jsonl",
        )

    def test_sync_keywords_machine_envelope_uses_returned_manifest_path(self):
        args = SimpleNamespace()
        export = {
            "manifest_path": "C:/jobs/sync-kw/manifest.json",
            "jsonl_path": "C:/jobs/sync-kw/keyword_candidates.jsonl",
            "markdown_path": "C:/jobs/sync-kw/keyword_candidates.md",
            "summary_jsonl_path": "C:/jobs/sync-kw/keyword_chunk_summaries.jsonl",
            "summary_markdown_path": "C:/jobs/sync-kw/keyword_chunk_summaries.md",
            "summary": {"candidate_count_deduped": 3},
            "history_evidence": {"occurrence_count": 1},
        }
        with mock.patch.object(batch, "load_manifest") as load_manifest:
            envelope = batch.build_machine_success_envelope(
                "sync-keywords",
                dict(export),
                args,
            )
        load_manifest.assert_not_called()
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "completed")
        self.assertEqual(
            envelope["result"]["manifest_path"],
            "C:/jobs/sync-kw/manifest.json",
        )
        self.assertEqual(
            envelope["result"]["summary"]["candidate_count_deduped"],
            3,
        )
        self.assertEqual(
            envelope["artifacts"]["keyword_candidates"],
            "C:/jobs/sync-kw/keyword_candidates.jsonl",
        )

    def test_sync_commands_return_dispatch_payload_for_machine_envelope(self):
        parser = batch.build_arg_parser()
        keyword_payload = {"manifest_path": "C:/jobs/sync-kw/manifest.json"}
        revision_payload = {"_manifest_path": "C:/jobs/sync-rev/manifest.json"}
        with (
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch, "sync_keyword_candidates", return_value=keyword_payload
            ) as sync_keywords,
        ):
            args = parser.parse_args(["sync-keywords", "--output", "json"])
            self.assertIs(batch.dispatch_command(parser, args), keyword_payload)
            sync_keywords.assert_called_once()
        with (
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch, "sync_revisions", return_value=revision_payload
            ) as sync_revisions,
        ):
            args = parser.parse_args(["sync-revisions", "--apply", "--output", "json"])
            self.assertIs(batch.dispatch_command(parser, args), revision_payload)
            sync_revisions.assert_called_once_with(
                display_name_override="",
                skip_prepare=False,
                chunk_size=0,
                limit=0,
                offset=0,
                output_jsonl="",
                output_markdown="",
                apply=True,
                force=False,
                api_key_index=None,
            )

    def test_keyword_merge_machine_envelope_reports_preview_merge_and_no_work(self):
        args = SimpleNamespace(target="candidates.jsonl")
        base = {
            "candidates_path": "candidates.jsonl",
            "glossary_path": "glossary.json",
            "candidates_read": 4,
            "accepted": 2,
            "overwritten": 1,
            "skipped_duplicate": 1,
            "skipped_low_confidence": 0,
            "skipped_empty": 0,
            "skipped_user": 0,
            "backup_path": "glossary.backup.json",
        }
        preview = batch.build_machine_success_envelope(
            "merge-keywords-to-glossary",
            {**base, "dry_run": True, "wrote_glossary": False},
            args,
        )
        self.assertEqual(preview["status"], "previewed")
        self.assertFalse(preview["result"]["wrote_glossary"])
        self.assertEqual(
            preview["artifacts"]["glossary_backup"],
            "glossary.backup.json",
        )

        merged = batch.build_machine_success_envelope(
            "merge-keywords-to-glossary",
            {**base, "dry_run": False, "wrote_glossary": True},
            args,
        )
        self.assertEqual(merged["status"], "merged")

        no_work = batch.build_machine_success_envelope(
            "merge-keywords-to-glossary",
            {**base, "dry_run": False, "wrote_glossary": False},
            args,
        )
        self.assertEqual(no_work["status"], "no_work")

    def test_final_review_campaign_envelopes_cover_status_export_resume_ingest(self):
        args = SimpleNamespace(target="campaign.json", force=False)
        campaign_status = {
            "status": "running",
            "manifest_path": "C:/jobs/fr/manifest.json",
            "package_dir": "C:/jobs/fr",
            "unit_count": 3,
            "finding_count": 1,
            "status_counts": {"done": 2, "pending": 1},
        }
        status = batch.build_machine_success_envelope(
            "final-review-status",
            dict(campaign_status),
            args,
        )
        self.assertTrue(status["ok"])
        self.assertEqual(status["status"], "running")
        self.assertEqual(
            status["result"]["status_counts"],
            {"done": 2, "pending": 1},
        )
        self.assertEqual(
            status["artifacts"]["manifest"],
            "C:/jobs/fr/manifest.json",
        )

        export = batch.build_machine_success_envelope(
            "final-review-export",
            {
                "jsonl_path": "C:/jobs/fr/findings.jsonl",
                "markdown_path": "C:/jobs/fr/report.md",
                "finding_count": 1,
                "status": dict(campaign_status),
            },
            args,
        )
        self.assertEqual(export["status"], "completed")
        self.assertEqual(export["result"]["campaign_status"], "running")
        self.assertEqual(
            export["artifacts"]["findings_jsonl"],
            "C:/jobs/fr/findings.jsonl",
        )

        resume = batch.build_machine_success_envelope(
            "final-review-resume",
            {
                "paths": {
                    "manifest": "C:/jobs/fr/manifest.json",
                    "package_dir": "C:/jobs/fr",
                    "review_units": "C:/jobs/fr/review_units.jsonl",
                    "report": "C:/jobs/fr/report.md",
                },
                "run_count": 2,
                "skip_count": 1,
                "to_run_unit_ids": ["u1", "u2"],
                "status": {"status": "running"},
            },
            args,
        )
        self.assertEqual(resume["status"], "rebuilt")
        self.assertEqual(resume["result"]["run_count"], 2)
        self.assertEqual(resume["result"]["to_run_unit_ids"], ["u1", "u2"])
        self.assertEqual(
            resume["artifacts"]["review_units"],
            "C:/jobs/fr/review_units.jsonl",
        )

        up_to_date = batch.build_machine_success_envelope(
            "final-review-resume",
            {
                "paths": {},
                "run_count": 0,
                "skip_count": 3,
                "status": {"status": "done"},
            },
            args,
        )
        self.assertEqual(up_to_date["status"], "no_work")

        ingest = batch.build_machine_success_envelope(
            "final-review-ingest-results",
            {
                "paths": {
                    "manifest": "C:/jobs/fr/manifest.json",
                    "package_dir": "C:/jobs/fr",
                    "findings": "C:/jobs/fr/findings.jsonl",
                    "quality_findings": "C:/jobs/fr/quality_findings.jsonl",
                    "report": "C:/jobs/fr/report.md",
                },
                "summary": {
                    "result_rows": 2,
                    "done_units": 2,
                    "failed_units": 0,
                    "finding_count": 1,
                },
                "status": {"status": "done"},
            },
            args,
        )
        self.assertEqual(ingest["status"], "done")
        self.assertEqual(ingest["result"]["summary"]["done_units"], 2)
        self.assertEqual(
            ingest["artifacts"]["quality_findings"],
            "C:/jobs/fr/quality_findings.jsonl",
        )

    def test_final_review_status_machine_mode_wraps_campaign_in_envelope(self):
        campaign = {
            "status": "running",
            "manifest_path": "C:/jobs/fr/manifest.json",
            "unit_count": 3,
            "finding_count": 1,
            "status_counts": {"done": 2, "pending": 1},
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        seen = {}

        def fake_status(target=None, as_json=False):
            seen["as_json"] = as_json
            print("campaign text summary")
            return dict(campaign)

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch, "run_final_review_status", side_effect=fake_status
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                [
                    "final-review-status",
                    "C:/jobs/fr/manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "running")
        self.assertEqual(
            payload["result"]["status_counts"],
            {"done": 2, "pending": 1},
        )
        self.assertEqual(
            payload["artifacts"]["manifest"],
            "C:/jobs/fr/manifest.json",
        )
        # Machine mode must not depend on the legacy --json flag; text
        # diagnostics stay on stderr and the envelope reads the handler
        # return value.
        self.assertFalse(seen["as_json"])
        self.assertIn("campaign text summary", stderr.getvalue())

    def test_merge_keywords_machine_mode_requires_yes_or_dry_run(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        merge = mock.Mock()
        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "resolve_keyword_candidates_path",
                return_value="candidates.jsonl",
            ),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "merge_keywords_to_glossary",
                merge,
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                [
                    "merge-keywords-to-glossary",
                    "candidates.jsonl",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(payload["error"]["code"], "INTERACTIVE_REVIEW_UNSUPPORTED")
        self.assertEqual(
            payload["error"]["suggested_action"],
            "pass_yes_or_dry_run",
        )
        merge.assert_not_called()

    def test_merge_keywords_non_interactive_text_mode_requires_yes_or_dry_run(self):
        stderr = io.StringIO()
        merge = mock.Mock()
        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "resolve_keyword_candidates_path",
                return_value="candidates.jsonl",
            ),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "merge_keywords_to_glossary",
                merge,
            ),
            contextlib.redirect_stderr(stderr),
        ):
            with self.assertRaisesRegex(SystemExit, "--yes or --dry-run"):
                batch.main(
                    [
                        "merge-keywords-to-glossary",
                        "candidates.jsonl",
                        "--non-interactive",
                    ]
                )
        merge.assert_not_called()

    def test_merge_keywords_machine_mode_runs_non_interactively_with_yes(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        summary = batch.keyword_glossary_merge.MergeSummary(
            candidates_read=2,
            accepted=1,
            wrote_glossary=True,
            glossary_path="glossary.json",
            candidates_path="candidates.jsonl",
            backup_path="glossary.backup.json",
        )
        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "resolve_keyword_candidates_path",
                return_value="candidates.jsonl",
            ),
            mock.patch.object(
                batch.keyword_glossary_merge,
                "merge_keywords_to_glossary",
                return_value=summary,
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                [
                    "merge-keywords-to-glossary",
                    "candidates.jsonl",
                    "--yes",
                    "--output",
                    "json",
                ]
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "merged")
        self.assertEqual(payload["result"]["accepted"], 1)
        self.assertEqual(payload["artifacts"]["glossary"], "glossary.json")

    def test_proposal_import_machine_envelope_exposes_status_actions_and_artifacts(self):
        args = SimpleNamespace(command="import-revision-proposals")
        envelope = batch.build_machine_success_envelope(
            "import-revision-proposals",
            {
                "status": "stale",
                "input_count": 2,
                "requested_selected_count": 1,
                "selected_count": 0,
                "candidate_count": 0,
                "diagnostics": [{"code": "CURRENT_TRANSLATION_STALE"}],
                "suggested_action": "re_export_corpus_and_regenerate_proposals",
                "paths": {"import_report": "C:/jobs/import/report.json"},
            },
            args,
        )
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["status"], "stale")
        self.assertEqual(envelope["result"]["requested_selected_count"], 1)
        self.assertEqual(envelope["result"]["selected_count"], 0)
        self.assertEqual(
            envelope["result"]["suggested_action"],
            "re_export_corpus_and_regenerate_proposals",
        )
        self.assertEqual(
            envelope["artifacts"]["import_report"],
            "C:/jobs/import/report.json",
        )

    def test_quality_ack_machine_envelope_is_stable_and_reports_no_work(self):
        listed_args = SimpleNamespace(finding_ids=[], all_findings=False)
        listed = batch.build_machine_success_envelope(
            "quality-ack",
            {
                "manifest": {
                    "_manifest_path": "C:/jobs/demo/manifest.json",
                    "last_quality_findings_path": "C:/jobs/demo/quality_findings.jsonl",
                },
                "old_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "new_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "selected_ids": set(),
                "unmatched": [],
                "previous_acknowledged_finding_ids": [],
                "acknowledged_finding_ids": [],
            },
            listed_args,
        )
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["status"], "listed")
        self.assertEqual(listed["result"]["selected_finding_ids"], [])

        updated_args = SimpleNamespace(
            finding_ids=["w2", "w1"],
            all_findings=False,
        )
        updated = batch.build_machine_success_envelope(
            "quality-ack",
            {
                "manifest": {"_manifest_path": "C:/jobs/demo/manifest.json"},
                "old_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "new_gate": {"decision": "needs_review", "acknowledged_count": 2},
                "selected_ids": {"w2", "w1"},
                "unmatched": ["missing"],
                "previous_acknowledged_finding_ids": [],
                "acknowledged_finding_ids": ["w1", "w2"],
            },
            updated_args,
        )
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["result"]["selected_finding_ids"], ["w1", "w2"])
        self.assertEqual(updated["result"]["unmatched_finding_ids"], ["missing"])
        self.assertEqual(updated["result"]["acknowledged_finding_ids"], ["w1", "w2"])

        no_work = batch.build_machine_success_envelope(
            "quality-ack",
            {
                "manifest": {"_manifest_path": "C:/jobs/demo/manifest.json"},
                "old_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "new_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "selected_ids": set(),
                "unmatched": ["missing"],
                "previous_acknowledged_finding_ids": [],
                "acknowledged_finding_ids": [],
            },
            SimpleNamespace(finding_ids=["missing"], all_findings=False),
        )
        self.assertEqual(no_work["status"], "no_work")
        self.assertEqual(no_work["result"]["selected_finding_ids"], [])
        self.assertEqual(no_work["result"]["unmatched_finding_ids"], ["missing"])

        pruned = batch.build_machine_success_envelope(
            "quality-ack",
            {
                "manifest": {"_manifest_path": "C:/jobs/demo/manifest.json"},
                "old_gate": {"decision": "needs_review", "acknowledged_count": 1},
                "new_gate": {"decision": "needs_review", "acknowledged_count": 0},
                "selected_ids": set(),
                "unmatched": [],
                "previous_acknowledged_finding_ids": ["stale"],
                "acknowledged_finding_ids": [],
            },
            SimpleNamespace(finding_ids=[], all_findings=False),
        )
        self.assertEqual(pruned["status"], "updated")
        self.assertEqual(pruned["result"]["acknowledged_finding_ids"], [])

    def test_build_without_pending_work_does_not_load_latest_manifest(self):
        args = SimpleNamespace(target="")

        with mock.patch.object(batch, "load_manifest") as load_manifest:
            envelope = batch.build_machine_success_envelope("build", None, args)

        self.assertEqual(envelope["status"], "no_work")
        self.assertEqual(
            envelope["result"]["reason"],
            "no_pending_translation_work",
        )
        load_manifest.assert_not_called()

    def test_sync_commands_without_source_items_do_not_load_latest_manifest(self):
        args = SimpleNamespace(target="", apply=False)

        with mock.patch.object(batch, "load_manifest") as load_manifest:
            for command in ("sync-revisions", "sync-keywords"):
                with self.subTest(command=command):
                    envelope = batch.build_machine_success_envelope(
                        command, None, args
                    )
                    self.assertEqual(envelope["status"], "no_work")
                    self.assertEqual(
                        envelope["result"]["reason"],
                        "no_source_items",
                    )
        load_manifest.assert_not_called()

    def test_doctor_json_keeps_stdout_parseable_and_moves_text_to_stderr(self):
        report = {
            "mode": "existing_tl_only",
            "workflow_state": "ready_to_build",
            "recommendations": [],
            "warnings": ["optional note"],
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner", side_effect=lambda: print("banner")),
            mock.patch.object(batch, "collect_doctor_report", return_value=report),
            mock.patch.object(
                batch,
                "print_doctor_report",
                side_effect=lambda _report: print("doctor text"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["doctor", "--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["command"], "doctor")
        self.assertEqual(payload["status"], "ready_to_build")
        self.assertEqual(payload["warnings"], ["optional note"])
        self.assertNotIn("banner", stdout.getvalue())
        self.assertIn("banner", stderr.getvalue())
        self.assertIn("doctor text", stderr.getvalue())

    def test_machine_mode_routes_prepare_child_stdout_away_from_json(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as child_diagnostics:
            def dispatch(_parser, _args):
                self.assertTrue(batch.cli_contract.machine_output_active())
                print("python progress")
                ok = batch.legacy._run_prepare_command(
                    [sys.executable, "-c", "print('child stdout')"],
                    cwd=str(Path.cwd()),
                    step_name="contract smoke",
                )
                self.assertTrue(ok)
                return {}

            with (
                mock.patch.object(batch, "dispatch_command", side_effect=dispatch),
                mock.patch.object(
                    batch.legacy,
                    "_machine_subprocess_diagnostic_stream",
                    return_value=child_diagnostics,
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = batch.main(["doctor", "--output", "json", "--compact"])

            child_diagnostics.flush()
            child_diagnostics.seek(0)
            child_output = child_diagnostics.read()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["command"], "doctor")
        self.assertNotIn("child stdout", stdout.getvalue())
        self.assertIn("child stdout", child_output)
        self.assertIn("python progress", stderr.getvalue())

    def test_check_json_exposes_safety_and_artifacts(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "mode": "translation",
            "last_check_summary": {
                "safety_level": "warn",
                "failure_items": 1,
            },
            "last_check_report_path": "C:/jobs/demo/check_failures.jsonl",
            "last_quality_findings_path": "C:/jobs/demo/quality_findings.jsonl",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner", side_effect=lambda: print("banner")),
            mock.patch.object(
                batch,
                "check_results",
                side_effect=lambda _target: (print("check text"), manifest)[1],
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["check", "C:/jobs/demo/manifest.json", "--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "warn")
        self.assertEqual(payload["result"]["check"]["failure_items"], 1)
        self.assertEqual(
            payload["artifacts"]["check_report"],
            "C:/jobs/demo/check_failures.jsonl",
        )
        self.assertEqual(
            payload["artifacts"]["quality_findings"],
            "C:/jobs/demo/quality_findings.jsonl",
        )
        self.assertIn("check text", stderr.getvalue())

    def test_json_mode_turns_system_exit_into_error_envelope(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch,
                "check_results",
                side_effect=SystemExit("Manifest or results changed after the last check."),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["check", "manifest.json", "--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "COMMAND_REFUSED")
        self.assertIn("changed after the last check", payload["error"]["message"])

    def test_non_strict_exception_keeps_internal_error_contract(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(
                batch,
                "dispatch_command",
                side_effect=RuntimeError("503 UNAVAILABLE"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(["status", "manifest.json", "--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("semantic_exit_code", payload["error"]["details"])

    def test_strict_check_ready_with_warnings_reports_needs_action(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_check_summary": {
                "safety_level": "safe",
                "check_status": "ready_with_warnings",
                "writeback_gate": {"decision": "allow", "can_apply": True},
                "quality_gate": {"decision": "needs_review", "warning_count": 2},
            },
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "check",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )

        self.assertEqual(exit_code, batch.cli_contract.EXIT_NEEDS_ACTION)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready_with_warnings")

    def test_strict_check_ready_exit_code_is_ok(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_check_summary": {
                "safety_level": "safe",
                "check_status": "ready",
                "writeback_gate": {"decision": "allow", "can_apply": True},
                "quality_gate": {"decision": "pass", "warning_count": 0},
            },
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                ["check", "manifest.json", "--output", "json", "--strict-exit-codes"]
            )

        self.assertEqual(exit_code, batch.cli_contract.EXIT_OK)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

    def test_strict_check_exit_code_reports_needs_action(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_check_summary": {"safety_level": "warn"},
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(batch, "check_results", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "check",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )

        self.assertEqual(exit_code, batch.cli_contract.EXIT_NEEDS_ACTION)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "warn")

    def test_non_strict_check_block_keeps_compatible_zero_exit(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "last_check_summary": {"safety_level": "block"},
        }
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=manifest),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(["check", "manifest.json", "--output", "json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "block")

    def test_strict_success_state_matrix(self):
        cases = (
            (
                "check",
                {"last_check_summary": {"safety_level": "block"}},
                batch.cli_contract.EXIT_BLOCKED,
            ),
            (
                "status",
                {"job_state": "JOB_STATE_FAILED"},
                batch.cli_contract.EXIT_BLOCKED,
            ),
            (
                "submit",
                {"job_state": "JOB_STATE_FAILED"},
                batch.cli_contract.EXIT_BLOCKED,
            ),
        )

        for command, manifest, expected_exit in cases:
            with self.subTest(command=command):
                stdout = io.StringIO()
                manifest["_manifest_path"] = "C:/jobs/demo/manifest.json"
                with (
                    mock.patch.object(batch, "dispatch_command", return_value=manifest),
                    contextlib.redirect_stdout(stdout),
                ):
                    exit_code = batch.main(
                        [
                            command,
                            "manifest.json",
                            "--output",
                            "json",
                            "--strict-exit-codes",
                        ]
                    )

                self.assertEqual(exit_code, expected_exit)
                self.assertTrue(json.loads(stdout.getvalue())["ok"])

    def test_strict_doctor_blocked_returns_blocked_exit(self):
        report = {"recommendations": [{"code": "blocking"}]}
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "dispatch_command", return_value=report),
            mock.patch.object(
                batch.doctor_rec,
                "recommendations_block_workflow_state",
                return_value=True,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(["doctor", "--output", "json", "--strict-exit-codes"])

        self.assertEqual(exit_code, batch.cli_contract.EXIT_BLOCKED)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "blocked")

    def test_strict_retryable_exception_returns_retryable_exit(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            mock.patch.object(
                batch,
                "dispatch_command",
                side_effect=RuntimeError("503 UNAVAILABLE"),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = batch.main(
                ["status", "manifest.json", "--output", "json", "--strict-exit-codes"]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_RETRYABLE)
        self.assertEqual(payload["error"]["code"], "REMOTE_RETRYABLE")
        self.assertTrue(payload["error"]["retryable"])

    def test_strict_apply_preflight_errors_return_invalid_state(self):
        messages = (
            "Manifest has no valid check summary. Run check before apply.",
            "Manifest check summary was produced by an older check contract. "
            "Run check again before apply.",
        )

        for message in messages:
            with self.subTest(message=message):
                stdout = io.StringIO()
                with (
                    mock.patch.object(
                        batch,
                        "dispatch_command",
                        side_effect=SystemExit(message),
                    ),
                    contextlib.redirect_stdout(stdout),
                ):
                    exit_code = batch.main(
                        ["apply", "manifest.json", "--output", "json", "--strict-exit-codes"]
                    )

                payload = json.loads(stdout.getvalue())
                self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
                self.assertEqual(payload["error"]["code"], "STALE_STATE")
                self.assertEqual(
                    payload["error"]["suggested_action"],
                    "run_check_again",
                )

    def test_strict_system_exit_uses_stale_state_code(self):
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch,
                "check_results",
                side_effect=SystemExit("Manifest or results changed after the last check."),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(
                [
                    "check",
                    "manifest.json",
                    "--output",
                    "json",
                    "--strict-exit-codes",
                ]
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_INVALID_STATE)
        self.assertEqual(payload["error"]["code"], "STALE_STATE")
        self.assertEqual(
            payload["error"]["suggested_action"],
            "run_check_again",
        )

    def test_strict_exit_codes_requires_json_output(self):
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                batch.main(["doctor", "--strict-exit-codes"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires --output json", stderr.getvalue())

    def test_output_trimming_options_require_json_output(self):
        cases = (
            ["doctor", "--compact"],
            ["doctor", "--fields", "status"],
            ["doctor", "--output-file", "result.json"],
        )

        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        batch.main(argv)

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("requires --output json", stderr.getvalue())


    def test_offline_batch_commands_skip_api_key_requirement(self):
        for command in sorted(batch.OFFLINE_BATCH_COMMANDS):
            with self.subTest(command=command):
                load_config = mock.Mock()
                with (
                    mock.patch.object(batch, "initialize_batch_logging"),
                    mock.patch.object(batch.legacy, "load_config", load_config),
                    mock.patch.object(batch.legacy, "load_translator_settings"),
                    mock.patch.object(batch.legacy, "load_glossary"),
                    mock.patch.object(batch, "load_batch_settings"),
                    mock.patch.object(batch, "print_banner"),
                    mock.patch.object(
                        batch,
                        "dispatch_command",
                        wraps=batch.dispatch_command,
                    ),
                ):
                    # Patch the actual offline command handlers after the load gate.
                    handler_patches = []
                    if command == "check":
                        handler_patches.append(
                            mock.patch.object(batch, "check_results", return_value={"ok": True})
                        )
                    elif command == "apply":
                        handler_patches.append(
                            mock.patch.object(batch, "apply_results", return_value={"ok": True})
                        )
                    elif command == "estimate-cost":
                        handler_patches.extend(
                            [
                                mock.patch.object(
                                    batch,
                                    "load_manifest",
                                    return_value={"_manifest_path": "manifest.json"},
                                ),
                                mock.patch.object(
                                    batch,
                                    "ensure_manifest_cost_estimate",
                                    return_value={"total": 1},
                                ),
                                mock.patch.object(
                                    batch.batch_cost_estimate,
                                    "format_cost_estimate_lines",
                                    return_value=["cost"],
                                ),
                            ]
                        )
                    elif command == "preview-revisions":
                        handler_patches.append(
                            mock.patch.object(batch, "preview_revisions", return_value={"ok": True})
                        )
                    elif command == "apply-revisions":
                        handler_patches.append(
                            mock.patch.object(batch, "apply_revisions", return_value={"ok": True})
                        )
                    elif command == "split":
                        handler_patches.append(
                            mock.patch.object(batch, "split_manifest", return_value=None)
                        )
                    elif command == "build-retry":
                        handler_patches.append(
                            mock.patch.object(batch, "build_retry_package", return_value=None)
                        )
                    elif command == "merge-retry":
                        handler_patches.append(
                            mock.patch.object(batch, "merge_retry_results", return_value=None)
                        )
                    elif command == "export-keywords":
                        handler_patches.append(
                            mock.patch.object(
                                batch, "export_keyword_candidates", return_value=None
                            )
                        )
                    elif command in {"quality-ack", "quality-unack"}:
                        handler_patches.extend(
                            [
                                mock.patch.object(
                                    batch,
                                    "quality_acknowledge_command",
                                    return_value={
                                        "manifest": {"_manifest_path": "manifest.json"},
                                        "findings": [],
                                        "old_gate": {},
                                        "new_gate": {},
                                        "selected_ids": set(),
                                        "unmatched": [],
                                        "acknowledged_finding_ids": [],
                                    },
                                ),
                                mock.patch.object(
                                    batch,
                                    "print_quality_acknowledgement_summary",
                                ),
                            ]
                        )
                    elif command == "export-revision-corpus":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_revision_corpus_export",
                                return_value={"paths": {}, "scope": {}},
                            )
                        )
                    elif command == "import-revision-proposals":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "import_revision_proposals",
                                return_value={"status": "previewed", "paths": {}},
                            )
                        )
                    elif command == "export-project-snapshot":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_project_snapshot_export",
                                return_value={"paths": {}, "coverage": {}},
                            )
                        )
                    elif command == "reconcile-project-snapshots":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_project_snapshot_reconciliation",
                                return_value={"paths": {}, "summary": {}},
                            )
                        )
                    elif command == "build-translation-records":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_translation_records_export",
                                return_value={"paths": {}},
                            )
                        )
                    elif command == "build-reuse-candidates":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_reuse_candidates_build",
                                return_value={"paths": {}, "summary": {}},
                            )
                        )
                    elif command == "import-reuse-decisions":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_reuse_decisions_import",
                                return_value={"paths": {}, "summary": {}},
                            )
                        )
                    elif command == "export-reuse-results":
                        handler_patches.append(
                            mock.patch.object(
                                batch,
                                "run_reuse_results_export",
                                return_value={"paths": {}},
                            )
                        )
                    elif command == "merge-keywords-to-glossary":
                        handler_patches.extend(
                            [
                                mock.patch.object(
                                    batch.keyword_glossary_merge,
                                    "resolve_keyword_candidates_path",
                                    return_value="candidates.jsonl",
                                ),
                                mock.patch.object(
                                    batch.keyword_glossary_merge,
                                    "merge_keywords_to_glossary",
                                    return_value=None,
                                ),
                                mock.patch.object(
                                    batch.legacy,
                                    "GLOSSARY_FILE",
                                    "glossary.json",
                                ),
                            ]
                        )

                    with contextlib.ExitStack() as stack:
                        for patcher in handler_patches:
                            stack.enter_context(patcher)
                        if command == "merge-retry":
                            argv = ["merge-retry", "parent.json", "retry.json"]
                        elif command == "merge-keywords-to-glossary":
                            argv = ["merge-keywords-to-glossary", "candidates.jsonl", "--yes"]
                        elif command == "export-revision-corpus":
                            argv = ["export-revision-corpus"]
                        elif command == "import-revision-proposals":
                            argv = ["import-revision-proposals", "proposals.jsonl"]
                        elif command == "export-project-snapshot":
                            argv = ["export-project-snapshot", "--version-id", "test-version"]
                        elif command == "reconcile-project-snapshots":
                            argv = [
                                "reconcile-project-snapshots",
                                "base-snapshot.json",
                                "target-snapshot.json",
                            ]
                        elif command == "build-translation-records":
                            argv = [
                                "build-translation-records",
                                "snapshot.json",
                                "manifest.json",
                            ]
                        elif command == "build-reuse-candidates":
                            argv = [
                                "build-reuse-candidates",
                                "base-snapshot.json",
                                "target-snapshot.json",
                                "reconciliation.json",
                                "records.json",
                            ]
                        elif command == "import-reuse-decisions":
                            argv = [
                                "import-reuse-decisions",
                                "reuse.json",
                                "decisions.jsonl",
                            ]
                        elif command == "export-reuse-results":
                            argv = [
                                "export-reuse-results",
                                "reuse.json",
                                "manifest.json",
                            ]
                        else:
                            argv = [command, "manifest.json"]
                        exit_code = batch.main(argv)

                self.assertEqual(exit_code, 0)
                if command in {
                    "export-revision-corpus",
                    "import-revision-proposals",
                    "export-project-snapshot",
                    "reconcile-project-snapshots",
                    "build-translation-records",
                    "build-reuse-candidates",
                    "import-reuse-decisions",
                    "export-reuse-results",
                }:
                    # Read-only export takes an early dispatch path that must
                    # not load (or rewrite) API-key / translator config.
                    load_config.assert_not_called()
                else:
                    load_config.assert_called_once_with(require_api_key=False)

    def test_merge_keywords_yes_explicitly_overrides_history_review(self):
        cases = (
            (("--yes",), False, True),
            (("--accept-confidence", "0.8"), True, False),
        )
        for extra_args, expected_interactive, expected_override in cases:
            with self.subTest(extra_args=extra_args):
                load_config = mock.Mock()
                merge = mock.Mock(return_value=None)
                with (
                    mock.patch.object(batch, "initialize_batch_logging"),
                    mock.patch.object(batch.legacy, "load_config", load_config),
                    mock.patch.object(batch.legacy, "load_translator_settings"),
                    mock.patch.object(batch.legacy, "load_glossary"),
                    mock.patch.object(batch, "load_batch_settings"),
                    mock.patch.object(batch, "print_banner"),
                    mock.patch.object(
                        batch.keyword_glossary_merge,
                        "resolve_keyword_candidates_path",
                        return_value="candidates.jsonl",
                    ),
                    mock.patch.object(
                        batch.keyword_glossary_merge,
                        "merge_keywords_to_glossary",
                        merge,
                    ),
                    mock.patch.object(batch.legacy, "GLOSSARY_FILE", "glossary.json"),
                ):
                    exit_code = batch.main(
                        [
                            "merge-keywords-to-glossary",
                            "candidates.jsonl",
                            *extra_args,
                        ]
                    )

                self.assertEqual(exit_code, 0)
                load_config.assert_called_once_with(require_api_key=False)
                merge.assert_called_once()
                kwargs = merge.call_args.kwargs
                self.assertEqual(kwargs["interactive"], expected_interactive)
                self.assertEqual(kwargs["allow_history_review"], expected_override)

    def test_remote_batch_commands_still_require_api_key(self):
        load_config = mock.Mock()
        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config", load_config),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(batch, "show_status", return_value={"job_state": "JOB_STATE_PENDING"}),
        ):
            exit_code = batch.main(["status", "manifest.json"])

        self.assertEqual(exit_code, 0)
        load_config.assert_called_once_with(require_api_key=True)

    def test_output_file_conflict_with_manifest_is_rejected_before_dispatch(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            manifest_path = package / "manifest.json"
            results_path = package / "results.jsonl"
            results_path.write_text("{}" + "\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "result_jsonl_path": "results.jsonl",
                        "input_jsonl_path": "input.jsonl",
                        "files": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with (
                mock.patch.object(batch, "dispatch_command") as dispatch,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = batch.main(
                    [
                        "status",
                        str(manifest_path),
                        "--output",
                        "json",
                        "--output-file",
                        str(manifest_path),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_PATH_CONFLICT")
        self.assertFalse(payload["error"]["details"]["workflow_started"])
        self.assertEqual(
            batch._normalized_abs_path(payload["error"]["details"]["output_file"]),
            batch._normalized_abs_path(str(manifest_path)),
        )
        self.assertEqual(
            batch._normalized_abs_path(payload["error"]["details"]["conflict_path"]),
            batch._normalized_abs_path(str(manifest_path)),
        )
        self.assertNotIn("Traceback", stderr.getvalue())
        dispatch.assert_not_called()

    def test_output_file_conflict_with_results_uses_normalized_paths(self):
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            manifest_path = package / "manifest.json"
            results_path = package / "results.jsonl"
            results_path.write_text("{}" + "\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"result_jsonl_path": "results.jsonl"}, ensure_ascii=False),
                encoding="utf-8",
            )
            # Keep a raw alias so pathlib cannot collapse it before CLI normalization.
            conflict_path = f"{package}{os.sep}.{os.sep}results.jsonl"
            with (
                mock.patch.object(batch, "dispatch_command") as dispatch,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = batch.main(
                    [
                        "check",
                        str(manifest_path),
                        "--output",
                        "json",
                        "--output-file",
                        conflict_path,
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_PATH_CONFLICT")
        self.assertEqual(
            batch._normalized_abs_path(payload["error"]["details"]["conflict_path"]),
            batch._normalized_abs_path(str(results_path)),
        )
        dispatch.assert_not_called()

    def test_output_file_conflict_without_output_json_uses_discovery_json_path(self):
        # Discovery commands allow --output-file without --output json and still
        # emit a structured conflict envelope on stdout.
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            manifest_path = package / "manifest.json"
            latest_path = package / "latest.txt"
            manifest_path.write_text("{}", encoding="utf-8")
            latest_path.write_text(str(manifest_path), encoding="utf-8")
            with (
                mock.patch.object(batch, "LATEST_MANIFEST_FILE", str(latest_path)),
                mock.patch.object(batch, "dispatch_command") as dispatch,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = batch.main(
                    [
                        "capabilities",
                        "--output-file",
                        str(manifest_path),
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, batch.cli_contract.EXIT_USAGE)
        self.assertEqual(payload["error"]["code"], "OUTPUT_FILE_PATH_CONFLICT")
        self.assertFalse(payload["error"]["details"]["workflow_started"])
        dispatch.assert_not_called()

    def test_default_glossary_is_protected_for_merge_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            glossary_path = package / "glossary.json"
            glossary_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(batch.legacy, "GLOSSARY_FILE", str(glossary_path)):
                args = SimpleNamespace(
                    command="merge-keywords-to-glossary",
                    target="candidates.jsonl",
                    parent="",
                    retry="",
                    report="",
                    jsonl="",
                    markdown="",
                    summary_jsonl="",
                    summary_markdown="",
                    variants_file="",
                    glossary="",
                    output_file=str(glossary_path),
                )
                with self.assertRaises(batch.cli_contract.MachineContractError) as raised:
                    batch._preflight_output_file(args)

        self.assertEqual(raised.exception.code_name, "OUTPUT_FILE_PATH_CONFLICT")
        self.assertEqual(
            batch._normalized_abs_path(raised.exception.details["conflict_path"]),
            batch._normalized_abs_path(str(glossary_path)),
        )

    def test_independent_output_file_still_allowed(self):
        manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "job_state": "JOB_STATE_SUCCEEDED",
        }
        stdout = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            manifest_path = package / "manifest.json"
            report_path = package / "status-report.json"
            manifest_path.write_text(
                json.dumps({"result_jsonl_path": "results.jsonl"}, ensure_ascii=False),
                encoding="utf-8",
            )
            with (
                mock.patch.object(batch, "dispatch_command", return_value=manifest),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = batch.main(
                    [
                        "status",
                        str(manifest_path),
                        "--output",
                        "json",
                        "--output-file",
                        str(report_path),
                    ]
                )

            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(payload["status"], "JOB_STATE_SUCCEEDED")

    def test_text_mode_preserves_human_stdout(self):
        stdout = io.StringIO()

        with (
            mock.patch.object(batch, "initialize_batch_logging"),
            mock.patch.object(batch.legacy, "load_config"),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner", side_effect=lambda: print("banner")),
            mock.patch.object(
                batch,
                "check_results",
                side_effect=lambda _target: print("check text"),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = batch.main(["check", "manifest.json"])

        self.assertEqual(exit_code, 0)
        self.assertIn("banner", stdout.getvalue())
        self.assertIn("check text", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
