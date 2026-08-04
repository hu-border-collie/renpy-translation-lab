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
    def test_core_commands_accept_json_output_after_subcommand(self):
        parser = batch.build_arg_parser()

        for command in sorted(batch.MACHINE_OUTPUT_COMMANDS):
            with self.subTest(command=command):
                args = parser.parse_args([command, "--output", "json"])
                self.assertEqual(args.command, command)
                self.assertEqual(args.output, "json")
                self.assertFalse(args.strict_exit_codes)
                strict_args = parser.parse_args(
                    [command, "--output", "json", "--strict-exit-codes"]
                )
                strict_invocation = parser.parse_args(
                    [command, "--non-interactive", "--require-explicit-target"]
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
                        else:
                            argv = [command, "manifest.json"]
                        exit_code = batch.main(argv)

                self.assertEqual(exit_code, 0)
                load_config.assert_called_once_with(require_api_key=False)

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
