import contextlib
import io
import json
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
