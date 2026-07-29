import contextlib
import io
import json
import unittest
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

    def test_machine_result_builder_covers_manifest_workflow(self):
        args = SimpleNamespace(target="")
        base_manifest = {
            "_manifest_path": "C:/jobs/demo/manifest.json",
            "mode": "translation",
            "job_state": "JOB_STATE_PENDING",
            "summary": {"item_count": 2},
            "last_check_summary": {"safety_level": "safe"},
            "apply_summary": {"applied_lines": 2},
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
