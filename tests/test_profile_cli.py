"""CLI tests for ModelProfile configuration commands (#348 P3)."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
import model_capability_probe as capability_probe
import translator_runtime as legacy
from model_routing_migration import preview_migration

FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def migrated_config(name: str = "gemini_batch") -> dict:
    payload = json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))
    return preview_migration(payload).config


class ProfileCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "translator_config.json"
        self.config = migrated_config()
        self.config["future_top_level"] = {"keep": True}
        self._write_config(self.config)
        self.config_patch = mock.patch.object(
            legacy,
            "TRANSLATOR_CONFIG",
            str(self.config_path),
        )
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def _write_config(self, config: dict) -> None:
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_json(self, *argv: str) -> tuple[int, dict]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = batch.main(list(argv))
        document = stream.getvalue().strip()
        try:
            envelope = cli_contract.parse_result_envelope(document)
        except ValueError:
            # Text-mode commands print human lines only.
            envelope = {}
        return exit_code, envelope

    def test_cli_choices_follow_shared_core_constants(self) -> None:
        import argparse

        import model_profiles_editor as editor

        parser = batch.build_arg_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        route_parser = subparsers.choices["profiles-set-route"]
        default_parser = subparsers.choices["profiles-set-default"]
        stage_action = next(
            action for action in route_parser._actions if action.dest == "stage"
        )
        route_strategy = next(
            action for action in route_parser._actions if action.dest == "strategy"
        )
        default_strategy = next(
            action for action in default_parser._actions if action.dest == "strategy"
        )

        self.assertEqual(tuple(stage_action.choices), tuple(editor.STAGE_ORDER))
        self.assertEqual(
            tuple(route_strategy.choices),
            tuple(editor.STRATEGY_ORDER),
        )
        self.assertEqual(
            tuple(default_strategy.choices),
            tuple(editor.STRATEGY_ORDER),
        )

    def test_profiles_show_machine_envelope_is_credential_free(self) -> None:
        exit_code, envelope = self._run_json(
            "profiles-show",
            "--output",
            "json",
            "--non-interactive",
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["command"], "profiles-show")
        result = envelope["result"]
        self.assertEqual(envelope["status"], "ready")
        self.assertEqual(
            result["defaults"]["primary_profile_id"],
            "legacy-batch",
        )
        text = json.dumps(result).lower()
        for forbidden in ("secret", "password", "authorization"):
            self.assertNotIn(forbidden, text)
        provider = result["providers"][0]
        self.assertEqual(
            set(provider["credential_ref"]),
            {"kind", "name", "env_name"},
        )
        self.assertEqual(result["issues"], [])

    def test_profiles_show_refuses_legacy_config(self) -> None:
        self._write_config({"sync": {"backend": "gemini"}})

        exit_code, envelope = self._run_json(
            "profiles-show",
            "--output",
            "json",
            "--non-interactive",
        )

        self.assertNotEqual(exit_code, 0)
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "MODEL_ROUTING_NOT_CONFIGURED")

    def test_profiles_validate_reports_invalid_section(self) -> None:
        section = self.config["model_routing"]
        section["defaults"]["primary_profile_id"] = "missing-profile"
        self._write_config(self.config)

        exit_code, envelope = self._run_json(
            "profiles-validate",
            "--output",
            "json",
            "--non-interactive",
        )

        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "MODEL_ROUTING_INVALID")
        self.assertTrue(envelope["error"]["details"]["issues"])
        self.assertNotEqual(exit_code, 0)

    def test_set_default_updates_file_and_preserves_unknown_fields(self) -> None:
        exit_code, envelope = self._run_json(
            "profiles-set-default",
            "--profile",
            "legacy-sync",
            "--strategy",
            "sync",
            "--output",
            "json",
            "--non-interactive",
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(envelope["ok"])
        self.assertTrue(envelope["result"]["changed"])
        on_disk = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk["model_routing"]["defaults"],
            {"primary_profile_id": "legacy-sync", "execution_strategy": "sync"},
        )
        self.assertEqual(on_disk["future_top_level"], {"keep": True})

    def test_set_default_is_idempotent(self) -> None:
        exit_code, _ = self._run_json(
            "profiles-set-default",
            "--profile",
            "legacy-batch",
            "--strategy",
            "gemini_batch",
            "--output",
            "json",
        )
        self.assertEqual(exit_code, 0)

        _exit_code, envelope = self._run_json(
            "profiles-set-default",
            "--profile",
            "legacy-batch",
            "--strategy",
            "gemini_batch",
            "--output",
            "json",
        )
        self.assertFalse(envelope["result"]["changed"])

    def test_set_default_rejects_incompatible_strategy(self) -> None:
        self._write_config(migrated_config("litellm_custom"))

        _exit_code, envelope = self._run_json(
            "profiles-set-default",
            "--profile",
            "legacy-sync",
            "--strategy",
            "gemini_batch",
            "--output",
            "json",
        )

        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "STRATEGY_NOT_SUPPORTED")
        self.assertEqual(
            envelope["error"]["details"]["reason"],
            "missing_gemini_adapter",
        )

    def test_set_route_and_clear_round_trip(self) -> None:
        _exit_code, envelope = self._run_json(
            "profiles-set-route",
            "--stage",
            "project_analysis",
            "--profile",
            "legacy-sync",
            "--strategy",
            "sync",
            "--output",
            "json",
        )
        self.assertTrue(envelope["ok"])
        on_disk = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk["model_routing"]["routes"]["project_analysis"]["profile_id"],
            "legacy-sync",
        )

        _exit_code, envelope = self._run_json(
            "profiles-set-route",
            "--stage",
            "project_analysis",
            "--clear",
            "--output",
            "json",
        )
        self.assertTrue(envelope["ok"])
        on_disk = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertNotIn("project_analysis", on_disk["model_routing"]["routes"])

    def test_route_argument_validation(self) -> None:
        _exit_code, envelope = self._run_json(
            "profiles-set-route",
            "--stage",
            "translation",
            "--clear",
            "--profile",
            "legacy-sync",
            "--output",
            "json",
        )
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "INVALID_ROUTE_ARGUMENTS")

    def test_profiles_probe_reports_capabilities(self) -> None:
        response = capability_probe.ProbeResponse(
            text=json.dumps(
                {
                    "translations": [
                        {"id": "probe-1", "translation": "你好"},
                    ]
                },
                ensure_ascii=False,
            ),
            usage={"total_tokens": 8, "reasoning_tokens": 0},
            finish_reason="STOP",
        )
        with (
            mock.patch.object(
                capability_probe,
                "default_generator",
                return_value=response,
            ),
            mock.patch.object(
                capability_probe,
                "default_credential_loader",
                return_value="probe-secret-key",
            ),
        ):
            exit_code, envelope = self._run_json(
                "profiles-probe",
                "--profile",
                "legacy-batch",
                "--acknowledge-billable-request",
                capability_probe.BILLABLE_ACK_TOKEN,
                "--output",
                "json",
                "--non-interactive",
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(envelope["ok"])
        result = envelope["result"]
        self.assertEqual(envelope["status"], "passed")
        self.assertEqual(result["requests"], 1)
        names = {item["name"] for item in result["capabilities"]}
        self.assertEqual(
            names,
            {
                "auth",
                "sync_generation",
                "structured_output",
                "reasoning",
                "usage",
                "remote_batch",
                "embedding",
            },
        )
        self.assertNotIn("probe-secret-key", json.dumps(result))

    def test_profiles_probe_requires_acknowledgement(self) -> None:
        exit_code, envelope = self._run_json(
            "profiles-probe",
            "--profile",
            "legacy-batch",
            "--acknowledge-billable-request",
            "wrong",
            "--output",
            "json",
        )

        self.assertNotEqual(exit_code, 0)
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "BILLABLE_ACK_REQUIRED")

    def test_text_mode_prints_human_summary(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exit_code = batch.main(["profiles-show", "--non-interactive"])

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("provider", output)
        self.assertIn("profile legacy-batch", output)

    def test_unknown_profile_is_rejected(self) -> None:
        _exit_code, envelope = self._run_json(
            "profiles-set-route",
            "--stage",
            "translation",
            "--profile",
            "missing",
            "--strategy",
            "sync",
            "--output",
            "json",
        )
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "UNKNOWN_PROFILE")


if __name__ == "__main__":
    unittest.main()
