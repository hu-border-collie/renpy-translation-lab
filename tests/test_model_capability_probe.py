"""Capability probe core tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from unittest import mock

import model_capability_probe as probe
import model_profiles_editor as editor


def gemini_section() -> dict:
    section = editor.add_provider(
        editor.empty_section(),
        label="Gemini",
        adapter="gemini",
        provider="gemini",
        credential_kind="api_keys_json",
        credential_name="api_keys",
        credential_env_name="GEMINI_API_KEY",
    )
    section = editor.add_profile(
        section,
        label="Gemini Main",
        provider_id="gemini",
        model="gemini-3.5-flash",
    )
    return editor.set_defaults(
        section,
        primary_profile_id="gemini-main",
        execution_strategy="sync",
    )


def litellm_section() -> dict:
    section = editor.add_provider(
        editor.empty_section(),
        label="Acme",
        adapter="litellm",
        provider="acme",
        base_url="https://acme.example/v1",
        credential_kind="env",
        credential_name="ACME_API_KEY",
    )
    section = editor.add_profile(
        section,
        label="Acme Main",
        provider_id="acme",
        model="acme/model-a",
    )
    return editor.set_defaults(
        section,
        primary_profile_id="acme-main",
        execution_strategy="sync",
    )


def valid_response() -> probe.ProbeResponse:
    return probe.ProbeResponse(
        text=json.dumps(
            {
                "translations": [
                    {"id": "probe-1", "translation": "你好"},
                ]
            },
            ensure_ascii=False,
        ),
        usage={"total_tokens": 12, "reasoning_tokens": 0, "completion_tokens": 6},
        finish_reason="STOP",
    )


class ProbeReportTests(unittest.TestCase):
    def _run(self, section, *, generate, credential="test-key", **kwargs):
        return probe.probe_profile(
            section,
            "gemini-main" if "gemini" in section["providers"] else "acme-main",
            acknowledge=probe.BILLABLE_ACK_TOKEN,
            generate=generate,
            credential_loader=lambda _provider: credential,
            **kwargs,
        )

    def test_valid_response_reports_each_capability(self) -> None:
        report = self._run(gemini_section(), generate=lambda _request: valid_response())

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["requests"], 1)
        statuses = {item["name"]: item["status"] for item in report["capabilities"]}
        self.assertEqual(statuses["auth"], "pass")
        self.assertEqual(statuses["sync_generation"], "pass")
        self.assertEqual(statuses["structured_output"], "pass")
        self.assertEqual(statuses["usage"], "pass")
        self.assertEqual(statuses["reasoning"], "pass")
        self.assertEqual(statuses["remote_batch"], "declared")
        self.assertEqual(statuses["embedding"], "declared")

    def test_non_json_response_fails_structured_output(self) -> None:
        report = self._run(
            gemini_section(),
            generate=lambda _request: probe.ProbeResponse(
                text="not json",
                usage={"total_tokens": 3},
            ),
        )

        self.assertEqual(report["status"], "failed")
        statuses = {item["name"]: item for item in report["capabilities"]}
        self.assertEqual(statuses["sync_generation"]["status"], "pass")
        self.assertEqual(statuses["structured_output"]["status"], "fail")
        self.assertEqual(statuses["structured_output"]["detail"], "invalid_json")

    def test_schema_mismatch_is_reported(self) -> None:
        report = self._run(
            gemini_section(),
            generate=lambda _request: probe.ProbeResponse(
                text=json.dumps({"translations": [{"id": "other"}]}),
                usage={"total_tokens": 3},
            ),
        )

        structured = next(
            item
            for item in report["capabilities"]
            if item["name"] == "structured_output"
        )
        self.assertEqual(structured["status"], "fail")
        self.assertEqual(structured["detail"], "schema_mismatch")

    def test_authentication_error_marks_auth_failed(self) -> None:
        def failing(_request):
            raise probe.ProbeTransportError("authentication")

        report = self._run(gemini_section(), generate=failing)

        self.assertEqual(report["status"], "failed")
        statuses = {item["name"]: item["status"] for item in report["capabilities"]}
        self.assertEqual(statuses["auth"], "fail")
        self.assertEqual(statuses["sync_generation"], "fail")

    def test_missing_credential_skips_without_calling_generate(self) -> None:
        calls: list[object] = []

        report = self._run(
            gemini_section(),
            generate=lambda request: calls.append(request) or valid_response(),
            credential="",
        )

        self.assertEqual(calls, [])
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["requests"], 0)
        statuses = {item["name"]: item["status"] for item in report["capabilities"]}
        self.assertEqual(statuses["auth"], "skipped")
        self.assertEqual(statuses["remote_batch"], "declared")

    def test_acknowledgement_is_required(self) -> None:
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            probe.probe_profile(
                gemini_section(),
                "gemini-main",
                acknowledge="wrong",
                generate=lambda _request: valid_response(),
                credential_loader=lambda _provider: "key",
            )
        self.assertEqual(caught.exception.code, "BILLABLE_ACK_REQUIRED")

    def test_invalid_section_is_rejected(self) -> None:
        section = gemini_section()
        section["defaults"]["primary_profile_id"] = "missing"
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            probe.probe_profile(
                section,
                "gemini-main",
                acknowledge=probe.BILLABLE_ACK_TOKEN,
                generate=lambda _request: valid_response(),
                credential_loader=lambda _provider: "key",
            )
        self.assertEqual(caught.exception.code, "MODEL_ROUTING_INVALID")

    def test_report_never_contains_credential_values(self) -> None:
        secret = "super-secret-key-value"
        report = self._run(
            litellm_section(),
            generate=lambda _request: valid_response(),
            credential=secret,
        )

        text = json.dumps(report)
        self.assertNotIn(secret, text)
        self.assertNotIn("api_key", text.lower())

    def test_litellm_credential_falls_back_to_native_env(self) -> None:
        import os

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            value = probe.default_credential_loader(
                {
                    "adapter": "litellm",
                    "provider": "openrouter",
                    "credential_ref": {"kind": "keyring", "name": "openrouter"},
                }
            )

        self.assertEqual(value, "env-key")

    def test_env_credential_ref_reads_the_named_variable(self) -> None:
        import os

        with mock.patch.dict(os.environ, {"ACME_CUSTOM_KEY": "env-key"}, clear=False):
            value = probe.default_credential_loader(
                {
                    "adapter": "litellm",
                    "provider": "acme",
                    "credential_ref": {"kind": "env", "name": "ACME_CUSTOM_KEY"},
                }
            )

        self.assertEqual(value, "env-key")

    def test_request_is_bounded(self) -> None:
        seen: list[probe.ProbeRequest] = []

        def capture(request):
            seen.append(request)
            return valid_response()

        self._run(
            gemini_section(),
            generate=capture,
            timeout_seconds=9999,
            max_output_tokens=9999,
        )

        self.assertEqual(len(seen), 1)
        self.assertLessEqual(seen[0].timeout_seconds, probe.MAX_TIMEOUT_SECONDS)
        self.assertLessEqual(seen[0].max_output_tokens, probe.MAX_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
