import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import run_provider_contract_smoke as smoke


class ProviderContractSmokeTests(unittest.TestCase):
    def test_catalog_covers_all_external_production_providers(self):
        expected = {"gemini", "openai", "anthropic", "openrouter", "deepseek", "xai"}
        self.assertEqual(set(smoke.PROVIDER_BY_NAME), expected)
        self.assertNotIn("ollama", smoke.PROVIDER_BY_NAME)

    def test_missing_credentials_skip_without_creating_backend(self):
        output = io.StringIO()
        with (
            mock.patch.object(smoke, "create_backend") as create_backend,
            redirect_stdout(output),
        ):
            code = smoke.run_selected(list(smoke.PROVIDER_SPECS), {})

        self.assertEqual(code, 0)
        create_backend.assert_not_called()
        self.assertIn("SUMMARY passed=0 skipped=6 failed=0", output.getvalue())

    def test_configured_provider_uses_one_bounded_production_request(self):
        spec = smoke.PROVIDER_BY_NAME["openai"]
        result = SimpleNamespace(
            provider="litellm",
            model=spec.model,
            response_text='{"ok":true}',
            usage_metadata={"total_tokens": 9},
        )
        backend = mock.Mock()
        backend.generate.return_value = result

        smoke.run_provider(spec, "secret", backend=backend)

        request = backend.generate.call_args.args[0]
        self.assertEqual(request.model, spec.model)
        self.assertEqual(
            request.config["max_output_tokens"],
            smoke.MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            request.config["timeout"],
            smoke.REQUEST_TIMEOUT_SECONDS,
        )

    def test_invalid_response_fails_with_provider_name_and_category(self):
        spec = smoke.PROVIDER_BY_NAME["openai"]
        backend = mock.Mock()
        backend.generate.return_value = SimpleNamespace(
            provider="litellm",
            model=spec.model,
            response_text="not-json",
            usage_metadata={},
        )
        error = io.StringIO()
        with (
            mock.patch.object(smoke, "create_backend", return_value=backend),
            redirect_stderr(error),
            redirect_stdout(io.StringIO()),
        ):
            code = smoke.run_selected([spec], {"OPENAI_API_KEY": "secret"})

        self.assertEqual(code, 1)
        self.assertIn("provider=openai", error.getvalue())
        self.assertIn("category=invalid_response", error.getvalue())

    def test_error_classification_covers_auth_rate_limit_and_outage(self):
        for status, expected in (
            (401, "authentication"),
            (429, "rate_limit"),
            (503, "service_unavailable"),
        ):
            error = RuntimeError("provider failed")
            error.status_code = status
            self.assertEqual(smoke.classify_error(error), expected)

    def test_workflow_is_scheduled_manual_and_not_a_pr_gate(self):
        workflow = (
            Path(smoke.__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "provider-contract-smoke.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("schedule:", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertIn("timeout-minutes: 10", workflow)
        for secret in (
            "PROVIDER_SMOKE_GEMINI_API_KEY",
            "PROVIDER_SMOKE_OPENAI_API_KEY",
            "PROVIDER_SMOKE_ANTHROPIC_API_KEY",
            "PROVIDER_SMOKE_OPENROUTER_API_KEY",
            "PROVIDER_SMOKE_DEEPSEEK_API_KEY",
            "PROVIDER_SMOKE_XAI_API_KEY",
        ):
            self.assertIn(secret, workflow)


if __name__ == "__main__":
    unittest.main()
