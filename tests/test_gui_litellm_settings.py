import unittest

from gui_qt.litellm_settings import (
    provider_credential_status,
    provider_from_model,
    read_sync_backend_models,
    validate_custom_provider_form,
    write_sync_backend_models,
)
from litellm_provider_config import custom_provider_registry


class _PresenceOnlyEnvironment(dict[str, str]):
    def get(self, key, default=None):
        raise AssertionError("credential values must not be read")


class GuiLiteLLMSettingsTests(unittest.TestCase):
    def test_provider_is_model_prefix(self):
        self.assertEqual(provider_from_model("openrouter/openai/gpt-5"), "openrouter")

    def test_openai_key_status_never_exposes_value(self):
        status = provider_credential_status(
            "openai/gpt-5",
            {"OPENAI_API_KEY": "secret-value"},
        )
        self.assertTrue(status.configured)
        self.assertIn("OPENAI_API_KEY", status.message)
        self.assertNotIn("secret-value", status.message)

    def test_missing_provider_key_is_reported(self):
        status = provider_credential_status("anthropic/claude", {})
        self.assertFalse(status.configured)
        self.assertIn("ANTHROPIC_API_KEY", status.message)

    def test_credential_presence_check_never_reads_values(self):
        cases = (
            ("openai/gpt-5", {"OPENAI_API_KEY": "secret-value"}),
            (
                "vertex_ai/gemini",
                {"VERTEXAI_PROJECT": "secret-project", "VERTEXAI_LOCATION": "region"},
            ),
        )
        for model, values in cases:
            with self.subTest(model=model):
                status = provider_credential_status(
                    model,
                    _PresenceOnlyEnvironment(values),
                )
                self.assertTrue(status.configured)

    def test_ollama_does_not_require_key(self):
        status = provider_credential_status("ollama/llama3", {})
        self.assertTrue(status.configured)
        self.assertEqual(status.environment_names, ())

    def test_unknown_provider_defers_to_provider_docs(self):
        status = provider_credential_status("custom/model", {})
        self.assertIsNone(status.configured)
        self.assertIn("未内置", status.message)

    def test_custom_provider_env_detection_uses_api_key_env(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "label": "OpenCode Go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                }
            ]
        )
        status = provider_credential_status(
            "opencode-go/gpt-4o-mini",
            {"OPENCODE_GO_API_KEY": "secret"},
            registry,
        )
        self.assertTrue(status.configured)
        self.assertEqual(status.environment_names, ("OPENCODE_GO_API_KEY",))
        self.assertNotIn("secret", status.message)
        self.assertIn("OPENCODE_GO_API_KEY", status.message)

        missing = provider_credential_status(
            "opencode-go/gpt-4o-mini",
            {},
            registry,
        )
        self.assertFalse(missing.configured)

    def test_custom_provider_without_env_hints_keyring(self):
        registry = custom_provider_registry(
            [
                {
                    "id": "opencode-go",
                    "base_url": "https://opencode.ai/zen/go/v1",
                }
            ]
        )
        status = provider_credential_status(
            "opencode-go/gpt-4o-mini",
            {},
            registry,
        )
        self.assertIsNone(status.configured)
        self.assertIn("保存 API Key", status.message)

    def test_custom_provider_form_validation(self):
        self.assertEqual(
            validate_custom_provider_form(
                "opencode-go",
                "https://opencode.ai/zen/go/v1",
            ),
            "",
        )
        self.assertIn(
            "http(s)",
            validate_custom_provider_form(
                "opencode-go",
                "not-a-url",
            ),
        )
        self.assertIn(
            "冲突",
            validate_custom_provider_form("openai", "https://example.com"),
        )
        self.assertIn(
            "环境变量名",
            validate_custom_provider_form(
                "opencode-go",
                "https://opencode.ai/zen/go/v1",
                api_key_env="BAD NAME",
            ),
        )
        self.assertEqual(
            validate_custom_provider_form(
                "opencode-go",
                "https://opencode.ai/zen/go/v1",
                models_url="https://opencode.ai/zen/go/v1/models",
                api_key_env="OPENCODE_GO_API_KEY",
            ),
            "",
        )

    def test_legacy_litellm_model_loads_without_overwriting_gemini_default(self):
        models = read_sync_backend_models(
            {"backend": "litellm", "model": "openai/gpt-5"},
            "litellm",
            "gemini-default",
        )
        self.assertEqual(models.gemini_model, "gemini-default")
        self.assertEqual(models.litellm_model, "openai/gpt-5")

    def test_backend_models_roundtrip_through_existing_runtime_fields(self):
        config = {"unknown": "preserved"}
        active = write_sync_backend_models(
            config,
            "litellm",
            "gemini-model",
            "anthropic/claude",
        )
        self.assertEqual(active, "anthropic/claude")
        self.assertEqual(config["backend"], "litellm")
        self.assertEqual(config["model"], "anthropic/claude")
        self.assertEqual(config["gemini_model"], "gemini-model")
        self.assertEqual(config["litellm_model"], "anthropic/claude")
        self.assertEqual(config["unknown"], "preserved")


if __name__ == "__main__":
    unittest.main()
