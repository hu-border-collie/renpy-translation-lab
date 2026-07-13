import unittest

from gui_qt.litellm_settings import (
    provider_credential_status,
    provider_from_model,
    read_sync_backend_models,
    write_sync_backend_models,
)


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

    def test_ollama_does_not_require_key(self):
        status = provider_credential_status("ollama/llama3", {})
        self.assertTrue(status.configured)
        self.assertEqual(status.environment_names, ())

    def test_unknown_provider_defers_to_provider_docs(self):
        status = provider_credential_status("custom/model", {})
        self.assertIsNone(status.configured)
        self.assertIn("未内置", status.message)

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
