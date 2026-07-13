import unittest
from types import SimpleNamespace

from litellm_provider_config import (
    KEYRING_SERVICE,
    delete_provider_api_key,
    load_provider_api_key,
    models_for_provider,
    provider_from_model,
    store_provider_api_key,
)


class _FakeKeyring:
    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def delete_password(self, service, username):
        del self.values[(service, username)]


class LiteLLMProviderConfigTests(unittest.TestCase):
    def test_key_is_stored_by_provider_in_keyring(self):
        keyring = _FakeKeyring()
        store_provider_api_key("OpenAI", " secret-value ", keyring)

        self.assertEqual(load_provider_api_key("openai", keyring), "secret-value")
        self.assertEqual(
            keyring.values[(KEYRING_SERVICE, "openai")],
            "secret-value",
        )
        self.assertTrue(delete_provider_api_key("openai", keyring))
        self.assertEqual(load_provider_api_key("openai", keyring), "")
        self.assertFalse(delete_provider_api_key("openai", keyring))

    def test_ollama_does_not_use_keyring(self):
        keyring = _FakeKeyring()
        self.assertEqual(load_provider_api_key("ollama", keyring), "")
        self.assertFalse(delete_provider_api_key("ollama", keyring))

    def test_catalog_filters_non_text_models_and_adds_provider_prefix(self):
        fake_litellm = SimpleNamespace(
            models_by_provider={"openai": ("gpt-text", "dall-e-test", "openai/gpt-prefixed")},
            model_cost={
                "gpt-text": {"mode": "chat"},
                "dall-e-test": {"mode": "image_generation"},
                "openai/gpt-prefixed": {"mode": "responses"},
            },
        )

        models = models_for_provider("openai", fake_litellm)

        self.assertIn("openai/gpt-text", models)
        self.assertIn("openai/gpt-prefixed", models)
        self.assertNotIn("openai/dall-e-test", models)

    def test_provider_is_derived_from_model_prefix(self):
        self.assertEqual(provider_from_model(" OpenRouter/openai/gpt-5 "), "openrouter")
        self.assertEqual(provider_from_model("gpt-5"), "")


if __name__ == "__main__":
    unittest.main()
