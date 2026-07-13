import unittest
from unittest import mock

import gemini_translate_batch as batch_mod


class LiteLLMConfigTests(unittest.TestCase):
    def test_batch_settings_load_explicit_sync_backend_and_model(self):
        previous_backend = batch_mod.SYNC_BACKEND
        previous_model = batch_mod.SYNC_MODEL
        try:
            with mock.patch.object(
                batch_mod,
                "load_json_file",
                side_effect=[{}, {"sync": {"backend": "litellm", "model": "openai/test"}}],
            ):
                batch_mod.load_batch_settings()
            self.assertEqual(batch_mod.SYNC_BACKEND, "litellm")
            self.assertEqual(batch_mod.SYNC_MODEL, "openai/test")
        finally:
            batch_mod.SYNC_BACKEND = previous_backend
            batch_mod.SYNC_MODEL = previous_model

    def test_batch_settings_reject_unknown_sync_backend(self):
        with mock.patch.object(
            batch_mod,
            "load_json_file",
            side_effect=[{}, {"sync": {"backend": "automatic"}}],
        ):
            with self.assertRaises(SystemExit) as captured:
                batch_mod.load_batch_settings()
        self.assertIn("Unsupported sync backend", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
