import unittest
from unittest import mock

import gemini_translate_batch as batch_mod


class LiteLLMConfigTests(unittest.TestCase):
    def test_batch_settings_load_explicit_sync_backend_and_model(self):
        previous_backend = batch_mod.SYNC_BACKEND
        previous_model = batch_mod.SYNC_MODEL
        previous_timeout = batch_mod.SYNC_TIMEOUT_SECONDS
        try:
            with mock.patch.object(
                batch_mod,
                "load_json_file",
                side_effect=[
                    {},
                    {
                        "sync": {
                            "backend": "litellm",
                            "model": "openai/test",
                            "timeout_seconds": 45,
                        }
                    },
                ],
            ):
                batch_mod.load_batch_settings()
            self.assertEqual(batch_mod.SYNC_BACKEND, "litellm")
            self.assertEqual(batch_mod.SYNC_MODEL, "openai/test")
            self.assertEqual(batch_mod.SYNC_TIMEOUT_SECONDS, 45)
        finally:
            batch_mod.SYNC_BACKEND = previous_backend
            batch_mod.SYNC_MODEL = previous_model
            batch_mod.SYNC_TIMEOUT_SECONDS = previous_timeout

    def test_batch_settings_reject_unknown_sync_backend(self):
        with mock.patch.object(
            batch_mod,
            "load_json_file",
            side_effect=[{}, {"sync": {"backend": "automatic"}}],
        ):
            with self.assertRaises(SystemExit) as captured:
                batch_mod.load_batch_settings()
        self.assertIn("Unsupported sync backend", str(captured.exception))
        self.assertEqual(captured.exception.code_name, "MODEL_PROFILE_INVALID")
        self.assertEqual(captured.exception.semantic_exit_code, 5)

    def test_doctor_can_tolerate_unknown_backend_for_routing_report(self):
        previous_backend = batch_mod.SYNC_BACKEND
        try:
            with mock.patch.object(
                batch_mod,
                "load_json_file",
                side_effect=[{}, {"sync": {"backend": "automatic"}}],
            ):
                batch_mod.load_batch_settings(tolerate_routing_errors=True)
            self.assertEqual(batch_mod.SYNC_BACKEND, "automatic")
            status = batch_mod.collect_doctor_model_routing_status()
            self.assertEqual(status["status"], "attention")
            self.assertTrue(all(
                issue["code"] == "MODEL_PROFILE_INVALID"
                for issue in status["issues"]
            ))
        finally:
            batch_mod.SYNC_BACKEND = previous_backend


if __name__ == "__main__":
    unittest.main()
