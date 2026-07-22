# -*- coding: utf-8 -*-
import unittest

import translator_runtime as runtime


class RotationSettingsTests(unittest.TestCase):
    def setUp(self):
        self._old = {
            "API_KEYS": list(runtime.API_KEYS),
            "MODELS": list(runtime.MODELS),
            "CURRENT_KEY_INDEX": runtime.CURRENT_KEY_INDEX,
            "CURRENT_MODEL_INDEX": runtime.CURRENT_MODEL_INDEX,
            "API_KEY_ROTATION_ENABLED": runtime.API_KEY_ROTATION_ENABLED,
            "MODEL_ROTATION_ENABLED": runtime.MODEL_ROTATION_ENABLED,
            "MODEL_ROTATION_MODELS": list(runtime.MODEL_ROTATION_MODELS),
        }

    def tearDown(self):
        runtime.API_KEYS = list(self._old["API_KEYS"])
        runtime.MODELS = list(self._old["MODELS"])
        runtime.CURRENT_KEY_INDEX = self._old["CURRENT_KEY_INDEX"]
        runtime.CURRENT_MODEL_INDEX = self._old["CURRENT_MODEL_INDEX"]
        runtime.API_KEY_ROTATION_ENABLED = self._old["API_KEY_ROTATION_ENABLED"]
        runtime.MODEL_ROTATION_ENABLED = self._old["MODEL_ROTATION_ENABLED"]
        runtime.MODEL_ROTATION_MODELS = list(self._old["MODEL_ROTATION_MODELS"])

    def test_defaults_disable_model_rotation_and_enable_key_rotation(self):
        runtime.load_rotation_settings({})
        self.assertTrue(runtime.API_KEY_ROTATION_ENABLED)
        self.assertFalse(runtime.MODEL_ROTATION_ENABLED)
        self.assertEqual(runtime.MODEL_ROTATION_MODELS, [])

    def test_rotate_api_key_respects_enabled_flag(self):
        runtime.API_KEYS = ["key-a", "key-b"]
        runtime.CURRENT_KEY_INDEX = 0
        runtime.API_KEY_ROTATION_ENABLED = False
        self.assertFalse(runtime.rotate_api_key())
        self.assertEqual(runtime.CURRENT_KEY_INDEX, 0)
        self.assertEqual(runtime.api_key_rotation_attempts(), 1)

        runtime.API_KEY_ROTATION_ENABLED = True
        self.assertTrue(runtime.rotate_api_key())
        self.assertEqual(runtime.CURRENT_KEY_INDEX, 1)
        self.assertEqual(runtime.api_key_rotation_attempts(), 2)

    def test_rotate_model_disabled_by_default(self):
        runtime.MODELS = ["gemini-a", "gemini-b"]
        runtime.CURRENT_MODEL_INDEX = 0
        runtime.MODEL_ROTATION_ENABLED = False
        self.assertFalse(runtime.rotate_model())
        self.assertEqual(runtime.CURRENT_MODEL_INDEX, 0)

    def test_rotate_model_uses_explicit_pool_when_enabled(self):
        runtime.MODELS = ["gemini-selected"]
        runtime.CURRENT_MODEL_INDEX = 0
        runtime.MODEL_ROTATION_ENABLED = True
        runtime.MODEL_ROTATION_MODELS = ["gemini-a", "gemini-b", "gemini-c"]
        self.assertTrue(runtime.rotate_model())
        self.assertEqual(runtime.get_current_model(), "gemini-b")
        self.assertEqual(runtime.MODELS, ["gemini-a", "gemini-b", "gemini-c"])

    def test_load_sync_pins_single_model_when_rotation_disabled(self):
        runtime.load_sync_translation_settings(
            {
                "sync": {
                    "backend": "gemini",
                    "model": "gemini-3.1-flash-lite",
                },
                "rotation": {
                    "model": {"enabled": False},
                },
            }
        )
        self.assertEqual(runtime.MODELS, ["gemini-3.1-flash-lite"])
        self.assertFalse(runtime.MODEL_ROTATION_ENABLED)

    def test_load_sync_builds_rotation_pool_when_enabled(self):
        runtime.load_sync_translation_settings(
            {
                "sync": {
                    "backend": "gemini",
                    "model": "gemini-3.1-flash-lite",
                },
                "rotation": {
                    "model": {
                        "enabled": True,
                        "models": [
                            "gemini-3.5-flash-lite",
                            "gemini-3.6-flash",
                            "gemini-3.1-flash-lite",
                        ],
                    }
                },
            }
        )
        self.assertTrue(runtime.MODEL_ROTATION_ENABLED)
        self.assertEqual(runtime.MODELS[0], "gemini-3.1-flash-lite")
        self.assertIn("gemini-3.5-flash-lite", runtime.MODELS)
        self.assertIn("gemini-3.6-flash", runtime.MODELS)


if __name__ == "__main__":
    unittest.main()
