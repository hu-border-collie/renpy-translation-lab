import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import model_profile as mp
from litellm_sync_backend import LiteLLMBackendError


def _explicit_translation_plan(stage_model, *, backend="litellm", primary="sync-primary-should-lose"):
    plan = mp.resolve_routing_plan(
        {"sync": {"backend": backend, "model": primary}},
        stage_overrides={"translation": stage_model},
    )
    return plan.routes["translation"], plan


class LiteLLMSyncIntegrationTests(unittest.TestCase):
    @staticmethod
    def _fake_result(provider="litellm", model="openai/test"):
        return type("Result", (), {
            "response_payload": {"choices": []},
            "response_text": "[]",
            "finish_reason": "stop",
            "usage_metadata": {"total_tokens": 3},
            "provider": provider,
            "model": model,
            "execution_mode": "sync",
        })()

    def test_sync_runner_uses_explicit_route_model_not_sync_model(self):
        explicit_model = "openai/explicit-stage"
        route, plan = _explicit_translation_plan(explicit_model)
        fake_result = self._fake_result(model=explicit_model)
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = fake_result
        custom_providers = {"opencode-go": object()}

        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/sync-overlay"),
            mock.patch.object(batch_mod, "SYNC_TIMEOUT_SECONDS", 45),
            mock.patch.object(
                batch_mod.legacy,
                "CUSTOM_LITELLM_PROVIDERS",
                custom_providers,
            ),
            mock.patch.object(batch_mod, "create_batch_client") as create_client,
            mock.patch.object(
                mp,
                "build_sync_backend",
                return_value=fake_backend,
            ) as backend_builder,
        ):
            result = batch_mod.run_sync_request(
                {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
                route,
                plan=plan,
            )

        create_client.assert_not_called()
        backend_builder.assert_called_once()
        built_profile = backend_builder.call_args.args[0]
        self.assertEqual(built_profile.model, explicit_model)
        self.assertEqual(
            backend_builder.call_args.kwargs["custom_providers"],
            custom_providers,
        )
        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.model, explicit_model)
        self.assertNotEqual(request.model, "openai/sync-overlay")
        self.assertEqual(request.config["timeout"], 45)
        self.assertEqual(result["provider"], "litellm")
        self.assertEqual(result["execution_mode"], "sync")

    def test_sync_runner_passes_timeout_to_gemini_backend(self):
        route, plan = _explicit_translation_plan(
            "gemini-explicit",
            backend="gemini",
            primary="gemini-primary-should-lose",
        )
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = self._fake_result(
            provider="gemini",
            model="gemini-explicit",
        )
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "gemini"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "gemini-sync-overlay"),
            mock.patch.object(batch_mod, "SYNC_TIMEOUT_SECONDS", 45),
            mock.patch.object(batch_mod, "create_batch_client", return_value=object()),
            mock.patch.object(
                mp,
                "build_sync_backend",
                return_value=fake_backend,
            ),
        ):
            batch_mod.run_sync_request({"contents": []}, route, plan=plan)

        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.model, "gemini-explicit")
        self.assertEqual(request.config["timeout"], 45)

    def test_litellm_rejects_gemini_api_key_index(self):
        route, plan = _explicit_translation_plan("openai/test")
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/sync-overlay"),
        ):
            with self.assertRaises(SystemExit) as captured:
                batch_mod.run_sync_request(
                    {},
                    route,
                    plan=plan,
                    api_key_index=0,
                )
        self.assertIn("only supported by the Gemini", str(captured.exception))

    def test_sync_manifest_records_effective_request_timeout(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            batch_mod,
            "SYNC_TIMEOUT_SECONDS",
            45,
        ):
            manifest_path = batch_mod.make_sync_manifest(
                package_dir=tmp,
                mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                display_name="timeout-test",
                chunks=[],
                request_rows=[],
                settings={"max_output_tokens": 64},
            )

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(manifest["settings"]["timeout_seconds"], 45)
        self.assertIn("model_routing", manifest)

    def test_litellm_authentication_failure_is_not_retried(self):
        route, plan = _explicit_translation_plan("openai/test")
        fake_backend = mock.Mock()
        fake_backend.generate.side_effect = LiteLLMBackendError(
            "provider echoed secret-value",
            category="authentication",
        )
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/sync-overlay"),
            mock.patch.object(
                mp,
                "build_sync_backend",
                return_value=fake_backend,
            ),
        ):
            with self.assertRaises(LiteLLMBackendError):
                batch_mod.run_sync_request({"contents": []}, route, plan=plan)

        self.assertEqual(fake_backend.generate.call_count, 1)

    def test_litellm_timeout_retry_is_bounded(self):
        route, plan = _explicit_translation_plan("openai/test")
        fake_backend = mock.Mock()
        fake_backend.generate.side_effect = [
            LiteLLMBackendError("timeout one", category="timeout"),
            LiteLLMBackendError("timeout two", category="timeout"),
            self._fake_result(),
        ]
        with (
            mock.patch.object(batch_mod, "SYNC_BACKEND", "litellm"),
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/sync-overlay"),
            mock.patch.object(batch_mod.time, "sleep"),
            mock.patch.object(
                mp,
                "build_sync_backend",
                return_value=fake_backend,
            ),
        ):
            result = batch_mod.run_sync_request(
                {"contents": []},
                route,
                plan=plan,
            )

        self.assertEqual(fake_backend.generate.call_count, 3)
        self.assertEqual(result["provider"], "litellm")
        request = fake_backend.generate.call_args.args[0]
        self.assertEqual(request.model, "openai/test")


if __name__ == "__main__":
    unittest.main()
