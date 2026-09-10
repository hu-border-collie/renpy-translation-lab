"""Production routing activation, legacy isolation and frozen-run regression tests."""
import copy
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
from unittest import TestCase
from unittest.mock import patch

import model_profile
import translator_runtime as runtime
import gemini_translate_batch as batch
from model_routing_migration import preview_migration
from model_routing_reader import runtime_settings_view, read_routing_plan


FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def migrated(name="gemini_batch"):
    return preview_migration(json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))).config


class ProductionRoutingTests(TestCase):
    def test_legacy_poison_does_not_change_loaded_models_or_embedding(self):
        for name in ("gemini_batch", "gemini_sync", "litellm_builtin", "litellm_custom"):
            with self.subTest(name=name), runtime.runtime_config_scope(runtime.default_runtime_config()):
                config = migrated(name)
                expected = runtime_settings_view(config)
                config["sync"]["backend"] = "invalid-obsolete-backend"
                config["sync"]["models"] = ["obsolete/model"]
                config["sync"]["model"] = "obsolete/model"
                config["sync"]["custom_litellm_providers"] = "obsolete"
                config["sync"].setdefault("rag", {})["embedding_model"] = "obsolete"
                config["batch"]["model"] = "obsolete"
                before = copy.deepcopy(config)
                with redirect_stdout(io.StringIO()):
                    runtime.load_sync_translation_settings(config)
                self.assertEqual(runtime.MODELS, expected["sync"]["models"])
                self.assertEqual(runtime.SYNC_BACKEND, expected["sync"]["backend"])
                self.assertEqual(runtime_settings_view(config)["sync"]["rag"], expected["sync"]["rag"])
                self.assertEqual(config, before)
                plan = model_profile.resolve_routing_plan_from_runtime(
                    model_routing_config=runtime.MODEL_ROUTING_CONFIG,
                    sync_backend="obsolete", sync_model="obsolete", execution="sync")
                route = plan.routes["translation"]
                self.assertEqual(plan.profiles[route.profile_id].model, expected["sync"]["model"])

    def test_scope_restores_and_deep_copies_new_config(self):
        config = migrated()["model_routing"]
        before = copy.deepcopy(runtime.MODEL_ROUTING_CONFIG)
        with runtime.runtime_config_scope(runtime.RuntimeConfig(model_routing_config=config)):
            config["schema_version"] = 900
            self.assertEqual(runtime.MODEL_ROUTING_CONFIG["schema_version"], 1)
            snapshot = runtime.snapshot_runtime_config()
            snapshot.model_routing_config["schema_version"] = 700
            self.assertEqual(runtime.MODEL_ROUTING_CONFIG["schema_version"], 1)
        self.assertEqual(runtime.MODEL_ROUTING_CONFIG, before)

    def test_invalid_section_never_falls_back(self):
        for section in ({}, {"schema_version": 900}, False):
            with self.subTest(section=section), runtime.runtime_config_scope(runtime.default_runtime_config()):
                with self.assertRaises(model_profile.ModelRoutingConfigError):
                    runtime.load_sync_translation_settings({"model_routing": section, "sync": {"model": "gemini-3.1-flash-lite"}})

    def test_batch_service_freezes_new_section_and_manifest_wins(self):
        config = migrated()
        with runtime.runtime_config_scope(runtime.RuntimeConfig(model_routing_config=config["model_routing"])):
            plan = batch.freeze_runtime_routing_plan(execution="gemini_batch")
            self.assertEqual(plan.profiles[plan.routes["translation"].profile_id].model, config["batch"]["model"])
            runtime.MODEL_ROUTING_CONFIG = {"schema_version": 900}
            restored = batch.resolve_manifest_routing_plan({"model_routing": plan.to_manifest_dict()})
            self.assertEqual(restored.to_manifest_dict(), plan.to_manifest_dict())

    def test_old_load_clears_new_snapshot(self):
        with runtime.runtime_config_scope(runtime.RuntimeConfig(model_routing_config=migrated()["model_routing"])):
            with redirect_stdout(io.StringIO()):
                runtime.load_sync_translation_settings({"sync": {"model": "gemini-3.1-flash-lite"}})
            self.assertIsNone(runtime.MODEL_ROUTING_CONFIG)

    def test_backend_uses_frozen_custom_connection(self):
        plan = read_routing_plan(migrated("litellm_custom"), legacy_execution="sync")
        profile = plan.profiles[plan.routes["translation"].profile_id]
        with patch("litellm_sync_backend.LiteLLMSyncBackend") as factory:
            model_profile.build_sync_backend(profile, custom_providers={})
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["custom_providers"][profile.provider].base_url, profile.base_url)
        self.assertEqual(kwargs["credential_ref"], profile.credential_ref.to_manifest_dict())

    def test_settings_preserve_unknown_new_fields_and_block_invalid_section(self):
        from gui_qt.settings.save_apply import apply_collected_settings
        config = migrated()
        config["model_routing"]["future_extension"] = {"nested": [1, 2]}
        section = copy.deepcopy(config["model_routing"])
        result = apply_collected_settings(config, {"batch_model": "old-ui-model"}, original_config=copy.deepcopy(config))
        self.assertTrue(result.ok)
        self.assertEqual(config["model_routing"], section)
        config["model_routing"]["schema_version"] = 900
        before = copy.deepcopy(config)
        result = apply_collected_settings(config, {"batch_model": "other"}, original_config=copy.deepcopy(config))
        self.assertFalse(result.ok)
        self.assertEqual(result.block_page, "models")
        self.assertEqual(config, before)

    def test_unsupported_profile_params_are_not_silently_ignored(self):
        config = migrated()
        key = config["model_routing"]["legacy_entrypoints"]["sync_profile_id"]
        config["model_routing"]["profiles"][key]["params"] = {"temperature": 0.2}
        with self.assertRaisesRegex(model_profile.ModelRoutingConfigError, "params"):
            runtime_settings_view(config)

    def test_explicit_override_keeps_connection_and_rejects_provider_switch(self):
        config = migrated("litellm_custom")["model_routing"]
        plan = model_profile.resolve_routing_plan_from_runtime(
            model_routing_config=config, sync_backend="obsolete", execution="sync",
            stage_overrides={"translation": "acme-compatible/model-b"})
        profile = plan.profiles[plan.routes["translation"].profile_id]
        self.assertEqual(profile.model, "acme-compatible/model-b")
        self.assertEqual(profile.base_url, "https://models.example.test/v1")
        with self.assertRaises(model_profile.ModelRoutingConfigError):
            model_profile.resolve_routing_plan_from_runtime(
                model_routing_config=config, sync_backend="obsolete", execution="sync",
                stage_overrides={"translation": "other/model-b"})

    def test_rotation_policy_is_not_a_secret_but_embedded_values_are(self):
        config = {"sync": {"model": "gemini-3.1-flash-lite"}, "batch": {"model": "gemini-3.1-flash-lite"},
                  "rotation": {"api_key": {"enabled": True}}}
        self.assertEqual(preview_migration(config).config["rotation"], config["rotation"])
        for policy in ("embedded-value", {"enabled": True, "value": "embedded-value"}):
            config["rotation"]["api_key"] = policy
            with self.assertRaises(model_profile.ModelRoutingConfigError):
                preview_migration(config)

    def test_real_loader_rejects_invalid_routing_before_path_correction(self):
        from cli_contract import MachineContractError
        with runtime.runtime_config_scope(runtime.default_runtime_config()):
            with patch.object(runtime, "_read_json_object", return_value={"model_routing": {"schema_version": 900}}), \
                 patch.object(runtime, "persist_game_root") as persist:
                with self.assertRaises(MachineContractError) as caught:
                    runtime.load_translator_settings()
                self.assertEqual(caught.exception.code_name, model_profile.MODEL_PROFILE_INVALID)
                persist.assert_not_called()

    def test_environment_reference_does_not_read_provider_keyring(self):
        from litellm_sync_backend import LiteLLMSyncBackend, LiteLLMBackendError
        backend = LiteLLMSyncBackend(credential_ref={"kind": "env", "name": "ROUTING_TEST_KEY"})
        with patch.dict("os.environ", {"ROUTING_TEST_KEY": "synthetic-test-value"}, clear=True), \
             patch("litellm_provider_config.load_provider_api_key") as keyring:
            creds = backend._resolve_credentials("openai", None)
            self.assertEqual(len(creds), 1)
            keyring.assert_not_called()
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(LiteLLMBackendError):
                backend._resolve_credentials("openai", None)

    def test_direct_batch_loader_publishes_v1_snapshot(self):
        config = migrated()
        with patch.dict(batch.__dict__, batch.__dict__.copy()), runtime.runtime_config_scope(runtime.default_runtime_config()):
            with patch.object(batch, "load_json_file", side_effect=lambda path: config if path == runtime.TRANSLATOR_CONFIG else {}), \
                 redirect_stdout(io.StringIO()):
                batch.load_batch_settings()
            self.assertEqual(runtime.MODEL_ROUTING_CONFIG, config["model_routing"])
            plan = batch.freeze_runtime_routing_plan(execution="gemini_batch")
            self.assertEqual(plan.routes["translation"].profile_id, config["model_routing"]["legacy_entrypoints"]["batch_profile_id"])

    def test_obsolete_model_containers_do_not_override_v1(self):
        config = migrated()
        expected = runtime_settings_view(config)
        config["sync"] = "obsolete"
        config["batch"] = None
        view = runtime_settings_view(config)
        self.assertEqual(view["sync"]["model"], expected["sync"]["model"])
        self.assertEqual(view["batch"]["model"], expected["batch"]["model"])

    def test_batch_command_rejects_sync_only_stage_before_artifacts(self):
        from cli_contract import MachineContractError
        config = migrated()
        section = config["model_routing"]
        section["routes"]["keyword"] = {"profile_id": section["legacy_entrypoints"]["sync_profile_id"], "strategy": "sync"}
        with runtime.runtime_config_scope(runtime.RuntimeConfig(model_routing_config=section)):
            with self.assertRaises(MachineContractError):
                batch.freeze_runtime_routing_plan(execution="gemini_batch", required_stages={"keyword"})
            plan = batch.freeze_runtime_routing_plan(execution="sync")
            self.assertEqual(plan.routes["keyword"].strategy.value, "sync")

    def test_frozen_native_profile_does_not_acquire_live_custom_endpoint(self):
        plan = read_routing_plan(migrated("litellm_builtin"), legacy_execution="sync")
        profile = plan.profiles[plan.routes["translation"].profile_id]
        with patch("litellm_sync_backend.LiteLLMSyncBackend") as factory:
            model_profile.build_sync_backend(profile, custom_providers={profile.provider: object()})
        self.assertNotIn(profile.provider, factory.call_args.kwargs["custom_providers"])
