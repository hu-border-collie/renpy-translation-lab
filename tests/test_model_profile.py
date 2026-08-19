"""Tests for the model routing contract module (issue #345 P1/P2)."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import litellm_provider_config as lpc
import model_profile as mp
import translation_ab_experiment as ab_mod
from litellm_sync_backend import LiteLLMBackendError
from sync_model_backend import GeminiSyncBackend, SyncGenerationRequest


def _custom_providers() -> dict[str, lpc.CustomLiteLLMProvider]:
    provider = lpc.CustomLiteLLMProvider(
        id="opencode-go",
        label="OpenCode Go",
        base_url="https://api.opencode-go.example",
        models_url="https://api.opencode-go.example/models",
        api_key_env="OPENCODE_GO_API_KEY",
    )
    return {provider.id: provider}


def _no_key_custom_provider() -> dict[str, lpc.CustomLiteLLMProvider]:
    provider = lpc.CustomLiteLLMProvider(
        id="local-llama",
        label="Local llama",
        base_url="http://127.0.0.1:11434",
        models_url="http://127.0.0.1:11434/models",
        requires_key=False,
    )
    return {provider.id: provider}


class BuildProfileRegistryTests(unittest.TestCase):
    def test_gemini_defaults_without_config(self) -> None:
        registry = mp.build_profile_registry({})
        self.assertEqual(set(registry), {"primary", "batch"})
        for profile in registry.values():
            self.assertEqual(profile.adapter, mp.ADAPTER_GEMINI)
            self.assertEqual(profile.provider, "gemini")
            self.assertEqual(profile.model, "gemini-3.1-flash-lite")
            self.assertEqual(
                profile.credential_ref,
                mp.CredentialRef(mp.CREDENTIAL_KIND_API_KEYS_JSON, "api_keys"),
            )

    def test_sync_model_over_batch_model(self) -> None:
        registry = mp.build_profile_registry({
            "sync": {"backend": "gemini", "model": "gemini-3.5-flash"},
            "batch": {"model": "gemini-2.5-flash"},
        })
        self.assertEqual(registry["primary"].model, "gemini-3.5-flash")
        self.assertEqual(registry["batch"].model, "gemini-2.5-flash")

    def test_game_config_batch_model_fallback(self) -> None:
        registry = mp.build_profile_registry(
            {"batch": {}},
            game_config={"batch_model": "gemini-2.5-pro"},
        )
        # batch.model wins over the game-level batch_model ...
        self.assertEqual(registry["batch"].model, "gemini-2.5-pro")
        # ... and primary falls back to it when sync.model is unset.
        self.assertEqual(registry["primary"].model, "gemini-2.5-pro")

    def test_litellm_builtin_provider(self) -> None:
        registry = mp.build_profile_registry({
            "sync": {"backend": "litellm", "model": "deepseek/deepseek-chat"},
        })
        primary = registry["primary"]
        self.assertEqual(primary.adapter, mp.ADAPTER_LITELLM)
        self.assertEqual(primary.provider, "deepseek")
        self.assertEqual(primary.base_url, "")
        self.assertEqual(
            primary.credential_ref,
            mp.CredentialRef(mp.CREDENTIAL_KIND_KEYRING, "deepseek"),
        )

    def test_litellm_custom_provider(self) -> None:
        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        primary = registry["primary"]
        self.assertEqual(primary.provider, "opencode-go")
        self.assertEqual(primary.base_url, "https://api.opencode-go.example")
        self.assertEqual(
            primary.credential_ref,
            mp.CredentialRef(
                mp.CREDENTIAL_KIND_KEYRING,
                "opencode-go",
                env_name="OPENCODE_GO_API_KEY",
            ),
        )

    def test_custom_provider_without_key_requirement(self) -> None:
        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "local-llama/llama-4"}},
            custom_providers=_no_key_custom_provider(),
        )
        self.assertEqual(
            registry["primary"].credential_ref.kind,
            mp.CREDENTIAL_KIND_NONE,
        )

    def test_rotation_pool_from_sync_models(self) -> None:
        registry = mp.build_profile_registry({
            "sync": {
                "backend": "gemini",
                "model": "gemini-3.5-flash",
                "models": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
            },
        })
        self.assertEqual(
            registry["primary"].models,
            ("gemini-3.5-flash", "gemini-3.1-flash-lite"),
        )

    def test_stage_profiles_only_when_configured(self) -> None:
        base = {"batch": {"model": "gemini-2.5-flash"}}
        self.assertNotIn("project_analysis_model", mp.build_profile_registry(base))
        self.assertNotIn("final_review_model", mp.build_profile_registry(base))
        registry = mp.build_profile_registry({
            "sync": {"backend": "litellm", "model": "deepseek/deepseek-chat"},
            "batch": {
                "model": "gemini-2.5-flash",
                "project_analysis": {"model": "gemini-3.5-flash"},
                "final_review": {"model": "gemini-3.1-pro-preview"},
            },
        })
        # Project analysis runs through the sync path, so it follows the
        # configured sync backend adapter; a bare Gemini model id under the
        # litellm backend keeps an empty provider, which validation flags as
        # an invalid profile instead of rerouting it silently.
        pa = registry["project_analysis_model"]
        self.assertEqual(pa.adapter, mp.ADAPTER_LITELLM)
        self.assertEqual(pa.provider, "")
        self.assertEqual(pa.model, "gemini-3.5-flash")
        # Final review is batch-executed and always stays on Gemini.
        fr = registry["final_review_model"]
        self.assertEqual(fr.adapter, mp.ADAPTER_GEMINI)
        self.assertEqual(fr.model, "gemini-3.1-pro-preview")


class ResolveRoutingPlanTests(unittest.TestCase):
    def test_default_sync_plan(self) -> None:
        plan = mp.resolve_routing_plan({}, created_at="2026-08-16T00:00:00Z")
        self.assertEqual(plan.schema_version, mp.MODEL_PROFILE_SCHEMA_VERSION)
        self.assertEqual(plan.created_at, "2026-08-16T00:00:00Z")
        self.assertEqual(plan.primary_profile_id, "primary")
        expected = {
            "translation": ("primary", "sync", "builtin_default"),
            "keyword": ("primary", "sync", "inherited"),
            "revision": ("primary", "sync", "inherited"),
            "project_analysis": ("primary", "sync", "inherited"),
            "final_review": ("batch", "gemini_batch", "inherited"),
            "ab_experiment": ("primary", "sync", "inherited"),
        }
        for stage, (profile_id, strategy, source) in expected.items():
            route = plan.routes[stage]
            self.assertEqual(route.profile_id, profile_id, stage)
            self.assertEqual(route.strategy.value, strategy, stage)
            self.assertEqual(route.source, source, stage)

    def test_sync_run_source_labels(self) -> None:
        plan = mp.resolve_routing_plan({
            "sync": {"backend": "gemini", "model": "gemini-3.5-flash"},
            "batch": {"model": "gemini-2.5-flash"},
        })
        self.assertEqual(
            plan.routes["translation"].source, mp.ROUTE_SOURCE_STAGE_CONFIG,
        )
        plan_without_sync_model = mp.resolve_routing_plan({
            "batch": {"model": "gemini-2.5-flash"},
        })
        self.assertEqual(
            plan_without_sync_model.routes["translation"].source,
            mp.ROUTE_SOURCE_INHERITED,
        )

    def test_batch_execution_routes(self) -> None:
        plan = mp.resolve_routing_plan(
            {"sync": {"backend": "litellm", "model": "deepseek/deepseek-chat"}},
            execution=mp.ExecutionStrategy.GEMINI_BATCH,
        )
        for stage in ("translation", "keyword", "revision"):
            route = plan.routes[stage]
            self.assertEqual(route.profile_id, "batch", stage)
            self.assertEqual(route.strategy, mp.ExecutionStrategy.GEMINI_BATCH)
        # Project analysis stays sync; final review stays gemini_batch.
        self.assertEqual(
            plan.routes["project_analysis"].profile_id, "primary",
        )
        self.assertEqual(
            plan.routes["final_review"].strategy, mp.ExecutionStrategy.GEMINI_BATCH,
        )

    def test_explicit_stage_model_wins_over_primary(self) -> None:
        # Issue #345 option B: the stage's own config key must not be
        # silently overridden by sync.model anymore.
        plan = mp.resolve_routing_plan({
            "sync": {"backend": "litellm", "model": "deepseek/deepseek-chat"},
            "batch": {
                "model": "gemini-2.5-flash",
                "project_analysis": {"model": "gemini-3.5-flash"},
            },
        })
        route = plan.routes["project_analysis"]
        self.assertEqual(route.profile_id, "project_analysis_model")
        self.assertEqual(route.source, mp.ROUTE_SOURCE_STAGE_CONFIG)
        self.assertEqual(
            plan.profiles["project_analysis_model"].model, "gemini-3.5-flash",
        )
        # The primary profile keeps serving the unconfigured stages.
        self.assertEqual(plan.routes["keyword"].profile_id, "primary")

    def test_stage_override_beats_configured_stage_model(self) -> None:
        plan = mp.resolve_routing_plan(
            {"batch": {"project_analysis": {"model": "gemini-3.5-flash"}}},
            stage_overrides={"project_analysis": "deepseek/deepseek-chat"},
        )
        route = plan.routes["project_analysis"]
        self.assertEqual(route.profile_id, "project_analysis_override")
        self.assertEqual(route.source, mp.ROUTE_SOURCE_EXPLICIT)
        self.assertEqual(
            plan.profiles["project_analysis_override"].model,
            "deepseek/deepseek-chat",
        )

    def test_translation_stage_override_wins_over_primary(self) -> None:
        plan = mp.resolve_routing_plan(
            {"sync": {"backend": "gemini", "model": "gemini-primary"}},
            stage_overrides={"translation": "gemini-explicit"},
        )
        route = plan.routes["translation"]
        self.assertEqual(route.profile_id, "translation_override")
        self.assertEqual(route.source, mp.ROUTE_SOURCE_EXPLICIT)
        self.assertEqual(plan.profiles["translation_override"].model, "gemini-explicit")

    def test_ab_experiment_override(self) -> None:
        plan = mp.resolve_routing_plan(
            {},
            stage_overrides={"ab_experiment": "gemini-2.5-pro"},
        )
        route = plan.routes["ab_experiment"]
        self.assertEqual(route.profile_id, "ab_experiment_override")
        self.assertEqual(route.source, mp.ROUTE_SOURCE_EXPLICIT)
        self.assertEqual(route.strategy, mp.ExecutionStrategy.SYNC)

    def test_final_review_override_stays_batch_transport(self) -> None:
        plan = mp.resolve_routing_plan(
            {},
            stage_overrides={"final_review": "gemini-2.5-pro"},
        )
        route = plan.routes["final_review"]
        self.assertEqual(route.profile_id, "final_review_override")
        self.assertEqual(route.strategy, mp.ExecutionStrategy.GEMINI_BATCH)
        self.assertEqual(
            plan.profiles["final_review_override"].adapter, mp.ADAPTER_GEMINI,
        )

    def test_unknown_sync_backend_refused(self) -> None:
        with self.assertRaises(ValueError):
            mp.resolve_routing_plan({"sync": {"backend": "openai"}})

    def test_plan_frozen_against_config_mutation(self) -> None:
        config = {"sync": {"backend": "gemini", "model": "gemini-3.5-flash"}}
        plan = mp.resolve_routing_plan(config, created_at="fixed")
        before = plan.to_manifest_dict()
        config["sync"]["model"] = "gemini-2.5-flash"
        config["batch"] = {"model": "gemini-2.5-pro"}
        self.assertEqual(plan.to_manifest_dict(), before)


class CapabilityTests(unittest.TestCase):
    def test_gemini_capability_defaults(self) -> None:
        profile = mp.build_profile_registry({})["primary"]
        caps = mp.resolve_capabilities(profile)
        self.assertTrue(caps.sync_generation.supported)
        self.assertTrue(caps.remote_batch.supported)
        self.assertTrue(caps.embedding.supported)
        self.assertTrue(caps.reasoning_request.supported)
        self.assertTrue(caps.usage_stats.supported)
        self.assertEqual(caps.structured_output.mode, "strict_json_schema")
        self.assertEqual(
            caps.structured_output.source, mp.CAPABILITY_SOURCE_ADAPTER_DEFAULT,
        )
        self.assertEqual(
            caps.structured_output.basis, mp.STRUCTURED_OUTPUT_BASIS_BUILTIN,
        )
        for flag in (
            caps.sync_generation, caps.reasoning_request, caps.remote_batch,
        ):
            self.assertEqual(flag.source, mp.CAPABILITY_SOURCE_ADAPTER_DEFAULT)
        self.assertIsNone(caps.context_limit_tokens)

    def test_litellm_capability_defaults(self) -> None:
        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        caps = mp.resolve_capabilities(
            registry["primary"], custom_providers=_custom_providers(),
        )
        self.assertTrue(caps.sync_generation.supported)
        self.assertFalse(caps.remote_batch.supported)
        self.assertFalse(caps.embedding.supported)
        self.assertFalse(caps.reasoning_request.supported)
        self.assertEqual(caps.structured_output.mode, "json_object")
        # Provenance and mode-basis are separate vocabularies: the adapter
        # table's "custom_openai_compatible" detail lives in basis, while
        # source shares the capability-source vocabulary.
        self.assertEqual(
            caps.structured_output.source, mp.CAPABILITY_SOURCE_ADAPTER_DEFAULT,
        )
        self.assertEqual(
            caps.structured_output.basis,
            mp.STRUCTURED_OUTPUT_BASIS_CUSTOM_OPENAI_COMPATIBLE,
        )

    def test_capability_overrides_flip_source(self) -> None:
        base = mp.build_profile_registry({})["primary"]
        profile = replace(base, capability_overrides={
            "remote_batch": False,
            "structured_output": {"mode": "json_object"},
            "context_limit_tokens": 1048576,
        })
        caps = mp.resolve_capabilities(profile)
        self.assertFalse(caps.remote_batch.supported)
        self.assertEqual(
            caps.remote_batch.source, mp.CAPABILITY_SOURCE_CONFIG_OVERRIDE,
        )
        self.assertEqual(
            caps.structured_output.source, mp.CAPABILITY_SOURCE_CONFIG_OVERRIDE,
        )
        # An override keeps the basis of the adapter default it replaced.
        self.assertEqual(
            caps.structured_output.basis, mp.STRUCTURED_OUTPUT_BASIS_BUILTIN,
        )
        self.assertEqual(caps.context_limit_tokens, 1048576)
        self.assertEqual(
            caps.context_source, mp.CAPABILITY_SOURCE_CONFIG_OVERRIDE,
        )


class ValidateRoutingPlanTests(unittest.TestCase):
    def test_default_plan_validates_cleanly(self) -> None:
        plan = mp.resolve_routing_plan({})
        self.assertEqual(
            mp.validate_routing_plan(plan, environ={}, keyring_has_credential=None),
            (),
        )

    def test_gemini_batch_rejects_non_gemini_profile(self) -> None:
        # A provider-prefixed batch model keeps the litellm adapter, so the
        # gemini_batch strategy must fail with machine-decidable missing
        # capabilities.
        plan = mp.resolve_routing_plan(
            {"batch": {"model": "deepseek/deepseek-chat"}},
            execution=mp.ExecutionStrategy.GEMINI_BATCH,
        )
        issues = mp.validate_routing_plan(plan, environ={})
        batch_issues = [
            issue for issue in issues
            if issue.stage in {"translation", "keyword", "revision"}
        ]
        self.assertTrue(batch_issues)
        for issue in batch_issues:
            self.assertEqual(issue.code, mp.MODEL_ROUTE_CAPABILITY_MISSING)
            self.assertIn("remote_batch", issue.missing_capabilities)
            self.assertIn("gemini_adapter", issue.missing_capabilities)

    def test_litellm_bare_model_invalid(self) -> None:
        plan = mp.resolve_routing_plan({
            "sync": {"backend": "litellm", "model": "no-prefix-model"},
        })
        issues = mp.validate_routing_plan(plan, environ={})
        self.assertTrue(any(
            issue.code == mp.MODEL_PROFILE_INVALID and "provider prefix" in issue.message
            for issue in issues
        ))

    def test_gemini_adapter_rejects_prefixed_model(self) -> None:
        plan = mp.resolve_routing_plan({
            "sync": {"backend": "gemini", "model": "deepseek/deepseek-chat"},
        })
        issues = mp.validate_routing_plan(plan, environ={})
        self.assertTrue(any(
            issue.code == mp.MODEL_PROFILE_INVALID
            and issue.profile_id == "primary"
            for issue in issues
        ))

    def test_env_credential_reference_missing(self) -> None:
        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        caps = mp.resolve_capabilities(registry["primary"])
        env_ref_profile = replace(
            registry["primary"],
            credential_ref=mp.CredentialRef(
                mp.CREDENTIAL_KIND_ENV, "OPENCODE_GO_API_KEY",
            ),
        )
        plan = mp.ModelRoutingPlan(
            schema_version=mp.MODEL_PROFILE_SCHEMA_VERSION,
            primary_profile_id="primary",
            profiles={"primary": env_ref_profile},
            routes={
                "translation": mp.TaskRoute(
                    "translation", "primary", mp.ExecutionStrategy.SYNC,
                    mp.ROUTE_SOURCE_STAGE_CONFIG,
                ),
            },
            capabilities={"primary": caps},
        )
        issues = mp.validate_routing_plan(plan, environ={})
        self.assertTrue(any(
            issue.code == mp.MODEL_PROFILE_CREDENTIAL_REF_MISSING
            and "OPENCODE_GO_API_KEY" in issue.message
            for issue in issues
        ))
        # The same plan validates when the environment provides the value.
        self.assertEqual(
            mp.validate_routing_plan(
                plan, environ={"OPENCODE_GO_API_KEY": "x"},
            ),
            (),
        )

    def test_keyring_credential_only_flagged_with_probe(self) -> None:
        plan = mp.resolve_routing_plan(
            {
                "sync": {
                    "backend": "litellm",
                    "model": "opencode-go/glm-5.3",
                },
            },
            custom_providers=_custom_providers(),
        )
        # Without a keyring probe the validator stays silent (it cannot know).
        self.assertEqual(
            mp.validate_routing_plan(
                plan, environ={}, keyring_has_credential=None,
            ),
            (),
        )
        issues = mp.validate_routing_plan(
            plan,
            environ={},
            keyring_has_credential=lambda name: False,
        )
        self.assertTrue(any(
            issue.code == mp.MODEL_PROFILE_CREDENTIAL_REF_MISSING
            and "opencode-go" in issue.message
            for issue in issues
        ))

    def test_unknown_capability_override_flagged(self) -> None:
        base = mp.build_profile_registry({})["primary"]
        profile = replace(base, capability_overrides={"telepathy": True})
        plan = mp.ModelRoutingPlan(
            schema_version=mp.MODEL_PROFILE_SCHEMA_VERSION,
            primary_profile_id="primary",
            profiles={"primary": profile},
            routes={
                "translation": mp.TaskRoute(
                    "translation", "primary", mp.ExecutionStrategy.SYNC,
                    mp.ROUTE_SOURCE_STAGE_CONFIG,
                ),
            },
            capabilities={"primary": mp.resolve_capabilities(profile)},
        )
        issues = mp.validate_routing_plan(plan, environ={})
        self.assertTrue(any(
            issue.code == mp.MODEL_PROFILE_INVALID and "telepathy" in issue.message
            for issue in issues
        ))

    def test_routing_validation_error_contract(self) -> None:
        plan = mp.resolve_routing_plan(
            {"batch": {"model": "deepseek/deepseek-chat"}},
            execution=mp.ExecutionStrategy.GEMINI_BATCH,
        )
        issues = mp.validate_routing_plan(plan, environ={})
        error = mp.routing_validation_error(issues)
        # MachineContractError extends SystemExit so CLI guards can raise it
        # straight through main().
        self.assertIsInstance(error, SystemExit)
        self.assertEqual(error.code_name, mp.MODEL_ROUTE_CAPABILITY_MISSING)
        self.assertFalse(error.retryable)
        # The next action is mapped per code, not hard-coded: a capability
        # mismatch asks for a different fix than an invalid profile.
        self.assertEqual(
            error.suggested_action, "choose_supported_strategy_or_profile",
        )
        self.assertEqual(error.details["missing_capabilities"], [
            "gemini_adapter", "remote_batch",
        ])

    def test_suggested_action_per_code(self) -> None:
        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        caps = mp.resolve_capabilities(
            registry["primary"], custom_providers=_custom_providers(),
        )
        routes = {
            "translation": mp.TaskRoute(
                "translation", "primary", mp.ExecutionStrategy.SYNC,
                mp.ROUTE_SOURCE_STAGE_CONFIG,
            ),
        }
        for code, expected_action in (
            (mp.MODEL_PROFILE_INVALID, "fix_translator_config"),
            (mp.MODEL_PROFILE_CREDENTIAL_REF_MISSING,
             "inspect_configuration_and_artifacts"),
        ):
            if code == mp.MODEL_PROFILE_INVALID:
                profile = replace(
                    registry["primary"], capability_overrides={"nope": True},
                )
            else:
                profile = replace(
                    registry["primary"],
                    credential_ref=mp.CredentialRef(
                        mp.CREDENTIAL_KIND_ENV, "DEFINITELY_UNSET_ENV",
                    ),
                )
            plan = mp.ModelRoutingPlan(
                schema_version=mp.MODEL_PROFILE_SCHEMA_VERSION,
                primary_profile_id="primary",
                profiles={"primary": profile},
                routes=routes,
                capabilities={"primary": caps},
            )
            issues = tuple(
                issue for issue in mp.validate_routing_plan(plan, environ={})
                if issue.code == code
            )
            self.assertTrue(issues, code)
            error = mp.routing_validation_error(issues)
            self.assertEqual(error.code_name, code)
            self.assertEqual(error.suggested_action, expected_action)


class SlotIdNamespaceTests(unittest.TestCase):
    """Resolver slot ids are reserved; #348 user ids must not collide."""

    def test_reserved_slot_detection(self) -> None:
        for slot in (
            "primary", "batch",
            "project_analysis_model", "final_review_model",
            "project_analysis_override", "ab_experiment_override",
        ):
            self.assertTrue(mp.is_profile_slot_id(slot), slot)
        for user_id in ("glm-main", "kimi-k3", "gemini33", "ds_v4", "x-y_z"):
            self.assertFalse(mp.is_profile_slot_id(user_id), user_id)

    def test_user_profile_id_rules(self) -> None:
        self.assertEqual(mp.user_profile_id_error("glm-main"), "")
        # Wrong shape.
        self.assertIn("must match", mp.user_profile_id_error("GLM Main"))
        self.assertIn("must match", mp.user_profile_id_error(""))
        # Reserved slots are off-limits for user ids.
        self.assertIn("reserved", mp.user_profile_id_error("primary"))
        self.assertIn(
            "reserved", mp.user_profile_id_error("translation_model"),
        )


class SerializationTests(unittest.TestCase):
    def test_manifest_roundtrip(self) -> None:
        plan = mp.resolve_routing_plan(
            {
                "sync": {
                    "backend": "litellm",
                    "model": "opencode-go/glm-5.3",
                    "models": ["opencode-go/glm-5.3", "opencode-go/glm-5.3-air"],
                },
                "batch": {
                    "model": "gemini-2.5-flash",
                    "project_analysis": {"model": "gemini-3.5-flash"},
                },
            },
            custom_providers=_custom_providers(),
            stage_overrides={"ab_experiment": "gemini-2.5-pro"},
            created_at="2026-08-16T00:00:00Z",
        )
        plan = replace(plan, config_origins=(
            mp.ConfigOrigin(
                kind="translator_config",
                path="work/translator_config.json",
                fingerprint="sha256:abc123",
            ),
        ))
        restored = mp.ModelRoutingPlan.from_manifest_dict(
            plan.to_manifest_dict(),
        )
        self.assertEqual(restored, plan)

    def test_manifest_has_no_credential_values(self) -> None:
        plan = mp.resolve_routing_plan(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        payload = plan.to_manifest_dict()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertNotIn(
                        key.lower(),
                        {"api_key", "apikey", "value", "secret", "token"},
                    )
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        text = json.dumps(payload)
        # The env var NAME is a reference and may appear; a planted fake key
        # value must never appear.
        self.assertNotIn("sk-test-secret", text)

    def test_manifest_credential_ref_shape(self) -> None:
        plan = mp.resolve_routing_plan(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        ref = plan.to_manifest_dict()["profiles"]["primary"]["credential_ref"]
        self.assertEqual(set(ref), {"kind", "name", "env_name"})
        self.assertEqual(ref["kind"], "keyring")
        self.assertEqual(ref["name"], "opencode-go")
        self.assertEqual(ref["env_name"], "OPENCODE_GO_API_KEY")


class BuildSyncBackendTests(unittest.TestCase):
    def test_litellm_backend_gets_custom_providers(self) -> None:
        from litellm_sync_backend import LiteLLMSyncBackend

        registry = mp.build_profile_registry(
            {"sync": {"backend": "litellm", "model": "opencode-go/glm-5.3"}},
            custom_providers=_custom_providers(),
        )
        backend = mp.build_sync_backend(
            registry["primary"],
            custom_providers=_custom_providers(),
        )
        self.assertIsInstance(backend, LiteLLMSyncBackend)
        self.assertEqual(backend._custom_providers.keys(), {"opencode-go"})

    def test_gemini_backend_wiring(self) -> None:
        import translator_runtime as runtime

        registry = mp.build_profile_registry({})
        with mock.patch.object(
            runtime, "configure_genai", lambda: None,
        ), mock.patch.object(
            runtime, "create_genai_client", lambda: object(),
        ):
            backend = mp.build_sync_backend(registry["primary"])
        self.assertIsInstance(backend, GeminiSyncBackend)
        self.assertEqual(backend.provider, "gemini")

    def test_unknown_adapter_refused(self) -> None:
        profile = mp.ModelProfile(
            id="x", label="x", adapter="carrier-pigeon", provider="x",
            model="x", credential_ref=mp.CredentialRef("none"),
        )
        with self.assertRaises(ValueError):
            mp.build_sync_backend(profile)


def _translation_chunk(package_dir: Path) -> dict:
    return {
        "key": "chunk-1",
        "file_path": str(package_dir / "script.rpy"),
        "file_rel_path": "script.rpy",
        "chunk_index": 1,
        "context_past": [],
        "context_future": [],
        "items": [
            {
                "id": "a",
                "text": "Hello",
                "file_rel_path": "script.rpy",
                "line": 0,
                "line_number": 1,
                "start": 4,
                "end": 11,
                "prefix": "",
                "quote": '"',
            },
            {
                "id": "b",
                "text": "World",
                "file_rel_path": "script.rpy",
                "line": 1,
                "line_number": 2,
                "start": 4,
                "end": 11,
                "prefix": "",
                "quote": '"',
            },
        ],
    }


def _walk_forbid_credential_slots(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            unittest.TestCase().assertNotIn(
                str(key).lower(),
                {"api_key", "apikey", "value", "secret", "token"},
            )
            _walk_forbid_credential_slots(value)
    elif isinstance(payload, list):
        for item in payload:
            _walk_forbid_credential_slots(item)


class SyncEntryWiringTests(unittest.TestCase):
    def _sync_response(self, payload, *, model="stage-explicit-model"):
        text = json.dumps(payload, ensure_ascii=False)
        return {
            "response_payload": {
                "candidates": [{"content": {"parts": [{"text": text}]}}],
            },
            "response_text": text,
            "finish_reason": "STOP",
            "usage_metadata": {"totalTokenCount": 1},
            "provider": "gemini",
            "model": model,
            "execution_mode": "sync",
        }

    def test_execute_sync_rows_use_explicit_route_on_first_pass_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "sync-run"
            package_dir.mkdir()
            chunk = _translation_chunk(package_dir)
            plan = mp.resolve_routing_plan(
                {"sync": {"backend": "gemini", "model": "primary-should-lose"}},
                stage_overrides={"translation": "stage-explicit-model"},
            )
            request_rows = [
                batch_mod.build_batch_request(chunk, model="stage-explicit-model"),
            ]
            manifest_path = batch_mod.make_sync_manifest(
                package_dir=str(package_dir),
                mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                display_name="explicit-route-test",
                chunks=[chunk],
                request_rows=request_rows,
                settings={},
                routing_plan=plan,
            )
            payloads = [
                {"translations": [{"id": "a", "translation": "你好"}]},
                {"translations": [{"id": "b", "translation": "世界"}]},
            ]
            seen_routes: list[mp.TaskRoute] = []
            seen_models: list[str] = []

            def fake_sync(request, route, plan=None, **_kwargs):
                seen_routes.append(route)
                seen_models.append(plan.profiles[route.profile_id].model)
                return self._sync_response(payloads[len(seen_routes) - 1])

            with (
                mock.patch.object(batch_mod, "SYNC_MODEL", "sync-overlay"),
                mock.patch.object(
                    batch_mod, "run_sync_request", side_effect=fake_sync,
                ),
            ):
                batch_mod.execute_sync_request_rows(manifest_path, request_rows)

        self.assertEqual(len(seen_routes), 2)
        self.assertTrue(all(isinstance(route, mp.TaskRoute) for route in seen_routes))
        self.assertEqual(seen_models, ["stage-explicit-model", "stage-explicit-model"])
        self.assertEqual({route.stage for route in seen_routes}, {"translation"})

    def test_keyword_revision_analysis_and_ab_callers_pass_task_route(self) -> None:
        captured: dict[str, list] = {
            "keyword": [],
            "revision": [],
            "project_analysis": [],
            "ab_experiment": [],
        }

        def record(stage):
            def _fake(request, route, plan=None, **_kwargs):
                captured[stage].append(route)
                if stage == "keyword":
                    payload = {
                        "candidates": [{
                            "source": "Void Gate",
                            "suggested_target": "虚空门",
                            "category": "term",
                            "confidence": 0.9,
                            "evidence": "script.rpy:2:keyword:0",
                            "source_item_ids": ["script.rpy:2:keyword:0"],
                        }],
                        "chunk_summary": "ok",
                        "summary_evidence_item_ids": ["script.rpy:2:keyword:0"],
                    }
                else:
                    payload = {"translations": [{"id": "a", "translation": "你好"}]}
                return self._sync_response(payload)
            return _fake

        old = {
            "tl_dir": batch_mod.legacy.TL_DIR,
            "log_dir": batch_mod.LOG_DIR,
            "jobs_dir": batch_mod.BATCH_JOBS_DIR,
            "repair_dir": batch_mod.REPAIR_RUNS_DIR,
            "sync_dir": batch_mod.SYNC_RUNS_DIR,
            "latest": batch_mod.LATEST_MANIFEST_FILE,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / "tl"
                jobs_dir = root / "batch_jobs"
                tl_dir.mkdir()
                jobs_dir.mkdir()
                (tl_dir / "script.rpy").write_text(
                    "translate schinese start:\n"
                    '    old "Void Gate"\n'
                    '    new "虚空门"\n',
                    encoding="utf-8",
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.LOG_DIR = str(root / "logs")
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / "repair")
                batch_mod.SYNC_RUNS_DIR = str(root / "sync")
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / "latest.txt")

                with mock.patch.object(
                    batch_mod, "run_sync_request", side_effect=record("keyword"),
                ):
                    batch_mod.sync_keyword_candidates(
                        skip_prepare=True, chunk_size=1, limit=1,
                    )
                with mock.patch.object(
                    batch_mod, "run_sync_request", side_effect=record("revision"),
                ), mock.patch.object(batch_mod, "preview_revisions", return_value={
                    "_manifest_path": "unused",
                }):
                    batch_mod.sync_revisions(skip_prepare=True, chunk_size=1, limit=1)
        finally:
            batch_mod.legacy.TL_DIR = old["tl_dir"]
            batch_mod.LOG_DIR = old["log_dir"]
            batch_mod.BATCH_JOBS_DIR = old["jobs_dir"]
            batch_mod.REPAIR_RUNS_DIR = old["repair_dir"]
            batch_mod.SYNC_RUNS_DIR = old["sync_dir"]
            batch_mod.LATEST_MANIFEST_FILE = old["latest"]

        analysis_plan = mp.resolve_routing_plan(
            {"sync": {"backend": "gemini", "model": "primary-should-lose"}},
            stage_overrides={"project_analysis": "analysis-explicit"},
        )
        analysis_route = analysis_plan.routes["project_analysis"]
        runner = batch_mod.build_project_analysis_sync_runner(
            analysis_plan, analysis_route,
        )
        with mock.patch.object(
            batch_mod, "run_sync_request", side_effect=record("project_analysis"),
        ):
            runner(SyncGenerationRequest(model="ignored", contents=[], config={}))

        fixture = (
            Path(__file__).parent / "fixtures" / "golden_batch_minimal"
            / "expected" / "manifest_snapshot.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest = json.loads(fixture.read_text(encoding="utf-8"))
            manifest["mode"] = batch_mod.MANIFEST_MODE_TRANSLATION
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8",
            )
            loaded = batch_mod.load_manifest(str(manifest_path))
            variants = ab_mod.load_variants_file(
                str(Path(__file__).parent / "fixtures" / "ab_variants_minimal.json"),
            )
            with mock.patch.object(
                batch_mod, "run_sync_request", side_effect=record("ab_experiment"),
            ), mock.patch.object(
                ab_mod, "enrich_chunk_for_current_settings", side_effect=lambda chunk, **_: chunk,
            ):
                ab_mod.run_translation_ab_experiment(
                    loaded,
                    variants,
                    limit=1,
                    offset=0,
                    output_dir=str(Path(tmp) / "ab"),
                    model_override="ab-explicit-model",
                    dry_run=False,
                )

        self.assertTrue(captured["keyword"])
        self.assertTrue(all(
            isinstance(route, mp.TaskRoute) and route.stage == "keyword"
            for route in captured["keyword"]
        ))
        self.assertTrue(captured["revision"])
        self.assertTrue(all(
            isinstance(route, mp.TaskRoute) and route.stage == "revision"
            for route in captured["revision"]
        ))
        self.assertTrue(captured["project_analysis"])
        self.assertEqual(captured["project_analysis"][0].stage, "project_analysis")
        self.assertTrue(captured["ab_experiment"])
        self.assertTrue(all(
            isinstance(route, mp.TaskRoute) and route.stage == "ab_experiment"
            for route in captured["ab_experiment"]
        ))

    def test_production_sync_backends_come_from_build_sync_backend(self) -> None:
        route, plan = (
            mp.resolve_routing_plan(
                {"sync": {"backend": "litellm", "model": "openai/primary-lose"}},
                stage_overrides={"translation": "openai/explicit-stage"},
            ).routes["translation"],
            mp.resolve_routing_plan(
                {"sync": {"backend": "litellm", "model": "openai/primary-lose"}},
                stage_overrides={"translation": "openai/explicit-stage"},
            ),
        )
        fake_backend = mock.Mock()
        fake_backend.generate.return_value = type("Result", (), {
            "response_payload": {"choices": []},
            "response_text": "[]",
            "finish_reason": "stop",
            "usage_metadata": {},
            "provider": "litellm",
            "model": "openai/explicit-stage",
            "execution_mode": "sync",
        })()
        with (
            mock.patch.object(batch_mod, "SYNC_MODEL", "openai/sync-overlay"),
            mock.patch.object(
                mp, "build_sync_backend", return_value=fake_backend,
            ) as builder,
        ):
            batch_mod.run_sync_request({"contents": []}, route, plan=plan)
        builder.assert_called_once()
        self.assertEqual(builder.call_args.args[0].model, "openai/explicit-stage")
        self.assertEqual(
            fake_backend.generate.call_args.args[0].model,
            "openai/explicit-stage",
        )

        repo = Path(__file__).resolve().parents[1]
        hits: list[str] = []
        for path in repo.rglob("*.py"):
            rel = path.relative_to(repo).as_posix()
            if rel.startswith(("gui_qt/", "tests/", "scripts/")):
                continue
            text = path.read_text(encoding="utf-8")
            for index, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if "LiteLLMSyncBackend(" not in stripped and "GeminiSyncBackend(" not in stripped:
                    continue
                if rel == "model_profile.py" and "return " in stripped:
                    continue
                hits.append(f"{rel}:{index}:{stripped}")
        self.assertEqual(hits, [])
        overlay = (repo / "gemini_translate_batch.py").read_text(encoding="utf-8")
        self.assertNotIn("effective_model = SYNC_MODEL or model_name", overlay)

    def test_mid_run_config_mutation_and_retry_keep_frozen_profile(self) -> None:
        previous_batch_model = batch_mod.BATCH_MODEL
        previous_sync_model = batch_mod.SYNC_MODEL
        try:
            with tempfile.TemporaryDirectory() as tmp:
                package_dir = Path(tmp) / "sync-run"
                package_dir.mkdir()
                chunk = _translation_chunk(package_dir)
                plan = mp.resolve_routing_plan(
                    {"sync": {"backend": "gemini", "model": "primary-should-lose"}},
                    stage_overrides={"translation": "frozen-stage-model"},
                    created_at="2026-08-18T00:00:00Z",
                )
                request_rows = [
                    batch_mod.build_batch_request(chunk, model="frozen-stage-model"),
                ]
                manifest_path = batch_mod.make_sync_manifest(
                    package_dir=str(package_dir),
                    mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                    display_name="freeze-retry-test",
                    chunks=[chunk],
                    request_rows=request_rows,
                    settings={},
                    routing_plan=plan,
                )
                seen: list[tuple[str, str, str]] = []
                payloads = [
                    {"translations": [{"id": "a", "translation": "你好"}]},
                    {"translations": [{"id": "b", "translation": "世界"}]},
                ]

                def fake_sync(request, route, plan=None, **_kwargs):
                    batch_mod.SYNC_MODEL = "mutated-sync-model"
                    batch_mod.BATCH_MODEL = "mutated-batch-model"
                    seen.append((
                        route.profile_id,
                        plan.profiles[route.profile_id].model,
                        plan.created_at,
                    ))
                    return self._sync_response(
                        payloads[len(seen) - 1],
                        model=plan.profiles[route.profile_id].model,
                    )

                with (
                    mock.patch.object(batch_mod, "SYNC_MODEL", "live-sync-model"),
                    mock.patch.object(
                        batch_mod, "run_sync_request", side_effect=fake_sync,
                    ),
                ):
                    batch_mod.execute_sync_request_rows(manifest_path, request_rows)

            self.assertEqual(len(seen), 2)
            self.assertEqual({item[1] for item in seen}, {"frozen-stage-model"})
            self.assertEqual({item[2] for item in seen}, {"2026-08-18T00:00:00Z"})
            self.assertEqual(seen[0], seen[1])

            retry_plan = mp.resolve_routing_plan(
                {"sync": {"backend": "litellm", "model": "openai/primary-lose"}},
                stage_overrides={"translation": "openai/frozen-retry"},
            )
            route = retry_plan.routes["translation"]
            models_seen: list[str] = []
            fake_backend = mock.Mock()

            def generate(request):
                batch_mod.SYNC_MODEL = "openai/mutated-after-start"
                models_seen.append(request.model)
                if len(models_seen) == 1:
                    raise LiteLLMBackendError("timeout one", category="timeout")
                return type("Result", (), {
                    "response_payload": {},
                    "response_text": "[]",
                    "finish_reason": "stop",
                    "usage_metadata": {},
                    "provider": "litellm",
                    "model": request.model,
                    "execution_mode": "sync",
                })()

            fake_backend.generate.side_effect = generate
            with (
                mock.patch.object(batch_mod, "SYNC_MODEL", "openai/live-overlay"),
                mock.patch.object(batch_mod.time, "sleep"),
                mock.patch.object(mp, "build_sync_backend", return_value=fake_backend),
            ):
                batch_mod.run_sync_request({"contents": []}, route, plan=retry_plan)
            self.assertEqual(models_seen, ["openai/frozen-retry", "openai/frozen-retry"])
        finally:
            batch_mod.BATCH_MODEL = previous_batch_model
            batch_mod.SYNC_MODEL = previous_sync_model

    def test_shipped_manifests_write_model_routing_without_credentials(self) -> None:
        planted = "sk-test-secret"
        old = {
            "tl_dir": batch_mod.legacy.TL_DIR,
            "log_dir": batch_mod.LOG_DIR,
            "jobs_dir": batch_mod.BATCH_JOBS_DIR,
            "repair_dir": batch_mod.REPAIR_RUNS_DIR,
            "sync_dir": batch_mod.SYNC_RUNS_DIR,
            "latest": batch_mod.LATEST_MANIFEST_FILE,
            "final_enabled": batch_mod.FINAL_REVIEW_ENABLED,
            "require_zero": batch_mod.FINAL_REVIEW_REQUIRE_ZERO_PENDING,
            "include_files": set(batch_mod.legacy.INCLUDE_FILES),
            "include_prefixes": set(batch_mod.legacy.INCLUDE_PREFIXES),
            "base_dir": batch_mod.legacy.BASE_DIR,
        }
        written: list[dict] = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / "tl"
                jobs_dir = root / "batch_jobs"
                tl_dir.mkdir()
                jobs_dir.mkdir()
                (tl_dir / "script.rpy").write_text(
                    "translate schinese start:\n"
                    '    old "Hello"\n'
                    '    new "你好"\n',
                    encoding="utf-8",
                )
                pending_tl = root / "game" / "tl" / "schinese"
                pending_tl.parent.mkdir(parents=True)
                shutil.copytree(
                    Path(__file__).parent / "fixtures" / "golden_batch_minimal" / "tl",
                    pending_tl,
                )
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.legacy.INCLUDE_FILES = set()
                batch_mod.legacy.INCLUDE_PREFIXES = set()
                batch_mod.LOG_DIR = str(root / "logs")
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / "repair")
                batch_mod.SYNC_RUNS_DIR = str(root / "sync")
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / "latest.txt")
                batch_mod.FINAL_REVIEW_ENABLED = True
                batch_mod.FINAL_REVIEW_REQUIRE_ZERO_PENDING = False

                sync_pkg = root / "sync-pkg"
                sync_pkg.mkdir()
                sync_path = batch_mod.make_sync_manifest(
                    package_dir=str(sync_pkg),
                    mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                    display_name="routing-scan",
                    chunks=[],
                    request_rows=[],
                    settings={},
                )
                written.append(json.loads(Path(sync_path).read_text(encoding="utf-8")))

                keyword_path = batch_mod.create_keyword_package(
                    skip_prepare=True, chunk_size=1,
                )
                written.append(json.loads(Path(keyword_path).read_text(encoding="utf-8")))

                revision_path = batch_mod.create_revision_package(
                    skip_prepare=True, chunk_size=1,
                )
                written.append(json.loads(Path(revision_path).read_text(encoding="utf-8")))

                batch_mod.legacy.TL_DIR = str(pending_tl)
                translation_path = batch_mod.create_batch_package(skip_prepare=True)
                self.assertIsNotNone(translation_path)
                written.append(
                    json.loads(Path(translation_path).read_text(encoding="utf-8"))
                )

                batch_mod.legacy.TL_DIR = str(tl_dir)
                review_path = batch_mod.create_final_review_package(
                    skip_prepare=True,
                    chunk_size=1,
                    allow_pending=True,
                )
                written.append(json.loads(Path(review_path).read_text(encoding="utf-8")))
        finally:
            batch_mod.legacy.TL_DIR = old["tl_dir"]
            batch_mod.LOG_DIR = old["log_dir"]
            batch_mod.BATCH_JOBS_DIR = old["jobs_dir"]
            batch_mod.REPAIR_RUNS_DIR = old["repair_dir"]
            batch_mod.SYNC_RUNS_DIR = old["sync_dir"]
            batch_mod.LATEST_MANIFEST_FILE = old["latest"]
            batch_mod.FINAL_REVIEW_ENABLED = old["final_enabled"]
            batch_mod.FINAL_REVIEW_REQUIRE_ZERO_PENDING = old["require_zero"]
            batch_mod.legacy.INCLUDE_FILES = old["include_files"]
            batch_mod.legacy.INCLUDE_PREFIXES = old["include_prefixes"]
            batch_mod.legacy.BASE_DIR = old["base_dir"]

        self.assertEqual(len(written), 5)
        for manifest in written:
            self.assertIn("model_routing", manifest)
            snapshot = manifest["model_routing"]
            _walk_forbid_credential_slots(snapshot)
            text = json.dumps(snapshot)
            self.assertNotIn(planted, text)
            self.assertIn("profiles", snapshot)
            self.assertIn("routes", snapshot)

    def test_old_manifest_without_model_routing_keeps_recorded_model(self) -> None:
        recorded = "gemini-old-recorded"
        live = "gemini-live-new"
        previous_batch = batch_mod.BATCH_MODEL
        previous_sync = batch_mod.SYNC_MODEL
        try:
            with tempfile.TemporaryDirectory() as tmp:
                package_dir = Path(tmp) / "old-run"
                package_dir.mkdir()
                chunk = _translation_chunk(package_dir)
                request_rows = [
                    batch_mod.build_batch_request(chunk, model=recorded),
                ]
                plan = mp.resolve_routing_plan(
                    {"sync": {"backend": "gemini", "model": recorded}},
                )
                manifest_path = batch_mod.make_sync_manifest(
                    package_dir=str(package_dir),
                    mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                    display_name="old-manifest-test",
                    chunks=[chunk],
                    request_rows=request_rows,
                    settings={},
                    routing_plan=plan,
                )
                raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                raw.pop("model_routing", None)
                raw["model"] = recorded
                raw["batch_model"] = recorded
                raw["provider"] = "gemini"
                Path(manifest_path).write_text(
                    json.dumps(raw, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                batch_mod.SYNC_MODEL = live
                batch_mod.BATCH_MODEL = live

                loaded = batch_mod.load_manifest(manifest_path)
                self.assertNotIn("model_routing", loaded)
                resolved = batch_mod.resolve_manifest_routing_plan(loaded)
                self.assertEqual(
                    batch_mod.route_model(
                        resolved,
                        batch_mod.route_for_manifest(resolved, loaded),
                    ),
                    recorded,
                )

                seen: list[str] = []

                def fake_sync(request, route, plan=None, **_kwargs):
                    seen.append(plan.profiles[route.profile_id].model)
                    return self._sync_response(
                        {
                            "translations": [
                                {"id": "a", "translation": "你好"},
                                {"id": "b", "translation": "世界"},
                            ]
                        },
                        model=plan.profiles[route.profile_id].model,
                    )

                with mock.patch.object(
                    batch_mod, "run_sync_request", side_effect=fake_sync,
                ):
                    batch_mod.probe_requests(manifest_path, limit=1)
                    batch_mod.execute_sync_request_rows(
                        manifest_path,
                        request_rows,
                    )

                self.assertGreaterEqual(len(seen), 2)
                self.assertEqual(set(seen), {recorded})
                self.assertNotIn(live, seen)
        finally:
            batch_mod.BATCH_MODEL = previous_batch
            batch_mod.SYNC_MODEL = previous_sync


if __name__ == "__main__":
    unittest.main()
