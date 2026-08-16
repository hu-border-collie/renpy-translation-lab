"""Tests for the model routing contract module (issue #345 P1)."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest import mock

import litellm_provider_config as lpc
import model_profile as mp
from sync_model_backend import GeminiSyncBackend


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


if __name__ == "__main__":
    unittest.main()
