"""Config/routing tests for the direct OpenAI-compatible adapter (#431 S1)."""

from __future__ import annotations

import unittest

import model_profile as routing
import model_profiles_editor as editor
import model_routing_config as config
import model_routing_reader as reader
import openai_compatible_provider_config as presets
from openai_compatible_sync_backend import OpenAICompatibleSyncBackend


def direct_section(
    *,
    base_url: str = "https://api.openai.com/v1",
    credential_kind: str = "keyring",
    extra_headers: dict | None = None,
    params: dict | None = None,
    structured_mode: str = "json_object",
):
    section = editor.empty_section()
    section = editor.add_provider(
        section,
        label="Gemini",
        adapter=routing.ADAPTER_GEMINI,
        provider="gemini",
        credential_kind="api_keys_json",
        credential_name="api_keys",
        credential_env_name="GEMINI_API_KEY",
    )
    gemini_provider_id = editor.provider_ids(section)[0]
    section = editor.add_profile(
        section,
        label="Gemini Batch",
        provider_id=gemini_provider_id,
        model="gemini-3.5-flash",
    )
    gemini_profile_id = editor.profile_ids(section)[0]

    section = editor.add_provider(
        section,
        label="OpenAI",
        adapter=routing.ADAPTER_OPENAI_COMPATIBLE,
        provider="openai",
        base_url=base_url,
        models_url="https://api.openai.com/v1/models",
        credential_kind=credential_kind,
        credential_name="openai",
        credential_env_name="OPENAI_API_KEY",
        extra_headers=extra_headers or {"X-Client-Version": "1.0"},
    )
    direct_provider_id = editor.provider_ids(section)[1]
    section = editor.add_profile(
        section,
        label="OpenAI GPT",
        provider_id=direct_provider_id,
        model="gpt-4.1-mini",
        params=params if params is not None else {"temperature": 0.2},
        capability_overrides={"structured_output": {"mode": structured_mode}},
    )
    direct_profile_id = editor.profile_ids(section)[1]
    section = editor.set_defaults(
        section,
        primary_profile_id=direct_profile_id,
        execution_strategy=routing.ExecutionStrategy.SYNC.value,
    )
    section = editor.set_route(
        section,
        "final_review",
        enabled=True,
        profile_id=gemini_profile_id,
        strategy=routing.ExecutionStrategy.GEMINI_BATCH.value,
    )
    section["legacy_entrypoints"] = {
        "sync_profile_id": direct_profile_id,
        "batch_profile_id": gemini_profile_id,
    }
    return section, direct_provider_id, direct_profile_id


def issue_codes(section: dict) -> set[str]:
    return {issue.code for issue in config.validate_model_routing_section(section)}


class EditorAndReaderTests(unittest.TestCase):
    def test_editor_round_trips_extra_headers(self) -> None:
        section, provider_id, _profile_id = direct_section()
        provider = next(
            item
            for item in editor.editor_view(section)["providers"]
            if item["id"] == provider_id
        )
        self.assertEqual(provider["adapter"], routing.ADAPTER_OPENAI_COMPATIBLE)
        self.assertEqual(provider["extra_headers"], {"X-Client-Version": "1.0"})

        updated = editor.update_provider(
            section,
            provider_id,
            extra_headers={"X-Trace": "abc"},
        )
        updated_provider = next(
            item
            for item in editor.editor_view(updated)["providers"]
            if item["id"] == provider_id
        )
        self.assertEqual(updated_provider["extra_headers"], {"X-Trace": "abc"})
        self.assertEqual(config.validate_model_routing_section(updated), ())

    def test_reader_builds_direct_profile_with_headers_and_params(self) -> None:
        section, _provider_id, profile_id = direct_section(
            params={"temperature": 0.4, "max_output_tokens": 512}
        )
        plan = reader.read_routing_plan({"model_routing": section})
        profile = plan.profiles[profile_id]
        self.assertEqual(profile.adapter, routing.ADAPTER_OPENAI_COMPATIBLE)
        self.assertEqual(profile.base_url, "https://api.openai.com/v1")
        self.assertEqual(profile.extra_headers, {"X-Client-Version": "1.0"})
        self.assertEqual(profile.params["max_output_tokens"], 512)
        self.assertEqual(plan.capabilities[profile_id].structured_output.mode, "json_object")
        self.assertTrue(plan.capabilities[profile_id].sync_generation.supported)

    def test_runtime_settings_view_projects_direct_backend(self) -> None:
        section, _provider_id, _profile_id = direct_section()
        view = reader.runtime_settings_view({"model_routing": section})
        self.assertEqual(view["sync"]["backend"], routing.ADAPTER_OPENAI_COMPATIBLE)

    def test_resolve_runtime_plan_accepts_direct_generation_params(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            params={"temperature": 0.4, "max_output_tokens": 512}
        )
        plan = reader.resolve_runtime_plan(
            {"model_routing": section},
            execution=routing.ExecutionStrategy.SYNC.value,
        )
        self.assertEqual(
            plan.routes[routing.STAGE_TRANSLATION].profile_id,
            next(
                profile_id
                for profile_id, profile in plan.profiles.items()
                if profile.adapter == routing.ADAPTER_OPENAI_COMPATIBLE
            ),
        )

    def test_build_sync_backend_returns_direct_backend(self) -> None:
        section, _provider_id, profile_id = direct_section()
        plan = reader.read_routing_plan({"model_routing": section})
        backend = routing.build_sync_backend(plan.profiles[profile_id])
        self.assertIsInstance(backend, OpenAICompatibleSyncBackend)
        self.assertEqual(backend.provider, "openai")

    def test_direct_capabilities_default_to_prompt_only_json(self) -> None:
        profile = routing.ModelProfile(
            id="direct",
            label="Direct",
            adapter=routing.ADAPTER_OPENAI_COMPATIBLE,
            provider="openai",
            model="gpt-4.1-mini",
            credential_ref=routing.CredentialRef("none", "openai"),
            base_url="https://api.openai.com/v1",
        )
        caps = routing.resolve_capabilities(profile)
        self.assertEqual(caps.structured_output.mode, "prompt_only_json")
        self.assertEqual(
            caps.structured_output.source,
            routing.CAPABILITY_SOURCE_ADAPTER_DEFAULT,
        )


class ConfigValidationTests(unittest.TestCase):
    def test_valid_direct_section_passes(self) -> None:
        section, _provider_id, _profile_id = direct_section()
        self.assertEqual(config.validate_model_routing_section(section), ())

    def test_missing_base_url_is_rejected(self) -> None:
        section, provider_id, _profile_id = direct_section(base_url="")
        self.assertIn("missing_provider_url", issue_codes(section))
        self.assertIn(provider_id, editor.provider_ids(section))

    def test_api_keys_json_credential_is_rejected(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="OpenAI",
            adapter=routing.ADAPTER_OPENAI_COMPATIBLE,
            provider="openai",
            base_url="https://api.openai.com/v1",
            credential_kind="api_keys_json",
            credential_name="api_keys",
        )
        section = editor.add_profile(
            section,
            label="OpenAI GPT",
            provider_id=editor.provider_ids(section)[0],
            model="gpt-4.1-mini",
        )
        self.assertIn(
            "unsupported_credential_kind_for_adapter",
            issue_codes(section),
        )

    def test_sensitive_extra_header_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            extra_headers={"Authorization": "Bearer nope"}
        )
        codes = issue_codes(section)
        self.assertTrue(
            codes & {"sensitive_extra_header", "credential_value_forbidden"},
            codes,
        )

    def test_non_ascii_extra_header_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            extra_headers={"X-Trace": "中文"}
        )
        self.assertIn("invalid_extra_header", issue_codes(section))

    def test_sensitive_extra_header_value_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            extra_headers={"X-Auth": "Bearer sk-test-secret"}
        )
        self.assertIn("sensitive_extra_header", issue_codes(section))

    def test_api_key_shaped_extra_header_value_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            extra_headers={"X-Auth": "sk-abcdefghijklmnop"}
        )
        self.assertIn("sensitive_extra_header", issue_codes(section))

    def test_benign_extra_header_values_are_allowed(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            extra_headers={
                "X-Run-Mode": "task-1",
                "X-Phase": "risk-free",
            }
        )
        self.assertNotIn("sensitive_extra_header", issue_codes(section))

    def test_structured_output_modes_share_one_contract_source(self) -> None:
        import openai_compatible_contract as contract
        from openai_compatible_sync_backend import STRUCTURED_OUTPUT_MODES

        self.assertEqual(config.STRUCTURED_OUTPUT_MODES, contract.STRUCTURED_OUTPUT_MODES)
        self.assertEqual(STRUCTURED_OUTPUT_MODES, contract.STRUCTURED_OUTPUT_MODES)

    def test_unknown_generation_param_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            params={"temperature": 0.2, "top_k": 5}
        )
        self.assertIn("unsupported_generation_param", issue_codes(section))

    def test_unknown_structured_output_mode_is_rejected(self) -> None:
        section, _provider_id, _profile_id = direct_section(
            structured_mode="freeform"
        )
        self.assertIn("unsupported_structured_output_mode", issue_codes(section))

    def test_extra_headers_are_rejected_for_non_direct_adapter(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Gemini",
            adapter=routing.ADAPTER_GEMINI,
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id=provider_id,
            model="gemini-3.5-flash",
        )
        section = editor.update_provider(
            section,
            provider_id,
            extra_headers={"X-Test": "1"},
        )
        self.assertIn("unsupported_provider_field", issue_codes(section))

    def test_legacy_sync_backend_direct_is_rejected_with_clear_error(self) -> None:
        with self.assertRaises(routing.ModelRoutingConfigError) as captured:
            routing.resolve_routing_plan({
                "sync": {"backend": "openai_compatible", "model": "gpt-4.1-mini"},
            })
        self.assertIn("model_routing", str(captured.exception))


class PresetTests(unittest.TestCase):
    def test_preset_ids_and_payload_shape(self) -> None:
        self.assertIn("openai", presets.preset_ids())
        self.assertIn("custom", presets.preset_ids())
        payload = presets.preset_provider_payload("openai")
        self.assertEqual(payload["adapter"], routing.ADAPTER_OPENAI_COMPATIBLE)
        self.assertEqual(payload["base_url"], "https://api.openai.com/v1")
        self.assertEqual(payload["credential_ref"]["env_name"], "OPENAI_API_KEY")
        self.assertEqual(payload["extra_headers"], {})

    def test_ollama_preset_is_keyless(self) -> None:
        preset = presets.get_preset("ollama")
        self.assertEqual(preset.credential_kind, "none")
        self.assertFalse(preset.requires_key)

    def test_custom_preset_defaults_to_conservative_mode(self) -> None:
        preset = presets.get_preset("custom")
        self.assertEqual(preset.structured_output_mode, "prompt_only_json")


if __name__ == "__main__":
    unittest.main()
