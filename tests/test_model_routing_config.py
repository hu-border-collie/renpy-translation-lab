from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import model_profile
import model_routing_config as contract


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(relative_path: str) -> dict[str, object]:
    return json.loads((FIXTURES / relative_path).read_text(encoding="utf-8"))


class ModelRoutingConfigContractTests(unittest.TestCase):
    def test_malformed_json_field_types_return_diagnostics(self) -> None:
        mutations = [
            (lambda s, v: s.update(schema_version=v)),
            (lambda s, v: s["providers"]["openrouter"].update(adapter=v)),
            (lambda s, v: s["providers"]["openrouter"]["credential_ref"].update(kind=v)),
            (lambda s, v: s["defaults"].update(primary_profile_id=v)),
            (lambda s, v: s["defaults"].update(execution_strategy=v)),
            (lambda s, v: s["profiles"]["gemini-main"].update(embedding_profile_id=v)),
            (lambda s, v: s["routes"].update(translation={"profile_id": v})),
        ]
        for mutate in mutations:
            for value in ([], {}, True, 1.0, None):
                section = _load_fixture("model_routing_config_v1.json")["model_routing"]
                mutate(section, value)
                with self.subTest(mutation=mutate, value=value):
                    self.assertTrue(contract.validate_model_routing_section(section))

    def test_malformed_url_and_reserved_ab_slot_return_diagnostics(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        for url in ("http://[broken", "https://example.test:wrong"):
            section["providers"]["openrouter"]["base_url"] = url
            self.assertIn("invalid_provider_url", {
                issue.code for issue in contract.validate_model_routing_section(section)
            })
        section["profiles"]["ab_experiment_override"] = section["profiles"]["gemini-main"]
        self.assertIn("reserved_profile_id", {
            issue.code for issue in contract.validate_model_routing_section(section)
        })

    def test_v1_fixture_is_valid_and_validation_is_read_only(self) -> None:
        config = _load_fixture("model_routing_config_v1.json")
        before = copy.deepcopy(config)

        self.assertEqual(contract.validate_translator_config_model_routing(config), ())
        self.assertEqual(config, before)

    def test_legacy_only_config_remains_accepted_during_transition(self) -> None:
        config = _load_fixture("model_routing_legacy/gemini_sync.json")
        self.assertEqual(contract.validate_translator_config_model_routing(config), ())

    def test_contract_ids_match_existing_resolver_namespace(self) -> None:
        fixture = _load_fixture("model_routing_config_v1.json")
        profiles = fixture["model_routing"]["profiles"]
        for profile_id in profiles:
            self.assertEqual(model_profile.user_profile_id_error(profile_id), "")

        self.assertEqual(
            contract.KNOWN_EXECUTION_STRATEGIES,
            frozenset(strategy.value for strategy in model_profile.ExecutionStrategy),
        )

    def test_reserved_slots_match_legacy_resolver(self) -> None:
        ids = {"primary", "batch", "foo_model", "foo_override", "gemini-main"}
        ids.update(stage + suffix for stage in model_profile.KNOWN_STAGES
                   for suffix in ("_model", "_override"))
        for profile_id in ids:
            with self.subTest(profile_id=profile_id):
                self.assertEqual(contract._reserved_profile_id(profile_id),
                                 model_profile.is_profile_slot_id(profile_id))

    def test_empty_stage_models_require_legacy_specific_routes(self) -> None:
        config = _load_fixture("model_routing_legacy/example_defaults.json")
        for execution in ("sync", "gemini_batch"):
            plan = model_profile.resolve_routing_plan(config, execution=execution)
            self.assertEqual(plan.routes["project_analysis"].profile_id, "primary")
            self.assertEqual(plan.routes["project_analysis"].strategy.value, "sync")
            self.assertEqual(plan.routes["final_review"].profile_id, "batch")
            self.assertEqual(plan.routes["final_review"].strategy.value, "gemini_batch")

    def test_unknown_non_sensitive_fields_are_forward_compatible(self) -> None:
        config = _load_fixture("model_routing_config_v1.json")
        section = config["model_routing"]
        section["future_extension"] = {"new_policy": {"enabled": True}}
        section["profiles"]["gemini-main"]["future_model_option"] = "keep-me"

        self.assertEqual(contract.validate_model_routing_section(section), ())

    def test_credential_values_are_forbidden_even_in_unknown_fields(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["future_extension"] = {"api_key": "must-not-be-here"}

        issues = contract.validate_model_routing_section(section)

        self.assertIn("credential_value_forbidden", {issue.code for issue in issues})
        self.assertIn(
            "model_routing.future_extension.api_key",
            {issue.path for issue in issues},
        )

    def test_unsupported_version_and_missing_keyring_name_fail_closed(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["schema_version"] = 2
        section["providers"]["openrouter"]["credential_ref"]["name"] = ""

        codes = {issue.code for issue in contract.validate_model_routing_section(section)}

        self.assertIn("unsupported_schema_version", codes)
        self.assertIn("missing_credential_reference", codes)

    def test_profile_must_reference_provider_and_embedding_profile(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["profiles"]["gemini-main"]["provider_id"] = "missing-provider"
        section["profiles"]["gemini-main"]["embedding_profile_id"] = "missing-profile"

        codes = {issue.code for issue in contract.validate_model_routing_section(section)}

        self.assertIn("unknown_provider", codes)
        self.assertIn("unknown_embedding_profile", codes)

    def test_gemini_batch_rejects_non_gemini_profile(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["defaults"] = {
            "primary_profile_id": "openrouter-review",
            "execution_strategy": "gemini_batch",
        }

        codes = {issue.code for issue in contract.validate_model_routing_section(section)}

        self.assertIn("strategy_profile_mismatch", codes)

    def test_route_can_override_only_profile_or_only_strategy(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["routes"] = {
            "project_analysis": {"profile_id": "gemini-main"},
            "revision": {"strategy": "sync"},
        }

        self.assertEqual(contract.validate_model_routing_section(section), ())

    def test_invalid_route_and_url_fail_closed(self) -> None:
        section = _load_fixture("model_routing_config_v1.json")["model_routing"]
        section["routes"]["unknown-stage"] = {}
        section["providers"]["acme-compatible"]["base_url"] = (
            "https://user:password@models.example.test/v1?secret=yes"
        )

        codes = {issue.code for issue in contract.validate_model_routing_section(section)}

        self.assertIn("unsupported_task_stage", codes)
        self.assertIn("empty_route_override", codes)
        self.assertIn("invalid_provider_url", codes)

    def test_four_legacy_fixtures_cover_frozen_mapping_inputs(self) -> None:
        expected = {
            "gemini_sync.json": {
                ("sync", "backend"),
                ("sync", "model"),
                ("sync", "models"),
                ("batch", "model"),
            },
            "gemini_batch.json": {
                ("sync", "backend"),
                ("sync", "model"),
                ("batch", "model"),
                ("batch", "project_analysis", "model"),
                ("batch", "final_review", "model"),
            },
            "litellm_builtin.json": {
                ("sync", "backend"),
                ("sync", "model"),
                ("sync", "models"),
                ("batch", "model"),
            },
            "litellm_custom.json": {
                ("sync", "backend"),
                ("sync", "model"),
                ("sync", "custom_litellm_providers"),
                ("batch", "model"),
            },
        }
        rag_keys = {
            "embedding_backend", "embedding_provider", "embedding_endpoint",
            "embedding_api_key_env", "embedding_model", "output_dimensionality",
            "embedding_timeout_seconds", "query_task_type", "document_task_type",
        }
        for name in ("gemini_sync.json", "litellm_custom.json"):
            expected[name].update((scope, "rag", key) for scope in ("sync", "batch") for key in rag_keys)
        expected["gemini_sync.json"].add(("rotation", "model"))
        for name, paths in expected.items():
            with self.subTest(name=name):
                config = _load_fixture(f"model_routing_legacy/{name}")
                self.assertEqual(set(contract.legacy_fields_present(config)), paths)

    def test_legacy_inventory_is_read_only(self) -> None:
        config = _load_fixture("model_routing_legacy/litellm_custom.json")
        before = copy.deepcopy(config)

        contract.legacy_fields_present(config)

        self.assertEqual(config, before)


if __name__ == "__main__":
    unittest.main()
