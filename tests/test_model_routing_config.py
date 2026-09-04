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
            "project_analysis": {"profile_id": "gemini-embedding"},
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
