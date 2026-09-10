"""Model Profiles editor core tests (#348 P3)."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import model_profiles_editor as editor

FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def gemini_section() -> dict:
    section = editor.empty_section()
    section = editor.add_provider(
        section,
        label="Gemini Main",
        adapter="gemini",
        provider="gemini",
        credential_kind="api_keys_json",
        credential_name="api_keys",
        credential_env_name="GEMINI_API_KEY",
    )
    section = editor.add_profile(
        section,
        label="Gemini Main",
        provider_id="gemini-main",
        model="gemini-3.5-flash",
    )
    section = editor.set_defaults(
        section,
        primary_profile_id="gemini-main",
        execution_strategy="sync",
    )
    return section


def litellm_section() -> dict:
    section = editor.add_provider(
        editor.empty_section(),
        label="Acme",
        adapter="litellm",
        provider="acme",
        base_url="https://acme.example/v1",
        credential_kind="env",
        credential_name="ACME_API_KEY",
    )
    section = editor.add_profile(
        section,
        label="Acme Main",
        provider_id="acme",
        model="acme/model-a",
    )
    return editor.set_defaults(
        section,
        primary_profile_id="acme-main",
        execution_strategy="sync",
    )


class ProviderProfileCrudTests(unittest.TestCase):
    def test_add_provider_and_profile_generate_unique_ids(self) -> None:
        section = gemini_section()
        section = editor.add_provider(
            section,
            label="Gemini Main",
            adapter="gemini",
            provider="gemini",
        )
        self.assertEqual(editor.provider_ids(section), ("gemini-main", "gemini-main-2"))

        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id="gemini-main",
            model="gemini-3.1-flash-lite",
        )
        self.assertIn("gemini-main-2", editor.profile_ids(section))

    def test_copy_profile_keeps_connection_and_isolates_edits(self) -> None:
        section = copy.deepcopy(gemini_section())
        copied = editor.copy_profile(section, "gemini-main", label="Gemini Backup")

        backup = copied["profiles"]["gemini-backup"]
        self.assertEqual(backup["provider_id"], "gemini-main")
        self.assertEqual(backup["model"], "gemini-3.5-flash")
        self.assertNotIn("gemini-backup", section["profiles"])

        updated = editor.update_profile(
            copied,
            "gemini-backup",
            model="gemini-3.1-flash-lite",
        )
        self.assertEqual(
            updated["profiles"]["gemini-main"]["model"],
            "gemini-3.5-flash",
        )
        self.assertEqual(
            updated["profiles"]["gemini-backup"]["model"],
            "gemini-3.1-flash-lite",
        )

    def test_mutations_preserve_unknown_fields(self) -> None:
        section = gemini_section()
        section["future_top_level"] = {"keep": True}
        section["providers"]["gemini-main"]["future_provider_field"] = "keep"
        section["profiles"]["gemini-main"]["future_profile_field"] = ["keep"]

        updated = editor.update_profile(
            section,
            "gemini-main",
            models=["gemini-3.1-flash-lite"],
        )
        updated = editor.update_provider(updated, "gemini-main", label="Renamed")

        self.assertEqual(updated["future_top_level"], {"keep": True})
        self.assertEqual(
            updated["providers"]["gemini-main"]["future_provider_field"],
            "keep",
        )
        self.assertEqual(
            updated["profiles"]["gemini-main"]["future_profile_field"],
            ["keep"],
        )

    def test_delete_profile_refuses_references(self) -> None:
        section = gemini_section()
        section = editor.set_route(
            section,
            "project_analysis",
            enabled=True,
            profile_id="gemini-main",
            strategy="sync",
        )

        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.delete_profile(section, "gemini-main")

        self.assertEqual(caught.exception.code, "PROFILE_IN_USE")
        self.assertIn("routes.project_analysis.profile_id", caught.exception.details["references"])

    def test_delete_provider_refuses_use(self) -> None:
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.delete_provider(gemini_section(), "gemini-main")
        self.assertEqual(caught.exception.code, "PROVIDER_IN_USE")
        self.assertEqual(caught.exception.details["profile_ids"], ["gemini-main"])

    def test_delete_profile_and_provider_after_unlink(self) -> None:
        section = gemini_section()
        section = editor.set_defaults(
            section,
            primary_profile_id="gemini-main",
            execution_strategy="sync",
        )
        section = editor.update_profile(section, "gemini-main", embedding_profile_id="")
        # defaults still references the primary profile
        with self.assertRaises(editor.ModelProfilesEditorError):
            editor.delete_profile(section, "gemini-main")

        # Rebuild a free profile and remove it cleanly.
        free = editor.add_profile(
            section,
            label="Free",
            provider_id="gemini-main",
            model="gemini-3.1-flash-lite",
        )
        free = editor.delete_profile(free, "free")
        self.assertNotIn("free", free["profiles"])

        # Provider can be removed once no profile uses it.
        no_profiles = copy.deepcopy(free)
        no_profiles["profiles"] = {}
        no_profiles["defaults"] = {"primary_profile_id": "", "execution_strategy": "sync"}
        no_profiles["routes"] = {}
        trimmed = editor.delete_provider(no_profiles, "gemini-main")
        self.assertEqual(editor.provider_ids(trimmed), ())


class StrategyGatingTests(unittest.TestCase):
    def test_litellm_profile_cannot_be_default_for_batch(self) -> None:
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.set_defaults(
                litellm_section(),
                primary_profile_id="acme-main",
                execution_strategy="gemini_batch",
            )
        self.assertEqual(caught.exception.code, "STRATEGY_NOT_SUPPORTED")
        self.assertEqual(
            caught.exception.details["reason"],
            "missing_gemini_adapter",
        )

    def test_strategy_choices_match_capabilities(self) -> None:
        choices = editor.strategy_choices(litellm_section())
        self.assertEqual(choices["acme-main"], ("sync",))
        self.assertEqual(
            editor.strategy_choices(gemini_section())["gemini-main"],
            ("sync", "gemini_batch"),
        )

    def test_route_override_and_inheritance(self) -> None:
        section = gemini_section()
        section = editor.set_route(
            section,
            "final_review",
            enabled=True,
            profile_id="gemini-main",
            strategy="gemini_batch",
        )

        routes = {row["stage"]: row for row in editor.resolved_routes(section)}
        self.assertFalse(routes["translation"]["explicit"])
        self.assertEqual(routes["translation"]["profile_id"], "gemini-main")
        self.assertTrue(routes["final_review"]["explicit"])
        self.assertEqual(routes["final_review"]["strategy"], "gemini_batch")

        section = editor.set_route(section, "final_review", enabled=False)
        routes = {row["stage"]: row for row in editor.resolved_routes(section)}
        self.assertFalse(routes["final_review"]["explicit"])

    def test_models_pool_always_includes_primary_model(self) -> None:
        section = editor.add_provider(
            editor.empty_section(),
            label="Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
        )
        section = editor.add_profile(
            section,
            label="Main",
            provider_id="gemini",
            model="gemini-primary",
            models=["gemini-extra", "gemini-primary"],
        )

        self.assertEqual(
            section["profiles"]["main"]["models"],
            ["gemini-primary", "gemini-extra"],
        )
        view = editor.editor_view(section)
        self.assertEqual(view["profiles"][0]["models"], ("gemini-primary", "gemini-extra"))
        self.assertEqual(view["profiles"][0]["rotation_extras"], ("gemini-extra",))

        updated = editor.update_profile(section, "main", label="Renamed")
        self.assertEqual(
            updated["profiles"]["main"]["models"],
            ["gemini-primary", "gemini-extra"],
        )

        rotated = editor.update_profile(
            updated,
            "main",
            models=["gemini-new"],
        )
        self.assertEqual(
            rotated["profiles"]["main"]["models"],
            ["gemini-primary", "gemini-new"],
        )

    def test_sync_capability_override_gates_strategies(self) -> None:
        section = gemini_section()
        section = editor.update_profile(
            section,
            "gemini-main",
            capability_overrides={"sync_generation": False},
        )

        choices = editor.strategy_choices(section)
        self.assertEqual(choices["gemini-main"], ("gemini_batch",))
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.set_defaults(
                section,
                primary_profile_id="gemini-main",
                execution_strategy="sync",
            )
        self.assertEqual(
            caught.exception.details["reason"],
            "missing_sync_generation",
        )

    def test_embedding_profile_cannot_serve_generation_stages(self) -> None:
        section = gemini_section()
        section = editor.add_profile(
            section,
            label="Embedding",
            provider_id="gemini-main",
            model="gemini-embedding-001",
            purpose="embedding",
        )

        choices = editor.strategy_choices(section)
        self.assertEqual(choices["embedding"], ())
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.set_route(
                section,
                "translation",
                enabled=True,
                profile_id="embedding",
                strategy="sync",
            )
        self.assertEqual(
            caught.exception.details["reason"],
            "embedding_profile",
        )

    def test_corrupt_profile_entry_does_not_crash_views(self) -> None:
        section = gemini_section()
        section["profiles"]["broken"] = "not-an-object"

        view = editor.editor_view(section)
        choices = editor.strategy_choices(section)

        self.assertIn("broken", {item["id"] for item in view["profiles"]})
        self.assertEqual(choices["broken"], ())
        self.assertTrue(editor.section_issues(section))

    def test_unknown_capability_override_is_rejected(self) -> None:
        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.update_profile(
                gemini_section(),
                "gemini-main",
                capability_overrides={"telepathy": True},
            )
        self.assertEqual(caught.exception.code, "UNKNOWN_CAPABILITY_OVERRIDE")


class EditorViewTests(unittest.TestCase):
    def test_view_is_credential_free_and_resolves_capabilities(self) -> None:
        view = editor.editor_view(gemini_section())
        text = json.dumps(view).lower()
        for forbidden in ("secret", "password", "authorization", "api_key_value"):
            self.assertNotIn(forbidden, text)
        # Only reference fields are exposed, never resolved credential values.
        self.assertEqual(
            set(view["providers"][0]["credential_ref"]),
            {"kind", "name", "env_name"},
        )
        profile = view["profiles"][0]
        self.assertEqual(profile["strategies"], ("sync", "gemini_batch"))
        self.assertTrue(profile["capabilities"]["remote_batch"])
        self.assertEqual(profile["capabilities"]["structured_output_source"], "adapter_default")
        provider = view["providers"][0]
        self.assertEqual(provider["credential_ref"]["kind"], "api_keys_json")
        self.assertEqual(provider["credential_ref"]["name"], "api_keys")

    def test_view_survives_invalid_draft(self) -> None:
        section = editor.empty_section()
        section["profiles"]["draft"] = {
            "label": "Draft",
            "provider_id": "missing-provider",
            "model": "",
        }
        view = editor.editor_view(section)
        self.assertEqual(view["profiles"][0]["model"], "")
        self.assertEqual(view["profiles"][0]["strategies"], ())

    def test_section_issues_report_invalid_draft(self) -> None:
        issues = editor.section_issues(editor.empty_section())
        codes = {issue["code"] for issue in issues}
        self.assertIn("missing_profiles", codes)
        self.assertIn("unknown_primary_profile", codes)

    def test_broken_containers_are_reported_and_mutations_refuse(self) -> None:
        broken = {
            "schema_version": 1,
            "providers": [],
            "profiles": [],
            "defaults": {"primary_profile_id": "x", "execution_strategy": "sync"},
            "routes": [],
        }

        codes = {issue["code"] for issue in editor.section_issues(broken)}
        self.assertIn("missing_providers", codes)
        self.assertIn("invalid_routes", codes)

        with self.assertRaises(editor.ModelProfilesEditorError) as caught:
            editor.add_provider(
                broken,
                label="X",
                adapter="gemini",
                provider="gemini",
            )
        self.assertEqual(caught.exception.code, "INVALID_SECTION_CONTAINER")
        self.assertEqual(caught.exception.details["field"], "providers")

        # Read-only views still render something instead of crashing.
        view = editor.editor_view(broken)
        self.assertEqual(view["profiles"], ())

    def test_migrated_fixture_round_trips_through_view(self) -> None:
        payload = json.loads((FIXTURES / "gemini_batch.json").read_text(encoding="utf-8"))
        from model_routing_migration import preview_migration

        section = preview_migration(payload).config["model_routing"]
        view = editor.editor_view(section)
        self.assertIn("legacy-batch", {item["id"] for item in view["profiles"]})
        self.assertEqual(view["defaults"]["primary_profile_id"], "legacy-batch")
        self.assertFalse(editor.section_issues(section))


if __name__ == "__main__":
    unittest.main()
