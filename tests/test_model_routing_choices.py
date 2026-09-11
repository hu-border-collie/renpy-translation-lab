"""ModelProfile / ExecutionStrategy selection tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gemini_translate_batch as batch
import model_routing_reader as reader
from model_routing_migration import preview_migration

FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def migrated(name: str) -> dict:
    payload = json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))
    return preview_migration(payload).config


class ProfileStrategyChoicesTests(unittest.TestCase):
    def test_gemini_profiles_offer_both_strategies(self) -> None:
        choices = reader.profile_strategy_choices(migrated("gemini_batch"))

        by_id = {entry["id"]: entry for entry in choices["profiles"]}
        self.assertNotIn("legacy-sync-embedding", by_id)
        self.assertEqual(
            by_id["legacy-sync"]["strategies"],
            ("sync", "gemini_batch"),
        )
        self.assertEqual(by_id["legacy-sync"]["unsupported"], {})
        self.assertEqual(
            choices["defaults"]["primary_profile_id"],
            "legacy-batch",
        )
        self.assertEqual(
            choices["defaults"]["execution_strategy"],
            "gemini_batch",
        )

    def test_litellm_profile_is_sync_only_with_reason(self) -> None:
        choices = reader.profile_strategy_choices(migrated("litellm_custom"))

        by_id = {entry["id"]: entry for entry in choices["profiles"]}
        litellm = by_id["legacy-sync"]
        self.assertEqual(litellm["adapter"], "litellm")
        self.assertEqual(litellm["strategies"], ("sync",))
        self.assertEqual(
            litellm["unsupported"]["gemini_batch"],
            "missing_gemini_adapter",
        )
        self.assertEqual(
            by_id["legacy-batch"]["strategies"],
            ("sync", "gemini_batch"),
        )

    def test_choices_never_expose_credentials(self) -> None:
        text = json.dumps(
            reader.profile_strategy_choices(migrated("litellm_custom"))
        ).lower()
        for forbidden in ("api_key", "secret", "token", "password"):
            self.assertNotIn(forbidden, text)


class PrimaryProfileOverrideTests(unittest.TestCase):
    def test_override_pins_translation_and_legacy_execution(self) -> None:
        config = migrated("gemini_batch")

        with reader.primary_profile_override("legacy-sync"):
            plan = reader.resolve_runtime_plan(config, execution="sync")

        route = plan.routes["translation"]
        self.assertEqual(route.profile_id, "legacy-sync")
        self.assertEqual(route.strategy.value, "sync")
        self.assertEqual(plan.primary_profile_id, "legacy-sync")

    def test_override_without_scope_is_a_noop(self) -> None:
        config = migrated("gemini_batch")

        baseline = reader.resolve_runtime_plan(config, execution="sync")
        scoped = reader.resolve_runtime_plan(config, execution="sync")

        self.assertEqual(
            baseline.routes["translation"].profile_id,
            scoped.routes["translation"].profile_id,
        )
        self.assertIsNone(reader.active_primary_profile_override())

    def test_litellm_override_keeps_batch_projection_on_default(self) -> None:
        config = migrated("litellm_custom")
        baseline = reader.runtime_settings_view(config)

        with reader.primary_profile_override("legacy-sync"):
            view = reader.runtime_settings_view(config)

        self.assertEqual(view["sync"]["backend"], "litellm")
        self.assertEqual(view["sync"]["model"], "acme-compatible/model-a")
        self.assertEqual(view["batch"]["model"], baseline["batch"]["model"])

    def test_gemini_override_projects_into_batch_scope(self) -> None:
        config = migrated("gemini_batch")

        with reader.primary_profile_override("legacy-sync"):
            view = reader.runtime_settings_view(config)

        self.assertEqual(view["sync"]["model"], "gemini-3.1-flash-lite")
        self.assertEqual(view["batch"]["model"], "gemini-3.1-flash-lite")

    def test_incompatible_strategy_fails_strict_and_skips_non_strict(self) -> None:
        config = migrated("litellm_custom")

        with reader.primary_profile_override("legacy-sync"):
            with self.assertRaises(reader.routing.ModelRoutingConfigError):
                reader.resolve_runtime_plan(config, execution="gemini_batch")
            plan = reader.resolve_runtime_plan(
                config,
                execution="gemini_batch",
                strict_override=False,
            )

        self.assertEqual(
            plan.routes["translation"].profile_id,
            "legacy-batch",
        )

    def test_unknown_profile_is_rejected(self) -> None:
        with reader.primary_profile_override("missing-profile"):
            with self.assertRaises(reader.routing.ModelRoutingConfigError):
                reader.resolve_runtime_plan(
                    migrated("gemini_batch"),
                    execution="sync",
                )

    def test_nested_scopes_restore_the_outer_override(self) -> None:
        with reader.primary_profile_override("legacy-sync"):
            with reader.primary_profile_override("legacy-batch"):
                self.assertEqual(
                    reader.active_primary_profile_override(),
                    "legacy-batch",
                )
            self.assertEqual(
                reader.active_primary_profile_override(),
                "legacy-sync",
            )
        self.assertIsNone(reader.active_primary_profile_override())


class ProfileCliWiringTests(unittest.TestCase):
    def test_build_profile_is_active_while_loading_runtime(self) -> None:
        parser = batch.build_arg_parser()
        args = parser.parse_args(["build", "--profile", "legacy-sync"])
        seen: dict[str, object] = {}

        def fake_load_config(*, require_api_key=True):
            seen["override"] = reader.active_primary_profile_override()

        with (
            mock.patch.object(batch.legacy, "load_config", side_effect=fake_load_config),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch,
                "create_batch_package",
                return_value={},
            ) as create,
        ):
            batch.dispatch_command(parser, args)

        self.assertEqual(seen["override"], "legacy-sync")
        create.assert_called_once()
        self.assertIsNone(reader.active_primary_profile_override())

    def test_translate_preflight_profile_is_active_while_loading_runtime(self) -> None:
        parser = batch.build_arg_parser()
        args = parser.parse_args(
            [
                "translate-preflight",
                "--strategy",
                "sync",
                "--profile",
                "legacy-sync",
                "--output",
                "json",
            ]
        )
        seen: dict[str, object] = {}

        def fake_load_config(*, require_api_key=True):
            seen["override"] = reader.active_primary_profile_override()
            seen["require_api_key"] = require_api_key

        with (
            mock.patch.object(batch.legacy, "load_config", side_effect=fake_load_config),
            mock.patch.object(batch.legacy, "load_translator_settings"),
            mock.patch.object(batch.legacy, "load_glossary"),
            mock.patch.object(batch, "load_batch_settings"),
            mock.patch.object(batch, "print_banner"),
            mock.patch.object(
                batch,
                "run_translate_preflight",
                return_value={"status": "ready"},
            ) as run_preflight,
        ):
            result = batch.dispatch_command(parser, args)

        self.assertEqual(seen["override"], "legacy-sync")
        self.assertFalse(bool(seen["require_api_key"]))
        self.assertEqual(result, {"status": "ready"})
        run_preflight.assert_called_once()
        self.assertIsNone(reader.active_primary_profile_override())

    def test_sync_start_profile_is_active_while_building_service(self) -> None:
        parser = batch.build_arg_parser()
        args = parser.parse_args(
            ["sync-start", "--profile", "legacy-sync", "--output", "json"]
        )
        seen: dict[str, object] = {}

        def fake_service(*, require_provider):
            seen["override"] = reader.active_primary_profile_override()
            service = mock.Mock()
            service.start.return_value = {
                "run_id": "sync-run-v1-demo",
                "run_status": "completed",
            }
            context = SimpleNamespace(plan_build=SimpleNamespace(requests=[object()]))
            return service, context

        with (
            mock.patch.object(
                batch,
                "_durable_sync_production_service",
                side_effect=fake_service,
            ),
            mock.patch.object(batch, "_print_durable_sync_snapshot"),
        ):
            snapshot = batch.run_durable_sync_command(args)

        self.assertEqual(seen["override"], "legacy-sync")
        self.assertEqual(snapshot["run_id"], "sync-run-v1-demo")
        self.assertIsNone(reader.active_primary_profile_override())


if __name__ == "__main__":
    unittest.main()
