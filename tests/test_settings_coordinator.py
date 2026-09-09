"""Coordinator tests for #202 Settings lazy build and page boundaries (no Qt)."""
from __future__ import annotations

import unittest

from gui_qt.settings.coordinator import SettingsCoordinator
from gui_qt.settings.page_contract import SettingsIssue, SettingsPageActions
from gui_qt.settings.registry import SettingsPageRegistry, SettingsPageSpec


class _FakeSettingsPage:
    def __init__(self, spec: SettingsPageSpec) -> None:
        self.page_key = spec.key
        self.nav_label = spec.nav_label
        self.config_keys = spec.config_keys
        self.immediate_action_ids = spec.immediate_action_ids
        self.values: dict[str, object] = {}
        self.load_calls: list[dict[str, object]] = []
        self.reset_calls = 0
        self.focused: str | None = None
        self.running: bool | None = None
        self.actions: SettingsPageActions | None = None
        self.issues: list[SettingsIssue] = []

    def set_action_callbacks(self, actions: SettingsPageActions) -> None:
        self.actions = actions

    def load(self, snapshot):
        self.values.update(dict(snapshot))
        self.load_calls.append(dict(snapshot))

    def collect(self):
        return dict(self.values)

    def validate(self):
        return list(self.issues)

    def reset(self):
        self.values.clear()
        self.reset_calls += 1

    def focus_issue(self, issue):
        self.focused = issue.field_key
        return issue.field_key in self.config_keys

    def set_task_running(self, running):
        self.running = running


def _registry() -> SettingsPageRegistry:
    return SettingsPageRegistry(
        [
            SettingsPageSpec(
                "alpha",
                "Alpha",
                "_build_alpha",
                config_page=True,
                config_keys=frozenset({"alpha_value", "alpha_flag"}),
                immediate_action_ids=frozenset({"alpha_action"}),
            ),
            SettingsPageSpec(
                "beta",
                "Beta",
                "_build_beta",
                config_page=True,
                config_keys=frozenset({"beta_value"}),
            ),
            SettingsPageSpec(
                "gamma",
                "Gamma",
                "_build_gamma",
            ),
        ]
    )


class SettingsCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _registry()
        self.built: list[str] = []
        self.shown: list[str] = []
        self.installed: list[str] = []
        self.actions = SettingsPageActions(
            save=lambda: self.shown.append("save"),
            reload=lambda: self.shown.append("reload"),
            navigate=lambda key: self.shown.append(f"nav:{key}"),
            show_status=lambda message: self.shown.append(f"status:{message}"),
        )

        def builder(spec):
            self.built.append(spec.key)
            return _FakeSettingsPage(spec)

        def on_page_built(spec, _page):
            self.installed.append(spec.key)

        self.coordinator = SettingsCoordinator(
            self.registry,
            builder=builder,
            show_page=lambda key: self.shown.append(key),
            on_page_built=on_page_built,
            actions=self.actions,
        )

    def test_activate_builds_only_requested_page(self) -> None:
        page = self.coordinator.activate("alpha")
        self.assertIsNotNone(page)
        self.assertEqual(self.built, ["alpha"])
        self.assertEqual(self.installed, ["alpha"])
        self.assertEqual(self.shown, ["alpha"])
        self.assertTrue(self.coordinator.is_built("alpha"))
        self.assertFalse(self.coordinator.is_built("beta"))
        self.assertIs(self.coordinator.page("alpha").actions, self.actions)

    def test_ensure_page_is_idempotent(self) -> None:
        first = self.coordinator.ensure_page("alpha")
        second = self.coordinator.ensure_page("alpha")
        self.assertIs(first, second)
        self.assertEqual(self.built, ["alpha"])

    def test_ensure_page_without_build_returns_none(self) -> None:
        self.assertIsNone(self.coordinator.ensure_page("alpha", build=False))
        self.assertEqual(self.built, [])

    def test_load_only_touches_built_pages_and_owned_keys(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.load(
            {
                "alpha_value": 1,
                "alpha_flag": True,
                "beta_value": 2,
                "unowned": 3,
            }
        )
        page = self.coordinator.page("alpha")
        self.assertEqual(page.values, {"alpha_value": 1, "alpha_flag": True})
        self.assertEqual(self.coordinator.loaded_keys(), frozenset({"alpha"}))
        self.assertFalse(self.coordinator.is_built("beta"))

    def test_load_can_limit_to_requested_built_pages(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.ensure_page("beta")
        self.coordinator.load(
            {"alpha_value": 1, "beta_value": 2},
            pages={"beta"},
        )
        self.assertEqual(self.coordinator.page("alpha").values, {})
        self.assertEqual(self.coordinator.page("beta").values, {"beta_value": 2})
        self.assertEqual(self.coordinator.loaded_keys(), frozenset({"beta"}))

    def test_collect_merges_page_owned_values(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.ensure_page("beta")
        self.coordinator.load({"alpha_value": 1, "beta_value": 2})
        self.assertEqual(
            self.coordinator.collect(),
            {"alpha_value": 1, "beta_value": 2},
        )

    def test_collect_rejects_unowned_key(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.page("alpha").values["outside"] = True
        with self.assertRaises(ValueError):
            self.coordinator.collect()

    def test_collect_rejects_duplicate_key_ownership(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.ensure_page("beta")
        self.coordinator.page("beta").values["alpha_value"] = True
        with self.assertRaises(ValueError):
            self.coordinator.collect()

    def test_validate_aggregates_page_issues(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.ensure_page("beta")
        self.coordinator.page("alpha").issues.append(
            SettingsIssue("alpha", "alpha_value", "bad alpha")
        )
        self.coordinator.page("beta").issues.append(
            SettingsIssue("beta", "beta_value", "bad beta")
        )
        self.assertEqual(
            [issue.field_key for issue in self.coordinator.validate()],
            ["alpha_value", "beta_value"],
        )

    def test_validate_rejects_issue_for_wrong_page(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.page("alpha").issues.append(
            SettingsIssue("beta", "beta_value", "wrong page")
        )
        with self.assertRaises(ValueError):
            self.coordinator.validate()

    def test_focus_issue_activates_and_delegates(self) -> None:
        issue = SettingsIssue("beta", "beta_value", "bad")
        self.assertTrue(self.coordinator.focus_issue(issue))
        self.assertEqual(self.built, ["beta"])
        self.assertEqual(self.shown, ["beta"])
        self.assertEqual(self.coordinator.page("beta").focused, "beta_value")
        self.assertEqual(self.coordinator.active_key, "beta")

    def test_focus_issue_returns_false_for_unhandled_field(self) -> None:
        self.coordinator.ensure_page("alpha")
        issue = SettingsIssue("alpha", "outside", "bad")
        self.assertFalse(self.coordinator.focus_issue(issue))

    def test_reset_clears_values_and_loaded_state(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.load({"alpha_value": 1})
        self.coordinator.reset(pages={"alpha"})
        self.assertEqual(self.coordinator.page("alpha").values, {})
        self.assertEqual(self.coordinator.page("alpha").reset_calls, 1)
        self.assertEqual(self.coordinator.loaded_keys(), frozenset())

    def test_set_task_running_reaches_built_pages(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.ensure_page("beta")
        self.coordinator.set_task_running(True)
        self.assertTrue(self.coordinator.page("alpha").running)
        self.assertTrue(self.coordinator.page("beta").running)

    def test_has_unsaved_changes_compares_host_baseline(self) -> None:
        self.coordinator.ensure_page("alpha")
        self.coordinator.load({"alpha_value": 1})
        self.assertFalse(
            self.coordinator.has_unsaved_changes({"alpha_value": 1})
        )
        self.coordinator.page("alpha").values["alpha_value"] = 2
        self.assertTrue(
            self.coordinator.has_unsaved_changes({"alpha_value": 1})
        )

    def test_page_contract_mismatch_is_rejected(self) -> None:
        class WrongPage(_FakeSettingsPage):
            def __init__(self, spec):
                super().__init__(spec)
                self.page_key = "wrong"

        coordinator = SettingsCoordinator(
            self.registry,
            builder=lambda spec: WrongPage(spec),
        )
        with self.assertRaises(ValueError):
            coordinator.ensure_page("alpha")

    def test_issue_348_model_profiles_page_can_use_contract(self) -> None:
        """A Model Profiles page only needs the frozen contract to integrate."""

        spec = SettingsPageSpec(
            "model_profiles",
            "Model Profiles",
            "_build_model_profiles",
            config_page=True,
            config_keys=frozenset({"model_routing"}),
        )
        registry = SettingsPageRegistry([spec])
        page = _FakeSettingsPage(spec)
        save_calls: list[str] = []
        coordinator = SettingsCoordinator(
            registry,
            builder=lambda _spec: page,
            actions=SettingsPageActions(
                save=lambda: save_calls.append("save"),
                navigate=lambda key: save_calls.append(key),
            ),
        )
        coordinator.ensure_page("model_profiles")
        coordinator.load({"model_routing": {"schema_version": 1}})
        self.assertEqual(
            coordinator.collect(),
            {"model_routing": {"schema_version": 1}},
        )
        self.assertEqual(coordinator.validate(), [])
        page.issues.append(
            SettingsIssue("model_profiles", "model_routing", "invalid")
        )
        self.assertTrue(
            coordinator.focus_issue(
                SettingsIssue("model_profiles", "model_routing", "invalid")
            )
        )
        page.actions.save()
        self.assertEqual(save_calls, ["save"])


if __name__ == "__main__":
    unittest.main()
