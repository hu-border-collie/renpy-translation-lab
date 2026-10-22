"""Contract tests for the #202 Settings page boundary (no Qt required)."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import (
    SettingsIssue,
    SettingsPage,
    SettingsPageActions,
)


class _FakePage:
    page_key = "fake"
    nav_label = "Fake"
    config_keys = frozenset({"fake_value"})
    immediate_action_ids = frozenset({"immediate"})

    def load(self, snapshot):
        self.snapshot = dict(snapshot)

    def collect(self):
        return {"fake_value": self.snapshot.get("fake_value")}

    def validate(self):
        return []

    def reset(self):
        self.snapshot = {}

    def focus_issue(self, issue):
        return issue.field_key == "fake_value"

    def set_task_running(self, running):
        self.running = running


class SettingsPageContractTests(unittest.TestCase):
    def test_issue_defaults_to_error_severity(self) -> None:
        issue = SettingsIssue("models", "batch_model", "must not be empty")
        self.assertEqual(issue.page_key, "models")
        self.assertEqual(issue.field_key, "batch_model")
        self.assertEqual(issue.severity, "error")

    def test_issue_accepts_warning_severity(self) -> None:
        issue = SettingsIssue("models", "batch_model", "check this", "warning")
        self.assertEqual(issue.severity, "warning")

    def test_page_actions_default_to_no_callbacks(self) -> None:
        actions = SettingsPageActions()
        self.assertIsNone(actions.save)
        self.assertIsNone(actions.reload)
        self.assertIsNone(actions.navigate)
        self.assertIsNone(actions.run_immediate)
        self.assertIsNone(actions.show_status)

    def test_page_actions_carry_host_callbacks(self) -> None:
        calls: list[object] = []
        actions = SettingsPageActions(
            save=lambda: calls.append("save"),
            reload=lambda: calls.append("reload"),
            navigate=lambda key: calls.append(key),
            run_immediate=lambda action_id, payload: bool(
                calls.append((action_id, dict(payload))) or True
            ),
            show_status=lambda message: calls.append(message),
        )
        actions.save()
        actions.reload()
        actions.navigate("models")
        actions.run_immediate("preview_theme", {"theme": "dark"})
        actions.show_status("ok")
        self.assertEqual(
            calls,
            [
                "save",
                "reload",
                "models",
                ("preview_theme", {"theme": "dark"}),
                "ok",
            ],
        )

    def test_fake_page_satisfies_runtime_protocol(self) -> None:
        self.assertIsInstance(_FakePage(), SettingsPage)


if __name__ == "__main__":
    unittest.main()
