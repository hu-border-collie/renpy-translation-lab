"""Independent construction tests for the #202 Phase C LiteLLM Settings page."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from gui_qt.litellm_catalog_cache import LiteLLMCatalogCache
from gui_qt.settings.page_contract import SettingsIssue, SettingsPage, SettingsPageActions

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.settings.litellm_page import LiteLLMSettingsPage
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    LiteLLMSettingsPage = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(LiteLLMSettingsPage is None, IMPORT_ERROR)
class LiteLLMSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])
        cls._temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def setUp(self) -> None:
        self.cache = LiteLLMCatalogCache(
            Path(self._temp_dir.name) / f"{self._testMethodName}.json"
        )
        self.page = LiteLLMSettingsPage(
            cache=self.cache,
            start_warmup=False,
        )

    def tearDown(self) -> None:
        self.page.request_shutdown()
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "litellm")
        self.assertEqual(self.page.nav_label, "LiteLLM")
        self.assertEqual(
            self.page.config_keys,
            frozenset({"sync_backend", "litellm_model", "custom_litellm_providers"}),
        )

    def test_load_collect_round_trip_owned_keys(self) -> None:
        self.page.load(
            {
                "sync_backend": "litellm",
                "litellm_model": "openai/gpt-test",
                "custom_litellm_providers": (),
            }
        )
        collected = self.page.collect()
        self.assertEqual(collected["sync_backend"], "litellm")
        self.assertEqual(collected["litellm_model"], "openai/gpt-test")
        self.assertEqual(collected["custom_litellm_providers"], ())
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_validate_requires_model_when_backend_is_litellm(self) -> None:
        self.page.load({"sync_backend": "litellm", "litellm_model": ""})
        issues = self.page.validate()
        self.assertTrue(issues)
        self.assertEqual(issues[0].page_key, "litellm")
        self.assertEqual(issues[0].field_key, "litellm_model")

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load(
            {
                "sync_backend": "gemini",
                "litellm_model": "openai/kept",
            }
        )
        self.page.sync_backend_combo.setCurrentIndex(
            self.page.sync_backend_combo.findData("litellm")
        )
        self.page.litellm_model_combo.setEditText("openai/dirty")
        self.page.reset()
        collected = self.page.collect()
        self.assertEqual(collected["sync_backend"], "gemini")
        self.assertEqual(collected["litellm_model"], "openai/kept")

    def test_focus_issue_targets_owned_widget(self) -> None:
        issue = SettingsIssue("litellm", "litellm_model", "missing")
        self.assertTrue(self.page.focus_issue(issue))

    def test_immediate_actions_go_through_injected_callbacks(self) -> None:
        calls: list[tuple[str, dict]] = []
        self.page.set_action_callbacks(
            SettingsPageActions(
                run_immediate=lambda action_id, payload: bool(
                    calls.append((action_id, dict(payload))) or True
                )
            )
        )
        self.page.sync_backend_combo.setCurrentIndex(
            self.page.sync_backend_combo.findData("litellm")
        )
        self.page._populate_litellm_providers(("openai",), selected="openai")
        self.page._on_manage_litellm_keys()
        self.page._on_install_litellm()
        self.assertEqual(
            [item[0] for item in calls],
            ["manage_provider_keys", "install_litellm"],
        )


if __name__ == "__main__":
    unittest.main()
