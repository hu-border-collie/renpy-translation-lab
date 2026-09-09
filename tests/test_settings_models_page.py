"""Independent construction tests for the #202 Phase D Models Settings page."""
from __future__ import annotations

import unittest

from gui_qt.settings.page_contract import SettingsIssue, SettingsPage
from gui_qt.settings.models_page import (
    ModelsSettingsPage,
    batch_thinking_value_for_load,
    batch_thinking_value_for_model_change,
    should_save_batch_thinking_level,
    supports_batch_thinking,
)

try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
    ModelsSettingsPage = None  # type: ignore[misc,assignment]
else:
    IMPORT_ERROR = None

from tests import gui_test_support


class ModelsThinkingHelperTests(unittest.TestCase):
    def test_gemini3_supports_thinking(self) -> None:
        self.assertTrue(supports_batch_thinking("gemini-3.1-flash-lite"))
        self.assertFalse(supports_batch_thinking("gemini-2.5-flash"))

    def test_missing_thinking_defaults_for_supported_model(self) -> None:
        self.assertEqual(
            batch_thinking_value_for_load({}, "gemini-3.1-flash-lite"),
            "minimal",
        )

    def test_explicit_empty_thinking_is_kept(self) -> None:
        self.assertEqual(
            batch_thinking_value_for_load(
                {"thinking_level": ""},
                "gemini-3.1-flash-lite",
            ),
            "",
        )

    def test_empty_thinking_saved_after_user_change(self) -> None:
        self.assertTrue(
            should_save_batch_thinking_level(
                {},
                "gemini-3.1-flash-lite",
                "",
                True,
            )
        )

    def test_supported_model_switch_defaults_empty_to_minimal(self) -> None:
        self.assertEqual(
            batch_thinking_value_for_model_change(
                "gemini-3.1-flash-lite",
                "",
                False,
                False,
            ),
            "minimal",
        )

    def test_supported_model_switch_preserves_user_empty(self) -> None:
        self.assertIsNone(
            batch_thinking_value_for_model_change(
                "gemini-3.1-flash-lite",
                "",
                False,
                True,
            )
        )


@gui_test_support.skip_unless_gui(QApplication is None, IMPORT_ERROR)
class ModelsSettingsPageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ModelsSettingsPage()

    def tearDown(self) -> None:
        self.page.widget.deleteLater()
        self._app.processEvents()

    def test_page_satisfies_settings_protocol(self) -> None:
        self.assertIsInstance(self.page, SettingsPage)
        self.assertEqual(self.page.page_key, "models")
        self.assertEqual(
            self.page.config_keys,
            frozenset(
                {
                    "sync_model",
                    "sync_embedding_model",
                    "batch_model",
                    "batch_embedding_model",
                    "batch_thinking_level",
                }
            ),
        )

    def test_load_collect_round_trip_owned_keys(self) -> None:
        self.page.load(
            {
                "sync_model": "gemini-3.1-flash-lite",
                "sync_embedding_model": "gemini-embedding-001",
                "batch_model": "gemini-3.1-pro",
                "batch_embedding_model": "gemini-embedding-001",
                "batch_thinking_level": "low",
            }
        )
        collected = self.page.collect()
        self.assertEqual(collected["sync_model"], "gemini-3.1-flash-lite")
        self.assertEqual(collected["batch_model"], "gemini-3.1-pro")
        self.assertEqual(collected["batch_thinking_level"], "low")
        self.assertEqual(set(collected), set(self.page.config_keys))

    def test_reset_restores_last_loaded_baseline(self) -> None:
        self.page.load(
            {
                "sync_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "minimal",
            }
        )
        self.page.sync_model_combo.setCurrentIndex(0)
        self.page.reset()
        self.assertEqual(self.page.collect()["sync_model"], "gemini-3.1-flash-lite")

    def test_focus_issue_targets_owned_widget(self) -> None:
        issue = SettingsIssue("models", "batch_model", "missing")
        self.assertTrue(self.page.focus_issue(issue))

    def test_set_task_running_disables_model_combos(self) -> None:
        self.assertTrue(self.page.sync_model_combo.isEnabled())
        self.page.set_task_running(True)
        self.assertFalse(self.page.sync_model_combo.isEnabled())
        self.assertFalse(self.page.batch_model_combo.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.sync_model_combo.isEnabled())

    def test_catalog_extras_round_trip_through_load(self) -> None:
        self.page.set_catalog(
            ["gemini-3.1-flash-lite", "custom-flash"],
            ["gemini-embedding-001"],
        )
        self.page.load(
            {
                "sync_model": "custom-flash",
                "sync_embedding_model": "gemini-embedding-001",
                "batch_model": "gemini-3.1-flash-lite",
                "batch_embedding_model": "gemini-embedding-001",
                "batch_thinking_level": "minimal",
            }
        )
        collected = self.page.collect()
        self.assertEqual(collected["sync_model"], "custom-flash")
        texts = [
            self.page.sync_model_combo.itemText(i)
            for i in range(self.page.sync_model_combo.count())
        ]
        self.assertIn("custom-flash", texts)

    def test_thinking_combo_tracks_supported_batch_model(self) -> None:
        self.page.load(
            {
                "batch_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "low",
            }
        )
        self.assertTrue(self.page.batch_thinking_combo.isEnabled())
        self.page.load(
            {
                "batch_model": "gemini-2.5-flash",
                "batch_thinking_level": "",
            }
        )
        self.assertFalse(self.page.batch_thinking_combo.isEnabled())
        self.assertEqual(self.page.collect()["batch_thinking_level"], "")

    def test_task_running_keeps_thinking_disabled_for_supported_model(self) -> None:
        self.page.load(
            {
                "batch_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "minimal",
            }
        )
        self.page.set_task_running(True)
        self.assertFalse(self.page.batch_thinking_combo.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.batch_thinking_combo.isEnabled())

    def test_gemini_sync_lock_survives_task_running_idle(self) -> None:
        self.page.set_gemini_sync_allowed(
            False,
            tooltip="当前同步后端为 LiteLLM；切回 Gemini 后可选择此模型。",
        )
        self.assertFalse(self.page.sync_model_combo.isEnabled())
        self.page.set_task_running(False)
        self.assertFalse(self.page.sync_model_combo.isEnabled())
        self.assertIn("LiteLLM", self.page.sync_model_combo.toolTip())
        self.page.set_gemini_sync_allowed(True, tooltip="")
        self.assertTrue(self.page.sync_model_combo.isEnabled())
        self.assertEqual(self.page.sync_model_combo.toolTip(), "")

    def test_restore_keeps_user_changed_thinking_flag(self) -> None:
        self.page.load(
            {
                "batch_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "minimal",
            }
        )
        empty_index = self.page.batch_thinking_combo.findData("")
        self.page.batch_thinking_combo.setCurrentIndex(empty_index)
        self.assertTrue(self.page._batch_thinking_user_changed)
        self.page.load(
            {
                "batch_model": "gemini-3.1-flash-lite",
                "batch_thinking_level": "",
            },
            restore=True,
        )
        self.assertTrue(self.page._batch_thinking_user_changed)
        self.assertEqual(self.page.collect()["batch_thinking_level"], "")
        self.page.reset()
        self.assertEqual(self.page.collect()["batch_thinking_level"], "minimal")


if __name__ == "__main__":
    unittest.main()
