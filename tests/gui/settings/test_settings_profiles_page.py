"""Model Profiles Settings page tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from foundation import cli_contract
from providers import model_profiles_editor as editor
from gui_qt.user_copy import MODEL_PROFILES_PAGE_COPY
from providers.model_routing_migration import preview_migration

try:
    from PySide6.QtWidgets import QApplication, QGroupBox

    from gui_qt.app import MainWindow
    from gui_qt.settings.page_contract import SettingsPageActions
    from gui_qt.settings.profiles_page import ProfilesSettingsPage
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    ProfilesSettingsPage = None  # type: ignore[assignment,misc]
    SettingsPageActions = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests.support import gui_test_support

FIXTURES = Path(__file__).parents[2] / "fixtures" / "model_routing_legacy"


def migrated_section(name: str = "gemini_batch") -> dict:
    payload = json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))
    return preview_migration(payload).config["model_routing"]


@gui_test_support.skip_unless_gui(ProfilesSettingsPage is None, IMPORT_ERROR)
class ProfilesPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.messages: list[str] = []
        self.page = ProfilesSettingsPage(
            actions=SettingsPageActions(show_status=self.messages.append)
        )

    def tearDown(self) -> None:
        self.page.widget.deleteLater()

    def test_page_contract_and_empty_collect(self) -> None:
        self.assertEqual(self.page.page_key, "profiles")
        self.assertEqual(self.page.nav_label, "模型与供应商")
        self.assertEqual(self.page.config_keys, frozenset({"model_routing"}))
        self.assertEqual(self.page.collect(), {})
        self.assertEqual(self.page.validate(), [])
        self.assertFalse(self.page.profiles_list.isEnabled())
        self.assertTrue(self.page.profiles_group.isHidden())
        self.assertIsInstance(self.page.notice_group, QGroupBox)
        self.assertEqual(self.page.hint_label.objectName(), "config_hint_label")

    def test_load_and_collect_round_trip(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})

        collected = self.page.collect()
        self.assertEqual(collected["model_routing"], section)
        self.assertGreater(self.page.profiles_list.count(), 0)
        self.assertGreater(self.page.providers_list.count(), 0)
        self.assertTrue(self.page.profiles_add_btn.isEnabled())
        self.assertFalse(self.page.profiles_group.isHidden())
        self.assertEqual(self.page.profiles_group.title(), "模型方案")
        self.page.widget.resize(1200, 800)
        self.page.widget.show()
        self.page.section_tabs.setCurrentIndex(1)
        self._app.processEvents()
        self.assertLess(self.page.profiles_group.width(), self.page.profile_editor_group.width())
        self.assertLess(self.page.profile_panels.width(), self.page.widget.viewport().width())

    def test_edit_connection_reveals_the_selected_profiles_provider(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})
        # Deliberately start with a different connection selected.
        self.page.providers_list.setCurrentRow(self.page.providers_list.count() - 1)
        expected = self.page.profile_provider_combo.currentData()
        self.assertNotEqual(self.page._selected_provider_id, expected)
        self.page.section_tabs.setCurrentIndex(1)
        self.page.edit_provider_btn.click()
        self.assertEqual(self.page.section_tabs.currentIndex(), 2)
        self.assertEqual(self.page._selected_provider_id, expected)
        self.assertEqual(self.page.collect(), {"model_routing": section})

    def test_model_search_can_be_selected_with_the_keyboard(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        self.page.load({"model_routing": migrated_section()})
        profile_id = self.page._selected_profile_id
        self.page.widget.resize(1100, 800)
        self.page.widget.show()
        self.page.section_tabs.setCurrentIndex(1)
        self._app.processEvents()
        edit = self.page.profile_model_edit
        edit.setFocus()
        edit.selectAll()
        QTest.keyClicks(edit, "argon")
        self._app.processEvents()
        popup = self.page.profile_model_combo.completer().popup()
        self.assertTrue(popup.isVisible())
        QTest.keyClick(popup, Qt.Key.Key_Down)
        QTest.keyClick(popup, Qt.Key.Key_Return)
        self._app.processEvents()
        self.assertEqual(self.page.profile_model_combo.currentText(), "gemini-4-argon")
        self.assertEqual(self.page.collect()["model_routing"]["profiles"][profile_id]["model"], "gemini-4-argon")

    def test_management_disclosure_preserves_config_and_invalid_removal_access(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})
        self.assertTrue(self.page.notice_group.isHidden())
        self.page.management_toggle.click()
        self.assertFalse(self.page.notice_group.isHidden())
        self.page.management_toggle.click()
        self.assertTrue(self.page.notice_group.isHidden())
        self.assertEqual(self.page.collect(), {"model_routing": section})
        self.page.load({"model_routing": "invalid"})
        self.assertTrue(self.page.section_tabs.isHidden())
        self.assertTrue(self.page.management_toggle.isHidden())
        self.assertFalse(self.page.notice_group.isHidden())
        self.assertFalse(self.page.remove_btn.isHidden())

    def test_create_section_stores_a_valid_gemini_batch_default(self) -> None:
        from providers.gemini_model_catalog import DEFAULT_GEMINI_TRANSLATION_MODEL

        self.page.load({})
        self.page.create_btn.click()

        collected = self.page.collect()["model_routing"]
        self.assertEqual(self.page.validate(), [])
        self.assertEqual(
            collected["profiles"]["gemini-main"]["model"],
            DEFAULT_GEMINI_TRANSLATION_MODEL,
        )
        self.assertEqual(
            collected["defaults"],
            {
                "primary_profile_id": "gemini-main",
                "execution_strategy": "gemini_batch",
            },
        )
        self.assertIn("Gemini Batch", self.page.create_btn.toolTip())

    def test_null_model_routing_is_an_empty_state(self) -> None:
        self.page.load({"model_routing": None})

        self.assertTrue(self.page.create_btn.isEnabled())
        self.assertEqual(self.page.collect(), {})
        self.assertEqual(self.page.validate(), [])

    def test_legacy_fields_disable_blank_creation_and_explain_migration(self) -> None:
        self.page.set_creation_context(legacy_fields=("sync.model", "batch.model"))
        self.page.load({})

        self.assertFalse(self.page.create_btn.isEnabled())
        self.assertTrue(self.page.remove_btn.isHidden())
        self.assertTrue(self.page.profiles_group.isHidden())
        self.assertEqual(self.page.notice_group.title(), "需先迁移旧模型配置")
        self.assertIn("诊断与运行日志", self.page.hint_label.text())
        self.assertIn("完整脚本路径", self.page.hint_label.text())
        self.assertIn("preview / migrate", self.page.hint_label.text())
        self.assertTrue(self.page.legacy_fields_label.isHidden())
        self.assertIn("2 项", self.page.legacy_fields_toggle.text())
        self.page.legacy_fields_toggle.click()
        self.assertFalse(self.page.legacy_fields_label.isHidden())
        self.assertIn("sync.model", self.page.legacy_fields_label.text())
        self.assertIn("batch.model", self.page.legacy_fields_label.text())
        self.page._refresh_all()
        self.assertTrue(self.page.legacy_fields_toggle.isChecked())
        self.assertFalse(self.page.legacy_fields_label.isHidden())
        self.page.load({})
        self.assertTrue(self.page.legacy_fields_toggle.isChecked())
        self.assertFalse(self.page.legacy_fields_label.isHidden())
        self.assertEqual(self.page.collect(), {})

        # The read-only detection result is never part of the saved config.
        self.page.load({"model_routing": migrated_section()})
        self.assertFalse(self.page.legacy_fields_toggle.isChecked())
        self.assertTrue(self.page.legacy_fields_label.isHidden())
        self.assertEqual(set(self.page.collect()), {"model_routing"})
        self.page.load({})
        self.assertFalse(self.page.legacy_fields_toggle.isChecked())
        self.assertTrue(self.page.legacy_fields_label.isHidden())

    def test_invalid_raw_model_routing_is_not_treated_as_empty(self) -> None:
        for raw in ("broken", ["broken"], 7, False):
            with self.subTest(raw=raw):
                self.page.load({"model_routing": raw})

                self.assertFalse(self.page.create_btn.isEnabled())
                self.assertTrue(self.page.remove_btn.isEnabled())
                self.assertFalse(self.page.remove_btn.isHidden())
                self.assertTrue(self.page.profiles_group.isHidden())
                self.assertEqual(self.page.collect(), {"model_routing": raw})
                issues = self.page.validate()
                self.assertEqual(len(issues), 1)
                self.assertIn("model_routing", issues[0].message)

    def test_remove_invalid_raw_is_explicit_and_reversible(self) -> None:
        raw = ["broken"]
        self.page.load({"model_routing": raw})
        self.assertEqual(len(self.page.validate()), 1)

        self.page.remove_btn.click()

        self.assertEqual(self.page.collect(), {"model_routing": None})
        self.assertEqual(self.page.validate(), [])
        self.assertEqual(self.page.notice_group.title(), "等待保存移除操作")
        self.assertIn("不会修复", self.messages[-1])

        self.page.reset()
        self.assertEqual(self.page.collect(), {"model_routing": raw})
        self.assertEqual(len(self.page.validate()), 1)

    def test_created_section_is_discarded_by_reset(self) -> None:
        self.page.load({})
        self.page.create_btn.click()
        self.assertIn("model_routing", self.page.collect())

        self.page.reset()

        self.assertEqual(self.page.collect(), {})

    def test_existing_section_blocks_create_even_with_legacy_context(self) -> None:
        self.page.set_creation_context(legacy_fields=("sync.model",))
        self.page.load({"model_routing": migrated_section()})

        self.assertFalse(self.page.create_btn.isEnabled())
        self.assertEqual(
            self.page.collect()["model_routing"]["schema_version"],
            1,
        )

    def test_invalid_integer_keeps_other_capability_overrides(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id
        self.page._context_edits["context_limit_tokens"].setText("abc")
        combo = self.page._capability_combos["usage_stats"]
        combo.setCurrentIndex(combo.findData(False))

        self.page._on_profile_capabilities_changed()

        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertIs(overrides["usage_stats"], False)
        self.assertNotIn("context_limit_tokens", overrides)
        self.assertTrue(self.messages)
        self.assertIn("上下文上限 tokens", self.messages[-1])
        self.assertNotIn("context_limit_tokens", self.messages[-1])

    def test_missing_primary_profile_requires_explicit_selection(self) -> None:
        section = migrated_section()
        section["defaults"]["primary_profile_id"] = ""
        section["defaults"]["execution_strategy"] = "gemini_batch"
        self.page.load({"model_routing": section})

        # Loading never rewrites the config; the user must pick a profile.
        collected = self.page.collect()["model_routing"]
        self.assertEqual(collected["defaults"]["primary_profile_id"], "")
        self.assertEqual(self.page.default_profile_combo.currentData(), "")
        self.assertTrue(self.page.validate())

        target_row = next(
            row
            for row in range(self.page.default_profile_combo.count())
            if self.page.default_profile_combo.itemData(row)
        )
        self.page.default_profile_combo.setCurrentIndex(target_row)

        collected = self.page.collect()["model_routing"]
        self.assertTrue(collected["defaults"]["primary_profile_id"])
        self.assertEqual(self.page.validate(), [])

    def test_invalid_integer_keeps_existing_numeric_override(self) -> None:
        section = migrated_section()
        profile_id = section["defaults"]["primary_profile_id"]
        section["profiles"][profile_id]["capability_overrides"] = {
            "context_limit_tokens": 4096,
        }
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(self.page.profiles_list.count())
                if str(self.page.profiles_list.item(row).data(256)) == profile_id
            )
        )
        self.page._context_edits["context_limit_tokens"].setText("abc")
        combo = self.page._capability_combos["usage_stats"]
        combo.setCurrentIndex(combo.findData(False))

        self.page._on_profile_capabilities_changed()

        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertEqual(overrides["context_limit_tokens"], 4096)
        self.assertIs(overrides["usage_stats"], False)
        self.assertTrue(self.messages)

    def test_adapter_change_requires_explicit_strategy_reselection(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Google Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
            credential_env_name="GEMINI_API_KEY",
        )
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id="google-gemini",
            model="gemini-3.5-flash",
        )
        section = editor.set_defaults(
            section,
            primary_profile_id="gemini-main",
            execution_strategy="gemini_batch",
        )
        self.page.load({"model_routing": section})
        self.page.providers_list.setCurrentRow(0)

        litellm_index = self.page.provider_adapter_combo.findData("litellm")
        self.page.provider_adapter_combo.setCurrentIndex(litellm_index)

        # The adapter edit invalidates the stored strategy, but the page does
        # not rewrite it; validation reports it and the placeholder forces an
        # explicit re-selection.
        defaults = self.page.collect()["model_routing"]["defaults"]
        self.assertEqual(defaults["execution_strategy"], "gemini_batch")
        self.assertEqual(self.page.default_strategy_combo.currentData(), "")
        self.assertTrue(self.page.validate())

        sync_index = self.page.default_strategy_combo.findData("sync")
        self.page.default_strategy_combo.setCurrentIndex(sync_index)

        defaults = self.page.collect()["model_routing"]["defaults"]
        self.assertEqual(defaults["execution_strategy"], "sync")

    def test_remove_section_is_an_explicit_reversible_action(self) -> None:
        original = migrated_section()
        self.page.load({"model_routing": original})

        self.page.remove_btn.click()
        self.assertEqual(self.page.collect(), {"model_routing": None})
        self.assertTrue(self.page.create_btn.isEnabled())

        self.page.reset()
        self.assertEqual(self.page.collect()["model_routing"], original)

    def test_unknown_fields_survive_edits(self) -> None:
        section = migrated_section()
        section["future_top_level"] = {"keep": True}
        profile_id = next(iter(section["profiles"]))
        section["profiles"][profile_id]["future_profile_field"] = "keep"
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(self.page.profiles_list.count())
                if str(self.page.profiles_list.item(row).data(256)) == profile_id
            )
        )

        self.page.profile_label_edit.setText("Renamed")
        self.page._on_profile_fields_changed()

        collected = self.page.collect()["model_routing"]
        self.assertEqual(collected["future_top_level"], {"keep": True})
        self.assertEqual(
            collected["profiles"][profile_id]["future_profile_field"],
            "keep",
        )
        self.assertEqual(collected["profiles"][profile_id]["label"], "Renamed")

    def test_add_copy_delete_profile(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        before = self.page.profiles_list.count()

        self.page.profiles_add_btn.click()
        self.assertEqual(self.page.profiles_list.count(), before + 1)
        added_id = self.page._selected_profile_id

        self.page.profiles_copy_btn.click()
        self.assertEqual(self.page.profiles_list.count(), before + 2)

        # Added draft is unreferenced, so removal is allowed.
        self.page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(self.page.profiles_list.count())
                if str(self.page.profiles_list.item(row).data(256)) == added_id
            )
        )
        self.page.profiles_delete_btn.click()
        self.assertNotIn(added_id, editor.profile_ids(self.page.collect()["model_routing"]))

    def test_delete_referenced_profile_is_blocked(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})
        default_id = section["defaults"]["primary_profile_id"]
        self.page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(self.page.profiles_list.count())
                if str(self.page.profiles_list.item(row).data(256)) == default_id
            )
        )

        self.page.profiles_delete_btn.click()

        self.assertIn(default_id, editor.profile_ids(self.page.collect()["model_routing"]))
        self.assertTrue(self.messages)

    def test_provider_add_and_in_use_delete_block(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        providers_before = self.page.providers_list.count()
        self.page.provider_add_btn.click()
        self.assertEqual(self.page.providers_list.count(), providers_before + 1)

        used_provider = self.page.providers_list.item(0).data(256)
        self.page.providers_list.setCurrentRow(0)
        self.page.provider_delete_btn.click()
        self.assertIn(used_provider, editor.provider_ids(self.page.collect()["model_routing"]))
        self.assertTrue(self.messages)

    def test_validation_reports_draft_issues_and_clears_when_fixed(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        section = self.page.collect()["model_routing"]
        profile_id = next(iter(section["profiles"]))
        self.page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(self.page.profiles_list.count())
                if str(self.page.profiles_list.item(row).data(256)) == profile_id
            )
        )

        self.page.profile_model_edit.setText("")
        self.page._on_profile_fields_changed()
        issues = self.page.validate()
        self.assertTrue(issues)
        self.assertEqual(issues[0].field_key, "model_routing")

        self.page.profile_model_edit.setText("gemini-3.1-flash-lite")
        self.page._on_profile_fields_changed()
        self.assertEqual(self.page.validate(), [])

    def test_capability_override_writes_and_clears(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id

        combo = self.page._capability_combos["usage_stats"]
        combo.setCurrentIndex(combo.findData(False))
        self.page._on_profile_capabilities_changed()

        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertIs(overrides["usage_stats"], False)

        combo.setCurrentIndex(0)
        self.page._on_profile_capabilities_changed()
        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertNotIn("usage_stats", overrides)

    def test_route_override_toggle(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        widgets = self.page._route_widgets["final_review"]
        profile_id = str(widgets["profile"].currentData())

        widgets["override"].setChecked(True)
        self.page._loading = False
        self.page._on_route_changed("final_review")
        route = self.page.collect()["model_routing"]["routes"]["final_review"]
        self.assertEqual(route["profile_id"], profile_id)

        widgets["override"].setChecked(False)
        self.page._on_route_changed("final_review")
        self.assertNotIn(
            "final_review",
            self.page.collect()["model_routing"].get("routes", {}),
        )

    def test_reset_restores_baseline(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        original = self.page.collect()["model_routing"]
        self.page.profiles_add_btn.click()
        self.assertNotEqual(self.page.collect()["model_routing"], original)

        self.page.reset()
        self.assertEqual(self.page.collect()["model_routing"], original)

    def test_diagnostics_show_capability_sources(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        self.page.profiles_diagnose_btn.click()

        text = self.page.diagnostics_label.text()
        self.assertIn("来源 adapter_default", text)
        self.assertIn("远程 Batch", text)

    def test_default_profile_change_refreshes_strategy_capabilities(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Acme",
            adapter="litellm",
            provider="acme",
            base_url="https://acme.example/v1",
            credential_kind="none",
        )
        section = editor.add_profile(
            section,
            label="Acme Main",
            provider_id="acme",
            model="acme/model-a",
        )
        section = editor.add_provider(
            section,
            label="Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
            credential_env_name="GEMINI_API_KEY",
        )
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id="gemini",
            model="gemini-3.5-flash",
        )
        section = editor.set_defaults(
            section,
            primary_profile_id="acme-main",
            execution_strategy="sync",
        )
        self.page.load({"model_routing": section})

        gemini_index = self.page.default_profile_combo.findData("gemini-main")
        self.page.default_profile_combo.setCurrentIndex(gemini_index)

        batch_index = self.page.default_strategy_combo.findData("gemini_batch")
        item = self.page.default_strategy_combo.model().item(batch_index)
        self.assertTrue(item.isEnabled())

    def test_profile_switch_falls_back_to_supported_strategy(self) -> None:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="Gemini",
            adapter="gemini",
            provider="gemini",
            credential_kind="api_keys_json",
            credential_name="api_keys",
            credential_env_name="GEMINI_API_KEY",
        )
        section = editor.add_profile(
            section,
            label="Gemini Main",
            provider_id="gemini",
            model="gemini-3.5-flash",
        )
        section = editor.add_provider(
            section,
            label="Acme",
            adapter="litellm",
            provider="acme",
            base_url="https://acme.example/v1",
            credential_kind="none",
        )
        section = editor.add_profile(
            section,
            label="Acme Main",
            provider_id="acme",
            model="acme/model-a",
        )
        section = editor.set_defaults(
            section,
            primary_profile_id="gemini-main",
            execution_strategy="gemini_batch",
        )
        self.page.load({"model_routing": section})

        acme_index = self.page.default_profile_combo.findData("acme-main")
        self.page.default_profile_combo.setCurrentIndex(acme_index)

        defaults = self.page.collect()["model_routing"]["defaults"]
        self.assertEqual(defaults["primary_profile_id"], "acme-main")
        self.assertEqual(defaults["execution_strategy"], "sync")
        self.assertEqual(self.messages, [])

    def test_add_profile_without_providers_preserves_existing_section(self) -> None:
        section = editor.empty_section()
        section["future_top_level"] = {"keep": True}
        section["profiles"]["orphan"] = {
            "label": "Orphan",
            "provider_id": "missing-provider",
            "model": "acme/model-a",
        }
        self.page.load({"model_routing": section})

        self.page.profiles_add_btn.click()

        collected = self.page.collect()["model_routing"]
        self.assertEqual(collected["future_top_level"], {"keep": True})
        self.assertIn("orphan", collected["profiles"])
        self.assertTrue(collected["providers"])

    def test_editor_errors_map_to_localized_reasons(self) -> None:
        error = editor.ModelProfilesEditorError(
            "STRATEGY_NOT_SUPPORTED",
            "nope",
            details={
                "profile_id": "acme-main",
                "strategy": "gemini_batch",
                "reason": "missing_gemini_adapter",
            },
        )

        message = self.page._editor_error_message(error)

        self.assertIn("不是 Gemini 直连模型", message)
        self.assertNotIn("missing_gemini_adapter", message)
        self.assertIn("Gemini Batch", message)

    def test_probe_button_wires_immediate_action(self) -> None:
        calls: list[tuple[str, object]] = []
        self.page.set_action_callbacks(
            SettingsPageActions(
                run_immediate=lambda action, payload: (
                    calls.append((action, payload)) or True
                )
            )
        )
        self.page.load({"model_routing": migrated_section()})
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id

        self.page.profiles_probe_btn.click()

        self.assertEqual(
            calls,
            [("probe_profile", {"profile_id": profile_id})],
        )

    def test_probe_report_renders_per_capability_lines(self) -> None:
        self.page.set_probe_report(
            {
                "profile_id": "gemini-main",
                "adapter": "gemini",
                "status": "passed",
                "requests": 1,
                "capabilities": [
                    {"name": "auth", "status": "pass", "detail": ""},
                    {
                        "name": "remote_batch",
                        "status": "declared",
                        "detail": "adapter_default",
                    },
                ],
            }
        )

        text = self.page.diagnostics_label.text()
        self.assertIn("auth：pass", text)
        self.assertIn("remote_batch：declared", text)

    def test_probe_running_reflects_button_label(self) -> None:
        self.page.set_probe_running(True)
        self.assertEqual(
            self.page.profiles_probe_btn.text(),
            "正在测试能力…",
        )
        self.page.set_probe_running(False)
        self.assertEqual(
            self.page.profiles_probe_btn.text(),
            "测试所选 Profile 能力",
        )

    def test_task_running_disables_editing(self) -> None:
        self.page.load({"model_routing": migrated_section()})
        self.page.set_task_running(True)
        self.assertFalse(self.page.profiles_list.isEnabled())
        self.assertFalse(self.page.profiles_add_btn.isEnabled())
        self.page.set_task_running(False)
        self.assertTrue(self.page.profiles_list.isEnabled())



@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class ProfilesAppIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.config = preview_migration(
            json.loads((FIXTURES / "gemini_batch.json").read_text(encoding="utf-8"))
        ).config

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_profiles_page_is_registered_and_loads_config(self) -> None:
        page = self.window._settings_coordinator.ensure_page("profiles")
        self.assertIsNotNone(page)
        self.assertIs(self.window._profiles_page(), page)
        self.assertIn("settings_profiles", self.window._settings_page_bodies)
        self.assertEqual(
            self.window._settings_page_bodies["settings_profiles"].property(
                "settingsPage"
            ),
            "settings_profiles",
        )

        self.window.state.load_translator_config = lambda: self.config  # type: ignore[method-assign]
        self.window._load_config_to_ui(
            refresh_task_gates=False,
            pages={"profiles"},
        )
        self.assertGreater(page.profiles_list.count(), 0)

        snapshot = self.window._current_config_ui_snapshot()
        self.assertEqual(
            snapshot["model_routing"]["defaults"],
            self.config["model_routing"]["defaults"],
        )

    def test_legacy_config_load_blocks_blank_creation(self) -> None:
        legacy_config = json.loads(
            (FIXTURES / "gemini_batch.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("model_routing", legacy_config)

        self.window.state.load_translator_config = lambda: legacy_config  # type: ignore[method-assign]
        page = self.window._settings_coordinator.ensure_page("profiles")
        self.window._load_config_to_ui(
            refresh_task_gates=False,
            pages={"profiles"},
        )

        self.assertFalse(page.create_btn.isEnabled())
        self.assertIn("sync.model", page.legacy_fields_label.text())
        self.assertTrue(page.profiles_group.isHidden())
        self.assertEqual(page.collect(), {})

    def test_probe_action_runs_cli_and_renders_report(self) -> None:
        class _Runner:
            def __init__(self) -> None:
                self.calls: list[tuple[Path, list[str]]] = []

            def run(self, script, args):
                self.calls.append((Path(script), list(args)))
                return True

        self.window.runner = _Runner()
        self.window.state.get_batch_script_path = lambda: Path(  # type: ignore[method-assign]
            "C:/tool/gemini_translate_batch.py"
        )
        self.window.state.load_translator_config = lambda: self.config  # type: ignore[method-assign]
        page = self.window._settings_coordinator.ensure_page("profiles")
        self.assertIsNotNone(page)
        self.window._load_config_to_ui(
            refresh_task_gates=False,
            pages={"profiles"},
        )
        page.profiles_list.setCurrentRow(0)
        profile_id = page._selected_profile_id

        with mock.patch("gui_qt.app.message_box_question", return_value="yes"):
            handled = self.window._run_settings_immediate(
                "probe_profile",
                {"profile_id": profile_id},
            )

        self.assertTrue(handled)
        script, args = self.window.runner.calls[0]
        self.assertEqual(script, Path("C:/tool/gemini_translate_batch.py"))
        self.assertEqual(args[:2], ["profiles-probe", "--profile"])
        self.assertEqual(args[2], profile_id)
        self.assertIn("--acknowledge-billable-request", args)

        report = {
            "profile_id": profile_id,
            "adapter": "gemini",
            "status": "passed",
            "requests": 1,
            "capabilities": [
                {"name": "auth", "status": "pass", "detail": ""},
            ],
        }
        envelope = cli_contract.success_envelope(
            "profiles-probe",
            status="passed",
            result=report,
        )
        self.window._profile_probe_output_lines = [json.dumps(envelope)]
        self.window._on_finished(0)

        self.assertIn("auth：pass", page.diagnostics_label.text())
        self.assertEqual(self.window._active_command, "")

    def test_coordinator_collects_profile_edits(self) -> None:
        page = self.window._settings_coordinator.ensure_page("profiles")
        self.assertIsNotNone(page)
        profile_id = next(iter(self.config["model_routing"]["profiles"]))
        page.load({"model_routing": self.config["model_routing"]})
        page.profiles_list.setCurrentRow(
            next(
                row
                for row in range(page.profiles_list.count())
                if str(page.profiles_list.item(row).data(256)) == profile_id
            )
        )
        page.profile_label_edit.setText("Edited Label")
        page._on_profile_fields_changed()

        collected = self.window._settings_coordinator.collect()
        self.assertEqual(
            collected["model_routing"]["profiles"][profile_id]["label"],
            "Edited Label",
        )


@gui_test_support.skip_unless_gui(ProfilesSettingsPage is None, IMPORT_ERROR)
class ProfilesPageDirectAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.messages: list[str] = []
        self.page = ProfilesSettingsPage(
            actions=SettingsPageActions(show_status=self.messages.append)
        )
        self.page.load({"model_routing": migrated_section()})
        self.page.providers_list.setCurrentRow(0)

    def tearDown(self) -> None:
        self.page.widget.deleteLater()

    def _apply_openai_preset(self) -> str:
        provider_id = self.page._selected_provider_id
        index = self.page.provider_preset_combo.findData("openai")
        self.page.provider_preset_combo.setCurrentIndex(index)
        return provider_id

    def test_provider_preset_fills_direct_connection(self) -> None:
        provider_id = self._apply_openai_preset()
        provider = self.page.collect()["model_routing"]["providers"][provider_id]
        self.assertEqual(provider["adapter"], "openai_compatible")
        self.assertEqual(provider["provider"], "openai")
        self.assertEqual(provider["base_url"], "https://api.openai.com/v1")
        self.assertEqual(
            provider["credential_ref"]["env_name"], "OPENAI_API_KEY"
        )
        self.assertEqual(provider["extra_headers"], {})

    def test_provider_preset_applies_mode_to_unset_profile(self) -> None:
        provider_id = self._apply_openai_preset()
        profiles = [
            profile
            for profile in editor.editor_view(
                self.page.collect()["model_routing"]
            )["profiles"]
            if profile["provider_id"] == provider_id
        ]
        self.assertTrue(profiles)
        for profile in profiles:
            self.assertEqual(
                profile["capability_overrides"]["structured_output"],
                {"mode": "strict_json_schema"},
            )

    def test_provider_preset_keeps_explicit_profile_mode(self) -> None:
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id
        combo = self.page.profile_structured_output_combo
        combo.setCurrentIndex(combo.findData("json_object"))
        self.page._on_profile_capabilities_changed()

        self._apply_openai_preset()

        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertEqual(overrides["structured_output"], {"mode": "json_object"})

    def test_provider_extra_headers_round_trip(self) -> None:
        provider_id = self._apply_openai_preset()
        self.page.provider_extra_headers_edit.setText('{"X-Trace": "abc"}')
        self.page._on_provider_fields_changed()
        provider = self.page.collect()["model_routing"]["providers"][provider_id]
        self.assertEqual(provider["extra_headers"], {"X-Trace": "abc"})

    def test_invalid_extra_headers_does_not_write(self) -> None:
        provider_id = self._apply_openai_preset()
        self.page.provider_extra_headers_edit.setText("{not json")
        self.page._on_provider_fields_changed()
        provider = self.page.collect()["model_routing"]["providers"][provider_id]
        self.assertEqual(provider["extra_headers"], {})
        self.assertTrue(self.messages)

    def test_non_string_extra_header_value_does_not_write(self) -> None:
        provider_id = self._apply_openai_preset()
        self.page.provider_extra_headers_edit.setText('{"X-Test": null}')
        self.page._on_provider_fields_changed()
        provider = self.page.collect()["model_routing"]["providers"][provider_id]
        self.assertEqual(provider["extra_headers"], {})
        self.assertTrue(self.messages)

    def _direct_section(self) -> dict:
        section = editor.empty_section()
        section = editor.add_provider(
            section,
            label="OpenAI",
            adapter="openai_compatible",
            provider="openai",
            base_url="https://api.example/v1",
            models_url="https://api.example/v1/models",
            credential_kind="none",
        )
        provider_id = editor.provider_ids(section)[0]
        section = editor.add_profile(
            section,
            label="OpenAI GPT",
            provider_id=provider_id,
            model="gpt-4.1-mini",
        )
        profile_id = editor.profile_ids(section)[0]
        return editor.set_defaults(
            section,
            primary_profile_id=profile_id,
            execution_strategy="sync",
        )

    def test_model_catalog_button_invokes_immediate_action(self) -> None:
        calls: list[tuple[str, dict]] = []
        page = ProfilesSettingsPage(
            actions=SettingsPageActions(
                show_status=self.messages.append,
                run_immediate=lambda action, payload: calls.append((action, dict(payload))) or True,
            )
        )
        self.addCleanup(page.widget.deleteLater)
        section = self._direct_section()
        page.load({"model_routing": section})
        page.profiles_list.setCurrentRow(0)

        self.assertTrue(page.profile_model_catalog_btn.isEnabled())
        page.profile_model_catalog_btn.click()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "list_profile_models")
        self.assertEqual(
            calls[0][1]["profile_id"],
            editor.profile_ids(section)[0],
        )

    def test_gemini_catalog_is_available_for_generation_and_embedding_profiles(self) -> None:
        calls: list[tuple[str, object]] = []
        page = ProfilesSettingsPage(actions=SettingsPageActions(
            run_immediate=lambda name, payload: calls.append((name, payload)) or True,
        ))
        self.addCleanup(page.widget.deleteLater)
        section = migrated_section()
        page.load({"model_routing": section})
        kinds = set()
        for row in range(page.profiles_list.count()):
            page.profiles_list.setCurrentRow(row)
            profile_id = page._selected_profile_id
            kinds.add(section["profiles"][profile_id].get("purpose", "generation"))
            self.assertTrue(page.profile_model_catalog_btn.isEnabled())
            page.profile_model_catalog_btn.click()
            self.assertEqual(calls[-1], ("list_profile_models", {"profile_id": profile_id}))
        self.assertEqual(kinds, {"generation", "embedding"})

    def test_gemini_catalog_selection_still_needs_explicit_model_selection(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id
        original = self.page.collect()
        self.page.set_model_catalog(["gemini-4-argon", "gemini-3.8-flash"], profile_id=profile_id)
        self.assertEqual(self.page.collect(), original)
        combo = self.page.profile_model_combo
        combo.setCurrentIndex(combo.findText("gemini-4-argon"))
        self.assertEqual(self.page.collect()["model_routing"]["profiles"][profile_id]["model"], "gemini-4-argon")

    def test_model_catalog_populates_and_selects_model(self) -> None:
        section = self._direct_section()
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id

        self.page.set_model_catalog(
            ["gpt-4.1", "gpt-4.1-mini"],
            source="openai",
            profile_id=profile_id,
        )
        combo = self.page.profile_model_combo
        self.assertTrue(combo.isEditable())
        self.assertEqual(combo.count(), 2)
        combo.setCurrentIndex(combo.findText("gpt-4.1"))

        collected = self.page.collect()["model_routing"]["profiles"][profile_id]
        self.assertEqual(collected["model"], "gpt-4.1")
        self.assertTrue(self.messages)

        # Refreshing the same connection retains its fetched suggestions.
        self.page._populate_profile_editor()
        self.assertEqual(combo.count(), 2)
        self.assertEqual(combo.currentText(), "gpt-4.1")

    def test_model_catalog_ignores_result_for_other_profile(self) -> None:
        section = self._direct_section()
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(0)

        self.page.set_model_catalog(
            ["gpt-4.1"],
            source="供应商模型列表",
            profile_id="some-other-profile",
        )

        self.assertEqual(self.page.profile_model_combo.count(), 1)
        self.assertEqual(self.page.profile_model_combo.currentText(), "gpt-4.1-mini")
        self.assertTrue(
            any("some-other-profile" in message for message in self.messages)
        )

    def test_offline_dropdown_supports_search_selection_and_unknown_ids(self) -> None:
        section = migrated_section()
        self.page.load({"model_routing": section})
        original = self.page.collect()
        combo = self.page.profile_model_combo
        self.assertTrue(combo.isEditable())
        self.assertGreater(combo.count(), 1)
        self.assertEqual(combo.completer().filterMode().name, "MatchContains")
        self.assertEqual(self.page.collect(), original)
        profile_id = self.page._selected_profile_id
        combo.setCurrentIndex(combo.findText("gemini-4-argon"))
        self.assertEqual(self.page.collect()["model_routing"]["profiles"][profile_id]["model"], "gemini-4-argon")
        self.page.profile_model_edit.setText("future-unlisted-model")
        self.page.profile_model_edit.editingFinished.emit()
        self.assertEqual(self.page.collect()["model_routing"]["profiles"][profile_id]["model"], "future-unlisted-model")
        self.page.set_model_catalog(["another-model"], profile_id=profile_id)
        self.assertEqual(combo.currentText(), "future-unlisted-model")

    def test_litellm_dropdown_and_refresh_use_the_existing_connection(self) -> None:
        section = migrated_section("litellm_builtin")
        self.page.load({"model_routing": section})
        self.page.profiles_list.setCurrentRow(next(
            row for row in range(self.page.profiles_list.count())
            if self.page.profiles_list.item(row).data(256) == "legacy-sync"
        ))
        self.assertTrue(self.page.profile_model_catalog_btn.isEnabled())
        self.assertTrue(self.page.profile_model_combo.isEnabled())
        self.assertEqual(self.page.collect(), {"model_routing": section})

    def test_changed_connection_cannot_receive_a_late_catalog_or_reuse_old_choices(self) -> None:
        from providers import profile_model_catalog

        section = self._direct_section()
        self.page.load({"model_routing": section})
        profile_id = self.page._selected_profile_id
        identity = profile_model_catalog.connection_identity(section, profile_id)
        self.page.set_model_catalog(["old-host-only-model"], profile_id=profile_id, connection_id=identity)
        provider_id = self.page._selected_provider_id
        changed = editor.update_provider(section, provider_id, base_url="https://new.example/v1", models_url="https://new.example/v1/models")
        self.page.load({"model_routing": changed})
        self.assertEqual(self.page.profile_model_combo.findText("old-host-only-model"), -1)
        self.page.set_model_catalog(["late-model"], profile_id=profile_id, connection_id=identity)
        self.assertEqual(self.page.profile_model_combo.findText("late-model"), -1)
        self.assertEqual(self.page.collect(), {"model_routing": changed})

    def test_model_catalog_running_and_error_feedback(self) -> None:
        page = ProfilesSettingsPage(
            actions=SettingsPageActions(show_status=self.messages.append)
        )
        self.addCleanup(page.widget.deleteLater)
        section = self._direct_section()
        page.load({"model_routing": section})
        page.profiles_list.setCurrentRow(0)

        page.set_model_catalog_running(True)
        self.assertEqual(
            page.profile_model_catalog_btn.text(),
            MODEL_PROFILES_PAGE_COPY["model_catalog_running"],
        )
        page.set_model_catalog_running(False)
        self.assertEqual(
            page.profile_model_catalog_btn.text(),
            MODEL_PROFILES_PAGE_COPY["model_catalog_button"],
        )

        page.set_model_catalog_error(
            "MODEL_CATALOG_TIMEOUT",
            profile_id=page._selected_profile_id,
        )
        self.assertTrue(
            any("MODEL_CATALOG_TIMEOUT" in message for message in self.messages)
        )

        # An error for a different profile must not surface on this page.
        self.messages.clear()
        page.set_model_catalog_error(
            "MODEL_CATALOG_TIMEOUT",
            profile_id="some-other-profile",
        )
        self.assertFalse(
            any("MODEL_CATALOG_TIMEOUT" in message for message in self.messages)
        )

    def test_structured_output_mode_round_trip(self) -> None:
        self.page.profiles_list.setCurrentRow(0)
        profile_id = self.page._selected_profile_id
        combo = self.page.profile_structured_output_combo
        combo.setCurrentIndex(combo.findData("json_object"))
        self.page._on_profile_capabilities_changed()
        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertEqual(overrides["structured_output"], {"mode": "json_object"})

        combo.setCurrentIndex(0)
        self.page._on_profile_capabilities_changed()
        overrides = self.page.collect()["model_routing"]["profiles"][profile_id][
            "capability_overrides"
        ]
        self.assertNotIn("structured_output", overrides)


if __name__ == "__main__":
    unittest.main()
