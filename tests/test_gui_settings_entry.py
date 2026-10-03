"""Behavior and window-size contracts for issue #540 phases A and B."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests import gui_test_support
import model_profiles_editor as editor

try:
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QWidget
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    from gui_qt.app import MainWindow, _SETTINGS_PAGE_SPECS
    from gui_qt.theme import apply_theme
    IMPORT_ERROR = None


def example_config() -> dict:
    """Original offline fixture: two generation profiles, no credential values."""
    section = editor.initial_section(
        provider_label="示例连接", profile_label="示例主模型", model="example-main",
    )
    section = editor.add_profile(
        section, label="示例备用模型", provider_id=editor.provider_ids(section)[0],
        model="example-alternate",
    )
    return {"model_routing": section}


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class SettingsEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        # processEvents() alone leaves DeferredDelete events pending when no
        # Qt exec() loop runs. Drain closed windows before a global QSS change
        # can repolish hundreds of stale trees from preceding test modules.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        stylesheet = self.app.styleSheet()
        self.addCleanup(self.app.setStyleSheet, stylesheet)
        apply_theme(self.app, Path(__file__).parents[1] / "gui_qt/resources", "light")
        self.window = MainWindow()
        self.config = example_config()
        self.load = mock.patch.object(
            self.window.state, "load_translator_config",
            side_effect=lambda: copy.deepcopy(self.config),
        )
        self.load.start()
        self.addCleanup(self.load.stop)
        self.save = mock.patch.object(self.window.state, "save_translator_config")
        self.save_mock = self.save.start()
        self.addCleanup(self.save.stop)
        self.window.resize(960, 640)
        self.window.show()
        self.window._focus_settings_section("profiles")
        self.process()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.process()

    def process(self) -> None:
        for _ in range(8):
            self.app.processEvents()

    def assert_in_viewport(self, widget, viewport) -> None:
        top_left = widget.mapTo(viewport, QPoint(0, 0))
        self.assertTrue(viewport.rect().contains(top_left), widget.objectName())
        self.assertTrue(
            viewport.rect().contains(top_left + QPoint(widget.width() - 1, widget.height() - 1)),
            widget.objectName(),
        )

    def test_theme_setup_does_not_restyle_closed_pending_windows(self) -> None:
        restyled = []

        class ClosingWindow(QWidget):
            def event(self, event):
                if event.type() == QEvent.Type.StyleChange:
                    restyled.append(True)
                return super().event(event)

        previous = ClosingWindow()
        previous.close()
        previous.deleteLater()
        other = SettingsEntryTests("test_defaults_are_editable_at_top_before_provider_details")
        try:
            other.setUp()
            self.assertEqual(restyled, [])
        finally:
            other.tearDown()
            other.doCleanups()

    def test_defaults_are_editable_at_top_before_provider_details(self) -> None:
        page = self.window._profiles_page()
        self.assertEqual(page.widget.verticalScrollBar().value(), 0)
        for control in (page.default_profile_combo, page.default_strategy_combo):
            self.assert_in_viewport(control, page.widget.viewport())
        self.assertLess(page.defaults_group.y(), page.profiles_group.mapTo(page.body, QPoint(0, 0)).y())
        self.assertLess(page.defaults_group.y(), page.notice_group.y())
        self.assert_in_viewport(page.readiness_label, page.widget.viewport())
        self.assertIn("未验证", page.readiness_label.text())
        alternate = editor.profile_ids(self.config["model_routing"])[1]
        page.default_profile_combo.setCurrentIndex(page.default_profile_combo.findData(alternate))
        page.default_strategy_combo.setCurrentIndex(page.default_strategy_combo.findData("sync"))
        self.assertEqual(page.collect()["model_routing"]["defaults"], {
            "primary_profile_id": alternate, "execution_strategy": "sync",
        })
        self.save_mock.assert_not_called()

    def test_all_categories_select_real_pages_and_sync_with_direct_navigation(self) -> None:
        selector = self.window.settings_category_combo
        self.assertEqual(selector.count(), 11)
        self.assert_in_viewport(selector, self.window._config_tab)
        selector.showPopup()
        self.process()
        view = selector.view()
        QTest.keyClick(view, Qt.Key.Key_End)
        QTest.keyClick(view, Qt.Key.Key_Return)
        self.process()
        self.assertEqual(selector.currentData(), "advanced")
        self.assertEqual(self.window._settings_coordinator.active_key, "advanced")
        for index, (key, label, _builder) in enumerate(_SETTINGS_PAGE_SPECS):
            selector.setCurrentIndex(index)
            self.process()
            self.assertEqual(selector.currentData(), key)
            self.assertEqual(selector.currentText(), label)
            self.assertEqual(self.window.settings_nav.currentRow(), index)
            self.assertIs(
                self.window.settings_stack.currentWidget(),
                self.window._settings_coordinator.page(key).widget,
            )
        self.window._focus_settings_section("profiles")
        self.assertEqual(selector.currentData(), "profiles")
        self.window.settings_nav.setCurrentRow(self.window._settings_nav_rows["models"])
        self.assertEqual(selector.currentData(), "models")
        self.save_mock.assert_not_called()

    def test_category_selector_reflows_after_font_and_theme_changes_and_keeps_selection(self) -> None:
        from tests.test_gui_control_layout_audit import _combo_natural_width

        selector = self.window.settings_category_combo
        original_font = selector.font()
        selected_key = selector.currentData()
        page = self.window._profiles_page()
        original_config = page.collect()
        for extra_pixels in (4, 12, 0):
            font = selector.font()
            font.setPixelSize(original_font.pixelSize() + extra_pixels)
            selector.setFont(font)
            self.process()
            self.assertGreaterEqual(selector.width() + 1, _combo_natural_width(selector))
            self.assert_in_viewport(selector, self.window._config_tab)
            self.assertEqual(selector.currentData(), selected_key)
            self.assertIs(self.window._profiles_page(), page)
            self.assertEqual(page.collect(), original_config)
        for theme in ("dark", "light"):
            apply_theme(self.app, Path(__file__).parents[1] / "gui_qt/resources", theme)
            self.process()
            self.assertGreaterEqual(selector.width() + 1, _combo_natural_width(selector))
            self.assert_in_viewport(selector, self.window._config_tab)
            self.assertEqual(selector.currentData(), selected_key)
            self.assertIs(self.window._profiles_page(), page)
            self.assertEqual(page.collect(), original_config)
        self.save_mock.assert_not_called()

    def test_model_links_resize_and_refresh_preserve_unsaved_edits_and_selection(self) -> None:
        page = self.window._profiles_page()
        alternate = editor.profile_ids(self.config["model_routing"])[1]
        page.profiles_list.setCurrentRow(1)
        page.profile_label_edit.setText("未保存的示例名称")
        page.profile_label_edit.editingFinished.emit()
        edited = page.collect()
        for size in ((1280, 800), (1920, 1080), (960, 640)):
            self.window.resize(*size)
            self.window._focus_settings_section("models")
            self.window._models_page().model_navigation_buttons["litellm"].click()
            self.assertEqual(self.window.settings_category_combo.currentData(), "litellm")
            self.window._litellm_page().model_navigation_buttons["profiles"].click()
            self.assertIs(self.window._profiles_page(), page)
            page._refresh_all()
            self.process()
            self.assertEqual(page._selected_profile_id, alternate)
            self.assertEqual(page.collect(), edited)
            self.assertEqual(self.window.settings_category_combo.currentData(), "profiles")
        self.save_mock.assert_not_called()

    def test_model_list_detail_reflow_keeps_selection_and_unfinished_text(self) -> None:
        page = self.window._profiles_page()
        page.profiles_list.setCurrentRow(1)
        selected = page.profiles_list.currentItem().data(Qt.ItemDataRole.UserRole)
        page.profile_model_edit.setText("original-long-model-id/" * 12)
        page.provider_base_url_edit.setText("https://offline.example/" + "long-path/" * 12)
        page.capabilities_toggle.click()
        for size in ((1920, 1080), (960, 640), (1280, 800), (1920, 1080)):
            self.window.resize(*size)
            self.process()
            list_pos = page.profiles_group.mapTo(page.body, QPoint(0, 0))
            detail_pos = page.profile_editor_group.mapTo(page.body, QPoint(0, 0))
            if size[0] == 1920:
                self.assertGreater(detail_pos.x(), list_pos.x() + page.profiles_group.width())
                self.assertEqual(detail_pos.y(), list_pos.y())
                self.assertLess(page.profile_label_edit.width(), page.profile_model_edit.width())
            elif size[0] == 960:
                self.assertGreater(detail_pos.y(), list_pos.y())
            self.assertEqual(page.profiles_list.currentItem().data(Qt.ItemDataRole.UserRole), selected)
            self.assertEqual(page.profile_model_edit.text(), "original-long-model-id/" * 12)
            self.assertEqual(page.provider_base_url_edit.text(), "https://offline.example/" + "long-path/" * 12)
            self.assertTrue(page.capabilities_toggle.isChecked())
            page.widget.ensureWidgetVisible(page.profile_model_edit)
            self.process()
            self.assert_in_viewport(page.profile_model_edit, page.widget.viewport())
            page.widget.ensureWidgetVisible(page.provider_base_url_edit)
            self.process()
            self.assert_in_viewport(page.provider_base_url_edit, page.widget.viewport())
        self.save_mock.assert_not_called()

    def test_load_states_remain_distinct_and_legacy_expansion_survives_navigation_reload(self) -> None:
        page = self.window._profiles_page()
        for config in ({}, {"model_routing": []}, {"model_routing": {}}, example_config()):
            self.config = config
            self.window._load_config_to_ui(refresh_task_gates=False, pages={"profiles"})
            expected = {"model_routing": config["model_routing"]} if "model_routing" in config else {}
            self.assertEqual(page.collect(), expected)
            if config.get("model_routing") == []:
                self.assertTrue(page.remove_btn.isVisible())
                self.assert_in_viewport(page.remove_btn, page.widget.viewport())
                page.remove_btn.click()
                self.assertEqual(page.collect(), {"model_routing": None})
                page.reset()
                self.assertEqual(page.collect(), expected)
        self.config = {"sync": {"model": "legacy-example"}, "batch": {}}
        self.window._load_config_to_ui(refresh_task_gates=False, pages={"profiles"})
        page.legacy_fields_toggle.click()
        self.window._focus_settings_section("models")
        self.window._focus_settings_section("profiles")
        self.window._load_config_to_ui(refresh_task_gates=False, pages={"profiles"})
        self.process()
        self.assertTrue(page.legacy_fields_toggle.isChecked())
        self.assertTrue(page.legacy_fields_label.isVisible())
        self.assertEqual(page.collect(), {})
        self.save_mock.assert_not_called()

    def test_save_and_reload_use_host_transaction_keep_page_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            game_root = Path(directory) / "work"
            game_root.mkdir()
            config_path = Path(directory) / "translator_config.json"
            self.config["game_root"] = str(game_root)
            config_path.write_text(json.dumps(self.config), encoding="utf-8")

            def save_config(config):
                config_path.write_text(json.dumps(config), encoding="utf-8")
                self.config = json.loads(config_path.read_text(encoding="utf-8"))

            self.save_mock.side_effect = save_config
            with (
                mock.patch.object(self.window.state, "get_game_root", return_value=game_root),
                mock.patch.object(self.window.state, "normalize_game_root", return_value=(game_root, False)),
            ):
                page = self.window._profiles_page()
                alternate = editor.profile_ids(self.config["model_routing"])[1]
                page.profiles_list.setCurrentRow(1)
                page.providers_list.setCurrentRow(0)
                provider_id = page._selected_provider_id
                page.default_profile_combo.setCurrentIndex(page.default_profile_combo.findData(alternate))
                page.default_strategy_combo.setCurrentIndex(page.default_strategy_combo.findData("sync"))
                expected = page.collect()["model_routing"]
                self.window._focus_settings_section("advanced")
                advanced = self.window._settings_coordinator.page("advanced")
                advanced.field_widgets["sync_chunk_size"].setValue(73)
                advanced.search_edit.setText("没有匹配-original-fixture")
                self.window._focus_settings_section("profiles")
                self.save_mock.assert_not_called()
                self.assertTrue(self.window._on_save_config())
                self.assertEqual(self.save_mock.call_count, 1)
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["model_routing"], expected)
                self.assertEqual(saved["sync"]["chunk_size"], 73)
                self.assertTrue((game_root / "project_context_settings.json").exists())
                self.assertEqual(self.window.settings_category_combo.currentData(), "profiles")
                self.assertEqual(page._selected_profile_id, alternate)
                self.assertEqual(page._selected_provider_id, provider_id)
                self.assertFalse(self.window._config_tab_has_unsaved_changes())
                page.default_strategy_combo.setCurrentIndex(page.default_strategy_combo.findData("gemini_batch"))
                advanced.field_widgets["sync_chunk_size"].setValue(19)
                self.assertTrue(self.window._config_tab_has_unsaved_changes())
                self.window._on_reload_config()
                self.assertIs(self.window._profiles_page(), page)
                self.assertEqual(page.collect()["model_routing"], expected)
                self.assertEqual(advanced.collect()["sync_chunk_size"], 73)
                self.assertEqual(page.default_strategy_combo.currentData(), "sync")
                self.assertEqual(page._selected_profile_id, alternate)
                self.assertEqual(page._selected_provider_id, provider_id)
                self.assertEqual(self.window.settings_category_combo.currentData(), "profiles")
                self.assertFalse(self.window._config_tab_has_unsaved_changes())
                self.assertEqual(self.save_mock.call_count, 1)

    def test_first_save_materialization_preserves_explicit_invalid_raw_removal(self) -> None:
        self.config = {"model_routing": ["invalid-example"]}
        self.window._load_config_to_ui(refresh_task_gates=False, pages={"profiles"})
        page = self.window._profiles_page()
        page.remove_btn.click()
        self.window._ensure_settings_pages_for_config()
        self.assertEqual(page.collect(), {"model_routing": None})
        self.assertEqual(page.validate(), [])
        self.assertEqual(page.notice_group.title(), "等待保存移除操作")
        page.reset()
        self.assertEqual(page.collect(), {"model_routing": ["invalid-example"]})
        self.assertEqual(len(page.validate()), 1)
        self.save_mock.assert_not_called()

    def test_invalid_mapping_readiness_updates_after_explicit_default_repair(self) -> None:
        page = self.window._profiles_page()
        section = copy.deepcopy(self.config["model_routing"])
        section["defaults"]["primary_profile_id"] = ""
        page.load({"model_routing": section})
        self.assertEqual(page.collect()["model_routing"], section)
        self.assertIn("保存会被阻止", page.readiness_label.text())
        self.assertEqual(page.default_profile_combo.currentData(), "")
        primary = editor.profile_ids(section)[0]
        page.default_profile_combo.setCurrentIndex(page.default_profile_combo.findData(primary))
        self.assertEqual(page.validate(), [])
        self.assertIn("校验通过", page.readiness_label.text())
        self.save_mock.assert_not_called()
