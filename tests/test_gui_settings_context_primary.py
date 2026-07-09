"""Settings · 上下文 primary switches (GUI IA P2b / #165)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gui_qt.settings_schema import (
    ADVANCED_SETTING_FIELD_BY_KEY,
    CONTEXT_PRIMARY_SETTING_KEYS,
    apply_advanced_settings,
    context_primary_setting_fields,
    grouped_advanced_fields,
    read_advanced_settings,
    recommended_advanced_settings,
)

try:
    from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton

    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    QCheckBox = object  # type: ignore[misc,assignment]
    QPushButton = object  # type: ignore[misc,assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


class ContextPrimarySchemaTests(unittest.TestCase):
    def test_primary_keys_have_schema_fields(self) -> None:
        keys = {field.key for field in context_primary_setting_fields()}
        self.assertEqual(keys, set(CONTEXT_PRIMARY_SETTING_KEYS))

    def test_advanced_groups_exclude_primary_when_requested(self) -> None:
        with_primary = {
            f.key
            for _, fields in grouped_advanced_fields(include_context_primary=True)
            for f in fields
        }
        without = {
            f.key
            for _, fields in grouped_advanced_fields(include_context_primary=False)
            for f in fields
        }
        for key in CONTEXT_PRIMARY_SETTING_KEYS:
            self.assertIn(key, with_primary)
            self.assertNotIn(key, without)

    def test_apply_preserves_unknown_and_writes_primary(self) -> None:
        config: dict = {
            "sync": {"rag": {"enabled": False}, "custom_keep": 42},
            "batch": {"story_memory": {"enabled": False}},
            "orphan": "stay",
        }
        values = recommended_advanced_settings()
        values["sync_rag_enabled"] = True
        values["batch_story_memory_enabled"] = True
        values["sync_story_memory_enabled"] = True
        # Fill remaining keys from defaults
        for field in ADVANCED_SETTING_FIELD_BY_KEY.values():
            values.setdefault(field.key, field.default)
        values["game_root"] = "C:/Games/Demo/work"
        out = apply_advanced_settings(config, values)
        self.assertTrue(out["sync"]["rag"]["enabled"])
        self.assertTrue(out["batch"]["story_memory"]["enabled"])
        self.assertTrue(out["sync"]["story_memory"]["enabled"])
        self.assertEqual(out["sync"]["custom_keep"], 42)
        self.assertEqual(out["orphan"], "stay")
        # Round-trip read
        read_back = read_advanced_settings(out)
        self.assertTrue(read_back["sync_rag_enabled"])
        self.assertTrue(read_back["batch_story_memory_enabled"])


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiContextPrimaryUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_primary_widgets_only_on_context_page(self) -> None:
        for key in CONTEXT_PRIMARY_SETTING_KEYS:
            self.assertIn(key, self.window._advanced_setting_widgets)
        # Advanced page groups must not recreate the same keys as separate widgets.
        # We only registered once per key.
        for key in CONTEXT_PRIMARY_SETTING_KEYS:
            widget = self.window._advanced_setting_widgets[key]
            # Parent chain should include settings_context, not settings_advanced.
            names: list[str] = []
            parent = widget.parentWidget()
            while parent is not None:
                names.append(parent.objectName() or "")
                parent = parent.parentWidget()
            joined = " ".join(names)
            self.assertIn("settings_context", joined, msg=f"{key} not under context page: {names}")
            self.assertNotIn(
                "settings_advanced",
                joined,
                msg=f"{key} incorrectly under advanced page: {names}",
            )

    def test_restore_recommended_sets_primary_switches(self) -> None:
        widgets = self.window._advanced_setting_widgets
        for key in CONTEXT_PRIMARY_SETTING_KEYS:
            w = widgets[key]
            if isinstance(w, QCheckBox):
                w.setChecked(not ADVANCED_SETTING_FIELD_BY_KEY[key].recommended_value)
        self.window._on_restore_recommended_config()
        defaults = recommended_advanced_settings()
        for key in CONTEXT_PRIMARY_SETTING_KEYS:
            w = widgets[key]
            if isinstance(w, QCheckBox):
                self.assertEqual(w.isChecked(), bool(defaults[key]), msg=key)

    def test_diagnostics_tab_renamed_and_slim(self) -> None:
        labels = [
            self.window.tab_widget.tabText(i)
            for i in range(self.window.tab_widget.count())
        ]
        self.assertIn("诊断与工具", labels)
        self.assertNotIn("诊断日志", labels)
        diag = self.window.diagnostics_action_panel
        texts = {b.text() for b in diag.findChildren(QPushButton)}
        self.assertIn("翻译 A/B 对比", texts)
        self.assertIn("刷新上下文", texts)
        self.assertIn("清空日志", texts)
        self.assertNotIn("合并到 glossary", texts)
        self.assertNotIn("试跑样本请求", texts)
        self.assertNotIn("拆分翻译包", texts)
        # Hidden attribute retained for enable helpers.
        self.assertTrue(self.window.keyword_merge_btn.isHidden())


if __name__ == "__main__":
    unittest.main()
