"""Unified translation target selector tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.workbench.translation_page import TranslationTargetSection
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    TranslationTargetSection = None  # type: ignore[assignment,misc]
    WorkMode = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from model_routing_migration import preview_migration
from tests import gui_test_support

FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def routing_config(name: str = "gemini_batch") -> dict:
    payload = json.loads((FIXTURES / (name + ".json")).read_text(encoding="utf-8"))
    return preview_migration(payload).config


def selector_payload() -> dict:
    return {
        "profiles": [
            {
                "id": "gemini-main",
                "label": "Gemini Main",
                "model": "gemini-3.5-flash",
                "adapter": "gemini",
                "strategies": ("sync", "gemini_batch"),
                "unsupported": {},
                "is_default": True,
            },
            {
                "id": "litellm-main",
                "label": "LiteLLM Main",
                "model": "openai/gpt-x",
                "adapter": "litellm",
                "strategies": ("sync",),
                "unsupported": {"gemini_batch": "missing_gemini_adapter"},
                "is_default": False,
            },
        ],
        "selected_profile_id": "gemini-main",
        "selected_strategy": "gemini_batch",
    }


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str]]] = []

    def run(self, script, args) -> bool:
        self.calls.append((Path(script), list(args)))
        return True


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class TranslationTargetSectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.section = TranslationTargetSection()
        self.selections: list[tuple[str, str]] = []
        self.section.set_select_callback(
            lambda profile, strategy: self.selections.append((profile, strategy))
        )

    def tearDown(self) -> None:
        self.section.deleteLater()

    def test_profiles_and_strategy_capabilities_render(self) -> None:
        self.section.set_target_choices(selector_payload())

        self.assertEqual(self.section.profile_combo.count(), 2)
        self.assertEqual(self.section.current_selection(), ("gemini-main", "gemini_batch"))
        # LiteLLM profile is present but cannot run gemini_batch.
        index = self.section._index_for(self.section.profile_combo, "litellm-main")
        self.section.profile_combo.setCurrentIndex(index)
        gemini_batch_index = self.section._index_for(
            self.section.strategy_combo, "gemini_batch"
        )
        item = self.section.strategy_combo.model().item(gemini_batch_index)
        self.assertFalse(item.isEnabled())
        self.assertIn("不是 Gemini 直连模型", self.section.hint_label.text())

    def test_selecting_supported_strategy_emits_selection(self) -> None:
        self.section.set_target_choices(selector_payload())
        index = self.section._index_for(self.section.profile_combo, "litellm-main")
        self.section.profile_combo.setCurrentIndex(index)

        sync_index = self.section._index_for(self.section.strategy_combo, "sync")
        self.section.strategy_combo.setCurrentIndex(sync_index)

        self.assertEqual(self.selections[-1], ("litellm-main", "sync"))

    def test_legacy_payload_disables_the_selector(self) -> None:
        self.section.set_target_choices(
            {
                "profiles": [],
                "hint": "旧配置提示",
                "disabled": True,
            }
        )

        self.assertFalse(self.section.profile_combo.isEnabled())
        self.assertFalse(self.section.strategy_combo.isEnabled())
        self.assertEqual(self.section.hint_label.text(), "旧配置提示")


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class TranslationTargetAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.runner = _FakeRunner()
        self.window.runner = self.runner
        self.window.state.load_translator_config = lambda: routing_config()  # type: ignore[method-assign]
        self.window.state.get_game_root = lambda: "C:/game/work"  # type: ignore[method-assign]
        self.window.state.get_batch_script_path = lambda: Path(  # type: ignore[method-assign]
            "C:/tool/gemini_translate_batch.py"
        )
        self.window._doctor_check_completed = True
        self.window._doctor_summary_status = "ready"
        self.window._confirm_unsaved_config_before_workflow = lambda: True  # type: ignore[method-assign]
        self.window._sync_work_modes_requiring_api_key = lambda: frozenset()  # type: ignore[method-assign]
        self.window._append_log = lambda _text: None  # type: ignore[method-assign]
        self.window._clear_log_view = lambda: None  # type: ignore[method-assign]
        self.window._show_workbench_log_drawer = lambda: None  # type: ignore[method-assign]
        self.window._refresh_diagnostics_context = lambda: None  # type: ignore[method-assign]

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_routing_config_populates_both_selector_sections(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.assertTrue(self.window._translation_routing_active)
        for section in self.window._translation_target_sections.values():
            self.assertGreater(section.profile_combo.count(), 0)
        self.assertEqual(
            self.window._selected_translation_profile_id(),
            "legacy-batch",
        )

    def test_strategy_selection_switches_execution_page(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_translation_target_selected("legacy-sync", "sync")

        self.assertEqual(self.window._work_mode, WorkMode.SYNC_TRANSLATION)
        self.assertIs(
            self.window.workbench_stack.currentWidget(),
            self.window.sync_translation_page,
        )

    def test_sync_start_passes_selected_profile(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_start_translation()

        self.assertEqual(
            self.runner.calls[0][1],
            [
                "sync-start",
                "--profile",
                "legacy-batch",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_batch_build_passes_selected_profile(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_start_translation()

        self.assertEqual(
            self.runner.calls[0][1],
            [
                "build",
                "--profile",
                "legacy-batch",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_legacy_config_keeps_default_behavior_without_profile_flag(self) -> None:
        self.window.state.load_translator_config = lambda: {}  # type: ignore[method-assign]
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.assertFalse(self.window._translation_routing_active)
        self.window._on_start_translation()

        self.assertEqual(
            self.runner.calls[0][1],
            ["sync-start", "--output", "json", "--non-interactive"],
        )


if __name__ == "__main__":
    unittest.main()
