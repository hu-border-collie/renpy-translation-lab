"""Unified translation target selector tests (#348 P3)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import cli_contract
from unittest import mock

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


def litellm_primary_config() -> dict:
    """Migrated config whose primary profile can only run sync."""
    config = routing_config("litellm_custom")
    section = config["model_routing"]
    section["defaults"]["primary_profile_id"] = "legacy-sync"
    section["defaults"]["execution_strategy"] = "sync"
    return config


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
        self.assertIn("不是 Gemini 直连模型", item.text())
        self.assertNotIn("missing_gemini_adapter", item.text())
        self.assertIn("不是 Gemini 直连模型", self.section.hint_label.text())
        self.assertNotIn(
            "missing_gemini_adapter",
            self.section.hint_label.text(),
        )

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
        self.window._refresh_diagnostics_context = lambda **_kwargs: None  # type: ignore[method-assign]

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    @staticmethod
    def _preflight_envelope(
        *,
        status: str = "ready",
        risks: list[dict[str, str]] | None = None,
    ) -> str:
        payload = {
            "strategy": "sync",
            "profile": {"id": "legacy-batch", "model": "gemini-3.5-flash"},
            "project": {"root": "C:/game/work"},
            "counts": {
                "files_with_pending": 1,
                "pending_items": 12,
                "chunks": 2,
            },
            "chunk_policy": {"max_items": 60, "max_chars": 18000},
            "context_sources": {
                "rag": False,
                "source_index": False,
                "story_memory": False,
                "project_analysis_brief": False,
                "local_context": {"before": 30, "after": 10},
            },
            "risks": risks or [],
        }
        return json.dumps(
            cli_contract.success_envelope(
                "translate-preflight",
                status=status,
                result=payload,
            )
        )

    def _finish_preflight(self, *, confirmed: bool = True) -> None:
        self.window._translate_preflight_output_lines = [self._preflight_envelope()]
        with mock.patch(
            "gui_qt.app.message_box_question",
            return_value="yes" if confirmed else "no",
        ):
            self.window._on_finished(0)
            QApplication.processEvents()

    def test_preflight_envelope_hoists_status_out_of_result(self) -> None:
        envelope = json.loads(self._preflight_envelope(status="blocked"))

        self.assertEqual(envelope["status"], "blocked")
        self.assertNotIn("status", envelope["result"])

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

    def test_unsupported_profile_on_batch_page_blocks_start(self) -> None:
        self.window.state.load_translator_config = lambda: litellm_primary_config()  # type: ignore[method-assign]
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.assertEqual(
            self.window._translation_target["strategy"],
            "gemini_batch",
        )
        self.assertFalse(self.window._translation_target_is_runnable())
        with mock.patch("gui_qt.app.message_box_information") as info:
            self.window._on_start_translation()

        info.assert_called_once()
        self.assertEqual(self.runner.calls, [])
        self.assertIsNone(self.window._pending_translation_start)

    def test_switching_to_sync_makes_litellm_profile_runnable(self) -> None:
        self.window.state.load_translator_config = lambda: litellm_primary_config()  # type: ignore[method-assign]
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_translation_target_selected("legacy-sync", "sync")

        self.assertEqual(self.window._work_mode, WorkMode.SYNC_TRANSLATION)
        self.assertTrue(self.window._translation_target_is_runnable())
        self.window._on_start_translation()
        self._finish_preflight()

        self.assertEqual(
            self.runner.calls[1][1],
            [
                "sync-start",
                "--profile",
                "legacy-sync",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_same_mode_selection_uses_model_name_and_localized_strategy(self) -> None:
        self.window._set_work_mode(
            WorkMode.BATCH_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        status_bar = self.window.statusBar()
        with mock.patch.object(status_bar, "showMessage") as show_message:
            self.window._on_translation_target_selected(
                "legacy-batch",
                "gemini_batch",
            )

        message = show_message.call_args[0][0]
        self.assertIn("gemini-3.5-flash", message)
        self.assertIn("Gemini Batch", message)
        self.assertNotIn("legacy-batch", message)

    def test_start_runs_preflight_before_the_workflow(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_start_translation()

        self.assertEqual(
            self.runner.calls[0][1],
            [
                "translate-preflight",
                "--strategy",
                "sync",
                "--profile",
                "legacy-batch",
                "--output",
                "json",
                "--non-interactive",
            ],
        )
        self._finish_preflight()
        self.assertEqual(
            self.runner.calls[1][1],
            [
                "sync-start",
                "--profile",
                "legacy-batch",
                "--output",
                "json",
                "--non-interactive",
            ],
        )

    def test_workflow_start_is_deferred_out_of_the_finished_callback(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._on_start_translation()
        self.window._translate_preflight_output_lines = [self._preflight_envelope()]
        with mock.patch("gui_qt.app.message_box_question", return_value="yes"):
            self.window._on_finished(0)

            # Starting the next QProcess from inside runner.finished is unsafe;
            # the chain must resume on the next event-loop turn.
            self.assertEqual(len(self.runner.calls), 1)

            QApplication.processEvents()

        self.assertEqual(len(self.runner.calls), 2)

    def test_rejected_preflight_start_clears_pending_and_warns(self) -> None:
        class _RejectingRunner(_FakeRunner):
            def run(self, script, args) -> bool:
                super().run(script, args)
                return False

        self.window.runner = _RejectingRunner()
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        with mock.patch("gui_qt.app.message_box_information") as info:
            self.window._on_start_translation()

        self.assertEqual(len(self.window.runner.calls), 1)
        self.assertIsNone(self.window._pending_translation_start)
        self.assertIn("无法开始预检", info.call_args[0][1])

    def test_blocked_preflight_never_starts_the_workflow(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._on_start_translation()
        self.window._translate_preflight_output_lines = [
            self._preflight_envelope(
                status="blocked",
                risks=[
                    {
                        "code": "CREDENTIAL_UNAVAILABLE",
                        "severity": "error",
                        "message": "missing",
                    }
                ],
            )
        ]
        with mock.patch("gui_qt.app.message_box_information") as info:
            self.window._on_finished(0)

        info.assert_called_once()
        message = info.call_args[0][2]
        self.assertIn("错误", message)
        self.assertIn("CREDENTIAL_UNAVAILABLE", message)
        self.assertEqual(len(self.runner.calls), 1)
        self.assertIsNone(self.window._pending_translation_start)

    def test_cancelled_preflight_never_starts_the_workflow(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._on_start_translation()

        self._finish_preflight(confirmed=False)

        self.assertEqual(len(self.runner.calls), 1)
        self.assertIsNone(self.window._pending_translation_start)

    def test_sync_start_passes_selected_profile(self) -> None:
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )

        self.window._on_start_translation()
        self._finish_preflight()

        self.assertEqual(
            self.runner.calls[1][1],
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
        self._finish_preflight()

        self.assertEqual(
            self.runner.calls[1][1],
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
        self._finish_preflight()

        self.assertEqual(
            self.runner.calls[1][1],
            ["sync-start", "--output", "json", "--non-interactive"],
        )


if __name__ == "__main__":
    unittest.main()
