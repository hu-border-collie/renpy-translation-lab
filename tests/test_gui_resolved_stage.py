"""Per-stage resolved profile display tests (#348 P3 increment B)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from model_routing_migration import preview_migration

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import MainWindow
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    WorkMode = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support

FIXTURES = Path(__file__).parent / "fixtures" / "model_routing_legacy"


def migrated_config() -> dict:
    payload = json.loads((FIXTURES / "gemini_batch.json").read_text(encoding="utf-8"))
    return preview_migration(payload).config


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class ResolvedStageFactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.state.load_translator_config = lambda: migrated_config()  # type: ignore[method-assign]

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def _facts_for(self, mode) -> str:
        self.window._set_work_mode(mode, refresh_manifest_writeback=False)
        self.window._set_workflow_summary("idle", "标题", "说明", [])
        page = self.window._workflow_status_page()
        if page is None:
            # Batch translation keeps the shared status card.
            return self.window.workflow_facts_label.text()
        for attr in ("status_section", "bootstrap_status_section"):
            section = getattr(page, attr, None)
            if section is not None:
                return section.facts_label.text()
        return ""

    def test_every_generation_stage_shows_resolved_profile(self) -> None:
        cases = {
            WorkMode.BATCH_TRANSLATION: ("gemini-3.5-flash", "Gemini Batch", "继承默认"),
            WorkMode.SYNC_TRANSLATION: ("gemini-3.1-flash-lite", "同步", "继承默认"),
            WorkMode.KEYWORD_EXTRACTION: ("gemini-3.5-flash", "Gemini Batch", "继承默认"),
            WorkMode.SYNC_KEYWORD_EXTRACTION: ("gemini-3.1-flash-lite", "同步", "继承默认"),
            WorkMode.REVISION: ("gemini-3.5-flash", "Gemini Batch", "继承默认"),
            WorkMode.SYNC_REVISION: ("gemini-3.1-flash-lite", "同步", "继承默认"),
            WorkMode.PROJECT_ANALYSIS: ("gemini-3.5-pro", "同步", "显式覆盖"),
            WorkMode.FINAL_REVIEW: ("gemini-3.5-flash", "Gemini Batch", "显式覆盖"),
        }
        for mode, (model, strategy, origin) in cases.items():
            with self.subTest(mode=mode):
                facts = self._facts_for(mode)
                self.assertIn("本阶段模型：", facts)
                self.assertIn(model, facts)
                self.assertIn(strategy, facts)
                self.assertIn(origin, facts)

    def test_missing_stage_route_shows_visible_fallback_fact(self) -> None:
        with mock.patch.dict(
            "gui_qt.app._RESOLVED_STAGE_BY_WORK_MODE",
            {WorkMode.SYNC_TRANSLATION: "missing-stage"},
        ):
            facts = self._facts_for(WorkMode.SYNC_TRANSLATION)

        self.assertIn("无法解析", facts)

    def test_broken_routing_shows_visible_fallback_fact(self) -> None:
        self.window.state.load_translator_config = lambda: {  # type: ignore[method-assign]
            "model_routing": {"profiles": "broken"},
        }
        self.window._set_work_mode(
            WorkMode.SYNC_TRANSLATION,
            refresh_manifest_writeback=False,
        )
        self.window._set_workflow_summary("idle", "标题", "说明", [])
        page = self.window._workflow_status_page()
        assert page is not None
        facts = page.status_section.facts_label.text()

        self.assertIn("无法解析", facts)

    def test_legacy_config_has_no_resolved_stage_fact(self) -> None:
        self.window.state.load_translator_config = lambda: {}  # type: ignore[method-assign]

        self.assertEqual(self.window._resolved_stage_target_fact(), "")
        self.window._set_work_mode(
            WorkMode.KEYWORD_EXTRACTION,
            refresh_manifest_writeback=False,
        )
        self.window._set_workflow_summary("idle", "标题", "说明", [])
        page = self.window._workflow_status_page()
        self.assertIsNotNone(page)
        assert page is not None
        self.assertNotIn(
            "本阶段模型：",
            page.status_section.facts_label.text(),
        )


if __name__ == "__main__":
    unittest.main()
