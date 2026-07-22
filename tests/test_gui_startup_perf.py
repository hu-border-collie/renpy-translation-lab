"""Cold-start guards: keep MainWindow construction off expensive work."""
from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.app import (
        MainWindow,
        _SETTINGS_CONFIG_PAGE_KEYS,
        _SETTINGS_PAGE_SPECS,
    )
    from gui_qt.project_state import ProjectState
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    ProjectState = None  # type: ignore[assignment,misc]
    _SETTINGS_CONFIG_PAGE_KEYS = frozenset()  # type: ignore[assignment,misc]
    _SETTINGS_PAGE_SPECS = ()  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiStartupPerfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        window = getattr(self, "window", None)
        if window is not None:
            window.close()
            window.deleteLater()
            self._app.processEvents()

    def test_startup_skips_manifest_history_scan(self) -> None:
        """Probe/split readiness must not walk logs/ during MainWindow().__init__."""
        with mock.patch.object(
            ProjectState,
            "get_latest_manifest_path_for_mode",
            autospec=True,
            return_value=None,
        ) as latest_for_mode:
            self.window = MainWindow()
            latest_for_mode.assert_not_called()

    def test_startup_builds_no_settings_section_bodies(self) -> None:
        self.window = MainWindow()
        self.assertEqual(self.window._settings_pages_built, set())
        self.assertNotIn("batch_model_combo", self.window.__dict__)
        self.assertNotIn("_games_registry_panel", self.window.__dict__)
        for index, (key, _label, _builder) in enumerate(_SETTINGS_PAGE_SPECS):
            page = self.window.settings_stack.widget(index)
            self.assertIsNotNone(page)
            assert page is not None
            self.assertEqual(page.objectName(), f"settings_{key}_placeholder")

    def test_load_config_to_ui_does_not_materialize_settings_pages(self) -> None:
        """_load_theme_to_ui must not getattr(theme_combo) into full config build."""
        self.window = MainWindow()
        self.assertEqual(self.window._settings_pages_built, set())
        self.window._load_config_to_ui(refresh_task_gates=False)
        self.assertEqual(self.window._settings_pages_built, set())
        self.assertNotIn("theme_combo", self.window.__dict__)
        self.assertNotIn("batch_model_combo", self.window.__dict__)

    def test_opening_workspace_builds_only_workspace(self) -> None:
        self.window = MainWindow()
        self.window._focus_settings_section("workspace")
        self.assertEqual(self.window._settings_pages_built, {"workspace"})
        self.assertIn("_games_registry_panel", self.window.__dict__)
        self.assertNotIn("batch_model_combo", self.window.__dict__)

    def test_first_config_section_builds_only_that_page(self) -> None:
        """Tab switches must not materialize every config section at once."""
        self.window = MainWindow()
        self.window._focus_settings_section("models")
        self.assertEqual(self.window._settings_pages_built, {"models"})
        self.assertIn("batch_model_combo", self.window.__dict__)
        self.assertNotIn("advanced", self.window._settings_pages_built)
        self.assertNotIn("workspace", self.window._settings_pages_built)
        self.assertNotIn("extensions", self.window._settings_pages_built)
        self.assertNotIn("shortcuts", self.window._settings_pages_built)

        self.window._focus_settings_section("advanced")
        self.assertEqual(
            self.window._settings_pages_built,
            {"models", "advanced"},
        )
        self.assertFalse(
            _SETTINGS_CONFIG_PAGE_KEYS.issubset(self.window._settings_pages_built)
        )

    def test_doctor_details_toggle_has_stable_fixed_width(self) -> None:
        self.window = MainWindow()
        toggle = self.window.doctor_details_toggle
        self.assertGreaterEqual(toggle.minimumWidth(), 96)
        self.assertEqual(toggle.minimumWidth(), toggle.maximumWidth())

    def test_status_tab_change_still_refreshes_readiness(self) -> None:
        self.window = MainWindow()
        with (
            mock.patch.object(
                self.window,
                "_update_probe_btn_enabled",
            ) as probe,
            mock.patch.object(
                self.window,
                "_update_split_btn_enabled",
            ) as split,
        ):
            self.window._sync_workbench_status_chrome(refresh_readiness=True)
            probe.assert_called_once()
            split.assert_called_once()

        with (
            mock.patch.object(
                self.window,
                "_update_probe_btn_enabled",
            ) as probe,
            mock.patch.object(
                self.window,
                "_update_split_btn_enabled",
            ) as split,
        ):
            self.window._sync_workbench_status_chrome(refresh_readiness=False)
            probe.assert_not_called()
            split.assert_not_called()
