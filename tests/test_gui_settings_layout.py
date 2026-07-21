"""Layout contracts for the unified settings surface."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication, QFormLayout, QScrollArea

    from gui_qt.app import MainWindow, _SETTINGS_PAGE_SPECS
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    QFormLayout = None  # type: ignore[assignment,misc]
    QScrollArea = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    _SETTINGS_PAGE_SPECS = ()  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _process(app: QApplication, rounds: int = 8) -> None:
    for _ in range(rounds):
        app.processEvents()


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiSettingsLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.resize(1600, 960)
        self.window.show()
        self.window._activate_shell_route("settings")
        # Layout contracts inspect every settings body; materialize them all.
        for key, _label, _builder in _SETTINGS_PAGE_SPECS:
            self.window._ensure_settings_page(key)
        _process(self._app)

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()

    def test_all_sections_share_the_same_bounded_content_body(self) -> None:
        expected = {
            "settings_workspace",
            "settings_project",
            "settings_api_keys",
            "settings_models",
            "settings_litellm",
            "settings_extensions",
            "settings_context",
            "settings_appearance",
            "settings_shortcuts",
            "settings_advanced",
        }
        self.assertEqual(set(self.window._settings_page_bodies), expected)

    def test_shortcuts_section_lists_core_bindings(self) -> None:
        from PySide6.QtWidgets import QLabel

        row = self.window._settings_nav_rows.get("shortcuts")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(self.window.settings_nav.item(row).text(), "快捷键")
        self.window.settings_nav.setCurrentRow(row)
        _process(self._app, 5)
        body = self.window._settings_page_bodies["settings_shortcuts"]
        labels = [
            widget.text()
            for widget in body.findChildren(QLabel)
            if widget.text().strip()
        ]
        joined = "\n".join(labels)
        for needle in (
            "Ctrl+D",
            "Ctrl+T",
            "Ctrl+K",
            "Ctrl+L",
            "Ctrl+Shift+L",
            "Ctrl+S",
            "Ctrl+1",
            "Ctrl+2",
            "Ctrl+3",
            "Ctrl+7",
            "Ctrl+0",
            "项目与环境",
            "批量翻译",
            "设置",
            "诊断与运行日志",
            "导航",
        ):
            self.assertIn(needle, joined, msg=needle)

    def test_number_shortcuts_activate_shell_routes_not_legacy_tabs(self) -> None:
        """Ctrl+N must follow sidebar IA, not the hidden main tab indices."""
        entries = self.window._shell_nav_shortcut_entries()
        self.assertGreaterEqual(len(entries), 7)
        self.assertEqual(entries[0][0], "project_prepare")
        self.assertEqual(entries[0][1], "项目与环境")
        self.assertTrue(entries[1][0].startswith("workbench:"))
        self.assertEqual(entries[6][0], "settings")

        self.window._activate_shell_route(entries[0][0])
        self.assertEqual(self.window._current_shell_route(), "project_prepare")
        self.window._activate_shell_route(entries[1][0])
        self.assertEqual(self.window._current_shell_route(), entries[1][0])
        self.window._activate_shell_route("diagnostics")
        self.assertEqual(self.window._current_shell_route(), "diagnostics")

        for key, body in self.window._settings_page_bodies.items():
            with self.subTest(section=key):
                self.assertEqual(body.objectName(), "settings_page_body")
                self.assertEqual(body.property("settingsPage"), key)
                # Bodies expand with the viewport (no hard 1080px content cap).
                self.assertGreaterEqual(body.maximumWidth(), 10000)
                margins = body.layout().contentsMargins()
                self.assertEqual(
                    (margins.left(), margins.top(), margins.right(), margins.bottom()),
                    (20, 18, 20, 20),
                )
                self.assertEqual(body.layout().spacing(), 14)

    def test_settings_pages_do_not_require_horizontal_scrolling(self) -> None:
        for width in (960, 1280, 1600):
            self.window.resize(width, 800)
            _process(self._app)
            for key, row in self.window._settings_nav_rows.items():
                with self.subTest(width=width, section=key):
                    self.window.settings_nav.setCurrentRow(row)
                    _process(self._app, 5)
                    page = self.window.settings_stack.currentWidget()
                    self.assertIsInstance(page, QScrollArea)
                    self.assertEqual(page.horizontalScrollBar().maximum(), 0)
                    body_key = f"settings_{key}"
                    body = self.window._settings_page_bodies[body_key]
                    # Content body should track the viewport width (fullscreen-friendly).
                    self.assertGreaterEqual(body.width(), page.viewport().width() - 2)
                    self.assertLessEqual(body.width(), page.viewport().width() + 2)

    def test_settings_navigation_items_fit_at_supported_narrow_width(self) -> None:
        self.window.resize(960, 640)
        _process(self._app)

        viewport_rect = self.window.settings_nav.viewport().rect()
        for row in range(self.window.settings_nav.count()):
            item = self.window.settings_nav.item(row)
            with self.subTest(section=item.text()):
                self.assertTrue(
                    viewport_rect.contains(
                        self.window.settings_nav.visualItemRect(item)
                    )
                )

    def test_settings_forms_share_label_and_field_spacing(self) -> None:
        form_count = 0
        for page_index in range(self.window.settings_stack.count()):
            page = self.window.settings_stack.widget(page_index)
            for form in page.findChildren(QFormLayout, "settings_form"):
                form_count += 1
                margins = form.contentsMargins()
                self.assertEqual(
                    (margins.left(), margins.top(), margins.right(), margins.bottom()),
                    (14, 18, 14, 14),
                )
                self.assertEqual(form.horizontalSpacing(), 16)
                self.assertEqual(form.verticalSpacing(), 10)
                self.assertEqual(
                    form.rowWrapPolicy(),
                    QFormLayout.RowWrapPolicy.WrapLongRows,
                )
        self.assertGreaterEqual(form_count, 7)

    def test_model_selectors_use_consistent_editing_affordances(self) -> None:
        self.assertFalse(self.window.sync_model_combo.isEditable())
        self.assertFalse(self.window.batch_model_combo.isEditable())
        for combo in (
            self.window.sync_embedding_combo,
            self.window.batch_embedding_combo,
            self.window.batch_thinking_combo,
        ):
            with self.subTest(combo=combo.objectName()):
                self.assertFalse(combo.isEditable())


if __name__ == "__main__":
    unittest.main()
