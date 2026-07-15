"""Layout contracts for the unified settings surface."""
from __future__ import annotations

import unittest

try:
    from PySide6.QtWidgets import QApplication, QFormLayout, QScrollArea

    from gui_qt.app import MainWindow
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    QFormLayout = None  # type: ignore[assignment,misc]
    QScrollArea = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
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
            "settings_context",
            "settings_appearance",
            "settings_advanced",
        }
        self.assertEqual(set(self.window._settings_page_bodies), expected)

        for key, body in self.window._settings_page_bodies.items():
            with self.subTest(section=key):
                self.assertEqual(body.objectName(), "settings_page_body")
                self.assertEqual(body.property("settingsPage"), key)
                self.assertEqual(body.maximumWidth(), 1080)
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
                    self.assertLessEqual(body.width(), 1080)
                    expected_width = min(page.viewport().width(), 1080)
                    self.assertGreaterEqual(body.width(), expected_width - 2)

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
