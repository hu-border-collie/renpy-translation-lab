"""Native-palette regression checks for app-themed Qt control surfaces."""
from __future__ import annotations

import os
import unittest
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QColor, QImage, QPalette
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QScrollArea,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    from gui_qt.app import MainWindow
    from gui_qt.keyword_merge_dialog import KeywordMergeDialog
    from gui_qt.revision_selection_dialog import RevisionProposalSelectionDialog
    from gui_qt.settings.litellm_page import LiteLLMSettingsPage
    from gui_qt.settings.profiles_page import ProfilesSettingsPage
    from gui_qt.theme_helpers import load_theme_stylesheet
    from gui_qt.theme_tokens import tokens_for_theme
    from tests import gui_test_support

    IMPORT_ERROR = None


@unittest.skipIf(
    QApplication is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class GuiThemeControlSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._original_palette = QPalette(self._app.palette())
        self._original_stylesheet = self._app.styleSheet()
        self._pages: list[Any] = []
        self._root: QWidget | None = None
        self._deactivate_window: QWidget | None = None

    def tearDown(self) -> None:
        if self._root is not None:
            self._root.close()
            self._root.deleteLater()
        if self._deactivate_window is not None:
            self._deactivate_window.close()
            self._deactivate_window.deleteLater()
        for page in self._pages:
            page.deleteLater()
        self._app.setStyleSheet(self._original_stylesheet)
        self._app.setPalette(self._original_palette)
        self._app.processEvents()

    def test_explicit_theme_controls_survive_opposite_system_palette(self) -> None:
        cases = (("light", True), ("dark", False))
        for theme, system_is_dark in cases:
            with self.subTest(theme=theme, system_is_dark=system_is_dark):
                self._app.setPalette(self._system_palette(system_is_dark))
                self._app.setStyleSheet(
                    load_theme_stylesheet(
                        Path(__file__).resolve().parents[1]
                        / "gui_qt"
                        / "resources",
                        theme,
                    )
                )
                tokens = tokens_for_theme(theme)
                controls = self._build_controls()

                for width, height in ((960, 640), (1280, 800), (1600, 900)):
                    controls["root"].resize(width, height)
                    controls["root"].show()
                    controls["root"].layout().activate()
                    self._app.processEvents()
                    self._assert_surfaces_fit(controls, width, height)
                    if (width, height) == (960, 640):
                        self._assert_theme_colors(controls, tokens)
                        self._assert_control_states(controls, tokens)

                controls["root"].close()
                controls["root"].deleteLater()
                self._root = None
                for page in self._pages:
                    page.deleteLater()
                self._pages.clear()
                self._app.processEvents()

    def test_main_window_builds_doctor_content_as_a_styled_surface(self) -> None:
        window = MainWindow(bootstrap_config={})
        try:
            content = window.doctor_summary_scroll.widget()
            self.assertIsNotNone(content)
            self.assertEqual(content.objectName(), "doctor_summary_content")
            self.assertTrue(
                content.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
            )
            self.assertEqual(
                window.doctor_summary_scroll.viewport().objectName(),
                "doctor_summary_viewport",
            )
        finally:
            gui_test_support.close_main_window(window)
            window.deleteLater()
            self._app.processEvents()

    def test_keyword_review_uses_explicit_theme_with_opposite_native_palette(self) -> None:
        import keyword_glossary_merge as merge

        candidates = [{"source": "Moon Gate", "suggested_target": "月门", "confidence": 0.9}]
        rows = merge.build_candidate_merge_rows(candidates, {"normalize_map": {}})
        for theme, native_dark in (("light", True), ("dark", False)):
            with self.subTest(theme=theme):
                self._app.setPalette(self._system_palette(native_dark))
                self._app.setStyleSheet(load_theme_stylesheet(
                    Path(__file__).resolve().parents[1] / "gui_qt/resources", theme,
                ))
                dialog = KeywordMergeDialog(
                    None, rows=rows, candidates=candidates,
                    candidates_path="original-candidates.jsonl", glossary_path="original-glossary.json",
                )
                self._root = dialog
                dialog.show()
                self._app.processEvents()
                tokens = tokens_for_theme(theme)
                table = dialog.table
                self.assertEqual(self._pixel(table, self._blank_table_point(table)), QColor(tokens["bg_table"]).name())
                header = table.horizontalHeader()
                self._assert_color_present(header, QColor(tokens["fg_table_header"]))
                self.assertEqual(table.item(0, 1).foreground().color().name(), QColor(tokens["badge_warning_fg"]).name())
                check_item = table.item(0, 0)
                check_rect = table.visualItemRect(check_item).adjusted(1, 1, -1, -1)
                unchecked = table.viewport().grab(check_rect).toImage()
                border_color = QColor(tokens["fg_table_header"]).name()
                self.assertTrue(any(
                    unchecked.pixelColor(x, y).name() == border_color
                    for x in range(unchecked.width()) for y in range(unchecked.height())
                ), "The unchecked review indicator must be visible")
                check_item.setCheckState(Qt.CheckState.Checked)
                self._app.processEvents()
                checked = table.viewport().grab(check_rect).toImage()
                checked_color = QColor(tokens["accent_primary"]).name()
                self.assertTrue(any(
                    checked.pixelColor(x, y).name() == checked_color
                    for x in range(checked.width()) for y in range(checked.height())
                ), "The checked review indicator must differ from the unchecked box")
                self.assertEqual(dialog._selected_indices(), {0})
                check_item.setCheckState(Qt.CheckState.Unchecked)
                table.setCurrentCell(0, 1)
                table.selectRow(0)
                table.setFocus()
                self._app.processEvents()
                self._assert_color_present(table, QColor(tokens["bg_table_selected_solid"]))
                table.setEnabled(False)
                self._app.processEvents()
                self.assertEqual(self._pixel(table, self._blank_table_point(table)), QColor(tokens["bg_disabled"]).name())
                table.setEnabled(True)
                table.setRowCount(0)
                self._app.processEvents()
                self.assertEqual(self._pixel(table, self._blank_table_point(table)), QColor(tokens["bg_table"]).name())
                dialog.close()
                dialog.deleteLater()
                self._root = None
                self._app.processEvents()

    def test_revision_selection_indicators_survive_opposite_native_palette(self) -> None:
        import revision_corpus
        import revision_selection

        item = {"id": "original-lantern", "file_rel_path": "lantern.rpy",
                "source": "Carry the lantern.", "current_translation": "带上灯。"}
        row = {**item, "schema_version": 1, "occurrence_id": item["id"],
               "identity_v2": item["id"], "proposed_translation": "带好提灯。",
               "reason": "统一灯具用语", "selected": False, "disposition": "accepted",
               "producer": {"type": "agent", "tool": "original-offline-fixture"},
               "project_identity": {"tl_dir": "C:/original/tl"},
               "snapshot_digest": revision_corpus.item_snapshot_digest(item["source"], item["current_translation"]),
               "corpus_snapshot_digest": "a" * 64}
        stage = revision_selection.build_staged_selection(
            rows=[row], live_items={item["id"]: item}, live_snapshot_digest="a" * 64,
            project_identity={"game_root": "C:/original", "tl_dir": "C:/original/tl"},
            proposal_path="C:/original/proposals.jsonl", proposal_sha256="b" * 64,
            operation_id="original-offline-review",
        )
        for theme, native_dark in (("light", True), ("dark", False)):
            with self.subTest(theme=theme):
                self._app.setPalette(self._system_palette(native_dark))
                self._app.setStyleSheet(load_theme_stylesheet(
                    Path(__file__).resolve().parents[1] / "gui_qt/resources", theme,
                ))
                dialog = RevisionProposalSelectionDialog(stage)
                self._root = dialog
                dialog.show()
                self._app.processEvents()
                self.assertFalse(dialog._ok_button.isEnabled())
                item = dialog.table.item(0, 0)
                rect = dialog.table.visualItemRect(item).adjusted(1, 1, -1, -1)
                tokens = tokens_for_theme(theme)
                for state, color in ((Qt.CheckState.Unchecked, "fg_table_header"), (Qt.CheckState.Checked, "accent_primary")):
                    item.setCheckState(state)
                    self._app.processEvents()
                    rendered = dialog.table.viewport().grab(rect).toImage()
                    self.assertTrue(any(
                        rendered.pixelColor(x, y).name() == QColor(tokens[color]).name()
                        for x in range(rendered.width()) for y in range(rendered.height())
                    ), f"Review indicator is invisible: {theme}, {state}")
                self.assertTrue(dialog._ok_button.isEnabled())
                dialog.close()
                dialog.deleteLater()
                self._root = None
                self._app.processEvents()

    def _system_palette(self, dark: bool) -> QPalette:
        palette = QPalette(self._app.palette())
        if dark:
            values = {
                QPalette.ColorRole.Window: "#202020",
                QPalette.ColorRole.WindowText: "#171717",
                QPalette.ColorRole.Base: "#202020",
                QPalette.ColorRole.AlternateBase: "#303030",
                QPalette.ColorRole.Text: "#171717",
                QPalette.ColorRole.Button: "#242424",
                QPalette.ColorRole.ButtonText: "#171717",
                QPalette.ColorRole.Highlight: "#484848",
                QPalette.ColorRole.HighlightedText: "#171717",
            }
        else:
            values = {
                QPalette.ColorRole.Window: "#ffffff",
                QPalette.ColorRole.WindowText: "#171717",
                QPalette.ColorRole.Base: "#ffffff",
                QPalette.ColorRole.AlternateBase: "#f1f1f1",
                QPalette.ColorRole.Text: "#171717",
                QPalette.ColorRole.Button: "#f8f8f8",
                QPalette.ColorRole.ButtonText: "#171717",
                QPalette.ColorRole.Highlight: "#c0c0c0",
                QPalette.ColorRole.HighlightedText: "#171717",
            }
        for role, value in values.items():
            palette.setColor(QPalette.ColorGroup.All, role, QColor(value))
        if dark:
            inactive_highlight = "#101010"
            inactive_text = "#050505"
        else:
            inactive_highlight = "#e0e0e0"
            inactive_text = "#f8f8f8"
        palette.setColor(
            QPalette.ColorGroup.Inactive,
            QPalette.ColorRole.Highlight,
            QColor(inactive_highlight),
        )
        palette.setColor(
            QPalette.ColorGroup.Inactive,
            QPalette.ColorRole.HighlightedText,
            QColor(inactive_text),
        )
        return palette

    def _set_active_window(self, window: QWidget) -> None:
        window.show()
        window.activateWindow()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            QApplication.setActiveWindow(window)
        self._app.processEvents()

    def _build_controls(self) -> dict[str, Any]:
        profiles_page = ProfilesSettingsPage()
        litellm_page = LiteLLMSettingsPage(start_warmup=False)
        self._pages.extend((profiles_page, litellm_page))

        root = QWidget()
        self._root = root
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        profiles_list = profiles_page.profiles_list
        profiles_list.setParent(None)
        profiles_list.setEnabled(True)
        profiles_list.setFixedHeight(120)
        profiles_list.addItems(
            ("Primary sync model", "Gemini Batch", "Embedding model")
        )
        layout.addWidget(profiles_list)

        provider_table = litellm_page.custom_provider_table
        provider_table.setParent(None)
        provider_table.setEnabled(True)
        provider_table.setFixedHeight(150)
        provider_table.setRowCount(2)
        for row, values in enumerate(
            (
                ("demo", "Example provider", "https://example.invalid", "DEMO_KEY"),
                ("secondary", "Another example", "https://api.invalid", "OTHER_KEY"),
            )
        ):
            for column, value in enumerate(values):
                provider_table.setItem(row, column, QTableWidgetItem(value))
        layout.addWidget(provider_table)

        doctor_scroll = QScrollArea()
        doctor_scroll.setObjectName("doctor_summary_scroll")
        doctor_scroll.setWidgetResizable(True)
        doctor_scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        doctor_viewport = doctor_scroll.viewport()
        doctor_viewport.setObjectName("doctor_summary_viewport")
        doctor_viewport.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        doctor_content = QWidget()
        doctor_content.setObjectName("doctor_summary_content")
        doctor_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        doctor_content.setMinimumHeight(220)
        doctor_layout = QVBoxLayout(doctor_content)
        doctor_layout.setContentsMargins(0, 0, 0, 0)
        doctor_label = QLabel("Example environment report and recommendations")
        doctor_label.setObjectName("summary_body_label")
        doctor_label.setWordWrap(True)
        doctor_layout.addWidget(doctor_label)
        doctor_layout.addStretch(1)
        doctor_scroll.setWidget(doctor_content)
        layout.addWidget(doctor_scroll, 1)

        root.resize(960, 640)
        root.show()
        self._app.processEvents()
        return {
            "root": root,
            "profiles_list": profiles_list,
            "provider_table": provider_table,
            "provider_header": provider_table.horizontalHeader(),
            "doctor_scroll": doctor_scroll,
            "doctor_viewport": doctor_viewport,
            "doctor_content": doctor_content,
            "doctor_label": doctor_label,
        }

    def _assert_surfaces_fit(
        self,
        controls: dict[str, Any],
        width: int,
        height: int,
    ) -> None:
        root = controls["root"]
        self.assertEqual(root.size().width(), width)
        self.assertEqual(root.size().height(), height)
        for name in ("profiles_list", "provider_table", "doctor_scroll"):
            widget = controls[name]
            geometry = widget.geometry()
            self.assertGreater(geometry.width(), 0, name)
            self.assertGreater(geometry.height(), 0, name)
            self.assertGreaterEqual(geometry.left(), 0, name)
            self.assertGreaterEqual(geometry.top(), 0, name)
            self.assertLess(geometry.right(), width, name)
            self.assertLess(geometry.bottom(), height, name)
        self.assertLessEqual(
            controls["doctor_content"].width(),
            controls["doctor_viewport"].width(),
        )

    def _assert_theme_colors(
        self,
        controls: dict[str, Any],
        tokens: dict[str, str],
    ) -> None:
        profiles_list = controls["profiles_list"]
        provider_table = controls["provider_table"]
        provider_header = controls["provider_header"]
        doctor_content = controls["doctor_content"]
        doctor_label = controls["doctor_label"]

        self.assertEqual(
            self._pixel(profiles_list, self._blank_list_point(profiles_list)),
            QColor(tokens["bg_table"]).name(),
        )
        self.assertEqual(
            profiles_list.palette().color(QPalette.ColorRole.Text).name(),
            QColor(tokens["fg_table"]).name(),
        )
        self.assertEqual(
            self._pixel(provider_header, QPoint(provider_header.width() - 8, 15)),
            QColor(tokens["bg_table_header"]).name(),
        )
        self._assert_color_present(
            provider_header,
            QColor(tokens["fg_table_header"]),
        )
        self.assertEqual(
            self._pixel(provider_table, self._blank_table_point(provider_table)),
            QColor(tokens["bg_table"]).name(),
        )
        self.assertEqual(
            provider_table.palette().color(QPalette.ColorRole.Text).name(),
            QColor(tokens["fg_table"]).name(),
        )
        self.assertEqual(
            self._pixel(
                doctor_content,
                QPoint(doctor_content.width() - 8, doctor_content.height() - 8),
            ),
            QColor(tokens["bg_surface"]).name(),
        )
        self.assertEqual(
            doctor_label.palette().color(QPalette.ColorRole.Text).name(),
            QColor(tokens["fg_body"]).name(),
        )

    def _assert_control_states(
        self,
        controls: dict[str, Any],
        tokens: dict[str, str],
    ) -> None:
        profiles_list = controls["profiles_list"]
        provider_table = controls["provider_table"]
        provider_header = controls["provider_header"]

        self._set_active_window(controls["root"])
        profiles_list.setCurrentRow(1)
        profiles_list.item(1).setSelected(True)
        profiles_list.setFocus(Qt.FocusReason.OtherFocusReason)
        self._app.processEvents()
        self.assertTrue(profiles_list.item(1).isSelected())
        self.assertEqual(
            self._pixel(
                profiles_list,
                self._list_item_point(profiles_list, 1),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        self._assert_color_present(
            profiles_list,
            QColor(tokens["fg_table_selected_solid"]),
        )
        self.assertEqual(
            self._pixel(
                profiles_list,
                QPoint(profiles_list.width() // 2, 0),
            ),
            QColor(tokens["accent_primary"]).name(),
        )
        profiles_list.clearFocus()
        self._app.processEvents()
        self.assertEqual(
            self._pixel(
                profiles_list,
                self._list_item_point(profiles_list, 1),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        inactive_window = QWidget()
        self._deactivate_window = inactive_window
        inactive_window.resize(120, 80)
        self._set_active_window(inactive_window)
        self.assertFalse(controls["root"].isActiveWindow())
        self.assertEqual(
            profiles_list.palette().currentColorGroup(),
            QPalette.ColorGroup.Inactive,
        )
        self.assertEqual(
            self._pixel(
                profiles_list,
                self._list_item_point(profiles_list, 1),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        inactive_window.close()
        inactive_window.deleteLater()
        self._deactivate_window = None
        self._set_active_window(controls["root"])
        profiles_list.setEnabled(False)
        self._app.processEvents()
        self.assertEqual(
            self._pixel(
                profiles_list,
                QPoint(profiles_list.width() - 8, profiles_list.height() - 8),
            ),
            QColor(tokens["bg_disabled"]).name(),
        )
        self.assertEqual(
            self._pixel(
                profiles_list,
                self._list_item_point(profiles_list, 1),
            ),
            QColor(tokens["bg_disabled"]).name(),
        )
        self.assertEqual(
            profiles_list.palette()
            .color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            .name(),
            QColor(tokens["fg_disabled"]).name(),
        )
        self._assert_color_present(
            profiles_list,
            QColor(tokens["fg_disabled"]),
        )
        profiles_list.setEnabled(True)
        profiles_list.clear()
        self._app.processEvents()
        self.assertEqual(
            self._pixel(profiles_list, self._blank_list_point(profiles_list)),
            QColor(tokens["bg_table"]).name(),
        )

        self._set_active_window(controls["root"])
        provider_table.selectRow(0)
        provider_table.setCurrentCell(0, 0)
        provider_table.setFocus(Qt.FocusReason.OtherFocusReason)
        self._app.processEvents()
        self.assertTrue(provider_table.item(0, 0).isSelected())
        self.assertEqual(
            self._pixel(
                provider_table,
                self._table_item_point(provider_table, 0, 0),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        self._assert_color_present(
            provider_table,
            QColor(tokens["fg_table_selected_solid"]),
        )
        self._assert_color_present(
            provider_table,
            QColor(tokens["fg_table"]),
        )
        self.assertEqual(
            self._pixel(
                provider_table,
                QPoint(provider_table.width() // 2, 0),
            ),
            QColor(tokens["accent_primary"]).name(),
        )
        provider_table.clearFocus()
        self._app.processEvents()
        self.assertEqual(
            self._pixel(
                provider_table,
                self._table_item_point(provider_table, 0, 0),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        inactive_window = QWidget()
        self._deactivate_window = inactive_window
        inactive_window.resize(120, 80)
        self._set_active_window(inactive_window)
        self.assertFalse(controls["root"].isActiveWindow())
        self.assertEqual(
            provider_table.palette().currentColorGroup(),
            QPalette.ColorGroup.Inactive,
        )
        self.assertEqual(
            self._pixel(
                provider_table,
                self._table_item_point(provider_table, 0, 0),
            ),
            QColor(tokens["bg_table_selected_solid"]).name(),
        )
        inactive_window.close()
        inactive_window.deleteLater()
        self._deactivate_window = None
        self._set_active_window(controls["root"])
        provider_table.setEnabled(False)
        self._app.processEvents()
        self.assertEqual(
            self._pixel(
                provider_table,
                self._table_item_point(provider_table, 0, 0),
            ),
            QColor(tokens["bg_disabled"]).name(),
        )
        self.assertEqual(
            self._pixel(
                provider_header,
                QPoint(provider_header.width() - 8, 15),
            ),
            QColor(tokens["bg_disabled"]).name(),
        )
        self._assert_color_present(
            provider_table,
            QColor(tokens["fg_disabled"]),
        )
        self._assert_color_present(
            provider_header,
            QColor(tokens["fg_disabled"]),
        )
        self.assertEqual(
            provider_table.palette()
            .color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            .name(),
            QColor(tokens["fg_disabled"]).name(),
        )
        provider_table.setEnabled(True)
        provider_table.setRowCount(0)
        self._app.processEvents()
        self.assertEqual(
            self._pixel(provider_table, self._blank_table_point(provider_table)),
            QColor(tokens["bg_table"]).name(),
        )

    def _pixel(self, widget: QWidget, point: QPoint) -> str:
        image = widget.grab().toImage().convertToFormat(
            QImage.Format.Format_ARGB32
        )
        x = min(max(point.x(), 0), image.width() - 1)
        y = min(max(point.y(), 0), image.height() - 1)
        return image.pixelColor(x, y).name()

    def _assert_color_present(self, widget: QWidget, color: QColor) -> None:
        image = widget.grab().toImage().convertToFormat(
            QImage.Format.Format_ARGB32
        )
        for y in range(image.height()):
            for x in range(image.width()):
                pixel = image.pixelColor(x, y)
                if max(
                    abs(pixel.red() - color.red()),
                    abs(pixel.green() - color.green()),
                    abs(pixel.blue() - color.blue()),
                ) <= 20:
                    return
        self.fail(f"Expected rendered text color near {color.name()} on {widget.objectName()}")

    def _blank_list_point(self, widget: QWidget) -> QPoint:
        viewport = widget.viewport()
        return viewport.mapTo(
            widget,
            QPoint(viewport.width() - 8, viewport.height() - 8),
        )

    def _list_item_point(self, widget: QWidget, row: int) -> QPoint:
        viewport = widget.viewport()
        item_rect = widget.visualItemRect(widget.item(row))
        return viewport.mapTo(
            widget,
            QPoint(item_rect.right() - 8, item_rect.center().y()),
        )

    def _table_item_point(self, widget: QWidget, row: int, column: int) -> QPoint:
        viewport = widget.viewport()
        item_rect = widget.visualItemRect(widget.item(row, column))
        return viewport.mapTo(
            widget,
            QPoint(item_rect.right() - 5, item_rect.center().y()),
        )

    def _blank_table_point(self, widget: QWidget) -> QPoint:
        viewport = widget.viewport()
        return viewport.mapTo(
            widget,
            QPoint(viewport.width() - 8, viewport.height() - 8),
        )


if __name__ == "__main__":
    unittest.main()
