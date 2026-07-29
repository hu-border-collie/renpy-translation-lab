"""GUI coverage for registry source URLs and optional table columns."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.games_registry_panel import GamesRegistryPanel
    from gui_qt.games_registry_table import (
        REGISTRY_DEFAULT_VISIBLE_COLUMN_IDS,
        REGISTRY_TABLE_COLUMN_DEFS,
    )
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    GamesRegistryPanel = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    GamesRegistryPanel is None,
    f"GUI dependencies are unavailable: {IMPORT_ERROR}",
)
class GuiGamesRegistrySourceUrlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _write_registry(self, workspace: Path) -> None:
        payload = {
            "projects": [
                {
                    "id": "demo",
                    "name": "Example",
                    "path": "Game_Example",
                    "source_url": "https://studio.itch.io/example",
                    "translation_status": "翻译中",
                    "notes": "等待最终校对",
                    "engine": "unity",
                    "in_renpy_pipeline": False,
                    "auto": {
                        "dialogue_translated_pct": 73,
                        "pending_tasks": 412,
                        "last_refresh_at": "2026-07-29T12:00:00+00:00",
                    },
                }
            ]
        }
        (workspace / registry.REGISTRY_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    def test_source_column_optional_columns_and_visibility_persist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_registry(workspace)
            panel = GamesRegistryPanel(
                None,
                workspace_root=workspace,
                auto_discover_on_show=False,
            )
            self.addCleanup(panel.deleteLater)
            indexes = {column.id: index for index, column in enumerate(REGISTRY_TABLE_COLUMN_DEFS)}

            self.assertFalse(panel._table.isColumnHidden(indexes["source"]))
            for column_id in ("progress", "engine", "last_refresh", "notes"):
                self.assertTrue(panel._table.isColumnHidden(indexes[column_id]))
            self.assertEqual(
                panel._table.item(0, indexes["source"]).text(),
                "itch.io",
            )
            self.assertIn(
                "https://studio.itch.io/example",
                panel._table.item(0, indexes["source"]).toolTip(),
            )

            for column_id in ("progress", "engine", "last_refresh", "notes"):
                panel._set_table_column_visible(column_id, True)
                self.assertFalse(panel._table.isColumnHidden(indexes[column_id]))
            self.assertEqual(
                panel._table.item(0, indexes["progress"]).text(),
                "73% · 待译 412",
            )
            self.assertEqual(panel._table.item(0, indexes["engine"]).text(), "unity")
            self.assertEqual(
                panel._table.item(0, indexes["last_refresh"]).text(),
                "2026-07-29T12:00:00+00:00",
            )
            self.assertEqual(
                panel._table.item(0, indexes["notes"]).text(),
                "等待最终校对",
            )

            progress_index = indexes["progress"]
            panel._table.setColumnWidth(progress_index, 190)
            panel._persist_table_column_widths()
            panel._set_table_column_visible("progress", False)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(
                data["preferences"]["table_column_widths"]["progress"],
                190,
            )

            panel._set_table_column_visible("name", False)
            self.assertFalse(panel._table.isColumnHidden(indexes["name"]))
            panel._set_table_column_visible("source", False)
            panel._search_edit.setText("studio.itch.io")
            self.assertEqual(panel._table.rowCount(), 1)

            panel._reset_table_column_visibility()
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(
                data["preferences"]["visible_columns"],
                list(REGISTRY_DEFAULT_VISIBLE_COLUMN_IDS),
            )
            self.assertFalse(panel._table.isColumnHidden(indexes["source"]))
            self.assertTrue(panel._table.isColumnHidden(indexes["progress"]))

    def test_open_source_is_explicit_and_follows_task_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_registry(workspace)
            with mock.patch(
                "gui_qt.games_registry_panel.QDesktopServices.openUrl",
                return_value=True,
            ) as open_url:
                panel = GamesRegistryPanel(
                    None,
                    workspace_root=workspace,
                    auto_discover_on_show=False,
                )
                self.addCleanup(panel.deleteLater)
                open_url.assert_not_called()

                panel._table.selectRow(0)
                self.assertTrue(panel._open_source_btn.isEnabled())
                open_url.assert_not_called()
                panel._open_selected_source_url()
                self.assertEqual(open_url.call_count, 1)
                opened = open_url.call_args.args[0]
                self.assertEqual(
                    opened.toString(),
                    "https://studio.itch.io/example",
                )

                panel.set_host_task_running(True)
                self.assertFalse(panel._source_url_edit.isEnabled())
                self.assertFalse(panel._open_source_btn.isEnabled())
                panel._open_selected_source_url()
                self.assertEqual(open_url.call_count, 1)

                panel.set_host_task_running(False)
                panel._set_refresh_busy(True)
                self.assertFalse(panel._source_url_edit.isEnabled())
                self.assertFalse(panel._open_source_btn.isEnabled())
                panel._open_selected_source_url()
                self.assertEqual(open_url.call_count, 1)
                panel._set_refresh_busy(False)

    def test_invalid_save_preserves_input_then_valid_update_and_clear_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_registry(workspace)
            panel = GamesRegistryPanel(
                None,
                workspace_root=workspace,
                auto_discover_on_show=False,
            )
            self.addCleanup(panel.deleteLater)
            panel._table.selectRow(0)

            invalid = "ftp://example.com/game"
            panel._source_url_edit.setText(invalid)
            with mock.patch("gui_qt.games_registry_panel.message_box_warning") as warning:
                panel._save_selected_project_fields()
            warning.assert_called_once()
            self.assertEqual(panel._source_url_edit.text(), invalid)
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(
                data["projects"][0]["source_url"],
                "https://studio.itch.io/example",
            )

            panel._source_url_edit.setText("https://example.com/new")
            panel._save_selected_project_fields()
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(
                data["projects"][0]["source_url"],
                "https://example.com/new",
            )
            self.assertEqual(
                panel._source_url_edit.text(),
                "https://example.com/new",
            )

            panel._source_url_edit.clear()
            panel._save_selected_project_fields()
            data = registry.load_registry(workspace / registry.REGISTRY_FILENAME)
            self.assertEqual(data["projects"][0]["source_url"], "")
            self.assertEqual(panel._source_url_edit.text(), "")
            self.assertFalse(panel._open_source_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
