"""Tests for registry refresh worker."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry
import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.games_registry_actions import RegistryActionResult
    from gui_qt.games_registry_worker import RegistryIngestWorker, RegistryRefreshWorker
except ImportError as exc:
    RegistryRefreshWorker = None  # type: ignore[assignment,misc]
    RegistryIngestWorker = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(RegistryRefreshWorker is None, IMPORT_ERROR)
class GuiGamesRegistryWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_worker_emits_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            payload = {
                "projects": [
                    {"id": "demo", "name": "Example", "path": "Game_Example"},
                ]
            }
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            worker = RegistryRefreshWorker(
                workspace_root=workspace,
                project_id="demo",
                refresh_everything=False,
                mode=registry.REFRESH_MODE_LITE,
            )
            with mock.patch(
                "gui_qt.games_registry_worker.refresh_registry_projects",
                return_value=RegistryActionResult(True, "done"),
            ) as refresh_mock:
                worker.run()
                refresh_mock.assert_called_once()

    def test_worker_emits_failure_when_refresh_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            worker = RegistryRefreshWorker(
                workspace_root=workspace,
                refresh_everything=True,
                mode=registry.REFRESH_MODE_LITE,
            )
            with mock.patch(
                "gui_qt.games_registry_worker.refresh_registry_projects",
                side_effect=RuntimeError("boom"),
            ):
                results: list[RegistryActionResult] = []
                worker.completed.connect(results.append)
                worker.run()
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertIn("boom", results[0].message)

    def test_worker_cancel_flag_is_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            worker = RegistryRefreshWorker(
                workspace_root=workspace,
                refresh_everything=True,
                mode=registry.REFRESH_MODE_LITE,
            )
            worker.request_stop()
            self.assertTrue(worker._should_cancel())

    def test_ingest_worker_emits_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = Path(tmp) / "src"
            source.mkdir()
            worker = RegistryIngestWorker(
                workspace_root=workspace,
                source=source,
                game_name="Demo",
            )
            with mock.patch(
                "gui_qt.games_registry_worker.ingest_registry_project",
                return_value=RegistryActionResult(
                    True, "imported", project_id="game_demo", project_path="Game_Demo"
                ),
            ) as ingest_mock:
                results: list[RegistryActionResult] = []
                worker.completed.connect(results.append)
                worker.run()
                ingest_mock.assert_called_once()
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertEqual(results[0].project_path, "Game_Demo")

    def test_ingest_worker_emits_failure_when_action_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            worker = RegistryIngestWorker(
                workspace_root=workspace,
                source=Path(tmp) / "missing",
                game_name="Demo",
            )
            with mock.patch(
                "gui_qt.games_registry_worker.ingest_registry_project",
                side_effect=RuntimeError("boom"),
            ):
                results: list[RegistryActionResult] = []
                worker.completed.connect(results.append)
                worker.run()
            self.assertEqual(len(results), 1)
            self.assertFalse(results[0].ok)
            self.assertIn("boom", results[0].message)


if __name__ == "__main__":
    unittest.main()