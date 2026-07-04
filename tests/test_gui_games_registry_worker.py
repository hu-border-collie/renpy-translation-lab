"""Tests for registry refresh worker."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.games_registry_actions import RegistryActionResult
    from gui_qt.games_registry_worker import RegistryRefreshWorker
except ImportError as exc:
    RegistryRefreshWorker = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(RegistryRefreshWorker is None, f"GUI dependencies are unavailable: {IMPORT_ERROR}")
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


if __name__ == "__main__":
    unittest.main()