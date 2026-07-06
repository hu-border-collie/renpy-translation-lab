"""Tests for registry vs doctor layout comparison."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import games_registry as registry
from gui_qt.games_registry_doctor_compare import compare_registry_with_doctor_report


class RegistryDoctorCompareTests(unittest.TestCase):
    def test_compare_reports_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "layout_status": "ready",
                        "auto": {
                            "doctor_mode": "existing_tl_only",
                            "last_refresh_at": "2026-07-04T12:00:00+00:00",
                        },
                    }
                ]
            }
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            result = compare_registry_with_doctor_report(
                workspace,
                game_root=project_root / "work",
                report={"layout_status": "ready", "mode": "existing_tl_only"},
            )
            self.assertIsNotNone(result)
            self.assertTrue(result.matched)
            self.assertIn("一致", result.message)

    def test_compare_reports_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project_root = workspace / "Game_Example"
            (project_root / "work").mkdir(parents=True)
            (project_root / "original" / "game").mkdir(parents=True)
            payload = {
                "projects": [
                    {
                        "id": "demo",
                        "name": "Example",
                        "path": "Game_Example",
                        "layout_status": "ready",
                        "auto": {"doctor_mode": "existing_tl_only"},
                    }
                ]
            }
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            result = compare_registry_with_doctor_report(
                workspace,
                game_root=project_root / "work",
                report={"layout_status": "attention", "mode": "can_generate_template"},
            )
            self.assertIsNotNone(result)
            self.assertFalse(result.matched)
            self.assertIn("不一致", result.message)

    def test_compare_when_project_not_in_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps({"projects": []}),
                encoding="utf-8",
            )
            result = compare_registry_with_doctor_report(
                workspace,
                game_root=workspace / "Game_Missing" / "work",
                report={"layout_status": "ready", "mode": "existing_tl_only"},
            )
            self.assertIsNotNone(result)
            self.assertIsNone(result.matched)
            self.assertIn("不在工作区总表", result.message)


if __name__ == "__main__":
    unittest.main()