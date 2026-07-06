"""Tests for registry vs doctor layout comparison."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import games_registry as registry
from gui_qt.games_registry_doctor_compare import (
    RegistryDoctorCompareResult,
    compare_registry_with_doctor_report,
    format_registry_compare_hint,
)


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

    def test_compare_when_registry_json_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / registry.REGISTRY_FILENAME).write_text("{not json", encoding="utf-8")
            result = compare_registry_with_doctor_report(
                workspace,
                game_root=workspace / "Game_Example" / "work",
                report={"layout_status": "ready", "mode": "existing_tl_only"},
            )
            self.assertIsNotNone(result)
            self.assertIsNone(result.matched)
            self.assertIn("格式无效", result.message)
            self.assertIn("解析失败", result.log_line)

    @mock.patch(
        "gui_qt.games_registry_doctor_compare.find_project_id_for_game_root",
        return_value="orphan-id",
    )
    @mock.patch(
        "gui_qt.games_registry_doctor_compare.find_project",
        return_value=None,
    )
    def test_compare_when_project_record_missing_by_id(
        self,
        _mock_find_project,
        _mock_find_project_id,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / registry.REGISTRY_FILENAME).write_text(
                json.dumps({"projects": []}),
                encoding="utf-8",
            )
            result = compare_registry_with_doctor_report(
                workspace,
                game_root=workspace / "Game_Example" / "work",
                report={"layout_status": "ready", "mode": "existing_tl_only"},
            )
            self.assertIsNotNone(result)
            self.assertIsNone(result.matched)
            self.assertIn("orphan-id", result.message)
            self.assertIn("未找到项目记录", result.message)


class FormatRegistryCompareHintTests(unittest.TestCase):
    def _sample_compare(
        self,
        *,
        matched: bool | None = True,
    ) -> RegistryDoctorCompareResult:
        return RegistryDoctorCompareResult(
            matched=matched,
            registry_layout="ready",
            registry_mode="existing_tl_only",
            doctor_layout="ready",
            doctor_mode="existing_tl_only",
            last_refresh_at="2026-07-04T12:00:00+00:00",
            project_name="Example",
            message="与工作区总表一致：layout=ready，mode=existing_tl_only。",
            log_line="[总表对比] 与 games_registry 记录一致。",
        )

    def test_hint_when_compare_is_none_for_dialog(self):
        self.assertIn(
            "环境检查",
            format_registry_compare_hint(None, for_registry_dialog=True),
        )

    def test_hint_when_compare_is_none_for_summary(self):
        self.assertEqual(format_registry_compare_hint(None, for_registry_dialog=False), "")

    def test_hint_when_matched_is_none(self):
        compare = RegistryDoctorCompareResult(
            matched=None,
            registry_layout="",
            registry_mode="",
            doctor_layout="ready",
            doctor_mode="existing_tl_only",
            last_refresh_at="",
            project_name="",
            message="当前项目不在工作区总表中，无法与 registry 对比 layout。",
            log_line="[总表对比] 当前项目未登记在 games_registry.json。",
        )
        self.assertEqual(
            format_registry_compare_hint(compare, for_registry_dialog=True),
            compare.message,
        )

    def test_hint_when_matched_for_dialog(self):
        compare = self._sample_compare(matched=True)
        hint = format_registry_compare_hint(compare, for_registry_dialog=True)
        self.assertIn("与环境检查一致", hint)
        self.assertIn("layout=ready", hint)
        self.assertIn("mode=existing_tl_only", hint)

    def test_hint_when_mismatched_for_dialog(self):
        compare = RegistryDoctorCompareResult(
            matched=False,
            registry_layout="ready",
            registry_mode="existing_tl_only",
            doctor_layout="attention",
            doctor_mode="can_generate_template",
            last_refresh_at="",
            project_name="Example",
            message="与工作区总表不一致。",
            log_line="[总表对比] 不一致。",
        )
        hint = format_registry_compare_hint(compare, for_registry_dialog=True)
        self.assertIn("与环境检查不一致", hint)
        self.assertIn("attention", hint)

    def test_hint_when_matched_for_summary(self):
        compare = self._sample_compare(matched=True)
        self.assertEqual(
            format_registry_compare_hint(compare, for_registry_dialog=False),
            compare.message,
        )


if __name__ == "__main__":
    unittest.main()