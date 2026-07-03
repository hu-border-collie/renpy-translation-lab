import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
from project_asset_paths import (
    expected_project_asset_paths,
    sync_project_asset_paths_in_config,
)


class ProjectAssetPathsTests(unittest.TestCase):
    def test_sync_project_asset_paths_in_config(self):
        work_dir = "C:/Games/Example/work"
        config = {
            "game_root": "C:/Games/Other/work",
            "glossary_file": "C:/Games/Other/work/glossary.json",
            "batch": {
                "model": "gemini-test",
                "macro_setting_file": "C:/Games/Other/work/macro_setting.md",
            },
        }

        synced = sync_project_asset_paths_in_config(config, work_dir)

        self.assertEqual(
            synced["glossary_file"],
            expected_project_asset_paths(work_dir)["glossary_file"],
        )
        self.assertEqual(
            synced["batch"]["macro_setting_file"],
            expected_project_asset_paths(work_dir)["macro_setting_file"],
        )
        self.assertEqual(synced["batch"]["model"], "gemini-test")

    def test_collect_doctor_project_assets_warnings_for_missing_files(self):
        work_dir = "C:/Games/Example/work"
        assets = {
            "glossary_file": f"{work_dir}/glossary.json",
            "glossary_exists": False,
            "glossary_matches_project": True,
            "macro_setting_file": f"{work_dir}/macro_setting.md",
            "macro_exists": False,
            "macro_matches_project": True,
            "expected_glossary_file": f"{work_dir}/glossary.json",
            "expected_macro_setting_file": f"{work_dir}/macro_setting.md",
        }

        warnings = batch_mod.collect_doctor_project_assets_warnings(assets)

        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("glossary.json not found" in warning for warning in warnings))
        self.assertTrue(any("macro_setting.md not found" in warning for warning in warnings))

    def test_collect_doctor_report_warns_when_project_assets_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "Game" / "work"
            work_dir.mkdir(parents=True)
            config_path = Path(tmp) / "translator_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "game_root": str(work_dir),
                        "glossary_file": str(work_dir / "glossary.json"),
                        "batch": {
                            "macro_setting_file": str(work_dir / "macro_setting.md"),
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(batch_mod.legacy, "BASE_DIR", str(work_dir)),
                mock.patch.object(batch_mod.legacy, "TL_DIR", str(work_dir / "game" / "tl" / "schinese")),
                mock.patch.object(batch_mod.legacy, "TRANSLATOR_CONFIG", str(config_path)),
                mock.patch.object(batch_mod, "collect_tl_doctor_counts", return_value={"rpy_files": 1, "translate_blocks": 1, "string_sections": 0, "old_lines": 0, "new_lines": 0, "commented_original_lines": 0}),
                mock.patch.object(batch_mod, "collect_pending_file_jobs", return_value=[]),
                mock.patch.object(batch_mod.legacy, "_guess_source_game_dir", return_value=""),
                mock.patch.object(batch_mod.legacy, "get_prepare_template_command_info", return_value={"available": False, "kind": "", "reason": ""}),
                mock.patch.object(batch_mod.legacy, "resolve_original_game_dir", return_value=""),
                mock.patch.object(batch_mod.legacy, "work_dir_bootstrap_allowed", return_value=(False, str(work_dir), "")),
                mock.patch.object(batch_mod, "collect_doctor_context_status", return_value={"rag": {"enabled": False}, "source_index": {"enabled": False}}),
                mock.patch.object(batch_mod.legacy, "is_work_dir_empty", return_value=False),
                mock.patch("os.path.isdir", return_value=True),
            ):
                report = batch_mod.collect_doctor_report()

            self.assertFalse(report["project_assets"]["glossary_exists"])
            self.assertFalse(report["project_assets"]["macro_exists"])
            self.assertTrue(
                any("glossary.json not found" in warning for warning in report["warnings"])
            )
            self.assertTrue(
                any("macro_setting.md not found" in warning for warning in report["warnings"])
            )