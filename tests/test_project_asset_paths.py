import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
from project_asset_paths import (
    canonical_abs_path,
    expected_project_asset_paths,
    normalize_relative_project_assets_in_config,
    paths_match_project,
    resolve_configured_glossary_value,
    resolve_glossary_path,
    resolve_macro_setting_path,
    sync_project_asset_paths_in_config,
)


class ProjectAssetPathsTests(unittest.TestCase):
    def test_resolve_configured_glossary_value_falls_back_when_primary_empty(self):
        self.assertEqual(
            resolve_configured_glossary_value(
                {"glossary_file": "", "glossary_path": "legacy.json"}
            ),
            "legacy.json",
        )
        self.assertEqual(
            resolve_configured_glossary_value({"glossary_path": "legacy.json"}),
            "legacy.json",
        )
        self.assertEqual(
            resolve_configured_glossary_value(
                {"glossary_file": "primary.json", "glossary_path": "legacy.json"}
            ),
            "primary.json",
        )
        self.assertEqual(resolve_configured_glossary_value({}), "")
        self.assertEqual(resolve_configured_glossary_value(None), "")

    def test_canonical_abs_path_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "Game" / "work"
            nested.mkdir(parents=True)
            asset = nested / "glossary.json"
            asset.write_text("{}", encoding="utf-8")

            resolved = canonical_abs_path(asset)
            expected = expected_project_asset_paths(nested)["glossary_file"]

            self.assertTrue(
                paths_match_project(resolved, expected),
            )

    def test_relative_glossary_prefers_game_root_even_if_tool_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp) / "tool"
            work_dir = Path(tmp) / "Game" / "work"
            tool_dir.mkdir(parents=True)
            work_dir.mkdir(parents=True)
            (tool_dir / "glossary.json").write_text('{"from":"tool"}', encoding="utf-8")

            resolved = resolve_glossary_path(
                "glossary.json",
                game_root=work_dir,
                tool_dir=tool_dir,
            )

            self.assertTrue(
                paths_match_project(
                    resolved,
                    expected_project_asset_paths(work_dir)["glossary_file"],
                )
            )
            self.assertFalse(paths_match_project(resolved, tool_dir / "glossary.json"))

    def test_absolute_glossary_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = str(Path(tmp) / "shared" / "team-glossary.json")
            work_dir = str(Path(tmp) / "Game" / "work")
            tool_dir = str(Path(tmp) / "tool")
            Path(shared).parent.mkdir(parents=True)
            Path(work_dir).mkdir(parents=True)
            Path(tool_dir).mkdir(parents=True)

            resolved = resolve_glossary_path(
                shared,
                game_root=work_dir,
                tool_dir=tool_dir,
            )
            self.assertTrue(paths_match_project(resolved, shared))

    def test_normalize_relative_project_assets_keeps_absolute_custom_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = str(Path(tmp) / "Game" / "work")
            shared_macro = str(Path(tmp) / "shared" / "style.md")
            Path(work_dir).mkdir(parents=True)
            Path(shared_macro).parent.mkdir(parents=True)
            config = {
                "game_root": work_dir,
                "glossary_file": "glossary.json",
                "batch": {
                    "model": "gemini-test",
                    "macro_setting_file": shared_macro,
                },
            }

            normalize_relative_project_assets_in_config(config, work_dir)

            self.assertTrue(
                paths_match_project(
                    config["glossary_file"],
                    expected_project_asset_paths(work_dir)["glossary_file"],
                )
            )
            self.assertTrue(
                paths_match_project(
                    config["batch"]["macro_setting_file"],
                    shared_macro,
                )
            )
            self.assertEqual(config["batch"]["model"], "gemini-test")

    def test_normalize_relative_project_assets_leaves_empty_entries_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = str(Path(tmp) / "Game" / "work")
            Path(work_dir).mkdir(parents=True)
            config = {"game_root": work_dir}

            normalize_relative_project_assets_in_config(config, work_dir)

            self.assertNotIn("glossary_file", config)
            self.assertEqual(config["batch"], {})

    def test_resolve_glossary_path_without_bases_canonicalizes_relative_value(self):
        resolved = resolve_glossary_path("glossary.json")

        self.assertTrue(os.path.isabs(resolved))
        self.assertEqual(resolved, canonical_abs_path("glossary.json"))

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

    def test_doctor_relative_glossary_matches_project_when_tool_has_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            tool_dir = Path(tmp) / "tool"
            work_dir = Path(tmp) / "Game" / "work"
            tool_dir.mkdir(parents=True)
            work_dir.mkdir(parents=True)
            (tool_dir / "glossary.json").write_text("{}", encoding="utf-8")
            config_path = tool_dir / "translator_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "game_root": str(work_dir),
                        "glossary_file": "glossary.json",
                        "batch": {"macro_setting_file": "macro_setting.md"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(batch_mod.legacy, "BASE_DIR", str(work_dir)),
                mock.patch.object(batch_mod.legacy, "TOOL_DIR", str(tool_dir)),
                mock.patch.object(batch_mod.legacy, "TRANSLATOR_CONFIG", str(config_path)),
            ):
                assets = batch_mod.collect_doctor_project_assets_status(str(work_dir))

            self.assertTrue(assets["glossary_matches_project"])
            self.assertTrue(assets["macro_matches_project"])
            self.assertTrue(
                paths_match_project(
                    assets["glossary_file"],
                    expected_project_asset_paths(work_dir)["glossary_file"],
                )
            )
            warnings = batch_mod.collect_doctor_project_assets_warnings(assets)
            self.assertFalse(
                any("does not match current project" in warning for warning in warnings)
            )

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
