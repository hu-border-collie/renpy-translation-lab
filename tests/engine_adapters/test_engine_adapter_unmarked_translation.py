"""Policy tests for unmarked TL targets and dynamic strings (issue #462)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


class RenPyUnmarkedTranslationPolicyTests(unittest.TestCase):
    def make_project(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        tl_dir = root / "game" / "tl" / "schinese"
        tl_dir.mkdir(parents=True)
        (tl_dir / "script.rpy").write_text(text, encoding="utf-8")
        return root, tl_dir

    def snapshot_for(self, text: str):
        root, tl_dir = self.make_project(text)
        return build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            ),
        )

    def test_unmarked_single_quoted_target_is_already_translated(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            "    e '你好，世界。'\n"
        )

        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None
        )
        self.assertEqual(candidate.classification, "already_translated")
        self.assertIn(
            "renpy.catalog.translation_present_without_marker",
            candidate.reason_codes,
        )
        self.assertTrue(candidate.evidence.get("source_marker_missing"))
        self.assertTrue(candidate.unit.metadata.get("source_marker_missing"))
        self.assertEqual(snapshot.report.classification_counts["unknown"], 0)
        self.assertEqual(snapshot.report.classification_counts["parse_error"], 0)

    def test_unmarked_double_quoted_target_is_already_translated(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    e "你好。"\n'
        )

        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.unit is not None
        )
        self.assertEqual(candidate.classification, "already_translated")
        self.assertIn(
            "renpy.catalog.translation_present_without_marker",
            candidate.reason_codes,
        )

    def test_unmarked_non_target_short_literal_stays_unknown(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    e "e"\n'
        )

        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.structure_kind == "unknown_string_structure"
        )
        self.assertEqual(candidate.classification, "unknown")
        self.assertIn("renpy.visibility_unknown", candidate.reason_codes)
        self.assertNotIn(
            "renpy.catalog.translation_present_without_marker",
            candidate.reason_codes,
        )

    def test_excluded_code_and_asset_strings_do_not_expand_translation(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    $ internal = "中文内部值"\n'
            '    play music "音频/主题.ogg"\n'
        )

        classifications = {
            candidate.classification for candidate in snapshot.inventory.candidates
        }
        self.assertIn("explicitly_excluded", classifications)
        self.assertNotIn("already_translated", classifications)
        self.assertNotIn(
            "renpy.catalog.translation_present_without_marker",
            snapshot.report.reason_counts,
        )

    def test_dynamic_say_string_is_unsupported_not_auto_translated(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # e "Dynamic greeting."\n'
            '    e f"动态问候 {player_name}"\n'
        )

        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "unsupported"
        )
        self.assertEqual(candidate.structure_kind, "dynamic_string_expression")
        self.assertNotIn(
            "renpy.catalog.translation_present_without_marker",
            candidate.reason_codes,
        )


if __name__ == "__main__":
    unittest.main()
