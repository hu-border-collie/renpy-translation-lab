"""Policy tests for quoted comments and orphan old rows (issue #464)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


class RenPyCommentPolicyTests(unittest.TestCase):
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

    def test_comment_without_target_is_excluded_not_parse_error(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # TODO "internal-only-note"\n'
        )

        self.assertEqual(snapshot.report.classification_counts["parse_error"], 0)
        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.structure_kind == "comment_without_target"
        )
        self.assertEqual(candidate.classification, "explicitly_excluded")
        self.assertIn("renpy.non_player_visible_literal", candidate.reason_codes)
        self.assertEqual(candidate.translation_scope, "exclude")

    def test_dangling_source_like_comment_with_later_target_stays_parse_error(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    # "Dangling source"\n'
            '    old "Old without new"\n'
            '    text f"Dynamic {name}"\n'
            '    "unterminated\n'
        )

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertGreaterEqual(len(parse_errors), 2)
        self.assertTrue(
            any(
                "renpy.source_marker_unpaired" in candidate.reason_codes
                for candidate in parse_errors
            )
        )
        self.assertFalse(
            any(
                candidate.structure_kind == "comment_without_target"
                for candidate in snapshot.inventory.candidates
            )
        )

    def test_orphan_old_row_stays_fail_closed_parse_error(self):
        snapshot = self.snapshot_for(
            "translate schinese strings:\n"
            '    old "Only an old row remains."\n'
        )

        parse_errors = [
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(len(parse_errors), 1)
        self.assertEqual(parse_errors[0].structure_kind, "old_source_marker")
        self.assertIn("renpy.source_marker_unpaired", parse_errors[0].reason_codes)

    def test_old_marker_with_extra_token_is_unsupported(self):
        snapshot = self.snapshot_for(
            "translate schinese strings:\n"
            '    old weird "Extra token"\n'
        )

        candidate = next(
            candidate
            for candidate in snapshot.inventory.candidates
            if candidate.classification == "unsupported"
        )
        self.assertEqual(candidate.structure_kind, "nonstandard_old_source_marker")
        self.assertIn("renpy.custom_statement_unsupported", candidate.reason_codes)


if __name__ == "__main__":
    unittest.main()
