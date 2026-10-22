"""Reproducibility tests for canonical parse-error evidence (issue #463)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot


class RenPyDigestStabilityTests(unittest.TestCase):
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

    def test_parse_error_evidence_has_no_process_memory_address(self):
        snapshot = self.snapshot_for(
            "translate schinese chapter:\n"
            '    e f"动态问候 {player_name}"\n'
        )

        evidence_values = [
            str(candidate.evidence.get("parse_error") or "")
            for candidate in snapshot.inventory.candidates
            if candidate.evidence.get("parse_error")
        ]
        self.assertTrue(evidence_values)
        for value in evidence_values:
            self.assertNotIn("0x", value)
            self.assertNotIn(" at ", value)

    def test_inventory_and_coverage_digests_are_reproducible(self):
        text = (
            "translate schinese chapter:\n"
            '    e f"动态问候 {player_name}"\n'
            '    e "unterminated\n'
        )
        root, tl_dir = self.make_project(text)
        request = ProjectDiscoveryRequest(
            project_root=str(root),
            localization_root=str(tl_dir),
            target_language="schinese",
        )
        first = build_translation_snapshot(RenPyAdapter(legacy_module=runtime), request)
        second = build_translation_snapshot(RenPyAdapter(legacy_module=runtime), request)

        self.assertTrue(first.report.inventory_digest)
        self.assertTrue(first.report.coverage_digest)
        self.assertEqual(first.report.inventory_digest, second.report.inventory_digest)
        self.assertEqual(first.report.coverage_digest, second.report.coverage_digest)


if __name__ == "__main__":
    unittest.main()
