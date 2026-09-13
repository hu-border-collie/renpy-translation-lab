"""Offline end-to-end corpus for the Ren'Py and Tyrano engine adapters (#424 P6)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.coverage import export_coverage_package
from engine_adapters.renpy import (
    RenPyAdapter,
    build_translation_snapshot as build_renpy_snapshot,
)
from engine_adapters.tyrano import (
    TyranoAdapter,
    build_translation_snapshot as build_tyrano_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RENPY_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "golden_batch_minimal"
TYRANO_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tyranoscript_v600"

RENPY_CORPUS = {
    "candidate_count": 11,
    "occurrence_count": 8,
    "coverage_status": "attention",
    "classification_counts": {
        "already_translated": 2,
        "explicitly_excluded": 3,
        "parse_error": 0,
        "translatable": 6,
        "unknown": 0,
        "unsupported": 0,
    },
}

TYRANO_CORPUS = {
    "candidate_count": 37,
    "occurrence_count": 13,
    "coverage_status": "block",
    "classification_counts": {
        "already_translated": 12,
        "explicitly_excluded": 17,
        "parse_error": 4,
        "translatable": 1,
        "unknown": 1,
        "unsupported": 2,
    },
}


class EngineOfflineCorpusTests(unittest.TestCase):
    maxDiff = None

    def _assert_corpus(self, snapshot, expected: dict) -> None:
        report = snapshot.report
        self.assertEqual(len(snapshot.inventory.candidates), expected["candidate_count"])
        self.assertEqual(len(snapshot.occurrences), expected["occurrence_count"])
        self.assertEqual(report.coverage_status, expected["coverage_status"])
        self.assertEqual(
            dict(report.classification_counts),
            expected["classification_counts"],
        )
        self.assertEqual(report.invariant_errors, ())
        self.assertRegex(report.inventory_digest, r"^[0-9a-f]{64}$")
        self.assertRegex(report.coverage_digest, r"^[0-9a-f]{64}$")

    def _assert_deterministic(self, first, second) -> None:
        # Digests intentionally are not frozen across operating systems:
        # Git may check text fixtures out with CRLF on Windows, which changes
        # source bytes and hashes. Corpus regression freezes counts/classes and
        # requires stable digests within one checkout.
        self.assertEqual(
            first.report.inventory_digest,
            second.report.inventory_digest,
        )
        self.assertEqual(
            first.report.coverage_digest,
            second.report.coverage_digest,
        )

    def test_renpy_corpus_snapshot_and_coverage_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "work"
            tl_dir = root / "game" / "tl" / "schinese"
            tl_dir.parent.mkdir(parents=True)
            shutil.copytree(RENPY_FIXTURE / "tl", tl_dir)

            request = ProjectDiscoveryRequest(
                project_root=str(root),
                localization_root=str(tl_dir),
                target_language="schinese",
            )
            adapter = RenPyAdapter()
            first = build_renpy_snapshot(adapter, request)
            second = build_renpy_snapshot(adapter, request)

            package_dir = Path(tmp) / "coverage-renpy"
            paths = export_coverage_package(
                package_dir,
                first.project,
                first.inventory,
                first.report,
            )
            template = json.loads(
                Path(paths.review_template_path).read_text(encoding="utf-8")
            )
            self.assertTrue(Path(paths.report_path).is_file())

        self._assert_deterministic(first, second)
        self._assert_corpus(first, RENPY_CORPUS)
        self.assertEqual(template["review_policy"], "agent_or_human")

    def test_tyrano_corpus_snapshot_and_coverage_package(self):
        request = ProjectDiscoveryRequest(
            project_root=str(TYRANO_FIXTURE),
            localization_root=str(TYRANO_FIXTURE / "data" / "others" / "lang"),
            target_language="ch",
        )
        adapter = TyranoAdapter()
        first = build_tyrano_snapshot(adapter, request)
        second = build_tyrano_snapshot(adapter, request)

        self._assert_deterministic(first, second)
        self._assert_corpus(first, TYRANO_CORPUS)

        with tempfile.TemporaryDirectory() as tmp:
            paths = export_coverage_package(
                Path(tmp) / "coverage-tyrano",
                first.project,
                first.inventory,
                first.report,
            )
            template = json.loads(
                Path(paths.review_template_path).read_text(encoding="utf-8")
            )
            self.assertEqual(template["review_policy"], "agent_or_human")
            self.assertTrue(Path(paths.candidates_path).is_file())


if __name__ == "__main__":
    unittest.main()
