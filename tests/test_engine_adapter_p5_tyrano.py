# -*- coding: utf-8 -*-
"""Read-only TyranoScript V600+ adapter tests (#265 P5 / #399)."""

from __future__ import annotations

from collections import Counter
import json
import tempfile
import unittest
from pathlib import Path

from engine_adapters.coverage import build_coverage_report
from engine_adapters.contracts import InventoryPolicy, ProjectDiscoveryRequest
from engine_adapters.tyrano import (
    ADAPTER_VERSION,
    TyranoAdapter,
    build_translation_snapshot,
    parse_tyrano_scenario,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tyranoscript_v600"


class TyranoAdapterFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = TyranoAdapter()
        cls.snapshot = build_translation_snapshot(
            cls.adapter,
            ProjectDiscoveryRequest(
                project_root=str(FIXTURE_DIR),
                localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                target_language="ch",
            ),
        )

    def test_capabilities_are_read_only_hybrid(self):
        capabilities = self.adapter.capabilities()
        self.assertEqual(capabilities.engine, "tyrano")
        self.assertEqual(capabilities.adapter_version, ADAPTER_VERSION)
        self.assertTrue(capabilities.source_inventory)
        self.assertTrue(capabilities.native_catalog)
        self.assertTrue(capabilities.native_catalog_required_for_writeback)
        self.assertEqual(capabilities.declarative_writeback, ())
        self.assertEqual(capabilities.relocation, False)

    def test_discovery_scans_ks_and_fingerprint_behavior_config(self):
        project = self.snapshot.project
        docs = {document.file_rel_path for document in project.source_documents}
        self.assertEqual(
            docs,
            {
                "data/scenario/scene1.ks",
                "data/scenario/choices.ks",
                "data/scenario/broken.ks",
                "data/system/Config.tjs",
            },
        )
        self.assertTrue(project.source_fingerprint)
        self.assertTrue(project.project_snapshot_fingerprint)
        self.assertEqual(project.catalog_provenance["target_language"], "ch")
        self.assertEqual(
            project.catalog_provenance["catalog_rel_path"],
            "data/others/lang/ch.json",
        )

    def test_inventory_classification_contract(self):
        inventory = self.snapshot.inventory
        counts = Counter(candidate.classification for candidate in inventory.candidates)
        self.assertEqual(
            counts,
            Counter(
                {
                    "explicitly_excluded": 17,
                    "already_translated": 12,
                    "parse_error": 4,
                    "unsupported": 2,
                    "translatable": 1,
                    "unknown": 1,
                }
            ),
        )
        # No candidate is silently dropped: every parser node / comment line is
        # inventoried exactly once.
        self.assertEqual(len(inventory.candidates), 37)
        self.assertEqual(len({candidate.candidate_id for candidate in inventory.candidates}), 37)

    def test_candidate_catalog_links_resolve_to_live_catalog(self):
        catalog = json.loads((FIXTURE_DIR / "data" / "others" / "lang" / "ch.json").read_text(encoding="utf-8"))
        for candidate in self.snapshot.inventory.candidates:
            link = candidate.catalog_link
            if link is None:
                continue
            value = catalog
            for part in link["path"]:
                self.assertIn(part, value, candidate.candidate_id)
                value = value[part]
            self.assertEqual(link["translation"], value, candidate.candidate_id)

    def test_malformed_lines_are_parse_errors_not_translatable(self):
        parse_errors = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "parse_error"
        ]
        self.assertEqual(len(parse_errors), 4)
        for candidate in parse_errors:
            self.assertNotEqual(candidate.classification, "translatable")
            self.assertNotEqual(candidate.classification, "already_translated")
            self.assertIn(
                candidate.locator.locator["file_rel_path"],
                "data/scenario/broken.ks",
            )

    def test_escaped_quote_is_catalog_matched(self):
        candidates = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.locator.locator.get("name") == "ptext"
            and candidate.raw_excerpt == "It's fine"
        ]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].classification, "already_translated")
        self.assertEqual(candidates[0].catalog_link["translation"], "没问题")

    def test_iscript_text_is_explicitly_excluded(self):
        choices = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.locator.locator["file_rel_path"] == "data/scenario/choices.ks"
        ]
        # ``Candidate`` has no ``line`` property; use locator line.
        iscript_text = [
            candidate
            for candidate in choices
            if candidate.locator.locator["line"] in (14, 15)
            and candidate.structure_kind == "text"
        ]
        self.assertEqual(len(iscript_text), 2)
        self.assertTrue(
            all(candidate.classification == "explicitly_excluded" for candidate in iscript_text)
        )
        self.assertTrue(
            all("tyrano.iscript_content" in candidate.reason_codes for candidate in iscript_text)
        )

    def test_unknown_macro_and_dynamic_parameter_are_not_writable(self):
        unknown = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "unknown"
        ]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(
            unknown[0].reason_codes,
            ("tyrano.unregistered_macro_invocation",),
        )
        dynamic = [
            candidate
            for candidate in self.snapshot.inventory.candidates
            if candidate.classification == "unsupported"
            and "tyrano.dynamic_parameter_expression" in candidate.reason_codes
        ]
        self.assertEqual(len(dynamic), 1)
        self.assertEqual(dynamic[0].raw_excerpt, "&sf.button_label")

    def test_coverage_status_is_block_without_review_overrides(self):
        report = self.snapshot.report
        self.assertEqual(report.coverage_status, "block")
        self.assertEqual(report.catalog_freshness, "unknown")
        self.assertFalse(report.invariant_errors)
        self.assertEqual(report.engine, "tyrano")
        self.assertTrue(report.coverage_digest)

    def test_extract_occurrences_only_returns_extractable_candidates(self):
        inventory = self.snapshot.inventory
        approved = [
            candidate.candidate_id
            for candidate in inventory.candidates
            if candidate.classification in {"translatable", "already_translated"}
        ]
        self.assertEqual(len(approved), 13)
        occurrences = self.adapter.extract_occurrences(
            self.snapshot.project,
            inventory,
            approved,
        )
        self.assertEqual(len(occurrences), 13)
        self.assertEqual(
            {occurrence.locator.engine for occurrence in occurrences},
            {"tyrano"},
        )
        for occurrence in occurrences:
            self.assertTrue(occurrence.unit.id)
            self.assertTrue(occurrence.unit.source)
            self.assertTrue(occurrence.content_fingerprint)

        with self.assertRaises(ValueError):
            self.adapter.extract_occurrences(
                self.snapshot.project,
                inventory,
                [approved[0], approved[0]],
            )

    def test_write_side_operations_fail_closed(self):
        with self.assertRaises(NotImplementedError):
            self.adapter.relocate_occurrences(self.snapshot.project, (), ())
        with self.assertRaises(NotImplementedError):
            self.adapter.validate_translation(self.snapshot.occurrences[0], "译文")
        with self.assertRaises(NotImplementedError):
            self.adapter.build_writeback_plan(self.snapshot.project, (), ())


class TyranoAdapterNegativeFixtureTests(unittest.TestCase):
    def test_missing_catalog_blocks_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario_dir = root / "data" / "scenario"
            scenario_dir.mkdir(parents=True)
            (scenario_dir / "missing_catalog.ks").write_text(
                '; comment\n*start|Start\nHello, world!\n',
                encoding="utf-8",
                newline="\n",
            )
            adapter = TyranoAdapter()
            project = adapter.discover_project(
                ProjectDiscoveryRequest(
                    project_root=str(root),
                    localization_root=str(root / "data" / "others" / "lang"),
                    target_language="ch",
                )
            )
            inventory = adapter.inventory_candidates(project, InventoryPolicy())
            draft = adapter.audit_extraction(project, inventory)
            report = build_coverage_report(
                project,
                inventory,
                draft,
                adapter_behavior_digest=adapter.behavior_digest(),
            )
            self.assertEqual(draft.catalog_freshness, "missing")
            self.assertIn("tyrano.catalog.missing_file", draft.reason_codes)
            self.assertEqual(report.coverage_status, "block")

    def test_missing_project_root_fails_closed(self):
        adapter = TyranoAdapter()
        with self.assertRaises(ValueError):
            adapter.discover_project(
                ProjectDiscoveryRequest(
                    project_root="",
                    localization_root="",
                    target_language="ch",
                )
            )

    def test_include_files_filter(self):
        adapter = TyranoAdapter()
        project = adapter.discover_project(
            ProjectDiscoveryRequest(
                project_root=str(FIXTURE_DIR),
                localization_root=str(FIXTURE_DIR / "data" / "others" / "lang"),
                target_language="ch",
                include_files=("data/scenario/scene1.ks",),
            )
        )
        rel_paths = {document.file_rel_path for document in project.source_documents}
        self.assertIn("data/scenario/scene1.ks", rel_paths)
        self.assertNotIn("data/scenario/choices.ks", rel_paths)

    def test_parse_tyrano_scenario_normalizes_keep_space_level(self):
        lines = ['[ptext layer=0 x=10 text="  Keep  inner " ]']
        result = parse_tyrano_scenario(lines, keep_space="1")
        ptext = next(node for node in result.nodes if node.name == "ptext")
        self.assertEqual(ptext.pm["text"], "Keepinner")
        result2 = parse_tyrano_scenario(lines, keep_space="2")
        ptext2 = next(node for node in result2.nodes if node.name == "ptext")
        self.assertEqual(ptext2.pm["text"], "Keep  inner")
        result3 = parse_tyrano_scenario(lines, keep_space="3")
        ptext3 = next(node for node in result3.nodes if node.name == "ptext")
        self.assertEqual(ptext3.pm["text"], "  Keep  inner ")


if __name__ == "__main__":
    unittest.main()
