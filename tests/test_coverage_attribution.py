"""Tests for the issue #426 read-only coverage block attribution helper."""

from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr

import translator_runtime as runtime
from engine_adapters.contracts import ProjectDiscoveryRequest
from engine_adapters.renpy import RenPyAdapter, build_translation_snapshot
from scripts import coverage_block_attribution as attribution


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "renpy_coverage_attribution"
PROJECT_ROOT = FIXTURE_ROOT / "game"
LOCALIZATION_ROOT = PROJECT_ROOT / "tl" / "schinese"
PINNED_TIME = "2026-09-11T00:00:00+00:00"


class CoverageAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = RenPyAdapter(legacy_module=runtime)
        cls.request = ProjectDiscoveryRequest(
            project_root=str(PROJECT_ROOT),
            localization_root=str(LOCALIZATION_ROOT),
            target_language="schinese",
        )
        cls.snapshot = build_translation_snapshot(
            cls.adapter,
            cls.request,
            include_occurrences=False,
            include_task_payloads=False,
        )

    def payload(self, **kwargs):
        options = {
            "generated_at": PINNED_TIME,
        }
        options.update(kwargs)
        return attribution.build_attribution_report(
            self.snapshot,
            self.request,
            **options,
        )

    @staticmethod
    def find_group(payload, classification, structure_kind, reason_codes, review_flags=()):
        key = attribution.group_key(
            classification,
            structure_kind,
            reason_codes,
            review_flags,
        )
        matches = [group for group in payload["groups"] if group["group_key"] == key]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one group for {key!r}, found {len(matches)}")
        return matches[0]

    def test_groups_account_for_every_candidate_and_coverage_counts(self):
        payload = self.payload()

        groups = payload["groups"]
        self.assertEqual(
            sum(group["count"] for group in groups),
            payload["candidate_count"],
        )
        self.assertEqual(payload["candidate_count"], len(self.snapshot.inventory.candidates))

        grouped_classifications = Counter(
            {
                group["classification"]: sum(
                    item["count"] for item in groups if item["classification"] == group["classification"]
                )
                for group in groups
            }
        )
        for classification, expected in payload["coverage"]["classification_counts"].items():
            self.assertEqual(grouped_classifications[classification], expected, classification)

        candidate_ids = [
            locator["candidate_id"]
            for group in groups
            for locator in group["locators"]
        ]
        self.assertEqual(len(candidate_ids), payload["candidate_count"])
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))

        audit_reasons = set(payload["coverage"]["audit_reason_codes"])
        coverage_reasons = payload["coverage"]["reason_counts"]
        for code, count in payload["candidate_reason_counts"].items():
            expected = count + (1 if code in audit_reasons else 0)
            self.assertEqual(coverage_reasons[code], expected, code)
        self.assertEqual(payload["coverage"]["coverage_status"], "block")
        self.assertEqual(payload["coverage"]["invariant_errors"], [])
        self.assertTrue(payload["unresolved_requires_human_review"])
        self.assertEqual(
            payload["unresolved_requires_human_review"],
            sorted(payload["unresolved_requires_human_review"]),
        )
        automatic_categories = {
            group["automatic_evidence"]["category"] for group in groups
        }
        self.assertTrue(automatic_categories <= attribution.AUTO_CATEGORIES)
        effective_categories = {group["effective_category"] for group in groups}
        self.assertTrue(
            effective_categories
            <= attribution.AUTO_CATEGORIES | attribution.MANUAL_CATEGORIES
        )

    def test_fixture_exposes_major_root_causes_and_legitimate_controls(self):
        payload = self.payload()

        self.assertNotIn("renpy.tokenize_error", payload["candidate_reason_counts"])
        multiline_locators = [
            locator
            for group in payload["groups"]
            for locator in group["locators"]
            if locator.get("multiline")
        ]
        self.assertEqual(
            {
                (locator["line_hint"], locator["end_line_hint"])
                for locator in multiline_locators
            },
            {(22, 23), (27, 28)},
        )
        for group in payload["groups"]:
            if any(locator.get("multiline") for locator in group["locators"]):
                self.assertEqual(group["classification"], "already_translated")
                self.assertEqual(group["effective_category"], "translated")

        orphan_old = self.find_group(
            payload,
            "parse_error",
            "old_source_marker",
            ("renpy.old_new_pair", "renpy.source_marker_unpaired"),
            ("orphan_old_row_needs_catalog_evidence",),
        )
        self.assertEqual(orphan_old["count"], 1)
        self.assertEqual(orphan_old["effective_category"], "unknown")
        self.assertIn(
            "orphan_old_row_needs_catalog_evidence",
            orphan_old["review_flags"],
        )

        dynamic_template = self.find_group(
            payload,
            "unsupported",
            "dynamic_string_expression",
            ("renpy.dynamic_string_expression",),
            ("dynamic_text_may_be_player_visible",),
        )
        self.assertEqual(dynamic_template["count"], 1)
        self.assertEqual(dynamic_template["effective_category"], "unsupported_structure")
        self.assertIn(
            "dynamic_text_may_be_player_visible",
            dynamic_template["review_flags"],
        )

        legacy_old = self.find_group(
            payload,
            "unsupported",
            "nonstandard_old_source_marker",
            ("renpy.custom_statement_unsupported",),
            ("nonstandard_statement_needs_structure_evidence",),
        )
        self.assertEqual(legacy_old["count"], 1)
        self.assertEqual(legacy_old["effective_category"], "unsupported_structure")

        single_quote_unknown = self.find_group(
            payload,
            "unknown",
            "unknown_string_structure",
            ("renpy.visibility_unknown",),
            (
                "single_quote_literal_needs_official_parser_check",
                "unmarked_string_needs_source_evidence",
            ),
        )
        self.assertEqual(single_quote_unknown["count"], 1)
        self.assertEqual(single_quote_unknown["effective_category"], "unknown")

        voice = self.find_group(
            payload,
            "explicitly_excluded",
            "voice_statement",
            ("renpy.voice_asset",),
        )
        asset = self.find_group(
            payload,
            "explicitly_excluded",
            "asset_literal",
            ("renpy.asset_path",),
        )
        for group in (voice, asset):
            self.assertEqual(group["effective_category"], "legitimate_exclusion")
            self.assertFalse(group["requires_human_review"])

        pending = [
            group for group in payload["groups"] if group["classification"] == "translatable"
        ]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["effective_category"], "pending_translation")
        self.assertEqual(pending[0]["count"], 1)

        translated_speaker_label = [
            group
            for group in payload["groups"]
            if "renpy.speaker_label_sibling_translated" in group["reason_codes"]
        ]
        self.assertEqual(len(translated_speaker_label), 1)
        self.assertEqual(translated_speaker_label[0]["count"], 1)
        self.assertEqual(translated_speaker_label[0]["effective_category"], "translated")

    def test_scan_completeness_marks_multiline_regions_without_uncovered_spans(self):
        payload = self.payload()
        scan = payload["scan_completeness"]

        self.assertEqual(scan["uncovered_span_count"], 0)
        self.assertEqual(scan["raw_span_count"], 17)
        self.assertEqual(scan["matched_span_count"], 16)
        self.assertEqual(scan["parse_error_region_span_count"], 1)
        self.assertEqual(scan["status"], "parse_error_regions")
        self.assertEqual(len(scan["files"]), 1)
        file_entry = scan["files"][0]
        self.assertEqual(file_entry["file_rel_path"], "attribution_samples.rpy")
        self.assertEqual(file_entry["uncovered_spans"], [])
        regions = {
            (span["line_hint"], span["quote"])
            for span in file_entry["parse_error_region_spans"]
        }
        self.assertEqual(regions, {(34, '"')})

    def test_manual_decisions_override_automatic_categories(self):
        dynamic_key = attribution.group_key(
            "unsupported",
            "dynamic_string_expression",
            ("renpy.dynamic_string_expression",),
            ("dynamic_text_may_be_player_visible",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            decisions_path = Path(tmp) / "decisions.json"
            decisions_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reviewer": {"type": "human", "id": "fixture-reviewer"},
                        "decisions": [
                            {
                                "group_key": dynamic_key,
                                "category": "unknown",
                                "rationale": "Reviewer cannot confirm the dynamic text is player-visible.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            decisions = attribution.load_decisions(decisions_path)
            payload = self.payload(decisions=decisions)

        self.assertEqual(payload["attribution_status"], "manual_review_applied")
        self.assertEqual(payload["manual_decisions"]["reviewer"]["id"], "fixture-reviewer")
        group = self.find_group(
            payload,
            "unsupported",
            "dynamic_string_expression",
            ("renpy.dynamic_string_expression",),
            ("dynamic_text_may_be_player_visible",),
        )
        self.assertEqual(group["human_judgment"]["category"], "unknown")
        self.assertEqual(group["effective_category"], "unknown")
        self.assertTrue(group["requires_human_review"])

    def test_unknown_decision_group_and_invalid_schema_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_group_path = Path(tmp) / "bad-group.json"
            bad_group_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reviewer": {"type": "human", "id": "fixture-reviewer"},
                        "decisions": [
                            {
                                "group_key": "unknown|missing|nothing|",
                                "category": "unknown",
                                "rationale": "not part of this run",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            decisions = attribution.load_decisions(bad_group_path)
            with self.assertRaises(attribution.AttributionError):
                self.payload(decisions=decisions)

            invalid_reviewer_path = Path(tmp) / "invalid-reviewer.json"
            invalid_reviewer_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "reviewer": {"type": "robot", "id": "r2"},
                        "decisions": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(attribution.AttributionError):
                attribution.load_decisions(invalid_reviewer_path)

    def test_report_is_deterministic_and_redacts_absolute_paths(self):
        first = self.payload()
        fresh_snapshot = build_translation_snapshot(
            RenPyAdapter(legacy_module=runtime),
            self.request,
            include_occurrences=False,
            include_task_payloads=False,
        )
        second = attribution.build_attribution_report(
            fresh_snapshot,
            self.request,
            generated_at=PINNED_TIME,
        )
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )
        self.assertEqual(first["aggregation_digest"], second["aggregation_digest"])
        self.assertEqual(len(first["aggregation_digest"]), 64)
        serialized = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertNotIn(str(FIXTURE_ROOT), serialized)
        self.assertNotIn(str(PROJECT_ROOT), serialized)
        self.assertNotIn("基线对白", serialized)
        self.assertNotIn('"raw_excerpt"', serialized)
        self.assertEqual(
            first["inputs"]["source_fingerprint"],
            self.snapshot.report.source_fingerprint,
        )
        self.assertEqual(
            first["inputs"]["inventory_digest"],
            self.snapshot.report.inventory_digest,
        )
        self.assertEqual(
            first["inputs"]["coverage_digest"],
            self.snapshot.report.coverage_digest,
        )
        self.assertTrue(first["inputs"]["inventory_digest"])
        self.assertTrue(first["inputs"]["coverage_digest"])

        markdown = attribution.render_markdown(first)
        self.assertIn("# Ren'Py coverage block attribution", markdown)
        self.assertIn("## Independent scan completeness", markdown)
        self.assertIn("## Evidence gaps and unknowns", markdown)
        self.assertNotIn("基线对白", markdown)

        with_excerpts = self.payload(include_excerpts=True)
        serialized_excerpts = json.dumps(with_excerpts, ensure_ascii=False, sort_keys=True)
        self.assertIn("基线对白", serialized_excerpts)

    def test_aggregation_digest_is_independent_of_root_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied_root = Path(tmp) / "copied-project"
            shutil.copytree(PROJECT_ROOT, copied_root)
            request = ProjectDiscoveryRequest(
                project_root=str(copied_root),
                localization_root=str(copied_root / "tl" / "schinese"),
                target_language="schinese",
            )
            snapshot = build_translation_snapshot(
                RenPyAdapter(legacy_module=runtime),
                request,
                include_occurrences=False,
                include_task_payloads=False,
            )
            payload = attribution.build_attribution_report(
                snapshot,
                request,
                generated_at=PINNED_TIME,
            )

        self.assertEqual(payload["aggregation_digest"], self.payload()["aggregation_digest"])
        self.assertEqual(
            payload["coverage"]["classification_counts"],
            self.payload()["coverage"]["classification_counts"],
        )

    def test_multiline_strings_are_inventory_candidates_not_parse_errors(self):
        candidates_by_line = {}
        for candidate in self.snapshot.inventory.candidates:
            line = int(candidate.locator.locator.get("line_hint") or 0)
            candidates_by_line.setdefault(line, []).append(candidate)

        for line_hint in (22, 23, 27, 28):
            self.assertFalse(
                any(
                    candidate.classification == "parse_error"
                    for candidate in candidates_by_line.get(line_hint, [])
                ),
                line_hint,
            )
        multiline = candidates_by_line[22][0]
        self.assertTrue(multiline.locator.locator.get("multiline"))
        self.assertEqual(multiline.locator.locator.get("end_line_hint"), 23)
        self.assertEqual(multiline.unit.text if multiline.unit else "", "多行对白第一行\n第二行")
        continued = candidates_by_line[27][0]
        self.assertTrue(continued.locator.locator.get("multiline"))
        self.assertEqual(continued.locator.locator.get("end_line_hint"), 28)
        self.assertEqual(continued.unit.text if continued.unit else "", "续行对白第一段第二段")

    def test_scan_document_string_spans_handles_comments_and_escapes(self):
        lines = [
            '# "comment only" must not count\n',
            '$ value = "escaped \\" quote"  # trailing comment\n',
        ]
        spans = attribution._scan_document_string_spans("sample.rpy", lines)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].line_index, 1)
        self.assertEqual(spans[0].start_col, 10)
        self.assertEqual(spans[0].end_col, 28)
        self.assertEqual(spans[0].quote, '"')

    def test_main_writes_reports_and_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_output = Path(tmp) / "attribution.json"
            markdown_output = Path(tmp) / "attribution.md"
            exit_code = attribution.main(
                [
                    str(LOCALIZATION_ROOT),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--target-language",
                    "schinese",
                    "--generated-at",
                    PINNED_TIME,
                    "--json-output",
                    str(json_output),
                    "--markdown-output",
                    str(markdown_output),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["generated_at"], PINNED_TIME)
            self.assertEqual(payload["coverage"]["coverage_status"], "block")
            self.assertIn(
                "# Ren'Py coverage block attribution",
                markdown_output.read_text(encoding="utf-8"),
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = attribution.main(
                    [str(Path(tmp) / "missing"), "--project-root", str(PROJECT_ROOT)]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("error:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
