# -*- coding: utf-8 -*-
"""Coverage completion gate tests (#424 P6 slice 1)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
from engine_adapters import coverage as engine_coverage


def make_coverage_report(
    *,
    status: str = "ready",
    counts: dict | None = None,
    invariant_errors=(),
    source_changed: bool = False,
) -> engine_coverage.CoverageReport:
    classification_counts = dict(counts or {})
    return engine_coverage.CoverageReport(
        engine="renpy",
        adapter_version="1.1.8",
        adapter_behavior_digest="behavior",
        localization_mode="hybrid",
        source_fingerprint="source",
        project_snapshot_fingerprint="project",
        inventory_digest="inventory",
        classification_rules_digest="rules",
        extraction_overrides_digest="overrides",
        catalog_provenance={},
        catalog_freshness="known",
        audit_reason_codes=(),
        source_changed_during_scan=source_changed,
        files_scanned=(),
        candidate_count=sum(classification_counts.values()),
        classification_counts=classification_counts,
        translation_scope_counts={},
        analysis_scope_counts={},
        reason_counts={},
        coverage_status=status,
        coverage_digest="coverage-digest",
        generated_at="2026-09-12T00:00:00+00:00",
        invariant_errors=tuple(invariant_errors),
    )


class CoverageCompletionAssessmentTests(unittest.TestCase):
    def test_ready_and_attention_without_unresolved_structures_confirm(self):
        for status in ("ready", "attention"):
            with self.subTest(status=status):
                report = make_coverage_report(
                    status=status,
                    counts={"translatable": 1, "already_translated": 2, "unsupported": 1},
                )
                assessment = engine_coverage.evaluate_coverage_completion(report)
                self.assertEqual(assessment.completion, "confirmed")
                self.assertEqual(assessment.reasons, ())
                self.assertEqual(
                    assessment.to_dict()["coverage_status"],
                    status,
                )

    def test_unknown_and_parse_error_block_completion(self):
        report = make_coverage_report(
            status="block",
            counts={"translatable": 1, "unknown": 2, "parse_error": 1},
        )
        assessment = engine_coverage.evaluate_coverage_completion(report)
        self.assertEqual(assessment.completion, "unconfirmed")
        self.assertIn("coverage.status.block", assessment.reasons)
        self.assertIn("coverage.unknown_candidates", assessment.reasons)
        self.assertIn("coverage.parse_error_candidates", assessment.reasons)

    def test_invariant_and_source_change_block_completion(self):
        report = make_coverage_report(
            status="block",
            counts={"translatable": 1},
            invariant_errors=("candidate invariant failed",),
            source_changed=True,
        )
        assessment = engine_coverage.evaluate_coverage_completion(report)
        self.assertEqual(assessment.completion, "unconfirmed")
        self.assertIn("coverage.invariant_errors", assessment.reasons)
        self.assertIn("coverage.source_changed_during_scan", assessment.reasons)


class BatchBuildCoverageGuardTests(unittest.TestCase):
    def _run_create(self, report):
        snapshot = SimpleNamespace(report=report)
        jobs = batch.TranslationFileJobs(coverage_snapshot=snapshot)
        with (
            mock.patch.object(batch.legacy, "require_supported_generation_target"),
            mock.patch.object(batch, "freeze_runtime_routing_plan", return_value={}),
            mock.patch.object(batch.legacy, "run_prepare_steps"),
            mock.patch.object(batch.os.path, "isdir", return_value=True),
            mock.patch.object(batch, "collect_pending_file_jobs", return_value=jobs),
        ):
            return batch.create_batch_package(skip_prepare=True)

    def test_unconfirmed_zero_pending_is_a_structured_refusal(self):
        with self.assertRaises(cli_contract.MachineContractError) as ctx:
            self._run_create(
                make_coverage_report(
                    status="block",
                    counts={"unknown": 1, "parse_error": 1},
                )
            )
        self.assertEqual(ctx.exception.code_name, "COVERAGE_UNCONFIRMED")
        self.assertEqual(
            ctx.exception.semantic_exit_code,
            cli_contract.EXIT_BLOCKED,
        )
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(
            ctx.exception.details["coverage_status"],
            "block",
        )

    def test_confirmed_zero_pending_still_reports_no_work(self):
        self.assertIsNone(
            self._run_create(
                make_coverage_report(status="attention", counts={"unsupported": 1})
            )
        )

    def test_missing_coverage_evidence_is_a_structured_refusal(self):
        with self.assertRaises(cli_contract.MachineContractError) as ctx:
            self._run_create(None)
        self.assertEqual(ctx.exception.code_name, "COVERAGE_UNCONFIRMED")
        self.assertEqual(
            ctx.exception.details["reasons"],
            ["coverage.evidence_missing"],
        )


if __name__ == "__main__":
    unittest.main()
