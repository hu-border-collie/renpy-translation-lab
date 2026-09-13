# -*- coding: utf-8 -*-
"""CLI coverage status / review import tests (#424 P6 slice 3a)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
from engine_adapters import coverage as engine_coverage
from engine_adapters.contracts import Candidate, CandidateInventory, OpaqueLocator


def make_inventory(*candidates: Candidate) -> CandidateInventory:
    return CandidateInventory(
        engine="renpy",
        adapter_version="1.1.8",
        source_fingerprint="source",
        project_snapshot_fingerprint="project",
        candidates=tuple(candidates),
        files_scanned=(),
    )


def make_candidate(classification: str = "unknown") -> Candidate:
    return Candidate(
        candidate_id="candidate-1",
        engine="renpy",
        adapter_version="1.1.8",
        source_fingerprint="source",
        locator=OpaqueLocator(
            engine="renpy",
            locator_schema_version=1,
            locator={"file_rel_path": "chapter1.rpy", "line": 7},
        ),
        raw_excerpt='"unresolved text"',
        structure_kind="say",
        classification=classification,
        reason_codes=("renpy.dynamic_string_expression",),
        translation_scope="translation",
        analysis_scope="analysis",
    )


def make_report(inventory: CandidateInventory, *, status: str = "ready", counts=None):
    return engine_coverage.CoverageReport(
        engine="renpy",
        adapter_version="1.1.8",
        adapter_behavior_digest="behavior",
        localization_mode="hybrid",
        source_fingerprint="source",
        project_snapshot_fingerprint="project",
        inventory_digest=engine_coverage.inventory_digest(inventory),
        classification_rules_digest="rules",
        extraction_overrides_digest="overrides",
        catalog_provenance={},
        catalog_freshness="known",
        audit_reason_codes=(),
        source_changed_during_scan=False,
        files_scanned=(),
        candidate_count=len(inventory.candidates),
        classification_counts=dict(counts or {}),
        translation_scope_counts={},
        analysis_scope_counts={},
        reason_counts={},
        coverage_status=status,
        coverage_digest="coverage-digest",
        generated_at="2026-09-13T00:00:00+00:00",
    )


def make_review(report, *, status: str = "agent_reviewed", source_fingerprint=None):
    record = engine_coverage.build_review_template(
        report, review_policy="agent_or_human"
    )
    record["status"] = status
    if status != "pending":
        record["reviewer"] = {
            "type": "agent",
            "id": "agent-1",
            "tool": "unit-test",
            "model": "",
            "session": "",
        }
        record["confirmed_at"] = "2026-09-13T00:00:00Z"
    if source_fingerprint is not None:
        record["source_fingerprint"] = source_fingerprint
    return record


class CoverageStatusTests(unittest.TestCase):
    def test_confirmed_status_payload(self):
        inventory = make_inventory()
        report = make_report(inventory, status="attention", counts={"unsupported": 1})
        record = make_review(report)
        review_path = "/game/work/translation_context/coverage_review.json"
        with (
            mock.patch.object(
                batch,
                "build_translation_snapshot",
                return_value=SimpleNamespace(report=report, inventory=inventory),
            ),
            mock.patch.object(
                batch,
                "load_coverage_review_for_project",
                return_value=(record, review_path, ""),
            ),
        ):
            payload = batch.collect_coverage_status()
        self.assertEqual(payload["status"], "confirmed")
        self.assertTrue(payload["confirmed"])
        self.assertEqual(payload["coverage_status"], "attention")
        self.assertEqual(payload["review_path"], review_path)
        self.assertEqual(payload["gate"]["review_status"], "agent_reviewed")
        self.assertEqual(payload["unresolved_candidate_count"], 0)

    def test_missing_review_is_not_confirmed(self):
        inventory = make_inventory()
        report = make_report(inventory, status="attention")
        with (
            mock.patch.object(
                batch,
                "build_translation_snapshot",
                return_value=SimpleNamespace(report=report, inventory=inventory),
            ),
            mock.patch.object(
                batch,
                "load_coverage_review_for_project",
                return_value=(None, "/game/work/translation_context/coverage_review.json", ""),
            ),
        ):
            payload = batch.collect_coverage_status()
        self.assertEqual(payload["status"], "review_missing")
        self.assertFalse(payload["confirmed"])
        self.assertIn("coverage.review_missing", payload["gate"]["reasons"])

    def test_unresolved_candidates_are_summarized_with_locator(self):
        candidate = make_candidate("unknown")
        inventory = make_inventory(candidate)
        report = make_report(inventory, status="block", counts={"unknown": 1})
        with (
            mock.patch.object(
                batch,
                "build_translation_snapshot",
                return_value=SimpleNamespace(report=report, inventory=inventory),
            ),
            mock.patch.object(
                batch,
                "load_coverage_review_for_project",
                return_value=(None, "", ""),
            ),
        ):
            payload = batch.collect_coverage_status()
            output = io.StringIO()
            with redirect_stdout(output):
                batch.print_coverage_status(payload)
        self.assertEqual(payload["status"], "coverage_unconfirmed")
        self.assertEqual(payload["unresolved_candidate_count"], 1)
        summary = payload["unresolved_candidates"][0]
        self.assertEqual(summary["classification"], "unknown")
        self.assertEqual(
            summary["locator"]["locator"]["file_rel_path"],
            "chapter1.rpy",
        )
        self.assertIn("chapter1.rpy:7", output.getvalue())
        self.assertIn("coverage.status.block", payload["gate"]["reasons"])


class CoverageReviewImportTests(unittest.TestCase):
    def _run_import(self, *, record, tmp, dry_run=False, source_fingerprint=None):
        inventory = make_inventory()
        report = make_report(inventory, status="attention")
        record = record(report) if callable(record) else record
        review_file = Path(tmp) / "review.json"
        review_file.write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        args = SimpleNamespace(file=str(review_file), dry_run=dry_run)
        with (
            mock.patch.object(batch.legacy, "BASE_DIR", tmp),
            mock.patch.object(
                batch,
                "build_translation_snapshot",
                return_value=SimpleNamespace(report=report, inventory=inventory),
            ),
        ):
            return batch.run_coverage_review_import(args), record

    def test_import_installs_review_at_project_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, record = self._run_import(record=make_review, tmp=tmp)
            target = Path(tmp) / "translation_context" / "coverage_review.json"
            self.assertEqual(payload["status"], "imported")
            self.assertEqual(payload["review_path"], str(target))
            self.assertTrue(target.is_file())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), record)
            self.assertTrue(payload["gate"]["confirmed"])

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload, _record = self._run_import(
                record=make_review, tmp=tmp, dry_run=True
            )
            target = Path(tmp) / "translation_context" / "coverage_review.json"
            self.assertEqual(payload["status"], "dry_run")
            self.assertFalse(target.exists())

    def test_pending_review_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cli_contract.MachineContractError) as ctx:
                self._run_import(
                    record=lambda report: make_review(report, status="pending"),
                    tmp=tmp,
                )
        self.assertEqual(ctx.exception.code_name, "COVERAGE_REVIEW_PENDING")

    def test_stale_review_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cli_contract.MachineContractError) as ctx:
                self._run_import(
                    record=lambda report: make_review(
                        report, source_fingerprint="other-source"
                    ),
                    tmp=tmp,
                )
        self.assertEqual(ctx.exception.code_name, "COVERAGE_REVIEW_STALE")


class CoverageCliContractTests(unittest.TestCase):
    def test_parser_registers_coverage_commands(self):
        parser = batch.build_arg_parser()
        status_args = parser.parse_args(["coverage-status"])
        self.assertEqual(status_args.command, "coverage-status")
        import_args = parser.parse_args(
            ["coverage-review-import", "--file", "review.json"]
        )
        self.assertEqual(import_args.command, "coverage-review-import")
        self.assertEqual(import_args.file, "review.json")

    def test_machine_envelope_uses_gate_status(self):
        payload = {
            "status": "review_missing",
            "confirmed": False,
            "gate": {"status": "review_missing", "confirmed": False},
        }
        envelope = batch.build_machine_success_envelope(
            "coverage-status", payload, SimpleNamespace()
        )
        self.assertEqual(envelope["status"], "review_missing")
        self.assertIn("gate", envelope["result"])

    def test_strict_exit_codes_map_gate_statuses(self):
        def envelope(status):
            return cli_contract.success_envelope(
                "coverage-status", status=status, result={}
            )

        self.assertEqual(
            cli_contract.strict_exit_code(envelope("confirmed")),
            cli_contract.EXIT_OK,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(envelope("review_missing")),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(envelope("coverage_unconfirmed")),
            cli_contract.EXIT_NEEDS_ACTION,
        )
        self.assertEqual(
            cli_contract.strict_exit_code(envelope("review_invalid")),
            cli_contract.EXIT_BLOCKED,
        )

    def test_coverage_commands_are_offline(self):
        self.assertIn("coverage-status", batch.OFFLINE_BATCH_COMMANDS)
        self.assertIn("coverage-review-import", batch.OFFLINE_BATCH_COMMANDS)
        self.assertIn("coverage-status", batch.MACHINE_OUTPUT_COMMANDS)


if __name__ == "__main__":
    unittest.main()
