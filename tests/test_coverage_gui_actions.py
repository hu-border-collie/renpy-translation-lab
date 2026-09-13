"""Coverage/review GUI presentation and import contract helpers (#424 P6 S3c)."""

from __future__ import annotations

import unittest
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
from gui_qt.coverage_actions import (
    format_coverage_detail_facts,
    format_coverage_facts,
    import_coverage_review,
)
from gui_qt.doctor_report import summarize_doctor_report


def _coverage_payload():
    return {
        "status": "attention",
        "completion": "unconfirmed",
        "coverage_digest": "abc123",
        "candidate_count": 4,
        "classification_counts": {
            "translatable": 2,
            "already_translated": 1,
            "unknown": 1,
        },
        "unknown_count": 1,
        "parse_error_count": 0,
        "unsupported_count": 0,
        "gate": {
            "status": "review_missing",
            "confirmed": False,
            "coverage_status": "attention",
            "review_status": "missing",
            "review_policy": "agent_or_human",
            "review_policy_satisfied": False,
            "unresolved_findings": 0,
            "reasons": ["coverage.unknown_candidates", "coverage.review_missing"],
        },
        "review_status": "missing",
        "review_policy": "agent_or_human",
        "unresolved_findings": 0,
        "unresolved_candidate_count": 1,
        "unresolved_candidates": [
            {
                "candidate_id": "cand1:deadbeef",
                "classification": "unknown",
                "structure_kind": "dialogue_string",
                "reason_codes": ["renpy.unknown_structure"],
                "locator": {
                    "engine": "renpy",
                    "locator_schema_version": 1,
                    "locator": {
                        "file_rel_path": "game/tl/schinese/script.rpy",
                        "line_hint": 7,
                    },
                },
                "excerpt": 'say "???"',
            }
        ],
        "review_path": "C:/Game/work/translation_context/coverage_review.json",
    }


class CoverageGuiActionsTests(unittest.TestCase):
    def test_format_coverage_facts_reports_unconfirmed_gate(self):
        facts = format_coverage_facts(_coverage_payload())
        text = "\n".join(facts)
        self.assertIn("文本覆盖", text)
        self.assertIn("覆盖未确认", text)
        self.assertIn("存在未识别候选", text)
        self.assertIn("缺少独立核对", text)
        self.assertIn("未解决候选：1 条", text)

    def test_format_coverage_facts_reports_confirmed_gate(self):
        coverage = _coverage_payload()
        coverage["status"] = "ready"
        coverage["completion"] = "confirmed"
        coverage["classification_counts"] = {"translatable": 3, "unknown": 0}
        coverage["unresolved_candidate_count"] = 0
        coverage["unresolved_candidates"] = []
        coverage["gate"] = {
            "status": "confirmed",
            "confirmed": True,
            "review_status": "human_reviewed",
            "review_policy": "human_required",
            "review_policy_satisfied": True,
            "unresolved_findings": 0,
            "reasons": [],
        }
        facts = format_coverage_facts(coverage)
        text = "\n".join(facts)
        self.assertIn("覆盖已确认", text)
        self.assertIn("人工已核对", text)
        self.assertIn("必须人工", text)
        self.assertIn("核对门禁已确认", text)
        self.assertNotIn("未解决候选", text)

    def test_format_coverage_facts_reports_each_unconfirmed_gate_state(self):
        cases = (
            ("review_invalid", "invalid", "coverage.review_invalid", "核对记录无效"),
            ("review_stale", "stale", "coverage.review_stale", "核对记录已过期"),
            ("review_pending", "pending", "coverage.review_pending", "核对尚未完成"),
            (
                "review_unresolved_findings",
                "agent_reviewed",
                "coverage.review_unresolved_findings",
                "仍有未解决 finding",
            ),
            (
                "review_policy_unsatisfied",
                "agent_reviewed",
                "coverage.review_policy_unsatisfied",
                "核对策略未满足",
            ),
        )
        for gate_status, review_status, reason, label in cases:
            with self.subTest(gate_status=gate_status):
                coverage = _coverage_payload()
                coverage["gate"] = {
                    "status": gate_status,
                    "confirmed": False,
                    "review_status": review_status,
                    "review_policy": "human_required",
                    "review_policy_satisfied": False,
                    "unresolved_findings": 0,
                    "reasons": [reason],
                }
                facts = "\n".join(format_coverage_facts(coverage))
                self.assertIn(label, facts)

    def test_format_coverage_detail_facts_includes_locator(self):
        details = format_coverage_detail_facts(_coverage_payload())
        text = "\n".join(details)
        self.assertIn("game/tl/schinese/script.rpy:7", text)
        self.assertIn("[unknown]", text)
        self.assertIn("renpy.unknown_structure", text)
        self.assertIn("coverage_review.json", text)

    def test_format_coverage_facts_empty_is_empty(self):
        self.assertEqual(format_coverage_facts(None), [])
        self.assertEqual(format_coverage_detail_facts({}), [])

    def test_import_coverage_review_uses_shared_cli_contract(self):
        payload = {
            "status": "imported",
            "review_path": "C:/Game/work/translation_context/coverage_review.json",
            "review_status": "human_reviewed",
            "review_policy": "human_required",
            "unresolved_findings": 0,
        }
        with mock.patch.object(
            batch,
            "run_coverage_review_import",
            return_value=payload,
        ) as mocked:
            result = import_coverage_review("C:/review.json")

        self.assertTrue(result.ok)
        self.assertEqual(result.payload, payload)
        passed_args = mocked.call_args.args[0]
        self.assertEqual(passed_args.file, "C:/review.json")
        self.assertFalse(passed_args.dry_run)

    def test_import_coverage_review_surfaces_structured_refusal(self):
        error = cli_contract.MachineContractError(
            "coverage review is stale for the current source / coverage",
            code_name="COVERAGE_REVIEW_STALE",
            suggested_action="reexport_review",
            semantic_exit_code=cli_contract.EXIT_BLOCKED,
            retryable=False,
        )
        with mock.patch.object(
            batch,
            "run_coverage_review_import",
            side_effect=error,
        ):
            result = import_coverage_review("C:/review.json")

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "COVERAGE_REVIEW_STALE")
        self.assertEqual(result.suggested_action, "reexport_review")
        self.assertIn("COVERAGE_REVIEW_STALE", result.user_message())

    def test_import_coverage_review_requires_a_file(self):
        result = import_coverage_review("")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "COVERAGE_REVIEW_FILE_REQUIRED")

    def test_summarize_doctor_report_includes_coverage_facts_and_details(self):
        report = {
            "base_dir": "C:/Game/work",
            "tl_dir": "C:/Game/work/game/tl/schinese",
            "tl_subdir": "game/tl/schinese",
            "tl_exists": True,
            "language": "schinese",
            "catalog_language": "schinese",
            "generation_target": "schinese",
            "generation_target_supported": True,
            "mode": "existing_tl_only",
            "is_work_root": True,
            "work_dir": "C:/Game/work",
            "work_exists": True,
            "work_empty": False,
            "original_game_dir": "",
            "counts": {
                "rpy_files": 1,
                "translate_blocks": 1,
                "string_sections": 0,
                "old_lines": 0,
                "new_lines": 0,
                "commented_original_lines": 1,
            },
            "pending_task_count": 1,
            "pending_file_count": 1,
            "translated_task_count": 0,
            "total_task_count": 1,
            "warnings": [],
            "recommendations": [],
            "coverage": _coverage_payload(),
        }
        summary = summarize_doctor_report(report, exit_code=0, api_key_count=1)
        facts_text = "\n".join(summary.facts)
        detail_text = "\n".join(summary.detail_facts or [])
        self.assertIn("文本覆盖", facts_text)
        self.assertIn("game/tl/schinese/script.rpy:7", detail_text)


if __name__ == "__main__":
    unittest.main()
