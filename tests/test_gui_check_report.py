import unittest

from gui_qt.check_report import (
    WritebackSummary,
    build_recheck_cli_args,
    idle_writeback_summary_for_work_mode,
    parse_check_output,
    recheck_writeback_ready,
    summarize_apply_envelope,
    summarize_apply_output,
    summarize_check_envelope,
    summarize_check_output,
    summarize_manifest_writeback,
)
from gui_qt.work_modes import WorkMode


CHECK_OUTPUT_SAFE = """
Pending files: 2
Pending lines: 18
Failure items: 0
Recoverable valid items: 18
Safety status: safe
Check failure report: C:\\pkg\\check_failures.jsonl
"""

CHECK_OUTPUT_WARN = """
Pending files: 1
Pending lines: 4
Failure items: 2
Safety status: warn
Warn reasons:
- source_mismatch: 2
Check failure report: C:\\pkg\\check_failures.jsonl
"""

CHECK_OUTPUT_READY_WITH_WARNINGS = """
Pending files: 1
Pending lines: 4
Failure items: 0
Safety status: safe
Writeback gate: allow
Writeback blockers: 0
Quality gate: needs_review
Quality warnings: 2
Quality blockers: 0
Acknowledged warnings: 0
Check status: ready_with_warnings
Quality categories:
- quality.typography.cjk_latin_spacing: 2
Quality findings report: C:\\pkg\\quality_findings.jsonl
"""

APPLY_OUTPUT = """
Safety status: safe
Pending files: 2
Pending lines: 18
Applied files: 2
Applied lines: 18
Failures logged: 0
"""


class GuiCheckReportTests(unittest.TestCase):
    def test_parse_check_output_extracts_counts_and_reasons(self):
        parsed = parse_check_output(CHECK_OUTPUT_WARN)

        self.assertEqual(parsed["safety_status"], "warn")
        self.assertEqual(parsed["pending_files"], 1)
        self.assertEqual(parsed["pending_lines"], 4)
        self.assertEqual(parsed["failure_items"], 2)
        self.assertTrue(any("source_mismatch" in finding for finding in parsed["findings"]))

    def test_summarize_check_output_nonzero_exit_blocks_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=1,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "failed")
        self.assertFalse(summary.can_apply)

    def test_summarize_check_output_already_applied_blocks_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
            already_applied=True,
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)

    def test_summarize_ready_with_warnings_still_enables_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_READY_WITH_WARNINGS,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertTrue(summary.can_apply)
        self.assertIn("质量报警", summary.heading)
        self.assertIn("可交付", summary.message)
        self.assertIn("质量", "\n".join(summary.facts))

    def test_summarize_manifest_with_quality_warnings_still_enables_apply(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "last_check_summary": {
                    "safety_level": "safe",
                    "check_status": "ready_with_warnings",
                    "writeback_gate": {"decision": "allow", "can_apply": True},
                    "quality_gate": {
                        "decision": "needs_review",
                        "warning_count": 3,
                        "blocker_count": 0,
                    },
                    "quality_reason_counts": {
                        "quality.typography.cjk_latin_spacing": 3
                    },
                    "pending_files": 1,
                    "pending_lines": 4,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertTrue(summary.can_apply)
        self.assertIn("质量报警", summary.heading)
        self.assertTrue(any("cjk_latin_spacing" in item for item in summary.findings))

    def test_summarize_check_envelope_uses_ready_with_warnings(self):
        summary = summarize_check_envelope(
            {
                "ok": True,
                "status": "ready_with_warnings",
                "result": {
                    "check": {
                        "safety_level": "safe",
                        "check_status": "ready_with_warnings",
                        "writeback_gate": {"decision": "allow", "can_apply": True},
                        "quality_gate": {"decision": "needs_review", "warning_count": 2},
                        "pending_files": 1,
                        "pending_lines": 4,
                    }
                },
            },
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertTrue(summary.can_apply)
        self.assertIn("质量报警", summary.heading)

    def test_summarize_safe_check_enables_apply(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)
        self.assertIn("2 个文件", "\n".join(summary.facts))

    def test_summarize_warn_check_blocks_apply(self):
        summary = summarize_check_output(CHECK_OUTPUT_WARN, exit_code=0)

        self.assertEqual(summary.status, "warn")
        self.assertFalse(summary.can_apply)
        self.assertTrue(summary.findings)
        self.assertTrue(
            any(fact.startswith("注意：") and "source_mismatch" in fact for fact in summary.facts)
        )
        self.assertFalse(any(fact.startswith("- ") for fact in summary.facts))

    def test_summarize_warn_check_points_to_remediation_commands(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_WARN,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "warn")
        self.assertIn("查看问题清单", summary.message)
        self.assertEqual(summary.message.count("补译"), 1)
        self.assertIn("重新检查", summary.message)
        self.assertIn("可写回", summary.message)

    def test_summarize_check_envelope_uses_structured_counts_and_reasons(self):
        summary = summarize_check_envelope(
            {
                "ok": True,
                "status": "warn",
                "result": {
                    "check": {
                        "pending_files": 1,
                        "pending_lines": 4,
                        "failure_items": 2,
                        "safety_reasons": {"warn": {"source_mismatch": 2}},
                    }
                },
                "artifacts": {"check_report": r"C:\pkg\check.jsonl"},
            },
            exit_code=0,
            manifest_path=r"C:\pkg\manifest.json",
        )

        self.assertEqual(summary.status, "warn")
        self.assertIn("1 个文件", "\n".join(summary.facts))
        self.assertTrue(any("source_mismatch" in item for item in summary.findings))

    def test_summarize_apply_envelope_uses_structured_result(self):
        summary = summarize_apply_envelope(
            {
                "ok": True,
                "status": "applied",
                "result": {
                    "apply": {
                        "applied_files": 2,
                        "applied_lines": 18,
                        "next_split_manifest": r"C:\pkg\part02\manifest.json",
                    }
                },
            },
            exit_code=0,
            manifest_path=r"C:\pkg\part01\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertIn("已写回 2 个文件", "\n".join(summary.facts))
        self.assertIn("下一拆分包", "\n".join(summary.facts))

    def test_summarize_apply_output_marks_completed(self):
        summary = summarize_apply_output(
            APPLY_OUTPUT,
            exit_code=0,
            manifest_path="C:\\pkg\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertIn("已写回 2 个文件", "\n".join(summary.facts))

    def test_summarize_apply_output_shows_next_split_manifest(self):
        summary = summarize_apply_output(
            APPLY_OUTPUT + "Next split manifest: C:\\pkg\\part02\\manifest.json\n",
            exit_code=0,
            manifest_path="C:\\pkg\\part01\\manifest.json",
        )

        self.assertEqual(summary.status, "applied")
        self.assertIn("下一拆分包", "\n".join(summary.facts))
        self.assertIn("继续提交", summary.message)

    def test_summarize_manifest_writeback_shows_next_split_manifest(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\part01\\manifest.json",
                "applied_at": "2026-06-30T12:00:00",
                "apply_summary": {"applied_files": 1, "applied_lines": 5},
                "next_split_manifest_path": "C:\\pkg\\part02\\manifest.json",
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertIn("下一拆分包", "\n".join(summary.facts))
        self.assertIn("下一拆分包", summary.message)

    def test_summarize_manifest_warn_points_to_remediation_commands(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "last_check_summary": {
                    "safety_level": "warn",
                    "pending_files": 1,
                    "pending_lines": 4,
                    "failure_items": 2,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "warn")
        self.assertFalse(summary.can_apply)
        self.assertIn("查看问题清单", summary.message)
        self.assertEqual(summary.message.count("补译"), 1)
        self.assertIn("重新检查", summary.message)
        self.assertIn("可写回", summary.message)

    def test_summarize_manifest_writeback_from_last_check(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "last_check_summary": {
                    "safety_level": "safe",
                    "pending_files": 3,
                    "pending_lines": 12,
                    "failure_items": 0,
                },
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)

    def test_summarize_manifest_writeback_after_apply_failure(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": r"C:\pkg\manifest.json",
                "last_apply_failure_report_path": r"C:\pkg\apply_failure_report.json",
            }
        )

        self.assertEqual(summary.status, "failed")
        self.assertFalse(summary.can_apply)
        self.assertIn("查看写回失败报告", summary.message)

    def test_summarize_manifest_writeback_after_apply_failure_and_recheck(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": r"C:\pkg\manifest.json",
                "last_check_summary": {
                    "safety_level": "safe",
                    "pending_files": 2,
                    "pending_lines": 10,
                },
            }
        )

        self.assertEqual(summary.status, "safe")
        self.assertTrue(summary.can_apply)

    def test_manifest_writeback_after_apply_keeps_quality_alarms(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "applied_at": "2026-06-18T12:00:00",
                "apply_summary": {"applied_files": 1, "applied_lines": 5},
                "last_check_summary": {
                    "quality_gate": {
                        "decision": "needs_review",
                        "warning_count": 2,
                        "blocker_count": 0,
                    }
                },
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)
        self.assertTrue(any("写回后质量检查" in fact for fact in summary.facts))

    def test_summarize_manifest_writeback_after_apply(self):
        summary = summarize_manifest_writeback(
            {
                "_manifest_path": "C:\\pkg\\manifest.json",
                "applied_at": "2026-06-18T12:00:00",
                "apply_summary": {"applied_files": 1, "applied_lines": 5},
            }
        )

        self.assertEqual(summary.status, "applied")
        self.assertFalse(summary.can_apply)

    def test_idle_writeback_summary_for_keyword_mode_disables_apply(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.KEYWORD_EXTRACTION)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("关键词", summary.message)

    def test_idle_writeback_summary_for_bootstrap_mode_disables_apply(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.BOOTSTRAP_RAG)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("预建", summary.message)

    def test_idle_writeback_summary_for_sync_mode_explains_direct_writeback(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.SYNC_TRANSLATION)

        self.assertEqual(summary.status, "idle")
        self.assertFalse(summary.can_apply)
        self.assertIn("同步翻译", summary.message)
        self.assertNotIn("重新检查", summary.message)

    def test_idle_writeback_summary_for_final_review_explains_manual_selection(self):
        summary = idle_writeback_summary_for_work_mode(WorkMode.FINAL_REVIEW)

        self.assertFalse(summary.can_apply)
        self.assertIn("问题报告", summary.message)
        self.assertIn("人工选择", summary.message)
        self.assertIn("订正预览", summary.message)

    def test_build_recheck_cli_args(self):
        self.assertEqual(
            build_recheck_cli_args(r"C:\pkg\manifest.json"),
            ["check", r"C:\pkg\manifest.json", "--output", "json", "--non-interactive"],
        )

    def test_recheck_writeback_ready_requires_batch_translation_and_manifest(self):
        base = WritebackSummary(
            status="warn",
            heading="",
            message="",
            facts=[],
            findings=[],
            can_apply=False,
            manifest_path=r"C:\pkg\manifest.json",
        )

        self.assertTrue(
            recheck_writeback_ready(
                base,
                supports_translation_writeback=True,
            )
        )
        self.assertFalse(
            recheck_writeback_ready(
                base,
                supports_translation_writeback=False,
            )
        )
        self.assertFalse(
            recheck_writeback_ready(
                WritebackSummary(
                    status="warn",
                    heading="",
                    message="",
                    facts=[],
                    findings=[],
                    can_apply=False,
                    manifest_path="",
                ),
                supports_translation_writeback=True,
            )
        )

    def test_recheck_writeback_ready_blocks_idle_and_running(self):
        for status in ("idle", "running"):
            with self.subTest(status=status):
                summary = WritebackSummary(
                    status=status,
                    heading="",
                    message="",
                    facts=[],
                    findings=[],
                    can_apply=False,
                    manifest_path=r"C:\pkg\manifest.json",
                )
                self.assertFalse(
                    recheck_writeback_ready(
                        summary,
                        supports_translation_writeback=True,
                    )
                )

    def test_recheck_safe_check_enables_apply_gate(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=0,
            manifest_path=r"C:\pkg\manifest.json",
        )

        self.assertTrue(summary.can_apply)
        self.assertTrue(
            recheck_writeback_ready(
                summary,
                supports_translation_writeback=True,
            )
        )

    def test_recheck_failed_check_blocks_apply_gate(self):
        summary = summarize_check_output(
            CHECK_OUTPUT_SAFE,
            exit_code=1,
            manifest_path=r"C:\pkg\manifest.json",
        )

        self.assertFalse(summary.can_apply)
        self.assertTrue(
            recheck_writeback_ready(
                summary,
                supports_translation_writeback=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
