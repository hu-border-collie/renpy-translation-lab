import json
import tempfile
import unittest
from pathlib import Path

from gui_qt.check_failures_report import REASON_CATEGORY_REPAIR, REASON_CATEGORY_RETRY
from gui_qt.repair_report import (
    assess_repair_eligibility,
    build_derived_repair_report_path,
    build_repair_cli_args,
    discover_repair_report_candidates,
    parse_repair_output,
    repair_action_ready,
    select_repair_report_path,
    summarize_repair_output,
)


REPAIR_OUTPUT_OK = """
Repair report: C:\\work\\remaining_need_translate_demo.jsonl
Repair run dir: C:\\logs\\repair_runs\\20260707_demo
Requested items: 4
Repair jobs: 2
Applied items: 4
Applied files: 2
Failure items: 0
Request errors: 0
Parse errors: 0
Validation failures: 0
Missing item ids: 0
Unresolved items: 0
Story Memory repair hits: 1/2 jobs
Repair results: C:\\logs\\repair_runs\\20260707_demo\\repair_results.jsonl
Repair failures: C:\\logs\\repair_runs\\20260707_demo\\repair_failures.jsonl
"""

CHECK_FAILURES_REPAIR = "\n".join(
    [
        json.dumps(
            {
                "reason_code": "validation_failed",
                "status": "warn",
                "file_rel_path": "game/tl/schinese/script.rpy",
                "line": 12,
                "text": "Hello there",
                "error": "Validation failed: missing hanzi",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "reason_code": "response_missing_item_id",
                "status": "warn",
                "file_rel_path": "game/tl/schinese/common.rpy",
                "line": 3,
                "text": "Menu",
                "error": "Response missing item id",
            },
            ensure_ascii=False,
        ),
    ]
)


class GuiRepairReportTests(unittest.TestCase):
    def test_build_repair_cli_args_uses_defaults(self):
        args = build_repair_cli_args(r"C:\work\remaining.jsonl")
        self.assertEqual(
            args[:7],
            [
                "repair",
                r"C:\work\remaining.jsonl",
                "--batch-size",
                "2",
                "--context-before",
                "2",
                "--context-after",
            ],
        )

    def test_assess_repair_eligibility_accepts_dominant_repair_reasons(self):
        eligibility = assess_repair_eligibility(
            {
                "last_check_summary": {
                    "safety_level": "warn",
                    "safety_reasons": {
                        "warn": {
                            "validation_failed": 2,
                            "response_missing_item_id": 1,
                        },
                    },
                },
            }
        )

        self.assertTrue(eligibility.eligible)
        self.assertEqual(eligibility.repair_count, 2)
        self.assertEqual(eligibility.retry_count, 1)

    def test_assess_repair_eligibility_rejects_retry_dominant_reasons(self):
        eligibility = assess_repair_eligibility(
            {
                "last_check_summary": {
                    "safety_level": "warn",
                    "safety_reasons": {
                        "warn": {
                            "validation_failed": 1,
                            "response_missing_item_id": 3,
                        },
                    },
                },
            }
        )

        self.assertFalse(eligibility.eligible)
        self.assertIn("补译", eligibility.message)

    def test_repair_action_ready_follows_eligibility(self):
        manifest = {
            "last_check_summary": {
                "safety_level": "warn",
                "safety_reasons": {"warn": {"validation_failed": 2}},
            },
        }
        self.assertTrue(
            repair_action_ready(manifest, manifest_path=r"C:\pkg\manifest.json")
        )

    def test_discover_repair_report_candidates_finds_failures_and_derived_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / "batch_jobs" / "pkg1"
            package_dir.mkdir(parents=True)
            failures = package_dir / "failures.jsonl"
            failures.write_text(
                json.dumps(
                    {
                        "file_rel_path": "script.rpy",
                        "line": 0,
                        "text": "Hello",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            check_failures = package_dir / "check_failures.jsonl"
            check_failures.write_text(CHECK_FAILURES_REPAIR, encoding="utf-8")
            manifest = {
                "last_check_report_path": str(check_failures),
                "last_check_summary": {
                    "safety_level": "warn",
                    "safety_reasons": {"warn": {"validation_failed": 1}},
                },
            }

            candidates = discover_repair_report_candidates(
                manifest,
                manifest_path=str(package_dir / "manifest.json"),
            )
            labels = [candidate.label for candidate in candidates]
            self.assertIn("翻译包失败明细", labels)
            self.assertIn("从检查报告提取（repair 类）", labels)

            selected = select_repair_report_path(candidates)
            self.assertTrue(selected.endswith("repair_from_check_failures.jsonl"))

    def test_build_derived_repair_report_path_writes_only_repair_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            check_failures = package_dir / "check_failures.jsonl"
            check_failures.write_text(CHECK_FAILURES_REPAIR, encoding="utf-8")
            manifest = {"last_check_report_path": str(check_failures)}

            derived = build_derived_repair_report_path(
                manifest,
                manifest_path=str(package_dir / "manifest.json"),
            )

            self.assertTrue(derived.endswith("repair_from_check_failures.jsonl"))
            rows = [
                json.loads(line)
                for line in Path(derived).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["file_rel_path"], "game/tl/schinese/script.rpy")
            self.assertEqual(rows[0]["line"], 11)

    def test_summarize_repair_output_marks_success(self):
        summary = summarize_repair_output(
            REPAIR_OUTPUT_OK,
            0,
            report_path=r"C:\work\remaining.jsonl",
            manifest_path=r"C:\pkg\manifest.json",
        )
        self.assertEqual(summary.status, "ok")
        self.assertEqual(summary.applied_items, 4)
        self.assertTrue(any("重新检查" in finding for finding in summary.findings))

    def test_parse_repair_output_extracts_run_dir(self):
        parsed = parse_repair_output(REPAIR_OUTPUT_OK)
        self.assertEqual(parsed["applied_items"], 4)
        self.assertIn("repair_runs", str(parsed["run_dir"]))


if __name__ == "__main__":
    unittest.main()