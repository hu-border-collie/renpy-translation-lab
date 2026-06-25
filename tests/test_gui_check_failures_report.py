import json
import unittest

from gui_qt.check_failures_report import (
    REASON_CATEGORY_MANUAL,
    REASON_CATEGORY_REBUILD,
    REASON_CATEGORY_RETRY,
    build_check_issues_report,
    classify_reason_category,
    group_failure_items,
    normalize_failure_entry,
    parse_check_failures_jsonl,
    resolve_check_report_path,
)


SAMPLE_JSONL = "\n".join(
    [
        json.dumps(
            {
                "reason_code": "response_missing_item_id",
                "status": "warn",
                "file_rel_path": "game/tl/schinese/script.rpy",
                "line": 42,
                "id": "item-1",
                "error": "Response missing item id",
                "text": "Hello world",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "reason_code": "source_text_mismatch",
                "status": "block",
                "file_rel_path": "game/tl/schinese/common.rpy",
                "line": 8,
                "item_id": "item-2",
                "error": "Source text mismatch",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "reason_code": "v2_relocation_missing",
                "status": "block",
                "file_rel_path": "game/tl/schinese/menu.rpy",
                "line": 15,
                "item_id": "item-3",
                "error": "V2 relocation missing",
            },
            ensure_ascii=False,
        ),
    ]
)


class GuiCheckFailuresReportTests(unittest.TestCase):
    def test_parse_check_failures_jsonl_reads_entries(self):
        entries = parse_check_failures_jsonl(SAMPLE_JSONL)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["reason_code"], "response_missing_item_id")

    def test_parse_check_failures_jsonl_rejects_invalid_line(self):
        with self.assertRaisesRegex(ValueError, "无法解析"):
            parse_check_failures_jsonl('{"ok": true}\nnot-json')

    def test_normalize_failure_entry_extracts_fields(self):
        entries = parse_check_failures_jsonl(SAMPLE_JSONL)
        item = normalize_failure_entry(entries[0])

        self.assertEqual(item.reason_code, "response_missing_item_id")
        self.assertEqual(item.status, "warn")
        self.assertEqual(item.file_rel_path, "game/tl/schinese/script.rpy")
        self.assertEqual(item.line, 42)
        self.assertEqual(item.item_id, "item-1")
        self.assertEqual(item.text_preview, "Hello world")

    def test_classify_reason_category_covers_common_reasons(self):
        self.assertEqual(
            classify_reason_category("response_missing_item_id"),
            REASON_CATEGORY_RETRY,
        )
        self.assertEqual(
            classify_reason_category("source_text_mismatch"),
            REASON_CATEGORY_MANUAL,
        )
        self.assertEqual(
            classify_reason_category("missing_manifest_file"),
            REASON_CATEGORY_REBUILD,
        )

    def test_group_failure_items_sorts_by_count(self):
        duplicate_warn = json.dumps(
            {
                "reason_code": "response_missing_item_id",
                "status": "warn",
                "file_rel_path": "game/tl/schinese/other.rpy",
                "line": 99,
                "id": "item-4",
                "error": "Response missing item id",
            },
            ensure_ascii=False,
        )
        items = [
            normalize_failure_entry(entry)
            for entry in parse_check_failures_jsonl(f"{SAMPLE_JSONL}\n{duplicate_warn}")
        ]
        groups = group_failure_items(items)

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0].count, 2)
        self.assertEqual(groups[0].reason_code, "response_missing_item_id")
        self.assertIn("retry", groups[0].category_label)

    def test_resolve_check_report_path_prefers_manifest_field(self):
        manifest = {
            "last_check_report_path": r"C:\pkg\check_failures.jsonl",
            "_package_dir": r"C:\pkg",
        }

        self.assertEqual(
            resolve_check_report_path(manifest, manifest_path=r"C:\pkg\manifest.json"),
            r"C:\pkg\check_failures.jsonl",
        )

    def test_resolve_check_report_path_falls_back_to_package_dir(self):
        manifest = {"_package_dir": r"C:\pkg"}

        self.assertEqual(
            resolve_check_report_path(manifest, manifest_path=r"C:\pkg\manifest.json"),
            r"C:\pkg\check_failures.jsonl",
        )

    def test_build_check_issues_report_accepts_injected_text_without_report_path(self):
        manifest = {
            "last_check_summary": {"safety_level": "warn"},
        }

        report = build_check_issues_report(
            manifest,
            report_text=SAMPLE_JSONL,
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(len(report.items), 3)
        self.assertTrue(
            any("source_text_mismatch" in line for line in report.detail_lines)
        )

    def test_build_check_issues_report_from_jsonl_text(self):
        manifest = {
            "_manifest_path": r"C:\pkg\manifest.json",
            "last_check_summary": {
                "safety_level": "warn",
                "safety_reasons": {
                    "warn": {"response_missing_item_id": 1},
                    "block": {
                        "source_text_mismatch": 1,
                        "v2_relocation_missing": 1,
                    },
                },
            },
            "last_check_report_path": r"C:\pkg\check_failures.jsonl",
        }

        report = build_check_issues_report(
            manifest,
            manifest_path=r"C:\pkg\manifest.json",
            report_text=SAMPLE_JSONL,
        )

        self.assertEqual(report.status, "ok")
        self.assertEqual(report.safety_level, "warn")
        self.assertEqual(len(report.items), 3)
        self.assertEqual(report.category_counts[REASON_CATEGORY_RETRY], 1)
        self.assertEqual(report.category_counts[REASON_CATEGORY_MANUAL], 2)
        self.assertTrue(any("按原因汇总" in line for line in report.detail_lines))
        self.assertTrue(any("script.rpy" in line for line in report.detail_lines))
        self.assertTrue(
            any("response_missing_item_id" in line for line in report.detail_lines)
        )

    def test_build_check_issues_report_missing_file_uses_summary_fallback(self):
        manifest = {
            "_manifest_path": r"C:\pkg\manifest.json",
            "last_check_summary": {
                "safety_level": "warn",
                "safety_reasons": {
                    "warn": {"partial_result_items": 2},
                },
            },
            "last_check_report_path": r"C:\pkg\check_failures.jsonl",
        }

        report = build_check_issues_report(
            manifest,
            manifest_path=r"C:\pkg\manifest.json",
            path_exists=lambda _path: False,
        )

        self.assertEqual(report.status, "missing_report")
        self.assertEqual(len(report.reason_groups), 1)
        self.assertEqual(report.reason_groups[0].reason_code, "partial_result_items")
        self.assertIn("最近检查结果", report.message)

    def test_build_check_issues_report_unreadable_file(self):
        manifest = {
            "last_check_summary": {"safety_level": "warn"},
            "last_check_report_path": r"C:\pkg\check_failures.jsonl",
        }

        report = build_check_issues_report(
            manifest,
            report_text="not-json",
        )

        self.assertEqual(report.status, "unreadable")
        self.assertTrue(any("解析错误" in fact for fact in report.facts))

    def test_build_check_issues_report_truncates_display_items(self):
        lines = []
        for index in range(5):
            lines.append(
                json.dumps(
                    {
                        "reason_code": "response_missing_item_id",
                        "status": "warn",
                        "file_rel_path": f"file-{index}.rpy",
                        "line": index,
                        "id": f"id-{index}",
                        "error": "missing",
                    }
                )
            )

        report = build_check_issues_report(
            {"last_check_summary": {"safety_level": "warn"}},
            report_text="\n".join(lines),
            max_items=2,
        )

        self.assertEqual(len(report.items), 2)
        self.assertEqual(report.omitted_item_count, 3)
        self.assertTrue(any("另有 3 条未显示" in line for line in report.detail_lines))


if __name__ == "__main__":
    unittest.main()