import json
import tempfile
import unittest
from pathlib import Path

from quality_report_export import (
    DEFAULT_REPORT_FILENAME,
    QualityReportExportError,
    export_quality_report,
    load_quality_findings,
    render_quality_report_html,
)


FINDINGS = [
    {
        "finding_id": "f-warning",
        "schema_version": 1,
        "reason_code": "quality.typography.cjk_latin_spacing",
        "rule_id": "cjk_latin_spacing",
        "severity": "medium",
        "disposition": "warning",
        "item_id": "item-1",
        "file": "chapter<script>.rpy",
        "line": 12,
        "source": "Hello <script>alert('source')</script>",
        "translation": "你好iPhone",
        "evidence": "token=iPhone",
        "suggestion": "加入空格",
        "rule_version": 1,
    },
    {
        "finding_id": "f-blocker",
        "schema_version": 1,
        "reason_code": "quality.renpy.wait_tag_inside_cjk",
        "rule_id": "renpy_wait_inside_cjk",
        "severity": "high",
        "disposition": "blocker",
        "item_id": "item-2",
        "file": "chapter2.rpy",
        "line": 30,
        "source": "Hi",
        "translation": "你{w=0.2}好",
        "evidence": "你{w=0.2}好",
        "suggestion": "移动标签",
        "rule_version": 1,
    },
]


class QualityReportExportTests(unittest.TestCase):
    def test_render_is_self_contained_filterable_and_escapes_finding_content(self):
        document = render_quality_report_html(
            FINDINGS,
            acknowledged_finding_ids={"f-warning"},
            source_name="quality_findings.jsonl",
        )

        self.assertIn("译文质量体检报告", document)
        self.assertIn("显示 2 / 2 条", document)
        self.assertIn('data-state="acknowledged"', document)
        self.assertIn("CJK/拉丁字符间距", document)
        self.assertIn("&lt;script&gt;alert(&#x27;source&#x27;)&lt;/script&gt;", document)
        self.assertNotIn("<script>alert('source')</script>", document)
        self.assertNotIn("https://", document)

    def test_export_uses_current_manifest_report_and_default_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package = Path(tmp_dir)
            report_path = package / "quality_findings.jsonl"
            report_path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in FINDINGS
                ),
                encoding="utf-8",
            )
            manifest_path = package / "manifest.json"
            manifest = {
                "_manifest_path": str(manifest_path),
                "_package_dir": str(package),
                "last_quality_findings_path": "quality_findings.jsonl",
                "quality_acknowledged_finding_ids": ["f-warning"],
            }

            result = export_quality_report(
                manifest,
                manifest_path=str(manifest_path),
            )

            output_path = package / DEFAULT_REPORT_FILENAME
            self.assertEqual(Path(result["output_path"]), output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(result["finding_count"], 2)
            self.assertEqual(result["warning_count"], 1)
            self.assertEqual(result["blocker_count"], 1)
            self.assertEqual(result["acknowledged_count"], 1)

    def test_invalid_jsonl_reports_the_source_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "quality_findings.jsonl"
            path.write_text('{}\n{"broken"\n', encoding="utf-8")

            with self.assertRaisesRegex(QualityReportExportError, "第 2 行"):
                load_quality_findings(str(path))

    def test_export_refuses_to_overwrite_manifest_or_findings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            package = Path(tmp_dir)
            report_path = package / "quality_findings.jsonl"
            report_path.write_text(
                json.dumps(FINDINGS[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            manifest_path = package / "manifest.json"
            manifest = {
                "_manifest_path": str(manifest_path),
                "last_quality_findings_path": str(report_path),
            }

            for protected_path in (manifest_path, report_path):
                with self.subTest(path=protected_path):
                    with self.assertRaisesRegex(
                        QualityReportExportError,
                        "不能覆盖",
                    ):
                        export_quality_report(
                            manifest,
                            manifest_path=str(manifest_path),
                            output_path=str(protected_path),
                        )


if __name__ == "__main__":
    unittest.main()
