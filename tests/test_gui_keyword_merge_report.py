import json
import os
import tempfile
import unittest

import keyword_glossary_merge as merge_mod

from gui_qt.keyword_merge_report import (
    keyword_merge_candidates_path_from_manifest,
    keyword_merge_ready,
    load_keyword_merge_context,
    summarize_keyword_merge_result,
)


class GuiKeywordMergeReportTests(unittest.TestCase):
    def _write_jsonl(self, path: str, rows: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_keyword_merge_ready_requires_candidates_and_glossary(self):
        ready, message = keyword_merge_ready(candidates_path="", glossary_path="")
        self.assertFalse(ready)
        self.assertIn("候选", message)

    def test_keyword_merge_candidates_path_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            manifest = {
                "mode": "keyword_extraction",
                "keyword_export": {"jsonl_path": jsonl_path},
            }
            resolved = keyword_merge_candidates_path_from_manifest("", manifest)
            self.assertEqual(resolved, jsonl_path)

    def test_keyword_merge_candidates_path_ignores_non_keyword_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            manifest = {
                "mode": "batch_translation",
                "keyword_export": {"jsonl_path": jsonl_path},
            }
            resolved = keyword_merge_candidates_path_from_manifest("", manifest)
            self.assertEqual(resolved, "")

    def test_load_keyword_merge_context_builds_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            glossary_path = os.path.join(tmp, "glossary.json")
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"preserve_terms": [], "normalize_map": {}}, ensure_ascii=False))
            self._write_jsonl(
                jsonl_path,
                [
                    {
                        "source": "Void Gate",
                        "suggested_target": "虚空门",
                        "category": "place",
                        "confidence": 0.9,
                    },
                    {
                        "source": "Start",
                        "suggested_target": "开始",
                        "category": "other",
                        "confidence": 0.95,
                        "evidence": "common.rpy menu label",
                    },
                ],
            )
            rows, candidates, resolved_glossary, _macro = load_keyword_merge_context(
                candidates_path=jsonl_path,
                config={},
                game_root=tmp,
                tool_root=tmp,
            )
            self.assertEqual(len(candidates), 2)
            self.assertEqual(resolved_glossary, glossary_path)
            self.assertEqual(len(rows), 2)
            start_row = next(row for row in rows if row.candidate.get("source") == "Start")
            self.assertFalse(start_row.default_checked)

    def test_merge_selected_candidates_writes_only_checked_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            glossary_path = os.path.join(tmp, "glossary.json")
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"preserve_terms": [], "normalize_map": {}}, ensure_ascii=False))
            rows_data = [
                {
                    "source": "Void Gate",
                    "suggested_target": "虚空门",
                    "category": "place",
                    "confidence": 0.9,
                },
                {
                    "source": "Crystal Key",
                    "suggested_target": "水晶钥匙",
                    "category": "item",
                    "confidence": 0.8,
                },
            ]
            self._write_jsonl(jsonl_path, rows_data)
            candidates = merge_mod.load_keyword_candidates_jsonl(jsonl_path)
            summary = merge_mod.merge_selected_candidates(
                candidates,
                {0},
                glossary_path,
                candidates_path=jsonl_path,
                dry_run=False,
            )
            with open(glossary_path, encoding="utf-8") as handle:
                data = json.loads(handle.read())
            self.assertEqual(summary.accepted, 1)
            self.assertIn("Void Gate", data["normalize_map"])
            self.assertNotIn("Crystal Key", data["normalize_map"])

    def test_summarize_keyword_merge_result_for_dry_run(self):
        summary = merge_mod.MergeSummary(
            candidates_read=2,
            accepted=1,
            dry_run=True,
            preview_lines=["+ normalize_map: A -> 甲"],
        )
        payload = summarize_keyword_merge_result(summary)
        self.assertEqual(payload["status"], "ready")
        self.assertIn("预览", payload["heading"])


if __name__ == "__main__":
    unittest.main()