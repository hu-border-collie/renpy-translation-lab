import json
import os
import tempfile
import unittest

import keyword_glossary_merge as merge_mod
from project_asset_paths import canonical_abs_path

from gui_qt.keyword_merge_report import (
    KeywordCandidateSelection,
    format_keyword_candidate_selection,
    keyword_candidate_open_ready,
    keyword_merge_candidates_path_from_manifest,
    keyword_merge_ready,
    keyword_review_context_stale_reason,
    load_keyword_merge_context,
    summarize_keyword_merge_result,
)
from gui_qt.user_copy import KEYWORD_CANDIDATE_COPY


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

    def test_non_keyword_manifest_with_path_ignores_sibling_candidates(self):
        """batch_translation + path must not enable merge via leftover sibling jsonl."""
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"mode": "batch_translation"}, ensure_ascii=False))
            manifest = {"mode": "batch_translation", "_manifest_path": manifest_path}
            resolved = keyword_merge_candidates_path_from_manifest(manifest_path, manifest)
            self.assertEqual(resolved, "")

    def test_keyword_merge_candidates_path_uses_sibling_without_export(self):
        """Lite manifests omit keyword_export; resolve via package sibling, not full JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            manifest_path = os.path.join(tmp, "manifest.json")
            # Oversized-looking payload without keyword_export (simulates lite reader).
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"mode": "keyword_extraction", "chunks": [{"i": n} for n in range(50)]},
                        ensure_ascii=False,
                    )
                )
            lite = {"mode": "keyword_extraction", "_manifest_path": manifest_path}
            resolved = keyword_merge_candidates_path_from_manifest(manifest_path, lite)
            self.assertEqual(resolved, jsonl_path)
            # Path-only (no export field) still finds sibling quickly.
            resolved_path_only = keyword_merge_candidates_path_from_manifest(manifest_path, None)
            self.assertEqual(resolved_path_only, jsonl_path)

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
            self.assertEqual(
                canonical_abs_path(resolved_glossary),
                canonical_abs_path(glossary_path),
            )
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

    def test_load_keyword_merge_context_reports_corrupt_glossary(self):
        """Malformed glossary is a catchable ValueError, not a process exit."""
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            glossary_path = os.path.join(tmp, "glossary.json")
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("{not json}\n")
            with self.assertRaises(ValueError):
                load_keyword_merge_context(
                    candidates_path=jsonl_path,
                    config={"glossary_file": glossary_path},
                    game_root=tmp,
                    tool_root=tmp,
                )

    def test_keyword_candidate_open_ready_reports_each_restriction(self):
        ready, message = keyword_candidate_open_ready(
            running=False,
            game_root="C:/Games/Demo/work",
            project_ready=True,
        )
        self.assertTrue(ready)
        self.assertEqual(message, "")

        cases = (
            (True, "C:/Games/Demo/work", True, KEYWORD_CANDIDATE_COPY["open_running"]),
            (False, "", True, KEYWORD_CANDIDATE_COPY["open_no_project"]),
            (False, "C:/Games/Demo/work", False, KEYWORD_CANDIDATE_COPY["open_project_not_ready"]),
        )
        for running, game_root, project_ready, expected in cases:
            with self.subTest(message=expected):
                ready, message = keyword_candidate_open_ready(
                    running=running,
                    game_root=game_root,
                    project_ready=project_ready,
                )
                self.assertFalse(ready)
                self.assertEqual(message, expected)

    def test_format_keyword_candidate_selection_reports_context(self):
        selection = KeywordCandidateSelection(
            candidates_path="C:/tmp/keyword_candidates.jsonl",
            game_root="C:/Games/Demo/work",
            glossary_path="C:/Games/Demo/work/glossary.json",
            macro_path="",
            candidate_total=12,
            mergeable_total=9,
            source="external",
        )
        text = format_keyword_candidate_selection(selection)
        self.assertIn(KEYWORD_CANDIDATE_COPY["source_external"], text)
        self.assertIn(KEYWORD_CANDIDATE_COPY["format_name"], text)
        self.assertIn("C:/tmp/keyword_candidates.jsonl", text)
        self.assertIn("12 条（可审核 9 条）", text)
        self.assertIn("C:/Games/Demo/work/glossary.json", text)
        self.assertIn(KEYWORD_CANDIDATE_COPY["info_hint"], text)

    def test_format_keyword_candidate_selection_marks_missing_target(self):
        for source in ("extraction", "sync", "unknown-source"):
            with self.subTest(source=source):
                selection = KeywordCandidateSelection(
                    candidates_path="C:/tmp/keyword_candidates.jsonl",
                    game_root="C:/Games/Demo/work",
                    glossary_path="",
                    macro_path="",
                    candidate_total=1,
                    mergeable_total=1,
                    source=source,
                )
                text = format_keyword_candidate_selection(selection)
                self.assertIn("术语表目标：未配置", text)

    def test_keyword_review_context_stale_reason_matches_live_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            self.assertEqual(
                keyword_review_context_stale_reason(
                    candidates_path=jsonl_path,
                    glossary_path=os.path.join(tmp, "glossary.json"),
                    game_root=tmp,
                    current_game_root=tmp,
                    current_glossary_path=os.path.join(tmp, "glossary.json"),
                ),
                "",
            )

    def test_keyword_review_context_stale_reason_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "keyword_candidates.jsonl")
            self._write_jsonl(jsonl_path, [{"source": "A", "suggested_target": "甲"}])
            glossary_path = os.path.join(tmp, "glossary.json")
            other_root = os.path.join(tmp, "other")
            os.makedirs(other_root)

            cases = (
                (other_root, glossary_path, KEYWORD_CANDIDATE_COPY["stale_project"]),
                (tmp, os.path.join(other_root, "glossary.json"), KEYWORD_CANDIDATE_COPY["stale_glossary"]),
            )
            for current_root, current_glossary, expected in cases:
                with self.subTest(expected=expected):
                    reason = keyword_review_context_stale_reason(
                        candidates_path=jsonl_path,
                        glossary_path=glossary_path,
                        game_root=tmp,
                        current_game_root=current_root,
                        current_glossary_path=current_glossary,
                    )
                    self.assertEqual(reason, expected)

            self.assertEqual(
                keyword_review_context_stale_reason(
                    candidates_path=os.path.join(tmp, "missing.jsonl"),
                    glossary_path=glossary_path,
                    game_root=tmp,
                    current_game_root=tmp,
                    current_glossary_path=glossary_path,
                ),
                KEYWORD_CANDIDATE_COPY["stale_candidates"],
            )


if __name__ == "__main__":
    unittest.main()