from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import keyword_glossary_merge as merge_mod
import keyword_history
from tests import gui_test_support

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui_qt.keyword_merge_dialog import KeywordMergeDialog
except ImportError as exc:  # pragma: no cover - exercised only without GUI deps
    QApplication = None  # type: ignore[assignment,misc]
    KeywordMergeDialog = None  # type: ignore[assignment,misc]
    Qt = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(KeywordMergeDialog is None, IMPORT_ERROR)
class GuiKeywordMergeDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _consistent_history(self) -> dict:
        return keyword_history.build_keyword_history_evidence(
            {'source': 'Void Gate', 'suggested_target': '虚空门'},
            [
                {
                    'identity_v2': 'id-consistent',
                    'occurrence_id': 'id-consistent',
                    'file_rel_path': 'a.rpy',
                    'display_line': 1,
                    'locator': {'line_number': 1, 'start': 0},
                    'source': 'Void Gate',
                    'current_translation': '虚空门',
                }
            ],
        )

    def test_history_column_and_review_defaults_are_visible(self) -> None:
        candidates = [
            {
                "source": "Void Gate",
                "suggested_target": "虚空门",
                "category": "place",
                "confidence": 0.9,
                "history_evidence": self._consistent_history(),
            },
            {
                "source": "Light",
                "suggested_target": "光",
                "category": "term",
                "confidence": 0.9,
                "history_evidence": {
                    "status": "ambiguous",
                    "first_occurrence": {
                        "file_rel_path": "b.rpy",
                        "line_number": 2,
                        "identity_v2": "id-ambiguous",
                        "current_translation": "光",
                    },
                    "conflict_reasons": ["存在多个不同现译"],
                },
            },
            {
                "source": "Legacy Term",
                "suggested_target": "旧术语",
                "category": "term",
                "confidence": 0.9,
            },
        ]
        rows = merge_mod.build_candidate_merge_rows(candidates, {"normalize_map": {}})
        self.assertTrue(rows[0].default_checked)
        self.assertFalse(rows[1].default_checked)
        self.assertFalse(rows[2].default_checked)

        dialog = KeywordMergeDialog(
            None,
            rows=rows,
            candidates_path="keyword_candidates.jsonl",
            glossary_path="glossary.json",
            candidates=candidates,
        )
        try:
            self.assertEqual(dialog.table.columnCount(), 8)
            history_header = dialog.table.horizontalHeaderItem(6)
            self.assertIsNotNone(history_header)
            self.assertEqual(history_header.text(), "历史首次译法")
            self.assertEqual(
                dialog.table.item(0, 0).checkState(),
                Qt.CheckState.Checked,
            )
            self.assertEqual(
                dialog.table.item(1, 0).checkState(),
                Qt.CheckState.Unchecked,
            )
            self.assertEqual(
                dialog.table.item(2, 0).checkState(),
                Qt.CheckState.Unchecked,
            )
            self.assertIn("b.rpy:L2", dialog.table.item(1, 6).text())
            self.assertIn("ambiguous", dialog.table.item(1, 6).text())
            self.assertIn("无历史证据", dialog.table.item(2, 6).text())
        finally:
            dialog.close()
            dialog.deleteLater()
            self._app.processEvents()

    def test_explicit_selection_allows_history_review_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glossary_path = root / "glossary.json"
            glossary_path.write_text(
                json.dumps({"preserve_terms": [], "normalize_map": {}}, ensure_ascii=False),
                encoding="utf-8",
            )
            candidates = [
                {
                    "source": "Light",
                    "suggested_target": "光",
                    "category": "term",
                    "confidence": 0.99,
                    "history_evidence": {
                        "status": "ambiguous",
                        "review_required": True,
                        "first_occurrence": {
                            "file_rel_path": "b.rpy",
                            "line_number": 2,
                            "identity_v2": "id-ambiguous",
                            "current_translation": "光",
                        },
                        "conflict_reasons": ["存在多个不同现译"],
                    },
                }
            ]
            rows = merge_mod.build_candidate_merge_rows(candidates, {"normalize_map": {}})
            self.assertFalse(rows[0].default_checked)

            dialog = KeywordMergeDialog(
                None,
                rows=rows,
                candidates_path=str(root / "keyword_candidates.jsonl"),
                glossary_path=str(glossary_path),
                candidates=candidates,
            )
            try:
                dialog.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
                summary = dialog._run_merge(dry_run=False)
            finally:
                dialog.close()
                dialog.deleteLater()
                self._app.processEvents()

            self.assertEqual(summary.accepted, 1)
            self.assertTrue(summary.wrote_glossary)
            data = json.loads(glossary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["normalize_map"]["Light"], "光")


if __name__ == "__main__":
    unittest.main()
