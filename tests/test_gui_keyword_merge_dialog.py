from __future__ import annotations

import unittest

import keyword_glossary_merge as merge_mod
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

    def test_history_column_and_review_defaults_are_visible(self) -> None:
        candidates = [
            {
                "source": "Void Gate",
                "suggested_target": "虚空门",
                "category": "place",
                "confidence": 0.9,
                "history_evidence": {
                    "status": "consistent",
                    "first_occurrence": {
                        "file_rel_path": "a.rpy",
                        "line_number": 1,
                        "identity_v2": "id-consistent",
                        "current_translation": "虚空门",
                    },
                },
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


if __name__ == "__main__":
    unittest.main()
