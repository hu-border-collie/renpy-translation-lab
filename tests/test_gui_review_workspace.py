"""Review page rendering and late-worker isolation with real S1 artifacts."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from tests import gui_test_support
from tests.test_review_index import _write_corpus

try:
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication, QWidget
    from gui_qt.workbench.review_workspace import ReviewWorkspaceWidget
    import review_workspace
except ImportError as exc:
    QApplication = None
    ReviewWorkspaceWidget = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(ReviewWorkspaceWidget is None, IMPORT_ERROR)
class GuiReviewWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _new_panel(self) -> ReviewWorkspaceWidget:
        host = QWidget()
        panel = ReviewWorkspaceWidget(host)
        def cleanup() -> None:
            host.close()
            host.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
        self.addCleanup(cleanup)
        return panel

    def test_plain_item_browse_save_and_selected_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            panel = self._new_panel()
            panel.resize(640, 480)
            panel._generation = 1
            panel._current_context = lambda: "current"
            panel._loaded((1, "current"), manifest, entries)
            self.assertEqual(panel.table.rowCount(), 3)
            panel.table.selectRow(0)
            panel._next_open()
            self.assertEqual(panel.current_id, "occ-2")
            panel.table.selectRow(1)
            self.assertIn("Hello", panel.details.toPlainText())
            self.assertIn("未附着证据", panel.details.toPlainText())
            panel.proposed.setPlainText("完整建议译文" * 40)
            panel.reason.setText("逐条核对")
            panel.save_current_draft()
            paths: list[tuple[str, str]] = []
            panel.proposal_ready.connect(lambda path, corpus_path: paths.append((path, corpus_path)))
            panel._export_selected()
            self.assertEqual(len(paths), 1)
            self.assertEqual(panel.table.selectedItems()[0].row(), 1)
            self.assertTrue(Path(paths[0][0]).is_file())
            self.assertTrue(Path(paths[0][1]).samefile(corpus / "revision_corpus_manifest.json"))
            self.assertGreater(panel.minimumSizeHint().height(), 0)

    def test_late_result_ignored_after_project_or_task_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            panel = self._new_panel()
            panel._generation = 10
            panel._current_context = lambda: "project-a"
            panel.reset()
            panel._loaded((10, "project-a"), manifest, entries)
            self.assertEqual(panel.table.rowCount(), 0)
            panel._generation = 20
            panel._current_context = lambda: "project-b"
            panel._loaded((20, "project-a"), manifest, entries)
            self.assertEqual(panel.table.rowCount(), 0)

    def test_pending_edit_flushes_when_task_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            manifest, entries = review_workspace.open_workspace(
                str(corpus), expected_game_root=str(root / "game")
            )
            panel = self._new_panel()
            panel._generation = 1
            panel._current_context = lambda: "same"
            panel._loaded((1, "same"), manifest, entries)
            panel.table.selectRow(0)
            panel.proposed.setPlainText("切换前的完整编辑")
            panel.reason.setText("待补充")
            panel.reset()
            drafts = review_workspace.load_drafts(manifest["_manifest_path"], manifest)
            self.assertEqual(drafts["occ-1"]["proposed_translation"], "切换前的完整编辑")

    def test_background_load_finishes_without_blocking_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_corpus(root)
            panel = self._new_panel()
            panel.load(str(corpus), str(root / "game"), str(root / "game" / "tl"), lambda: "task-a")
            self.assertIsNone(panel.manifest)
            deadline = time.monotonic() + 5
            while panel.manifest is None and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            self.assertIsNotNone(panel.manifest, panel.message.text())
            self.assertEqual(panel.table.rowCount(), 3)
