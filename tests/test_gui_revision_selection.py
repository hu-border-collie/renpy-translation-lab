"""GUI staged-selection dialog and layout contracts for #322."""
from __future__ import annotations

import unittest

import revision_corpus
import revision_selection

from tests import gui_test_support

try:
    from PySide6.QtWidgets import QApplication

    from gui_qt.revision_selection_dialog import RevisionProposalSelectionDialog
    from gui_qt.work_modes import WorkMode
    from gui_qt.workbench.revision_page import RevisionPage
    from gui_qt.workbench.page_contract import WorkbenchPageActions
    from gui_qt.workbench_session import WorkbenchModeSession
except ImportError as exc:
    QApplication = None  # type: ignore[assignment,misc]
    RevisionProposalSelectionDialog = None  # type: ignore[assignment,misc]
    RevisionPage = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _stage() -> dict[str, object]:
    live = {
        "occ-1": {
            "id": "occ-1",
            "file_rel_path": "a.rpy",
            "source": "Repeat",
            "current_translation": "第一处",
        },
        "occ-2": {
            "id": "occ-2",
            "file_rel_path": "b.rpy",
            "source": "Repeat",
            "current_translation": "第二处",
        },
    }
    digest = "a" * 64

    def row(identity: str, file_name: str, current: str, selected: bool) -> dict[str, object]:
        return {
            "schema_version": 1,
            "occurrence_id": identity,
            "identity_v2": identity,
            "file_rel_path": file_name,
            "source": "Repeat",
            "current_translation": current,
            "proposed_translation": f"修改{current}",
            "reason": "语气",
            "selected": selected,
            "disposition": "accepted" if selected else "rejected",
            "producer": {"type": "human"},
            "project_identity": {"tl_dir": "C:/demo/tl"},
            "snapshot_digest": revision_corpus.item_snapshot_digest("Repeat", current),
            "corpus_snapshot_digest": digest,
        }

    return revision_selection.build_staged_selection(
        rows=[row("occ-1", "a.rpy", "第一处", True), row("occ-2", "b.rpy", "第二处", False)],
        live_items=live,
        live_snapshot_digest=digest,
        project_identity={"game_root": "C:/demo", "tl_dir": "C:/demo/tl"},
        proposal_path="C:/demo/proposals.jsonl",
        proposal_sha256="b" * 64,
        source_file_digests={"a.rpy": "c" * 64, "b.rpy": "d" * 64},
        operation_id="operation-1",
    )


@gui_test_support.skip_unless_gui(
    RevisionProposalSelectionDialog is None,
    IMPORT_ERROR,
)
class RevisionProposalSelectionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.dialog = RevisionProposalSelectionDialog(_stage())

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()

    def test_duplicate_source_rows_keep_distinct_identity_and_initial_selection(self) -> None:
        self.assertEqual(self.dialog.table.rowCount(), 2)
        self.assertTrue(self.dialog._ok_button.isEnabled())
        self.assertEqual(self.dialog.selected_identity_v2(), ["occ-1"])
        self.assertEqual(
            [self.dialog.table.item(row, 7).text() for row in range(2)],
            ["occ-1", "occ-2"],
        )

    def test_filter_reason_file_status_and_valid_only(self) -> None:
        self.dialog.file_combo.setCurrentIndex(self.dialog.file_combo.findData("b.rpy"))
        self.assertEqual(self.dialog.table.rowCount(), 1)
        self.assertEqual(self.dialog.table.item(0, 7).text(), "occ-2")
        self.dialog.valid_only_cb.setChecked(True)
        self.assertEqual(self.dialog.table.rowCount(), 1)
        self.assertEqual(self.dialog.table.item(0, 7).text(), "occ-2")
        self.dialog.status_combo.setCurrentIndex(
            self.dialog.status_combo.findData(revision_selection.STATUS_VALID)
        )
        self.assertEqual(self.dialog.table.rowCount(), 1)

    def test_select_all_and_clear_valid_candidates_control_confirmation(self) -> None:
        self.dialog._select_all_valid()
        self.assertEqual(self.dialog.selected_identity_v2(), ["occ-1", "occ-2"])
        self.assertTrue(self.dialog._ok_button.isEnabled())
        self.dialog._clear_selection()
        self.assertEqual(self.dialog.selected_identity_v2(), [])
        self.assertFalse(self.dialog._ok_button.isEnabled())

    def test_layout_keeps_filters_table_and_actions_inside_960_by_640(self) -> None:
        self.dialog.resize(940, 600)
        self.dialog.show()
        self._app.processEvents()
        self.assertTrue(self.dialog.reason_combo.isVisible())
        self.assertTrue(self.dialog.table.isVisible())
        self.assertTrue(self.dialog._ok_button.isVisible())
        self.assertLessEqual(self.dialog.height(), 640)
        self.assertLessEqual(self.dialog.sizeHint().height(), 640)


@gui_test_support.skip_unless_gui(RevisionPage is None, IMPORT_ERROR)
class RevisionProposalPageLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_stage_summary_and_selection_action_fit_960_by_640(self) -> None:
        page = RevisionPage()
        page.activate(WorkMode.REVISION, WorkbenchModeSession())
        page.set_project_ready(True)
        actions: list[str] = []
        page.set_action_callbacks(WorkbenchPageActions(action=actions.append))
        stage = _stage()
        page.set_proposal_stage_result(
            {
                "candidate_count": 2,
                "selectable_count": 2,
                "selected_count": 1,
                "unselected_count": 1,
                "invalid_count": 0,
                "stale_count": 0,
                "conflict_count": 0,
                "paths": {"staged_selection": "C:/jobs/staged_selection.json"},
                "candidates": stage["candidates"],
            }
        )
        page.set_controls(
            start_enabled=True,
            resume_enabled=False,
            resume_visible=True,
            resume_label="继续订正",
            writeback_enabled=False,
            result_message="可筛选候选",
            selection_enabled=True,
        )
        page.resize(960, 640)
        page.show()
        self._app.processEvents()
        self.assertTrue(page.select_proposals_btn.isEnabled())
        page.select_proposals_btn.click()
        self.assertEqual(actions, ["select_revision_proposals"])
        self.assertLessEqual(page.preferred_height(960), 640)
        page.close()
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
