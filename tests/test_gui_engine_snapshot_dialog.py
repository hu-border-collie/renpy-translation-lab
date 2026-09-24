"""GUI engine/snapshot/reuse dialog presentation tests (#424 P6)."""

from __future__ import annotations

import unittest
from unittest import mock

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

    from gui_qt.app import MainWindow
    from gui_qt.engine_snapshot_dialog import EngineSnapshotDialog
    from gui_qt.engine_snapshot_worker import EngineSnapshotTaskResult
    from gui_qt.workbench.context_library_page import ContextLibraryPage
    from gui_qt.workbench.page_contract import WorkbenchPageActions
except ImportError as exc:
    QObject = None  # type: ignore[assignment,misc]
    Signal = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    QMessageBox = None  # type: ignore[assignment,misc]
    QWidget = None  # type: ignore[assignment,misc]
    MainWindow = None  # type: ignore[assignment,misc]
    EngineSnapshotDialog = None  # type: ignore[assignment,misc]
    EngineSnapshotTaskResult = None  # type: ignore[assignment,misc]
    ContextLibraryPage = None  # type: ignore[assignment,misc]
    WorkbenchPageActions = None  # type: ignore[assignment,misc]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from tests import gui_test_support


def _overview_payload():
    base = {
        "name": "v1",
        "path": "C:/logs/project_snapshots/v1",
        "manifest_path": "C:/logs/project_snapshots/v1/project_snapshot.json",
        "version_id": "1.0.0",
        "generated_at": "2026-09-01T00:00:00+00:00",
        "engine": "renpy",
        "adapter_version": "1.1.8",
        "coverage_status": "attention",
        "review_status": "pending",
        "snapshot_digest": "aaaa",
    }
    newer = dict(base)
    newer.update(
        {
            "name": "v2",
            "path": "C:/logs/project_snapshots/v2",
            "manifest_path": "C:/logs/project_snapshots/v2/project_snapshot.json",
            "version_id": "1.1.0",
            "generated_at": "2026-09-02T00:00:00+00:00",
            "coverage_status": "block",
            "review_status": "stale",
            "snapshot_digest": "bbbb",
        }
    )
    return {
        "engine": "renpy",
        "adapter_version": "1.1.8",
        "protocol_version": 1,
        "locator_schema_version": 1,
        "behavior_digest": "digest",
        "capabilities": {
            "selected_localization_mode": "hybrid",
            "native_catalog": True,
            "relocation": True,
            "declarative_writeback": ["text_span_replace"],
        },
        "snapshot_root": "C:/logs/project_snapshots",
        "snapshot_count": 2,
        "snapshots": [newer, base],
        "latest_snapshot": newer,
        "coverage_review_path": "C:/Game/work/translation_context/coverage_review.json",
        "game_root": "C:/Game/work",
    }


def _diff_payload():
    return {
        "base_path": "C:/snap/v1",
        "target_path": "C:/snap/v2",
        "base_version_id": "1.0.0",
        "target_version_id": "1.1.0",
        "status": "attention",
        "summary": {"matched": 1, "ambiguous": 1},
        "coverage_changes": {"added": 1},
        "reconciliation_digest": "digest",
        "item_count": 1,
        "item_limit": 500,
        "items": [
            {
                "item_id": "item-1",
                "disposition": "ambiguous",
                "match_kind": "ambiguous",
                "confidence": 0.5,
                "evidence": {"source_equal": True},
                "base_locator": "old.rpy:1",
                "target_locator": "",
                "candidate_locators": ["new.rpy:2", "new.rpy:3"],
                "base_source": "Hello",
                "target_source": "",
                "ambiguous": True,
            }
        ],
    }


def _reuse_payload():
    return {
        "path": "C:/reuse/reuse_report.json",
        "review_path": "C:/reuse/reuse_review.md",
        "review_exists": True,
        "status": "attention",
        "stale_reasons": [],
        "summary": {"ambiguous": 1},
        "base_version_id": "1.0.0",
        "target_version_id": "1.1.0",
        "reconciliation_digest": "digest",
        "candidate_set_digest": "setdigest",
        "candidate_count": 1,
        "candidate_limit": 500,
        "candidate_start": 1,
        "candidate_end": 1,
        "candidates": [
            {
                "candidate_id": "reusecand1:abcdef",
                "reuse_class": "ambiguous",
                "status": "pending",
                "confidence": 0.42,
                "reference_only": False,
                "reference_origin": "model_initial",
                "candidate_target_occurrence_ids": ["target:1", "target:2"],
                "reference_translation": "回忆",
                "effective_translation": "回忆",
                "reference_translation_full": "回忆",
                "effective_translation_full": "回忆",
                "evidence": {"source_equal": True},
                "decision": {},
                "audit": [{"action": "reject", "reviewer_name": "测试审阅员"}],
            }
        ],
    }


if QObject is not None:

    class _FakeTaskWorker(QObject):
        completed = Signal(object)

        def __init__(self, task, parent=None) -> None:  # noqa: ARG002
            super().__init__(parent)
            self._task = task

        def isRunning(self) -> bool:  # noqa: N802 - Qt spelling
            return False

        def start(self) -> None:
            return None

        def requestInterruption(self) -> None:  # noqa: N802 - Qt spelling
            return None

        def wait(self, *_args) -> bool:
            return True


@gui_test_support.skip_unless_gui(EngineSnapshotDialog is None, IMPORT_ERROR)
class GuiEngineSnapshotDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self._worker_patch = mock.patch(
            "gui_qt.engine_snapshot_dialog.EngineSnapshotTaskWorker",
            _FakeTaskWorker,
        )
        self._worker_patch.start()
        self.dialog = EngineSnapshotDialog(game_root="C:/Game/work", start_tab=0)
        self.dialog._set_controls_enabled(True)

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        self._worker_patch.stop()

    def test_overview_populates_capabilities_snapshots_and_diff_choices(self) -> None:
        self.dialog._on_overview_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_overview_payload())
        )
        summary = self.dialog.engine_summary_label.text()
        self.assertIn("renpy", summary)
        self.assertIn("text_span_replace", summary)
        self.assertIn("C:/logs/project_snapshots", self.dialog.snapshot_root_label.text())
        self.assertEqual(self.dialog.snapshot_table.rowCount(), 2)
        self.assertEqual(self.dialog.diff_target_combo.count(), 1)
        self.assertEqual(self.dialog.diff_base_combo.count(), 1)
        self.assertEqual(
            self.dialog.diff_target_combo.currentData(),
            "C:/logs/project_snapshots/v2/project_snapshot.json",
        )

    def test_diff_result_populates_summary_and_item_table(self) -> None:
        self.dialog._on_diff_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_diff_payload())
        )
        self.assertIn("status：attention", self.dialog.diff_summary.toPlainText())
        self.assertEqual(self.dialog.diff_table.rowCount(), 1)
        self.assertEqual(
            self.dialog.diff_table.item(0, 0).text(),
            "base 歧义",
        )
        self.assertIn(
            "候选：new.rpy:2、new.rpy:3",
            self.dialog.diff_table.item(0, 5).text(),
        )

    def test_reuse_result_populates_candidates_and_review_action(self) -> None:
        self.dialog._on_reuse_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_reuse_payload())
        )
        self.assertIn("candidates：1", self.dialog.reuse_summary_label.text())
        self.assertEqual(self.dialog.reuse_table.rowCount(), 1)
        self.assertEqual(
            self.dialog.reuse_table.item(0, 1).text(),
            "歧义，需人工指定目标",
        )
        self.assertTrue(self.dialog.reuse_open_review_btn.isEnabled())
        self.assertIn("reuse_review.md", self.dialog.reuse_open_review_btn.toolTip())

    def test_reuse_selection_shows_full_record_and_requires_explicit_ambiguous_target(self) -> None:
        self.dialog.reuse_path_edit.setText("C:/reuse")
        self.dialog._on_reuse_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_reuse_payload())
        )
        self.assertEqual(self.dialog.reuse_reviewer_edit.text(), "")
        self.dialog.reuse_table.selectRow(0)
        details = self.dialog.reuse_detail.toPlainText()
        self.assertIn("旧译文：\n回忆", details)
        self.assertIn('"reviewer_name": "测试审阅员"', details)
        self.assertEqual(self.dialog.reuse_target_combo.count(), 3)
        self.assertEqual(self.dialog.reuse_target_combo.currentData(), "")
        self.assertTrue(self.dialog.reuse_accept_btn.isEnabled())
        self.dialog.reuse_reviewer_edit.setText("原创测试审阅者")
        with mock.patch.object(self.dialog, "_set_controls_enabled") as set_controls:
            self.dialog.reuse_accept_btn.click()
        set_controls.assert_not_called()
        self.assertIn("显式选择", self.dialog.reuse_status_label.text())

        self.dialog.reuse_target_combo.setCurrentIndex(2)
        submitted = {}
        with (
            mock.patch.object(self.dialog, "_start_task", side_effect=lambda task, handler: submitted.update(task=task, handler=handler)),
            mock.patch(
                "gui_qt.engine_snapshot_dialog.submit_reuse_candidate_decision",
                return_value={"status": "accepted", "paths": {"output_dir": "C:/reuse/out"}},
            ) as submit,
        ):
            self.dialog.reuse_accept_btn.click()
            submitted["task"]()

        submit.assert_called_once_with(
            "C:/reuse",
            "reusecand1:abcdef",
            "accept",
            "原创测试审阅者",
            note="",
            target_occurrence_id="target:2",
        )

    def test_reuse_table_pages_past_500_and_details_keep_long_translation(self) -> None:
        long_translation = "原创超长译文甲" * 30
        candidates = [
            {
                "candidate_id": f"reusecand1:{index}",
                "reuse_class": "exact_reuse",
                "status": "pending",
                "confidence": 0.9,
                "reference_only": False,
                "reference_origin": "synthetic_test_data",
                "target_occurrence_id": f"target:{index}",
                "reference_translation": "摘要",
                "effective_translation": "摘要",
                "reference_translation_full": long_translation,
                "effective_translation_full": long_translation,
                "evidence": {"source_equal": True},
                "decision": {},
                "audit": [],
            }
            for index in range(500)
        ]
        payload = _reuse_payload()
        payload.update(
            {
                "candidate_count": 501,
                "candidate_limit": 500,
                "candidate_start": 1,
                "candidate_end": 500,
                "candidates": candidates,
            }
        )
        self.dialog._on_reuse_loaded(EngineSnapshotTaskResult(ok=True, payload=payload))
        self.assertEqual(self.dialog.reuse_table.rowCount(), 500)
        self.assertIn("每页最多 500 条", self.dialog.reuse_summary_label.text())
        self.assertTrue(self.dialog.reuse_next_page_btn.isEnabled())
        self.dialog.reuse_table.selectRow(0)
        self.assertIn(long_translation, self.dialog.reuse_detail.toPlainText())
        self.assertGreater(len(long_translation), 160)

        last_page = dict(payload)
        last_page.update(
            {
                "candidate_start": 501,
                "candidate_end": 501,
                "candidates": [dict(candidates[-1], candidate_id="reusecand1:500")],
            }
        )
        self.dialog._reuse_page_offset = 500
        self.dialog._on_reuse_loaded(EngineSnapshotTaskResult(ok=True, payload=last_page))
        self.assertEqual(self.dialog.reuse_table.rowCount(), 1)
        self.assertIn("第 501–501 条 / 共 501 条", self.dialog.reuse_page_label.text())
        self.assertTrue(self.dialog.reuse_previous_page_btn.isEnabled())
        self.assertFalse(self.dialog.reuse_next_page_btn.isEnabled())

    def test_cancel_clears_only_unsubmitted_reuse_editor_fields(self) -> None:
        self.dialog._on_reuse_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_reuse_payload())
        )
        self.dialog.reuse_table.selectRow(0)
        self.dialog.reuse_reviewer_edit.setText("原创测试审阅者")
        self.dialog.reuse_note_edit.setText("未提交的原创备注")
        self.dialog.reuse_target_combo.setCurrentIndex(1)

        self.dialog.reuse_cancel_decision_btn.click()

        self.assertEqual(self.dialog.reuse_reviewer_edit.text(), "")
        self.assertEqual(self.dialog.reuse_note_edit.text(), "")
        self.assertEqual(self.dialog.reuse_table.currentRow(), -1)
        self.assertEqual(self.dialog.reuse_target_combo.currentData(), "")
        self.assertIn("没有提交决定", self.dialog.reuse_status_label.text())

    def test_manifest_switch_clears_unsubmitted_reuse_editor(self) -> None:
        self.dialog._on_reuse_loaded(
            EngineSnapshotTaskResult(ok=True, payload=_reuse_payload())
        )
        self.dialog.reuse_table.selectRow(0)
        self.dialog.reuse_reviewer_edit.setText("原创测试审阅者")
        self.dialog.reuse_note_edit.setText("未提交的原创备注")

        self.dialog.reuse_manifest_edit.setText("C:/batch/other/manifest.json")

        self.assertEqual(self.dialog.reuse_reviewer_edit.text(), "")
        self.assertEqual(self.dialog.reuse_note_edit.text(), "")
        self.assertEqual(self.dialog.reuse_table.currentRow(), -1)
        self.assertIn("manifest 已更改", self.dialog.reuse_status_label.text())

    def test_late_worker_result_is_discarded_after_project_package_or_manifest_switch(self) -> None:
        for change in (
            lambda: setattr(self.dialog, "_game_root", "C:/Game/other"),
            lambda: self.dialog.reuse_path_edit.setText("C:/reuse/other"),
            lambda: self.dialog.reuse_manifest_edit.setText(
                "C:/batch/other/manifest.json"
            ),
        ):
            with self.subTest(change=change):
                expected = self.dialog._async_context_key()
                handler = mock.Mock()
                change()

                self.dialog._on_task_completed(
                    EngineSnapshotTaskResult(
                        ok=True,
                        payload=_reuse_payload(),
                    ),
                    handler,
                    expected,
                )

                handler.assert_not_called()
                self.assertIn("已忽略", self.dialog.reuse_status_label.text())

    def test_error_result_shows_message_box(self) -> None:
        with mock.patch.object(QMessageBox, "warning") as warning:
            self.dialog._on_diff_loaded(
                EngineSnapshotTaskResult(
                    ok=False,
                    error="bad snapshot",
                    error_code="VersioningArtifactError",
                )
            )
        warning.assert_called_once()
        self.assertIn("bad snapshot", warning.call_args.args[2])

    def test_context_library_page_has_engine_actions(self) -> None:
        page = ContextLibraryPage()
        clicked: list[str] = []
        page.set_action_callbacks(
            WorkbenchPageActions(action=lambda name: clicked.append(name))
        )
        page._has_project = True
        page._refresh_action_states()
        page.engine_snapshot_btn.click()
        page.reuse_candidates_btn.click()
        page.deleteLater()
        self.assertEqual(clicked, ["engine_snapshot", "reuse_candidates"])


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiEngineSnapshotEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = QApplication.instance()
        if app is None:
            cls._app = QApplication([])
        else:
            cls._app = app

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_main_window_open_dialog_forwards_project_and_start_tab(self) -> None:
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch(
                "gui_qt.engine_snapshot_dialog.EngineSnapshotDialog"
            ) as dialog_cls,
        ):
            self.window._open_engine_snapshot_dialog(start_tab=2)

        dialog_cls.assert_called_once()
        self.assertEqual(dialog_cls.call_args.kwargs["game_root"], "C:/Game/work")
        self.assertEqual(dialog_cls.call_args.kwargs["start_tab"], 2)
        dialog_cls.return_value.exec.assert_called_once_with()

    def test_context_library_action_routes_to_engine_snapshot(self) -> None:
        with (
            mock.patch.object(
                self.window.state,
                "get_game_root",
                return_value="C:/Game/work",
            ),
            mock.patch.object(
                self.window,
                "_open_engine_snapshot_dialog",
            ) as opener,
        ):
            self.window._on_context_library_action("engine_snapshot")
            self.window._on_context_library_action("reuse_candidates")

        self.assertEqual(
            opener.call_args_list,
            [mock.call(start_tab=0), mock.call(start_tab=2)],
        )


if __name__ == "__main__":
    unittest.main()
