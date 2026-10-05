"""Offline revision stages through real window wiring and original artifacts."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import gui_test_support
import cli_contract
import gemini_translate_batch as batch
import revision_corpus
import revision_selection
import review_workspace

try:
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog
    from gui_qt.app import MainWindow
    from gui_qt.revision_workflow import RevisionBatchWorkflow, RevisionProposalImportWorkflow
    from gui_qt.revision_writeback_report import summarize_revision_writeback_from_preview_output
    from gui_qt.revision_report import summarize_revision_apply_output
    from gui_qt.work_modes import WorkMode
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@contextlib.contextmanager
def original_project():
    """Isolate all file writes and use one original, credential-free TL entry."""
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        root = Path(directory)
        tl = root / "game" / "tl" / "schinese"
        tl.mkdir(parents=True)
        script = tl / "lantern.rpy"
        script.write_text('translate schinese strings:\n    old "Carry the lantern, {name}."\n    new "带上灯，{name}。"\n', encoding="utf-8")
        for owner, values in (
            (batch.legacy, {"BASE_DIR": str(root), "TL_DIR": str(tl), "INCLUDE_FILES": set(), "INCLUDE_PREFIXES": set()}),
            (batch, {"BATCH_JOBS_DIR": str(root / "jobs"), "LATEST_MANIFEST_FILE": str(root / "latest.txt"), "RAG_ENABLED": False, "STORY_MEMORY_ENABLED": False}),
        ):
            stack.enter_context(mock.patch.multiple(owner, **values))
        yield root, tl, script


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class RevisionGuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window._set_work_mode(WorkMode.REVISION, refresh_manifest_writeback=False)
        self.page = self.window.revision_page
        self.page.set_project_ready(True)

    def tearDown(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def stage(self):
        return self.page.guidance_title.property("stage")

    def test_model_workflow_wait_resume_preview_and_confirm_cancel(self):
        workflow = RevisionBatchWorkflow.start_new()
        self.window._set_task_running(True)
        self.assertEqual(self.stage(), "running")
        update = workflow.complete_current_step(0, "Created revision package: C:/offline/lantern")
        self.assertTrue(update.should_continue)
        workflow.complete_current_step(0, "Manifest: C:/offline/lantern/manifest.json")
        update = workflow.complete_current_step(0, "State: JOB_STATE_RUNNING")
        self.window._workflow = workflow
        self.window._set_workflow_update(update)
        self.window._set_task_running(False)
        self.assertEqual(self.stage(), "waiting")
        resumed = RevisionBatchWorkflow.resume_manifest(workflow.manifest_path, {"job_state": "JOB_STATE_SUCCEEDED", "job_name": "offline"})
        self.assertEqual(resumed.current_step().key, "download")
        resumed.complete_current_step(0, "Downloaded results")
        output = "Recoverable revision items: 1\nPending files: 1\nPending lines: 1\nFailure items: 0\nRevision writeback gate: allow\nPreview Markdown: C:/offline/preview.md"
        update = resumed.complete_current_step(0, output)
        self.window._set_workflow_update(update)
        summary = summarize_revision_writeback_from_preview_output(output, 0, manifest_path=workflow.manifest_path)
        self.window._set_writeback_summary(summary)
        self.assertEqual(self.stage(), "ready")
        self.assertTrue(self.page.writeback_btn.isEnabled())
        with mock.patch("gui_qt.app.message_box_question", return_value="no"), mock.patch.object(self.window, "_start_cli_command") as start:
            self.page.writeback_btn.click()
        start.assert_not_called()
        self.assertIs(self.window._current_writeback_summary(), summary)
        self.assertEqual(self.stage(), "ready")
        empty_preview = summarize_revision_writeback_from_preview_output(output.replace("items: 1", "items: 0"), 0, manifest_path=workflow.manifest_path)
        self.window._set_writeback_summary(empty_preview)
        self.assertEqual(self.stage(), "preview")
        self.assertFalse(self.page.writeback_btn.isEnabled())
        applied = summarize_revision_apply_output("Revision apply state: applied\nApplied files: 1\nApplied lines: 1", 0, manifest_path=workflow.manifest_path)
        self.window._set_writeback_summary(applied)
        self.assertEqual(self.stage(), "applied")
        self.assertFalse(self.page.writeback_btn.isEnabled())

    def test_original_external_artifacts_import_select_preview_without_game_write(self):
        with original_project() as (root, tl, script), contextlib.redirect_stdout(io.StringIO()):
            before = script.read_bytes()
            corpus = batch.run_revision_corpus_export(root / "corpus")
            item = batch.collect_revision_file_jobs()[0]["items"][0]
            row = {
                "schema_version": 1, "occurrence_id": item["id"], "identity_v2": item["id"],
                "file_rel_path": item["file_rel_path"], "source": item["source"],
                "current_translation": item["current_translation"], "proposed_translation": "带好提灯，{name}。",
                "reason": "统一灯具用语", "selected": False, "disposition": "accepted", "producer": {"type": "agent", "tool": "offline-fixture"},
                "project_identity": {"tl_dir": str(tl)},
                "snapshot_digest": revision_corpus.item_snapshot_digest(item["source"], item["current_translation"]),
                "corpus_snapshot_digest": corpus["source"]["snapshot_digest"],
            }
            proposal = root / "proposals.jsonl"
            proposal.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            staged = batch.import_revision_proposals(str(proposal), stage=True)
            workflow = RevisionProposalImportWorkflow(str(proposal), stage=True, operation_identity=staged["operation_identity"])
            envelope = cli_contract.success_envelope("import-revision-proposals", status="staged", result=staged, artifacts=staged["paths"])
            self.window._workflow = workflow
            self.window._workflow_step_output_lines = [json.dumps(envelope)]
            with mock.patch.object(self.window, "_current_revision_proposal_operation_identity", return_value=workflow.operation_identity), mock.patch("gui_qt.app.QTimer.singleShot"):
                self.window._on_workflow_step_finished(0)
            self.assertEqual(self.stage(), "select", staged.get("diagnostics"))
            # Hidden-at-construction controls must stay in the actual window,
            # rather than becoming detached top-level buttons on first import.
            self.assertIs(self.page.select_proposals_btn.window(), self.window)
            self.assertFalse(self.page.writeback_btn.isEnabled())
            self.assertEqual(script.read_bytes(), before)
            stage = revision_selection.load_staged_selection(staged["paths"]["staged_selection"])
            with mock.patch.object(self.window, "_current_revision_proposal_operation_identity", return_value=workflow.operation_identity), mock.patch("gui_qt.app.RevisionProposalSelectionDialog") as dialog, mock.patch.object(self.window, "_begin_translation_workflow") as begin:
                dialog.return_value.exec.return_value = QDialog.DialogCode.Rejected
                self.window._open_revision_proposal_selection()
            begin.assert_not_called()
            self.assertEqual(self.stage(), "select")
            request = revision_selection.make_selection_request(stage, [item["id"]])
            selection = root / "selection.json"
            revision_selection.write_selection_request(str(selection), request)
            confirmed = batch.confirm_revision_proposals(staged["paths"]["staged_selection"], str(selection))
            self.window._refresh_writeback_from_revision_manifest(confirmed["paths"]["manifest"])
            self.assertEqual(self.stage(), "ready")
            self.assertTrue(self.page.writeback_btn.isEnabled())
            self.assertEqual(script.read_bytes(), before)
            # A changed authoritative selection still fails in the existing core.
            selection.write_text('{}', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "selection changed"):
                batch.apply_revisions(confirmed["paths"]["manifest"], force=True)
            self.assertEqual(script.read_bytes(), before)
            with mock.patch.object(self.window, "_current_revision_proposal_operation_identity", return_value="changed-project"), mock.patch("gui_qt.app.message_box_information"):
                self.window._open_revision_proposal_selection()
            self.assertEqual(self.stage(), "stale")
            self.assertFalse(self.page.writeback_btn.isEnabled())
            self.assertIsNone(self.window._revision_proposal_stage_result)

    def test_stop_completion_and_late_import_are_not_executable_results(self):
        workflow = RevisionBatchWorkflow.start_new()
        self.window._workflow = workflow
        self.window._set_task_running(True)
        with mock.patch.object(self.window.runner, "kill", side_effect=lambda: setattr(self.window.runner, "_stop_requested", True)):
            self.page.stop_btn.click()
        self.assertEqual(self.stage(), "stopping")
        self.window._on_workflow_step_finished(-1)
        self.assertEqual(self.stage(), "stopped")
        self.assertFalse(self.page.writeback_btn.isEnabled())
        def start_again(*_args):
            self.window.runner._stop_requested = False
            return True
        with mock.patch.object(self.window.runner, "run", side_effect=start_again):
            self.assertTrue(self.window._start_cli_command("translation_workflow", "C:/offline/preview.py", []))
        self.assertEqual(self.stage(), "running")
        self.window._set_task_running(False)
        workflow = RevisionProposalImportWorkflow("C:/offline/proposals.jsonl", stage=True, operation_identity="old-project")
        self.window._workflow = workflow
        self.window._workflow_step_output_lines = ['{"schema_version":1,"command":"import-revision-proposals","ok":true,"status":"staged","result":{"session_status":"ready","selectable_count":1},"artifacts":{},"warnings":[],"error":null}']
        with mock.patch.object(self.window, "_current_revision_proposal_operation_identity", return_value="new-project"):
            self.window._on_workflow_step_finished(0)
        self.assertEqual(self.stage(), "stale")
        self.assertIsNone(self.window._revision_proposal_stage_result)
        self.assertFalse(self.page.select_proposals_btn.isEnabled())
        self.assertFalse(self.page.writeback_btn.isEnabled())
        # Even buffered success output cannot publish candidates after stop.
        self.window.runner._stop_requested = True
        workflow = RevisionProposalImportWorkflow("C:/offline/proposals.jsonl", stage=True, operation_identity="new-project")
        self.window._workflow = workflow
        self.window._workflow_step_output_lines = ['{"schema_version":1,"command":"import-revision-proposals","ok":true,"status":"staged","result":{"session_status":"ready","selectable_count":1},"artifacts":{},"warnings":[],"error":null}']
        with mock.patch.object(self.window, "_current_revision_proposal_operation_identity", return_value="new-project"):
            self.window._on_workflow_step_finished(-1)
        self.assertEqual(self.stage(), "stopped")
        self.assertIsNone(self.window._revision_proposal_stage_result)
        self.assertFalse(self.page.select_proposals_btn.isEnabled())

    def test_file_and_import_cancellations_preserve_context(self):
        result = {"session_status": "ready", "selectable_count": 1}
        self.window._revision_proposal_stage_result = result
        self.window._sync_revision_page_controls()
        for selection, corpus in (("", None), ("C:/offline/proposals.jsonl", None)):
            with mock.patch("gui_qt.app.QFileDialog.getOpenFileName", return_value=(selection, "")), mock.patch.object(self.window, "_choose_revision_corpus_manifest", return_value=corpus), mock.patch.object(self.window, "_import_review_workspace_proposals") as begin:
                self.window._on_final_review_page_action("import_revision_proposals")
            begin.assert_not_called()
            self.assertIs(self.window._revision_proposal_stage_result, result)
            self.assertEqual(self.stage(), "select")
        for button_text, expected in (("取消导入", None), ("选择配套语料", None), ("无配套语料，继续校验", "")):
            def choose(text=button_text):
                dialog = self.app.activeModalWidget()
                next(button for button in dialog.buttons() if button.text() == text).click()
            QTimer.singleShot(0, choose)
            with mock.patch("gui_qt.app.QFileDialog.getOpenFileName", return_value=("", "")):
                self.assertEqual(self.window._choose_revision_corpus_manifest("C:/offline/proposals.jsonl"), expected)

    def test_review_drafts_switch_and_late_load_keep_game_untouched(self):
        with original_project() as (root, tl, script), contextlib.redirect_stdout(io.StringIO()):
            before = script.read_bytes()
            corpus = batch.run_revision_corpus_export(root / "corpus")
            manifest, entries = review_workspace.open_workspace(corpus["paths"]["manifest"], expected_game_root=str(root), expected_tl_dir=str(tl))
            panel = self.page.review_workspace
            panel._generation = 1
            panel._current_context = lambda: "current"
            panel._loaded((1, "current"), manifest, entries)
            self.assertEqual(self.stage(), "review")
            panel.table.selectRow(0)
            panel.proposed.setPlainText("带好提灯，{name}。")
            panel.reason.setText("统一用语")
            panel.save_current_draft()
            self.assertEqual(script.read_bytes(), before)
            panel.proposed.setPlainText("记得带提灯，{name}。")
            # Existing task leave guard flushes unsaved edits before switching.
            self.window._set_work_mode(WorkMode.SYNC_REVISION, refresh_manifest_writeback=False)
            drafts = review_workspace.load_drafts(manifest["_manifest_path"], manifest)
            self.assertEqual(next(iter(drafts.values()))["proposed_translation"], "记得带提灯，{name}。")
            panel._loaded((1, "current"), manifest, entries)
            self.assertIsNone(panel.manifest)
            self.assertEqual(self.stage(), "sync_empty")
            self.assertFalse(self.page.writeback_btn.isEnabled())
            self.assertEqual(script.read_bytes(), before)

    def test_keyboard_and_gate_projection_do_not_follow_button_visibility(self):
        from gui_qt.workbench.page_contract import WorkbenchPageActions
        events = []
        self.page.set_action_callbacks(WorkbenchPageActions(action=events.append))
        self.page.set_controls(start_enabled=True, resume_enabled=False, resume_visible=True, resume_label="继续订正", writeback_enabled=False, result_message="", export_enabled=True)
        self.page.show()
        self.page.import_proposals_btn.setFocus()
        QTest.keyClick(self.page.import_proposals_btn, Qt.Key.Key_Space)
        self.assertEqual(events, ["import_revision_proposals"])
        self.page.writeback_btn.hide()
        self.page.set_guidance_state(can_apply=True, writeback_status="warn")
        self.assertEqual(self.stage(), "ready")
        self.page.set_guidance_state(can_apply=False, writeback_status="failed")
        self.assertEqual(self.stage(), "failed")
        self.page.reset_project()
        self.assertEqual(self.stage(), "empty")
        self.assertFalse(self.page.writeback_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
