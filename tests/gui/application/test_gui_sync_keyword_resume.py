"""Recovery acknowledgement is required only for unresolved keyword dispatches."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.support import gui_test_support
from gui_qt.work_modes import WorkMode
from gui_qt.project_state import ProjectState
from gui_qt.user_copy import SYNC_KEYWORD_COPY
from gui_qt.workflows.workflow_factory import validate_resume_manifest

try:
    from gui_qt.app import MainWindow
except ImportError as exc:
    MainWindow = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiSyncKeywordResumeTests(unittest.TestCase):
    def test_heavy_sync_run_enables_resume_and_keeps_its_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            game = root / "work" / "www"
            (game / "game").mkdir(parents=True)
            (game / "game" / "script.rpy").write_text("label start:\n    return\n", encoding="utf-8")
            package = root / "logs" / "sync_runs" / "keyword-run"
            package.mkdir(parents=True)
            path = package / "manifest.json"
            candidates = package / "keyword_candidates.jsonl"
            candidates.write_text("", encoding="utf-8")
            path.write_text(json.dumps({
                "mode": "keyword_extraction", "execution": "sync", "base_dir": str(game),
                "job_state": "SYNC_PARTIAL", "chunks": [{"payload": "x" * 1_000_000}],
                "keyword_checkpoint_version": 1,
                "keyword_export": {"jsonl_path": str(candidates)},
            }), encoding="utf-8")
            state = ProjectState.__new__(ProjectState)
            state.tool_root = root
            state.config_path = root / "translator_config.json"
            state._game_root = game
            state._manifest_file_cache = {}
            window = MainWindow.__new__(MainWindow)
            window.state = state
            window._context_library_config_snapshot = {"engine": "renpy"}
            window._current_work_mode = lambda: WorkMode.SYNC_KEYWORD_EXTRACTION
            window._workflow = None
            window._task_running = False
            window.kill_btn = mock.Mock()
            window.kill_btn.isEnabled.return_value = False
            window.resume_btn = mock.Mock()
            window._resume_button_is_query_mode = lambda: False
            window._writeback_manifest_path = ""
            window._keyword_candidate_selection = None
            window._keyword_merge_candidates_path = str(candidates)
            window._load_diagnostics_manifest = lambda value: state.load_manifest_file(value) if value else None

            self.assertEqual(window._resume_task_available(), (True, ""))
            window._update_resume_btn_enabled()
            window.resume_btn.setEnabled.assert_called_once_with(True)
            window.resume_btn.setToolTip.assert_called_once_with(SYNC_KEYWORD_COPY['resume_tooltip'])
            self.assertEqual(window._resolve_keyword_merge_candidates_path(), str(candidates))
            self.assertEqual(window._keyword_merge_candidates_path, str(candidates))

            window._keyword_merge_candidates_path = ""
            self.assertEqual(window._resolve_keyword_merge_candidates_path(), str(candidates))

    def test_unknown_cancel_and_confirm_are_respected(self):
        for reply in ('no', 'yes'):
            with self.subTest(reply=reply):
                window = mock.Mock()
                window._current_work_mode.return_value = WorkMode.SYNC_KEYWORD_EXTRACTION
                window.state.get_game_root.return_value = Path('C:/original-fixture')
                window._latest_resume_manifest_path.return_value = Path('C:/fixture/manifest.json')
                window._resume_button_is_query_mode.return_value = False
                window._split_entries_for_manifest.return_value = []
                with (mock.patch('execution.keyword_checkpoint.unknown_keyword_requests',
                                 return_value=['kw-1']),
                      mock.patch('gui_qt.app.message_box_question', return_value=reply),
                      mock.patch('gui_qt.app.resume_workflow') as factory):
                    MainWindow._on_resume_translation(window)
                if reply == 'no':
                    factory.assert_not_called()
                    window._run_workflow_current_step.assert_not_called()
                else:
                    self.assertTrue(factory.call_args.kwargs['retry_unknown'])
                    window._run_workflow_current_step.assert_called_once()

    def test_sync_keyword_resume_rejects_batch_and_pre_checkpoint_packages(self):
        for manifest in (
                {'mode': 'keyword_extraction', 'execution': 'gemini_batch'},
                {'mode': 'keyword_extraction', 'execution': 'sync'}):
            with self.assertRaisesRegex(ValueError, '检查点'):
                validate_resume_manifest(WorkMode.SYNC_KEYWORD_EXTRACTION, manifest,
                                         game_root=None, normalized_path_text=str)

