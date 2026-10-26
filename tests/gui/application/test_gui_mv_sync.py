"""MV product gates, settings and late-result rejection using original material."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.support import gui_test_support
from tests.support.mv_project import make_mv_project, add_mv_display_material

try:
    from PySide6.QtWidgets import QApplication
    from gui_qt.app import MainWindow
    from gui_qt.project_state import ProjectState
    from gui_qt.reports.doctor_report import summarize_doctor_report
    from gui_qt.reports.readiness_display import count_line
    from gui_qt.settings.project_page import ProjectSettingsPage
    from gui_qt.settings.settings_schema import ADVANCED_SETTING_FIELD_BY_KEY, normalize_for_write, set_nested_value, validate_value
    from gui_qt.work_modes import WorkMode
    from gui_qt.workers.doctor_worker import DoctorWorkerResult
    from gui_qt import diagnostics_context
    from tests.gui.reports import test_gui_readiness_summary as readiness_tests
except ImportError as exc:
    MainWindow, IMPORT_ERROR = None, exc
else:
    IMPORT_ERROR = None


@gui_test_support.skip_unless_gui(MainWindow is None, IMPORT_ERROR)
class GuiMvSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.config_file, self.config, _positions = make_mv_project(self.root)
        import translator_runtime as runtime
        saved = runtime.snapshot_runtime_config()
        self.stack.callback(runtime.apply_runtime_config, saved)
        state = ProjectState()
        state.config_path = self.config_file
        state.api_keys_path = self.root / 'no-real-credentials.json'
        state._game_root = self.root / 'target'
        state.get_api_key_status = lambda **_kwargs: (1, 'original-fixture')
        self.window = MainWindow(project_state=state, bootstrap_config=self.config)
        self.addCleanup(self.close_window)
        self.stack.enter_context(mock.patch('gui_qt.app.DoctorWorker', readiness_tests.HeldDoctor))
        self.window._snapshot_runtime_config_for_job = lambda **_kwargs: None

    def close_window(self):
        gui_test_support.close_main_window(self.window)
        self.window.deleteLater()

    def test_mv_settings_round_trip_and_unsupported_modes_cannot_start(self):
        page = ProjectSettingsPage()
        self.addCleanup(page.widget.deleteLater)
        page.load({'engine': 'rpgmaker_mv', 'mv_source_root': 'original/www',
                   'mv_support_window_file': 'approved.json', 'mv_occurrence_scope_file': 'pilot.json',
                   'mv_display_fields': ['system_terms', 'map_display_name']})
        values = page.collect()
        self.assertEqual(values['engine'], 'rpgmaker_mv')
        self.assertEqual(values['mv_source_root'], 'original/www')
        self.assertEqual(values['mv_support_window_file'], 'approved.json')
        self.assertEqual(values['mv_occurrence_scope_file'], 'pilot.json')
        self.assertIn('mv_reviewed_history_file', values)
        self.assertEqual(values['mv_display_fields'], 'system_terms\nmap_display_name')
        field = ADVANCED_SETTING_FIELD_BY_KEY['mv_display_fields']
        configured = {}
        set_nested_value(configured, field.path, normalize_for_write(field, values['mv_display_fields']))
        self.assertEqual(configured['rpgmaker_mv']['display_fields'], ['map_display_name', 'system_terms'])
        self.assertTrue(validate_value(field, 'database'))
        self.assertTrue(validate_value(field, 'system_terms\nsystem_terms'))
        self.assertTrue(self.window._engine_allows_work_mode(WorkMode.SYNC_TRANSLATION))
        for mode in (WorkMode.BATCH_TRANSLATION, WorkMode.KEYWORD_EXTRACTION, WorkMode.REVISION):
            self.assertFalse(self.window._engine_allows_work_mode(mode))
        self.window._set_work_mode(WorkMode.BATCH_TRANSLATION, refresh_manifest_writeback=False)
        self.assertFalse(self.window.translate_btn.isEnabled())
        self.assertIn('Batch、外部工作包', self.window.work_mode_hint_label.text())
        self.assertFalse(self.window._resume_task_available()[0])
        self.window._sync_bootstrap_prep_button_chrome()
        self.assertFalse(self.window.bootstrap_work_btn.isEnabled())

    def test_display_selection_change_rejects_late_doctor_and_uses_shared_counts(self):
        add_mv_display_material(self.root)
        self.window._on_run_doctor()
        worker = self.window._doctor_worker
        self.config['rpgmaker_mv']['display_fields'] = ['system_terms', 'map_display_name']
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        report = {'mode': 'mv_narrative', 'layout_status': 'ready', 'tl_exists': True,
                  'counts': {'json_files': 7}, 'pending_task_count': 14, 'pending_file_count': 3,
                  'engine_status': {'engine': 'rpgmaker_mv', 'status': 'ready', 'issues': []}}
        worker.finish(DoctorWorkerResult(True, report, ''))
        self.assertIsNone(self.window._readiness_doctor_report)
        from execution.mv_sync import build_snapshot
        snapshot = build_snapshot(self.config, self.config_file, require_review=False)
        self.assertEqual(snapshot.pending_task_count, 25)
        self.assertTrue(snapshot.review_error)
        self.window._on_run_doctor()
        report['pending_task_count'] = snapshot.pending_task_count
        report['layout_status'] = 'failed'
        report['engine_status'].update(status='blocked', issues=[{'message': snapshot.review_error}])
        self.window._doctor_worker.finish(DoctorWorkerResult(True, report, ''))
        self.assertIn('待译：25', self.window.doctor_facts_label.text())

    def test_material_edit_rejects_late_doctor_and_invalidates_display(self):
        report = {'mode': 'mv_narrative', 'layout_status': 'ready', 'tl_exists': True,
                  'counts': {'json_files': 4}, 'pending_task_count': 14, 'pending_file_count': 3,
                  'engine_status': {'engine': 'rpgmaker_mv', 'status': 'ready', 'issues': []}}
        self.window._on_run_doctor()
        worker = self.window._doctor_worker
        path = self.root / 'source/data/Map001.json'
        path.write_bytes(path.read_bytes() + b' ')
        worker.finish(DoctorWorkerResult(True, report, ''))
        self.assertIsNone(self.window._readiness_doctor_report)
        self.assertFalse(self.window._doctor_check_completed)
        self.window._on_run_doctor()
        self.window._doctor_worker.finish(DoctorWorkerResult(True, report, ''))
        self.assertIsNotNone(self.window._readiness_doctor_report)
        self.assertIn('待译：14', self.window.doctor_facts_label.text())
        window_file = self.root / 'approved-window.json'
        content = json.loads(window_file.read_text())
        content['message_styles'] = False
        window_file.write_text(json.dumps(content), encoding='utf-8')
        self.window._sync_readiness_display()
        self.assertIsNone(self.window._readiness_doctor_report)
        self.assertNotIn('待译：14', self.window.doctor_facts_label.text())

    def test_mv_diagnostics_only_advertise_supported_routes_and_unknown_is_not_zero(self):
        context = diagnostics_context.sync_diagnostics_context(
            sync_script_path='legacy.py', batch_script_path='batch.py', engine='rpgmaker_mv')
        commands = '\n'.join(command.command for command in context.commands)
        self.assertIn('sync-start', commands)
        self.assertIn('coverage-status', commands)
        self.assertIn('--export-dir', commands)
        self.assertIn('sync-keywords', commands)
        self.assertIn('bootstrap-source-index', commands)
        self.assertIn('bootstrap-rag', commands)
        self.assertNotIn('legacy.py', commands)
        self.assertNotIn('work-export', commands)
        report = {'mode': 'mv_narrative', 'counts': {'json_files': None},
                  'engine_status': {'engine': 'rpgmaker_mv', 'status': 'blocked', 'issues': [
                      {'message': 'Original fixture: material unreadable'}]}}
        self.assertEqual(summarize_doctor_report(report).status, 'blocked')
        self.assertIn('待译：未知', count_line(report))
        self.assertNotIn('待译：0', count_line(report))

    def test_mv_context_worker_uses_source_status_and_project_switch_rejects_late_result(self):
        from gui_qt.workers.context_library_worker import collect_context_library_status
        self.config['sync']['source_index']['enabled'] = True
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        with mock.patch('gemini_translate_batch.compute_current_project_analysis_fingerprint') as renpy_scan:
            result = collect_context_library_status(str(self.root / 'target'), self.config, config_file=str(self.config_file))
        renpy_scan.assert_not_called()
        self.assertFalse(result.error, result.error)
        self.assertEqual(result.status['mv_preparation']['source_index']['status'], 'missing')
        self.assertEqual(result.status['overall_status'], 'unsupported')
        self.assertTrue(self.window._engine_allows_work_mode(WorkMode.SYNC_KEYWORD_EXTRACTION))
        self.assertTrue(self.window._engine_allows_work_mode(WorkMode.BOOTSTRAP_SOURCE_INDEX))
        self.assertFalse(self.window._engine_allows_renpy_prepare())
        from gui_qt.runtime.operation_identity import context_library_config_digest
        before = result.config_digest
        from execution.mv_preprocessing import store_paths
        metadata = store_paths(self.config, self.config_file)['source'] / 'source_metadata.json'
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text('{"mv_preparation":{"status":"building"}}', encoding='utf-8')
        after = context_library_config_digest(self.config, context_flags=result.context_flags,
                                             config_file=str(self.config_file))
        self.assertNotEqual(before, after)
        other = self.root / 'other-project'
        other.mkdir()
        other_file, other_config, _ = make_mv_project(other)
        self.window.state.config_path = other_file
        self.window.state._game_root = other / 'target'
        self.window._context_library_config_snapshot = other_config
        self.window._context_library_status_cache = None
        with mock.patch('gui_qt.app.QTimer.singleShot'), mock.patch.object(self.window, '_render_context_library_panel') as render:
            self.window._on_context_library_status_ready(result)
        render.assert_not_called()
        self.assertIsNone(self.window._context_library_status_cache)

    def test_mv_bootstrap_buttons_use_sync_switches_and_the_same_cli(self):
        self.config['batch'] = {'rag': {'enabled': False}, 'source_index': {'enabled': False}}
        self.config['sync']['rag'] = {'enabled': True}
        self.config['sync']['source_index']['enabled'] = True
        self.config_file.write_text(json.dumps(self.config), encoding='utf-8')
        with mock.patch.object(self.window, '_confirm_unsaved_config_before_workflow', return_value=True), \
                mock.patch.object(self.window, '_start_cli_command', return_value=True) as start:
            self.assertTrue(self.window._start_bootstrap_task('rag'))
            self.assertEqual(start.call_args.args[2], ['bootstrap-rag', '--skip-prepare'])
            self.assertTrue(self.window._start_bootstrap_task('source_index'))
            self.assertEqual(start.call_args.args[2], ['bootstrap-source-index', '--skip-prepare'])


if __name__ == '__main__':
    unittest.main()
