"""Wrong engine selections must not suggest templates or dispatch work."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemini_translate_batch as batch
import translator_runtime as runtime
from foundation import cli_contract
from engine_adapters.project_config import ProjectEngineMismatchError


class DoctorEngineMismatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.target = self.root / 'work'
        content = self.target / 'www'
        (content / 'data').mkdir(parents=True)
        (content / 'js').mkdir()
        (content / 'js/rpg_core.js').write_text('// Original inert marker.\n')
        self.config_file = self.root / 'translator_config.json'
        self.config_file.write_text(json.dumps({'engine': 'renpy', 'game_root': str(self.target),
            'prepare': {'enabled': True, 'template_command': ['must-never-run']}}), encoding='utf-8')
        self.stack.enter_context(mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(self.config_file)))
        self.stack.enter_context(runtime.runtime_config_scope(
            reload_translator_settings=True, persist_corrected_game_root=False))

    def test_doctor_preserves_unknown_counts_and_never_enters_renpy_checks(self):
        before = self.config_file.read_bytes()
        with mock.patch.object(runtime, 'get_prepare_template_command_info') as template, \
                mock.patch.object(batch, 'collect_tl_doctor_counts') as scan:
            report = batch.collect_doctor_report()
        template.assert_not_called()
        scan.assert_not_called()
        self.assertEqual(report['mode'], 'engine_mismatch')
        self.assertEqual(report['engine_status']['status'], 'blocked')
        self.assertEqual(report['engine_status']['issues'][0]['code'], 'engine.project_mismatch')
        self.assertNotIn('pending_task_count', report)
        self.assertEqual(report['counts'], {})
        self.assertFalse(report['can_generate_template'])
        self.assertFalse(report['recommendations'])
        envelope = batch.build_machine_success_envelope('doctor', report, mock.Mock())
        self.assertEqual(envelope['status'], 'blocked')
        self.assertEqual(cli_contract.strict_exit_code(envelope), cli_contract.EXIT_BLOCKED)
        output = io.StringIO()
        with redirect_stdout(output):
            batch.print_doctor_report(report)
        self.assertIn('Pending translation: unknown', output.getvalue())
        self.assertNotIn('Template generation: available', output.getvalue())
        self.assertEqual(self.config_file.read_bytes(), before)

    def test_cli_refuses_execution_before_prepare_or_provider(self):
        for command in ('generate-template', 'bootstrap-work', 'build', 'sync-start'):
            with self.subTest(command=command), \
                    mock.patch.object(runtime, 'run_prepare_steps') as prepare, \
                    mock.patch.object(batch, 'run_durable_sync_command') as sync:
                output = io.StringIO()
                with redirect_stdout(output):
                    if command in batch.MACHINE_OUTPUT_COMMANDS:
                        code = batch.main([command, '--output', 'json', '--strict-exit-codes'])
                        envelope = json.loads(output.getvalue())
                        self.assertEqual(code, cli_contract.EXIT_BLOCKED, envelope)
                        self.assertIn('ENGINE_PROJECT_MISMATCH', output.getvalue())
                    else:
                        with self.assertRaises(cli_contract.MachineContractError) as raised:
                            batch.main([command])
                        self.assertEqual(raised.exception.code_name, 'ENGINE_PROJECT_MISMATCH')
                prepare.assert_not_called()
                sync.assert_not_called()

    def test_direct_prepare_refuses_before_copy_or_custom_command(self):
        with mock.patch.object(runtime, 'bootstrap_work_from_original') as copy, \
                mock.patch.object(runtime, 'subprocess') as process:
            with self.assertRaises(ProjectEngineMismatchError):
                runtime.run_prepare_steps()
        copy.assert_not_called()
        process.run.assert_not_called()

    def test_omitted_engine_auto_detects_full_project_and_reports_setup_inputs_read_only(self):
        config = json.loads(self.config_file.read_text(encoding='utf-8'))
        config.pop('engine')
        config['include_files'] = []
        config['include_prefixes'] = []
        self.config_file.write_text(json.dumps(config), encoding='utf-8')
        before = self.config_file.read_bytes()
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(runtime, 'run_prepare_steps') as prepare:
            code = batch.main(['doctor', '--output', 'json', '--strict-exit-codes'])
        prepare.assert_not_called()
        self.assertEqual(code, cli_contract.EXIT_BLOCKED)
        report = json.loads(output.getvalue())['result']
        self.assertEqual(report['engine_status']['engine'], 'rpgmaker_mv')
        self.assertEqual(report['mode'], 'mv_setup_required')
        self.assertEqual(Path(report['base_dir']), self.target / 'www')
        self.assertEqual(report['mv_setup']['scope'], 'full_project')
        self.assertEqual(report['mv_setup']['missing'], ['source_root', 'support_window_file'])
        self.assertEqual(report['counts'], {})
        self.assertEqual(self.config_file.read_bytes(), before)

    def test_auto_detected_mv_refuses_provider_dispatch_until_materials_are_configured(self):
        config = json.loads(self.config_file.read_text(encoding='utf-8'))
        config.pop('engine')
        self.config_file.write_text(json.dumps(config), encoding='utf-8')
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(batch, 'run_durable_sync_command') as dispatch:
            code = batch.main(['sync-start', '--output', 'json', '--strict-exit-codes'])
        dispatch.assert_not_called()
        self.assertEqual(code, cli_contract.EXIT_BLOCKED)
        self.assertEqual(json.loads(output.getvalue())['error']['code'], 'MV_PROJECT_SETUP_REQUIRED')


class MvCoveragePreparationTests(unittest.TestCase):
    def test_review_states_keep_counts_and_provide_structured_next_steps(self):
        from engine_adapters.coverage import build_review_template, default_coverage_review_path
        from execution.mv_sync import build_snapshot
        from tests.support.mv_project import make_mv_project

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config_file, config, _ = make_mv_project(Path(directory))
            stack.enter_context(mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_file)))
            stack.enter_context(runtime.runtime_config_scope(
                reload_translator_settings=True, persist_corrected_game_root=False))
            snapshot = build_snapshot(config, config_file)
            review_file = Path(default_coverage_review_path(config['game_root']))
            template = build_review_template(snapshot.report)
            for state in ('missing', 'pending', 'stale', 'invalid'):
                with self.subTest(state=state):
                    if state == 'missing':
                        review_file.unlink()
                    elif state == 'invalid':
                        review_file.write_text('{', encoding='utf-8')
                    else:
                        record = dict(template)
                        if state == 'stale':
                            record['coverage_digest'] = 'previous-material'
                        review_file.write_text(json.dumps(record), encoding='utf-8')
                    report = batch.collect_doctor_report()
                    issue, = report['engine_status']['issues']
                    self.assertEqual(issue['code'], 'engine.mv.scope_review_required')
                    self.assertEqual(issue['details']['review_status'], state)
                    self.assertNotIn('Could not read', issue['message'])
                    self.assertEqual(report['pending_task_count'], 14)
                    self.assertEqual(report['workflow_state'], '')
                    self.assertEqual(report['engine_status']['status'], 'blocked')
                    if state == 'missing':
                        self.assertIn('export-dir', issue['suggested_action'])
                        self.assertIn('导出', issue['message'])
                        self.assertIn('No such file', issue['details']['review_error'])


if __name__ == '__main__':
    unittest.main()
