import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import translator_runtime as runtime


def _snapshot_prepare_settings():
    return {
        'PREP_ALLOW_SHELL_COMMANDS': runtime.PREP_ALLOW_SHELL_COMMANDS,
        'PREP_UNPACK_COMMAND': runtime.PREP_UNPACK_COMMAND,
        'PREP_TEMPLATE_COMMAND': runtime.PREP_TEMPLATE_COMMAND,
        'TL_SUBDIR': runtime.TL_SUBDIR,
        'TL_DIR': runtime.TL_DIR,
        'BASE_DIR': runtime.BASE_DIR,
        'WORK_GAME_DIR': runtime.WORK_GAME_DIR,
        'PREP_LANGUAGE': runtime.PREP_LANGUAGE,
    }


def _restore_prepare_settings(snapshot):
    for key, value in snapshot.items():
        setattr(runtime, key, value)


class CoercePrepareCommandTests(unittest.TestCase):
    def test_argv_list_is_preferred_and_accepted(self):
        cmd = runtime._coerce_command(
            ['python', 'unpack.py', '{archive}'],
            field_name='prepare.unpack_command',
            allow_shell=False,
        )
        self.assertEqual(cmd, ['python', 'unpack.py', '{archive}'])

    def test_string_command_rejected_without_shell_opt_in(self):
        with self.assertRaises(runtime.InvalidPrepareCommandError) as ctx:
            runtime._coerce_command(
                'python unpack.py {archive}',
                field_name='prepare.unpack_command',
                allow_shell=False,
            )
        self.assertIn('allow_shell_commands', str(ctx.exception))

    def test_string_command_allowed_with_shell_opt_in(self):
        cmd = runtime._coerce_command(
            'python unpack.py {archive}',
            field_name='prepare.unpack_command',
            allow_shell=True,
        )
        self.assertEqual(cmd, 'python unpack.py {archive}')

    def test_malformed_command_type_rejected(self):
        with self.assertRaises(runtime.InvalidPrepareCommandError):
            runtime._coerce_command(
                {'bin': 'python'},
                field_name='prepare.template_command',
                allow_shell=True,
            )

    def test_empty_string_is_treated_as_unset(self):
        self.assertIsNone(
            runtime._coerce_command('   ', field_name='prepare.unpack_command', allow_shell=False)
        )


class LoadPrepareCommandSettingsTests(unittest.TestCase):
    def test_load_accepts_argv_commands(self):
        snapshot = _snapshot_prepare_settings()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work_dir = workspace / 'work'
                work_dir.mkdir()
                config = {
                    'game_root': str(work_dir),
                    'prepare': {
                        'unpack_command': ['python', 'unpack.py', '{archive}'],
                        'template_command': ['{python_exe}', '{launcher_py}', '{base_dir}', 'translate', '{language}'],
                        'allow_shell_commands': False,
                    },
                }
                config_path = workspace / 'translator_config.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                with (
                    mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                    mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                    mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    runtime.load_translator_settings()
                self.assertFalse(runtime.PREP_ALLOW_SHELL_COMMANDS)
                self.assertEqual(runtime.PREP_UNPACK_COMMAND[0], 'python')
                self.assertIsInstance(runtime.PREP_TEMPLATE_COMMAND, list)
        finally:
            _restore_prepare_settings(snapshot)

    def test_load_rejects_shell_string_without_opt_in(self):
        snapshot = _snapshot_prepare_settings()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                work_dir = workspace / 'work'
                work_dir.mkdir()
                config = {
                    'game_root': str(work_dir),
                    'prepare': {
                        'unpack_command': 'python unpack.py',
                        'allow_shell_commands': False,
                    },
                }
                config_path = workspace / 'translator_config.json'
                config_path.write_text(json.dumps(config), encoding='utf-8')
                with (
                    mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                    mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                    mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
                    mock.patch.dict(os.environ, {}, clear=True),
                ):
                    with self.assertRaises(SystemExit) as ctx:
                        runtime.load_translator_settings()
                self.assertIn('Invalid prepare command', str(ctx.exception))
                self.assertIn('allow_shell_commands', str(ctx.exception))
        finally:
            _restore_prepare_settings(snapshot)


class RunPrepareCommandTests(unittest.TestCase):
    def test_run_prints_cwd_and_uses_argv_without_shell(self):
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(runtime, 'PREP_ALLOW_SHELL_COMMANDS', False),
            mock.patch.object(runtime.subprocess, 'run', return_value=completed) as run_mock,
            mock.patch('builtins.print') as print_mock,
        ):
            ok = runtime._run_prepare_command(
                ['python', 'tool.py'],
                cwd='/tmp/work',
                step_name='Custom RPA unpack',
            )
        self.assertTrue(ok)
        run_mock.assert_called_once()
        args, kwargs = run_mock.call_args
        self.assertEqual(args[0], ['python', 'tool.py'])
        self.assertEqual(kwargs['cwd'], '/tmp/work')
        self.assertFalse(kwargs['shell'])
        printed = ' '.join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn('cwd: /tmp/work', printed)
        self.assertIn('shell: False', printed)

    def test_run_refuses_shell_string_without_opt_in(self):
        with (
            mock.patch.object(runtime, 'PREP_ALLOW_SHELL_COMMANDS', False),
            mock.patch.object(runtime.subprocess, 'run') as run_mock,
            mock.patch('builtins.print'),
        ):
            ok = runtime._run_prepare_command('echo hi', cwd='/tmp/work', step_name='Custom')
        self.assertFalse(ok)
        run_mock.assert_not_called()

    def test_run_allows_shell_string_with_opt_in(self):
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(runtime, 'PREP_ALLOW_SHELL_COMMANDS', True),
            mock.patch.object(runtime.subprocess, 'run', return_value=completed) as run_mock,
            mock.patch('builtins.print'),
        ):
            ok = runtime._run_prepare_command('echo hi', cwd='/tmp/work', step_name='Custom')
        self.assertTrue(ok)
        _args, kwargs = run_mock.call_args
        self.assertTrue(kwargs['shell'])


class DoctorPrepareShellWarningTests(unittest.TestCase):
    def test_doctor_warns_when_shell_mode_enabled(self):
        with (
            mock.patch.object(batch_mod.legacy, 'BASE_DIR', 'C:/Games/Example/work'),
            mock.patch.object(batch_mod.legacy, 'TL_DIR', 'C:/Games/Example/work/game/tl/schinese'),
            mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/schinese'),
            mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'schinese'),
            mock.patch.object(batch_mod.legacy, 'PREP_ALLOW_SHELL_COMMANDS', True),
            mock.patch.object(batch_mod.legacy, 'PREP_UNPACK_COMMAND', 'python unpack.py'),
            mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', None),
            mock.patch.object(batch_mod.legacy, '_guess_source_game_dir', return_value=''),
            mock.patch.object(
                batch_mod.legacy,
                'get_prepare_template_command_info',
                return_value={'available': False, 'kind': 'auto', 'reason': 'missing', 'command': None, 'cwd': ''},
            ),
            mock.patch.object(batch_mod.legacy, 'resolve_original_game_dir', return_value=''),
            mock.patch.object(batch_mod.legacy, 'work_dir_bootstrap_allowed', return_value=(False, '', False)),
            mock.patch.object(batch_mod, 'collect_doctor_context_status', return_value={}),
            mock.patch.object(batch_mod, 'collect_doctor_project_assets_status', return_value={}),
            mock.patch.object(batch_mod, 'collect_doctor_layout_context', return_value={}),
            mock.patch.object(batch_mod, 'assess_doctor_layout_status', return_value='ok'),
            mock.patch.object(batch_mod, 'RAG_ENABLED', False),
            mock.patch('os.path.isdir', return_value=False),
            mock.patch('os.listdir', return_value=[]),
        ):
            report = batch_mod.collect_doctor_report()

        joined = ' '.join(report['warnings'])
        self.assertTrue(report['allow_shell_commands'])
        self.assertIn('prepare.unpack_command', report['shell_prepare_command_fields'])
        self.assertIn('HIGH RISK', joined)
        self.assertIn('allow_shell_commands', joined)


if __name__ == '__main__':
    unittest.main()
