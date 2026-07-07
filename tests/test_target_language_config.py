import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import translator_runtime as runtime
from gui_qt.batch_workflow_support import load_target_language_facts_from_manifest


class TargetLanguageConfigTests(unittest.TestCase):
    def test_load_translator_settings_resolves_japanese_tl_paths(self):
        fixture_path = Path(__file__).resolve().parent / 'fixtures' / 'translator_config_japanese.json'
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            work_dir = workspace / 'work'
            work_dir.mkdir()
            config = json.loads(fixture_path.read_text(encoding='utf-8'))
            config['game_root'] = str(work_dir)
            config_path = workspace / 'translator_config.json'
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')

            with (
                mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                runtime.load_translator_settings()

            self.assertEqual(runtime.TL_SUBDIR, 'game/tl/japanese')
            self.assertEqual(runtime.PREP_LANGUAGE, 'japanese')
            self.assertTrue(runtime.TL_DIR.replace('\\', '/').endswith('work/game/tl/japanese'))

    def test_manifest_target_language_fields_follow_runtime(self):
        with (
            mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/korean'),
            mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'korean'),
        ):
            fields = batch_mod._manifest_target_language_fields()
        self.assertEqual(fields['tl_subdir'], 'game/tl/korean')
        self.assertEqual(fields['target_language'], 'korean')

    def test_manifest_target_language_fields_inherit_from_parent_manifest(self):
        parent = {
            'tl_subdir': 'game/tl/japanese',
            'target_language': 'japanese',
        }
        with (
            mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/schinese'),
            mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'schinese'),
        ):
            fields = batch_mod._manifest_target_language_fields(parent)
        self.assertEqual(fields['tl_subdir'], 'game/tl/japanese')
        self.assertEqual(fields['target_language'], 'japanese')

    def test_collect_doctor_report_exposes_configured_language(self):
        with (
            mock.patch.object(batch_mod.legacy, 'BASE_DIR', 'C:/Games/Example/work'),
            mock.patch.object(batch_mod.legacy, 'TL_DIR', 'C:/Games/Example/work/game/tl/japanese'),
            mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/japanese'),
            mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'japanese'),
            mock.patch.object(batch_mod.legacy, '_guess_source_game_dir', return_value=''),
            mock.patch.object(batch_mod.legacy, 'get_prepare_template_command_info', return_value={'available': False}),
            mock.patch.object(batch_mod.legacy, 'resolve_original_game_dir', return_value=''),
            mock.patch.object(batch_mod.legacy, 'work_dir_bootstrap_allowed', return_value=(False, '', False)),
            mock.patch.object(batch_mod, 'collect_tl_doctor_counts', return_value={
                'rpy_files': 0,
                'translate_blocks': 0,
                'string_sections': 0,
                'old_lines': 0,
                'new_lines': 0,
                'commented_original_lines': 0,
            }),
            mock.patch.object(batch_mod, 'collect_doctor_context_status', return_value={}),
            mock.patch.object(batch_mod, 'collect_doctor_project_assets_status', return_value={}),
            mock.patch.object(batch_mod, 'collect_doctor_layout_context', return_value={}),
            mock.patch.object(batch_mod, 'assess_doctor_layout_status', return_value='ok'),
            mock.patch.object(batch_mod, 'collect_doctor_recommendations', return_value=[]),
            mock.patch.object(batch_mod, 'RAG_ENABLED', False),
            mock.patch('os.path.isdir', return_value=False),
            mock.patch('os.listdir', return_value=[]),
        ):
            report = batch_mod.collect_doctor_report()

        self.assertEqual(report['tl_subdir'], 'game/tl/japanese')
        self.assertEqual(report['language'], 'japanese')

    def test_print_doctor_report_includes_language_and_tl_subdir(self):
        report = {
            'base_dir': 'C:/work',
            'tl_dir': 'C:/work/game/tl/japanese',
            'tl_exists': True,
            'tl_subdir': 'game/tl/japanese',
            'language': 'japanese',
            'prepare_enabled': True,
            'generate_template': True,
            'refresh_existing_template': True,
            'renpy_sdk_dir': '',
            'launcher_py': '',
            'python_exe': '',
            'can_generate_template': False,
            'template_command_kind': '',
            'template_command': '',
            'template_reason': 'missing',
            'mode': 'existing_tl_only',
            'counts': {
                'rpy_files': 0,
                'translate_blocks': 0,
                'string_sections': 0,
                'old_lines': 0,
                'new_lines': 0,
                'commented_original_lines': 0,
            },
            'pending_task_count': 0,
            'pending_file_count': 0,
            'context_status': {'rag': {}, 'source_index': {}},
            'work_dir': '',
            'work_exists': False,
            'work_empty': True,
            'original_game_dir': '',
            'layout_status': '',
            'warnings': [],
            'recommendations': [],
        }
        stdout = io.StringIO()
        with mock.patch('sys.stdout', stdout):
            batch_mod.print_doctor_report(report)
        output = stdout.getvalue()
        self.assertIn('TL subdir: game/tl/japanese', output)
        self.assertIn('Language: japanese', output)

    def test_load_target_language_facts_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = os.path.join(tmp_dir, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'tl_subdir': 'game/tl/japanese',
                        'target_language': 'japanese',
                    },
                    handle,
                )
            facts = load_target_language_facts_from_manifest(manifest_path)
            self.assertIn('TL 路径：game/tl/japanese', facts)
            self.assertIn('目标语言：japanese', facts)


if __name__ == '__main__':
    unittest.main()