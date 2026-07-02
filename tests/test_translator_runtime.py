import ast
import hashlib
import importlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import prompt_context
import rag_memory
import story_memory
import translation_core
import translator_runtime as runtime


GOLDEN_BATCH_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'golden_batch_minimal'
GOLDEN_REVISION_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'golden_revision_minimal'
GOLDEN_KEYWORD_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'golden_keyword_minimal'
UPDATE_GOLDEN_BATCH_ENV = 'UPDATE_GOLDEN_BATCH'
UPDATE_GOLDEN_REVISION_ENV = 'UPDATE_GOLDEN_REVISION'
UPDATE_GOLDEN_KEYWORD_ENV = 'UPDATE_GOLDEN_KEYWORD'



class TranslatorRuntimeRegressionTests(unittest.TestCase):
    def test_batch_module_import_has_no_stdout_or_directory_side_effects(self):
        sentinel_stdout = io.StringIO()
        module_name = 'gemini_translate_batch'
        original_module = sys.modules.pop(module_name, None)
        try:
            with (
                mock.patch('sys.stdout', sentinel_stdout),
                mock.patch('os.makedirs') as makedirs_mock,
            ):
                imported = importlib.import_module(module_name)
                self.assertIs(sys.stdout, sentinel_stdout)
                self.assertEqual(imported.BATCH_MACRO_SETTING, '')

            makedirs_mock.assert_not_called()
        finally:
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                sys.modules.pop(module_name, None)

    def test_batch_cli_without_command_prints_help_without_submit(self):
        output = io.StringIO()

        with (
            mock.patch('sys.stdout', output),
            mock.patch.object(batch_mod, 'initialize_batch_logging') as logging_mock,
            mock.patch.object(batch_mod.legacy, 'load_config') as load_config_mock,
            mock.patch.object(batch_mod, 'submit_manifest') as submit_mock,
        ):
            batch_mod.main([])

        self.assertIn('usage:', output.getvalue())
        logging_mock.assert_not_called()
        load_config_mock.assert_not_called()
        submit_mock.assert_not_called()

    def test_batch_cli_help_does_not_load_runtime_settings(self):
        output = io.StringIO()

        with (
            mock.patch('sys.stdout', output),
            mock.patch.object(batch_mod, 'initialize_batch_logging') as logging_mock,
            mock.patch.object(batch_mod.legacy, 'load_config') as load_config_mock,
            mock.patch.object(batch_mod.legacy, 'load_translator_settings') as settings_mock,
            mock.patch.object(batch_mod.legacy, 'load_glossary') as glossary_mock,
            mock.patch.object(batch_mod, 'load_batch_settings') as batch_settings_mock,
        ):
            with self.assertRaises(SystemExit) as raised:
                batch_mod.main(['--help'])

        self.assertEqual(raised.exception.code, 0)
        self.assertIn('usage:', output.getvalue())
        logging_mock.assert_not_called()
        load_config_mock.assert_not_called()
        settings_mock.assert_not_called()
        glossary_mock.assert_not_called()
        batch_settings_mock.assert_not_called()

    def test_batch_system_instruction_has_readable_empty_macro_default(self):
        old_macro_setting = batch_mod.BATCH_MACRO_SETTING
        try:
            batch_mod.BATCH_MACRO_SETTING = ''
            instruction = batch_mod.build_system_instruction()
        finally:
            batch_mod.BATCH_MACRO_SETTING = old_macro_setting

        self.assertIn('Setting:', instruction)
        setting_block = instruction.split('Task:', 1)[0]
        self.assertNotIn('????', setting_block)

    def test_short_preserve_terms_allow_adjacent_chinese(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['Lou', 'Max', 'Mo']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'lou', 'max', 'mo'}),
        ):
            self.assertEqual(runtime.missing_preserved_terms('Lou laughs.', 'Lou笑了。'), [])
            self.assertEqual(runtime.missing_preserved_terms('Max cooks.', 'Max做饭。'), [])
            self.assertEqual(runtime.missing_preserved_terms('Moon rises.', '月亮升起。'), [])
            self.assertEqual(runtime.missing_preserved_terms('Lou laughs.', 'CloudLou笑了。'), ['Lou'])

    def test_short_preserve_terms_do_not_match_contractions(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['Don']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'don'}),
        ):
            self.assertEqual(runtime.missing_preserved_terms("Don't move.", '别动。'), [])
            self.assertEqual(runtime.missing_preserved_terms('Don moved.', '他动了。'), ['Don'])

    def test_preserve_terms_allow_renpy_field_translation_conversion(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['[Main]']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'[main]'}),
        ):
            self.assertEqual(
                runtime.missing_preserved_terms('[Main], are you listening?', '[Main!t]，你在听吗？'),
                [],
            )

    def test_preserve_terms_ignore_common_mark_my_words_idiom(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['Mark']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'mark'}),
        ):
            self.assertEqual(
                runtime.missing_preserved_terms('Mark my words, this will work.', '???????????'),
                [],
            )
            self.assertEqual(runtime.missing_preserved_terms('Mark said so.', '?????'), ['Mark'])

    def test_short_preserve_terms_allow_matching_renpy_name_field(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['Gil', 'Don']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'gil', 'don'}),
        ):
            self.assertEqual(runtime.missing_preserved_terms('coach Gil said so.', '[Gil_name!t] 教练这么说。'), [])
            self.assertEqual(runtime.missing_preserved_terms('Don said so.', '[Gil_name!t] 这么说。'), ['Don'])

    def test_preserve_terms_allow_known_aliases(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['H.U.']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'h.u.'}),
        ):
            self.assertEqual(runtime.missing_preserved_terms('I am in H.U. now.', '我现在在Highwell University。'), [])
            self.assertEqual(runtime.missing_preserved_terms('I am in H.U. now.', '我现在在学校。'), ['H.U.'])
    def test_non_chinese_term_translation_allows_repeated_preserved_phrase(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', ['Music Appreciation']),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'music appreciation'}),
        ):
            self.assertTrue(runtime.allow_non_chinese_term_translation(
                'Music Appreciation...Music Appreciation...',
                'Music Appreciation……Music Appreciation……',
            ))

    def test_non_translatable_accepts_ui_labels_formats_and_identifiers(self):
        self.assertTrue(runtime.is_non_translatable('Page Up'))
        self.assertTrue(runtime.is_non_translatable('+III'))
        self.assertTrue(runtime.is_non_translatable('%b %d, %H:%M'))
        self.assertTrue(runtime.is_non_translatable('AndersF_name'))
        self.assertFalse(runtime.is_non_translatable('I'))

    def test_translation_templates_skip_keyword_argument_strings(self):
        lines = [
            'translate schinese sample_block:\n',
            '\n',
            '    # "[Main]" "Hello there." (ctc="ctc_blink", ctc_position="nestled")\n',
            '    "[Main]" "Hello there." (ctc="ctc_blink", ctc_position="nestled")\n',
        ]

        tasks = runtime.collect_tasks(lines)
        task_texts = [task['text'] for task in tasks]
        mapping = runtime.scan_all_translation_units(lines, 'sample.rpy')
        scanned_texts = [value[3] for value in mapping.values()]

        self.assertIn('Hello there.', task_texts)
        self.assertNotIn('ctc_blink', task_texts)
        self.assertNotIn('nestled', task_texts)
        self.assertEqual(scanned_texts, ['Hello there.'])
        self.assertEqual(tasks[0]['block_index'], 1)
        self.assertTrue(all(':sample_block:1:' in identity for identity in mapping))

    def test_batch_non_chinese_allowance_accepts_say_speaker_label_item(self):
        line = '    "Terry" "Hello there." (ctc="ctc_blink", ctc_position="nestled")\n'
        start = line.index('"Terry"')
        end = start + len('"Terry"')
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / 'sample.rpy'
            sample_path.write_text(line, encoding='utf-8')
            item = {'start': start, 'end': end, 'line_number': 1}
            manifest = {'tl_dir': tmpdir}
            chunk = {'file_rel_path': 'sample.rpy', 'glossary_hits': [], 'history_hits': []}

            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                manifest,
                chunk,
                'Terry',
                'Terry',
                item=item,
            ))

    def test_batch_non_chinese_allowance_accepts_manifest_keyword_argument_item(self):
        line = '    "[Main]" "Hello there." (ctc="ctc_blink", ctc_position="nestled")\n'
        start = line.index('"ctc_blink"')
        end = start + len('"ctc_blink"')
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / 'sample.rpy'
            sample_path.write_text(line, encoding='utf-8')
            item = {'start': start, 'end': end, 'line_number': 1}
            manifest = {'tl_dir': tmpdir}
            chunk = {'file_rel_path': 'sample.rpy', 'glossary_hits': [], 'history_hits': []}

            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                manifest,
                chunk,
                'ctc_blink',
                'ctc_blink',
                item=item,
            ))
            self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
                manifest,
                chunk,
                'ctc_blink',
                'ctc_blink',
                item=None,
            ))

    def test_phrase_preserve_terms_allow_non_chinese_punctuation_change(self):
        with (
            mock.patch.object(runtime, 'PRESERVE_TERMS', [
                'Raven Three',
                'Mrs. de Bruin',
                'New Lutetia',
                'Pretty Bunny Angel Heart',
            ]),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {
                'raven three',
                'mrs. de bruin',
                'new lutetia',
                'pretty bunny angel heart',
            }),
        ):
            self.assertTrue(runtime.allow_non_chinese_term_translation('Raven Three!', 'Raven Three！'))
            self.assertTrue(runtime.allow_non_chinese_term_translation('Mrs. de Bruin?', 'Mrs. de Bruin？'))
            self.assertFalse(runtime.allow_non_chinese_term_translation('New Heart?', 'New Heart？'))

    def test_batch_request_includes_request_level_safety_settings(self):
        old_safety_settings = batch_mod.BATCH_SAFETY_SETTINGS
        try:
            batch_mod.BATCH_SAFETY_SETTINGS = [
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE',
                }
            ]
            request = batch_mod.build_batch_request({
                'key': 'chunk-1',
                'context_past': [],
                'context_future': [],
                'glossary_hits': [],
                'history_hits': [],
                'source_hits': [],
                'items': [{'id': 'item-1', 'text': 'Hello'}],
            })
        finally:
            batch_mod.BATCH_SAFETY_SETTINGS = old_safety_settings

        self.assertNotIn('safety_settings', request['request']['generation_config'])
        self.assertEqual(
            request['request']['safety_settings'],
            [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}],
        )

    def test_batch_safety_settings_normalizer_accepts_adult_preset(self):
        self.assertEqual(
            batch_mod.normalize_batch_safety_settings('relaxed_adult'),
            [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}],
        )
        self.assertEqual(
            batch_mod.normalize_batch_safety_settings([
                {'category': 'sexually_explicit', 'threshold': 'none'},
            ]),
            [{'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'}],
        )

    def test_batch_non_chinese_allowance_accepts_preserved_names_and_acronyms(self):
        with mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', {'gilly', 'dearmer'}):
            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                {},
                {'glossary_hits': [], 'history_hits': []},
                '"AR?"',
                '“AR？”',
            ))
            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                {},
                {'glossary_hits': [], 'history_hits': []},
                'G-gilly!?',
                'G-Gilly！？',
            ))
            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                {},
                {'glossary_hits': [], 'history_hits': []},
                'Dearmer',
                'Dearmer',
            ))
            self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
                {},
                {'glossary_hits': [], 'history_hits': []},
                '"Macroeconomics."',
                '"Macroeconomics。"',
            ))
    def test_batch_non_chinese_allowance_accepts_static_context_items(self):
        self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_patronlistitem.rpy'},
            'Alpha, Beta, Gamma',
            'Alpha, Beta, Gamma',
        ))
        self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_menu_about.rpy'},
            '{a=https://example.test}Dirk the Red Panda{/a}.',
            '{a=https://example.test}Dirk the Red Panda{/a}。',
        ))
        self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_menu_about.rpy'},
            'Avi, MJ, Sinta, Steven.',
            'Avi, MJ, Sinta, Steven。',
        ))
        self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_menu_about.rpy'},
            'Main Writer: Andy Peng',
            'Main Writer: Andy Peng',
        ))
        self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_charselect.rpy'},
            'Lars Dearmer',
            'Lars Dearmer',
        ))
        self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'screens_charselect.rpy'},
            'Birthday: April 25th',
            'Birthday: April 25th',
        ))
        self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'script_define.rpy'},
            'Theodore',
            'Theodore',
        ))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
            base_dir = Path(tmpdir)
            source_path = base_dir / 'game' / 'script.rpy'
            source_path.parent.mkdir(parents=True)
            source_path.write_text('if Main == _("Herbert"):\n    pass\n', encoding='utf-8')
            tl_dir = base_dir / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            tl_path = tl_dir / 'script.rpy'
            tl_path.write_text('# game/script.rpy:1\nold "Herbert"\nnew "Herbert"\n', encoding='utf-8')
            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                {'base_dir': str(base_dir), 'tl_dir': str(tl_dir)},
                {'file_rel_path': 'script.rpy'},
                'Herbert',
                'Herbert',
                item={'line_number': 3},
            ))
            self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
                {'base_dir': str(base_dir), 'tl_dir': str(tl_dir)},
                {'file_rel_path': 'script.rpy'},
                'No',
                'No',
                item={'line_number': 3},
            ))
        self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
            {},
            {'file_rel_path': 'script.rpy', 'glossary_hits': [], 'history_hits': []},
            'But who is this Campbell?',
            'But who is this Campbell?',
        ))

    def test_batch_non_chinese_allowance_uses_glossary_without_rag_enabled(self):
        chunk = {
            'glossary_hits': [{'source': 'Raven Three', 'target': 'Raven Three'}],
            'history_hits': [],
        }
        with (
            mock.patch.object(batch_mod, 'RAG_ENABLED', False),
            mock.patch.object(runtime, 'PRESERVE_TERMS', []),
            mock.patch.object(runtime, 'PRESERVE_TERMS_LOWER', set()),
            mock.patch.object(batch_mod, '_RAG_PRESERVED_TERMS_CACHE', None),
            mock.patch.object(batch_mod, '_RAG_PRESERVED_TERMS_CACHE_KEY', None),
        ):
            self.assertTrue(batch_mod.allow_non_chinese_batch_translation(
                {'rag_enabled': False},
                chunk,
                'Raven Three!',
                'Raven Three！',
            ))
            self.assertFalse(batch_mod.allow_non_chinese_batch_translation(
                {'rag_enabled': False},
                {'glossary_hits': [], 'history_hits': []},
                'Dawn Hound?',
                'Dawn Hound？',
            ))

    def test_collect_tasks_keeps_distinct_entries_on_same_line(self):
        tasks = runtime.collect_tasks(['call screen test("Hello", "World")\n'])
        self.assertEqual(len(tasks), 2)
        self.assertNotEqual(tasks[0]['start'], tasks[1]['start'])
        self.assertNotEqual(tasks[0]['progress_entry'], tasks[1]['progress_entry'])

    def test_collect_tasks_records_dialogue_speaker_id(self):
        tasks = runtime.collect_tasks([
            'e happy "Hello Noah"\n',
            'text "Start Game"\n',
        ])
        by_text = {task['text']: task for task in tasks}

        self.assertEqual(by_text['Hello Noah'].get('speaker_id'), 'e')
        self.assertEqual(by_text['Hello Noah'].get('speaker'), 'e')
        self.assertNotIn('speaker_id', by_text['Start Game'])

    def test_collect_tasks_uses_character_defines_for_speaker_name(self):
        tasks = runtime.collect_tasks([
            'define e = Character("Eileen")\n',
            'define n = Character(_("Noah"), color="#fff")\n',
            'e "Hello Noah"\n',
            'n "Hi Eileen"\n',
        ])
        by_text = {task['text']: task for task in tasks}

        self.assertNotIn('Eileen', by_text)
        self.assertNotIn('Noah', by_text)
        self.assertEqual(by_text['Hello Noah'].get('speaker_id'), 'e')
        self.assertEqual(by_text['Hello Noah'].get('speaker_name'), 'Eileen')
        self.assertEqual(by_text['Hi Eileen'].get('speaker_id'), 'n')
        self.assertEqual(by_text['Hi Eileen'].get('speaker_name'), 'Noah')

    def test_collect_tasks_keeps_character_kwargs_after_display_name(self):
        tasks = runtime.collect_tasks([
            'define e = Character("Eileen", what_prefix=" says ")\n',
            'e "Hello Noah"\n',
        ])
        by_text = {task['text']: task for task in tasks}

        self.assertNotIn('Eileen', by_text)
        self.assertIn(' says ', by_text)
        self.assertEqual(by_text['Hello Noah'].get('speaker_name'), 'Eileen')

    def test_collect_tasks_resolves_character_defines_in_file_order(self):
        tasks = runtime.collect_tasks([
            'define e = Character("Eileen")\n',
            'e "Before"\n',
            'define e = Character("Echo")\n',
            'e "After"\n',
        ])
        by_text = {task['text']: task for task in tasks}

        self.assertEqual(by_text['Before'].get('speaker_name'), 'Eileen')
        self.assertEqual(by_text['After'].get('speaker_name'), 'Echo')

    def test_collect_tasks_skips_multiline_character_display_name(self):
        tasks = runtime.collect_tasks([
            'define e = Character(\n',
            '    _("Eileen"),\n',
            '    what_prefix=" says ",\n',
            ')\n',
            'e "Hello Noah"\n',
        ])
        by_text = {task['text']: task for task in tasks}

        self.assertNotIn('Eileen', by_text)
        self.assertIn(' says ', by_text)
        self.assertEqual(by_text['Hello Noah'].get('speaker_name'), 'Eileen')

    def test_runtime_import_does_not_load_relation_analyzer_common(self):
        result = subprocess.run(
            [
                sys.executable,
                '-c',
                (
                    'import sys\n'
                    'import translator_runtime\n'
                    'print("COMMON_IMPORTED=%s" % ("relation_analyzer.common" in sys.modules))\n'
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn('COMMON_IMPORTED=False', result.stdout)

    def test_collect_tasks_allows_new_as_dialogue_speaker_id(self):
        tasks = runtime.collect_tasks(['new "Hello Noah"\n'])

        self.assertEqual(tasks[0].get('speaker_id'), 'new')

    def test_collect_tasks_skips_new_speaker_id_in_translation_templates(self):
        tasks = runtime.collect_tasks([
            'translate schinese start_123:\n',
            '    old "Hello Noah"\n',
            '    new "Hello Noah"\n',
        ])

        self.assertEqual(len(tasks), 1)
        self.assertNotIn('speaker_id', tasks[0])

    def test_infer_dialogue_speaker_skips_non_speaker_names(self):
        line = 'call e happy "Hello Noah"\n'
        self.assertEqual(
            runtime.infer_dialogue_speaker_id(line, line.index('"')),
            'e',
        )

    def test_infer_dialogue_speaker_uses_inline_dialogue_segment(self):
        line = 'if unlocked: e "Hello Noah"\n'
        self.assertEqual(
            runtime.infer_dialogue_speaker_id(line, line.index('"')),
            'e',
        )

    def test_infer_dialogue_speaker_skips_define_expressions(self):
        line = 'define e = Character("Eileen")\n'
        self.assertEqual(runtime.infer_dialogue_speaker_id(line, line.index('"')), '')

    def test_quote_with_round_trips_prefixed_literals(self):
        prefix, quote = runtime.parse_string_literal_format('u"Hello"')
        literal = runtime.quote_with('\u4f60\u597d', quote, prefix=prefix)
        self.assertEqual(ast.literal_eval(literal), '\u4f60\u597d')
        self.assertTrue(literal.startswith('u"'))

        prefix, quote = runtime.parse_string_literal_format('r"Hello"')
        literal = runtime.quote_with(r'A\B', quote, prefix=prefix)
        self.assertEqual(ast.literal_eval(literal), r'A\B')
        self.assertFalse(literal.startswith('r'))

        prefix, quote = runtime.parse_string_literal_format('"""Hello"""')
        literal = runtime.quote_with('\u4f60\u597d', quote, prefix=prefix)
        self.assertEqual(ast.literal_eval(literal), '\u4f60\u597d')

        literal = runtime.quote_with('a"b"', '"')
        self.assertEqual(ast.literal_eval(literal), 'a"b"')

        literal = runtime.quote_with('a\n"b"\n', '"')
        self.assertEqual(ast.literal_eval(literal), 'a\n"b"\n')

    def test_rpa_index_loads_primitive_pickle_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / 'archive.rpa'
            raw_index = {'game/script.rpy': [(123, 4, b'')]}
            payload = zlib.compress(pickle.dumps(raw_index, protocol=4))
            header = b'RPA-3.0 %016x %08x\n' % (34, 0)
            archive_path.write_bytes(header + payload)

            index = runtime._read_rpa_index(str(archive_path))

        self.assertEqual(index, raw_index)

    def test_rpa_index_rejects_pickle_globals_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / 'archive.rpa'
            marker_path = Path(tmp) / 'pickle-executed.txt'

            class Payload:
                def __reduce__(self):
                    return (os.system, (f'echo unsafe > "{marker_path}"',))

            payload = zlib.compress(pickle.dumps(Payload(), protocol=4))
            header = b'RPA-3.0 %016x %08x\n' % (34, 0)
            archive_path.write_bytes(header + payload)

            with self.assertRaises(pickle.UnpicklingError):
                runtime._read_rpa_index(str(archive_path))
            self.assertFalse(marker_path.exists())

    def test_prepare_launcher_does_not_guess_unrelated_python_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tools.py').write_text('print("maintenance helper")\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(root)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
            ):
                self.assertEqual(runtime._resolve_prepare_launcher(), '')

    def test_prepare_launcher_still_detects_renpy_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / 'Game.py'
            launcher.write_text('import renpy.bootstrap\n', encoding='utf-8')
            (root / 'tools.py').write_text('print("maintenance helper")\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(root)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
            ):
                self.assertEqual(runtime._resolve_prepare_launcher(), str(launcher))

    def test_prepare_launcher_uses_configured_renpy_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sdk = root / 'renpy-sdk'
            sdk.mkdir()
            launcher = sdk / 'renpy.py'
            launcher.write_text('import renpy.bootstrap\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(root / 'work')),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', str(sdk)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
            ):
                self.assertEqual(runtime._resolve_prepare_launcher(), str(launcher))

    def test_prepare_discovers_workspace_renpy_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            base = workspace / 'Game_Example' / 'work'
            sdk = workspace / 'renpy-8.5.2-sdk'
            sdk.mkdir(parents=True)
            (sdk / 'renpy.py').write_text('import renpy.bootstrap\n', encoding='utf-8')
            base.mkdir(parents=True)

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
            ):
                self.assertEqual(runtime._discover_renpy_sdk_dir(), str(sdk))

    def test_prepare_discovers_newest_sdk_by_parsed_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            base = workspace / 'Game_Example' / 'work'
            older = workspace / 'renpy-8.9-sdk'
            newer = workspace / 'renpy-8.10-sdk'
            for sdk in (older, newer):
                sdk.mkdir(parents=True)
                (sdk / 'renpy.py').write_text('import renpy.bootstrap\n', encoding='utf-8')
            base.mkdir(parents=True)

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
            ):
                self.assertEqual(runtime._discover_renpy_sdk_dir(), str(newer))

    def test_prepare_invalid_configured_sdk_falls_back_to_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            base = workspace / 'Game_Example' / 'work'
            sdk = workspace / 'renpy-8.5.2-sdk'
            sdk.mkdir(parents=True)
            (sdk / 'renpy.py').write_text('import renpy.bootstrap\n', encoding='utf-8')
            base.mkdir(parents=True)
            config_path = workspace / 'translator_config.json'
            config_path.write_text(
                json.dumps(
                    {
                        'game_root': str(base),
                        'prepare': {'renpy_sdk_dir': 'missing-sdk'},
                    }
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                mock.patch.object(runtime, 'ROOT_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                runtime.load_translator_settings()
                self.assertEqual(runtime.PREP_RENPY_SDK_DIR, str(sdk))

    def test_translator_config_sync_settings_override_sync_defaults(self):
        old_values = {
            'models': runtime.MODELS,
            'model_index': runtime.CURRENT_MODEL_INDEX,
            'max_items': runtime.MAX_ITEMS,
            'max_chars': runtime.MAX_CHARS,
            'max_output_tokens': runtime.SYNC_MAX_OUTPUT_TOKENS,
            'include_files': runtime.INCLUDE_FILES,
            'include_prefixes': runtime.INCLUDE_PREFIXES,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                base = workspace / 'work'
                base.mkdir()
                config_path = workspace / 'translator_config.json'
                config_path.write_text(
                    json.dumps({
                        'game_root': str(base),
                        'include_files': ['game/tl/schinese/script.rpy'],
                        'include_prefixes': ['game/tl/schinese/chapter1'],
                        'sync': {
                            'models': [' ', 'gemini-sync-test'],
                            'chunk_size': 7,
                            'max_source_chars': 1234,
                            'max_output_tokens': 5678,
                        },
                    }),
                    encoding='utf-8',
                )

                with (
                    mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    runtime.CURRENT_MODEL_INDEX = 3
                    runtime.MAX_ITEMS = 40
                    runtime.MAX_CHARS = 12000
                    runtime.SYNC_MAX_OUTPUT_TOKENS = 24576
                    runtime.INCLUDE_FILES = set()
                    runtime.INCLUDE_PREFIXES = set()
                    runtime.load_translator_settings()

                self.assertEqual(runtime.MODELS, ['gemini-sync-test'])
                self.assertEqual(runtime.CURRENT_MODEL_INDEX, 0)
                self.assertEqual(runtime.get_current_model(), 'gemini-sync-test')
                self.assertEqual(runtime.MAX_ITEMS, 7)
                self.assertEqual(runtime.MAX_CHARS, 1234)
                self.assertEqual(runtime.SYNC_MAX_OUTPUT_TOKENS, 5678)
                self.assertEqual(runtime.INCLUDE_FILES, {'game/tl/schinese/script.rpy'})
                self.assertEqual(runtime.INCLUDE_PREFIXES, {'game/tl/schinese/chapter1'})
        finally:
            runtime.MODELS = old_values['models']
            runtime.CURRENT_MODEL_INDEX = old_values['model_index']
            runtime.MAX_ITEMS = old_values['max_items']
            runtime.MAX_CHARS = old_values['max_chars']
            runtime.SYNC_MAX_OUTPUT_TOKENS = old_values['max_output_tokens']
            runtime.INCLUDE_FILES = old_values['include_files']
            runtime.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_legacy_api_config_still_sets_sync_chunk_defaults(self):
        old_values = {
            'api_keys': runtime.API_KEYS,
            'models': runtime.MODELS,
            'model_index': runtime.CURRENT_MODEL_INDEX,
            'max_items': runtime.MAX_ITEMS,
            'max_chars': runtime.MAX_CHARS,
            'max_output_tokens': runtime.SYNC_MAX_OUTPUT_TOKENS,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / 'api_keys.json'
                config_path.write_text(
                    json.dumps({
                        'api_keys': ['test-key'],
                        'models': [' ', 'gemini-legacy-test'],
                        'batch_size': 9,
                        'max_chars': 2222,
                        'sync_max_output_tokens': 3333,
                    }),
                    encoding='utf-8',
                )

                with (
                    mock.patch.object(runtime, 'CONFIG_FILE', str(config_path)),
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    runtime.API_KEYS = []
                    runtime.CURRENT_MODEL_INDEX = 2
                    runtime.MAX_ITEMS = 40
                    runtime.MAX_CHARS = 12000
                    runtime.SYNC_MAX_OUTPUT_TOKENS = 24576
                    runtime.load_config()

                self.assertEqual(runtime.MODELS, ['gemini-legacy-test'])
                self.assertEqual(runtime.CURRENT_MODEL_INDEX, 0)
                self.assertEqual(runtime.get_current_model(), 'gemini-legacy-test')
                self.assertEqual(runtime.MAX_ITEMS, 9)
                self.assertEqual(runtime.MAX_CHARS, 2222)
                self.assertEqual(runtime.SYNC_MAX_OUTPUT_TOKENS, 3333)
        finally:
            runtime.API_KEYS = old_values['api_keys']
            runtime.MODELS = old_values['models']
            runtime.CURRENT_MODEL_INDEX = old_values['model_index']
            runtime.MAX_ITEMS = old_values['max_items']
            runtime.MAX_CHARS = old_values['max_chars']
            runtime.SYNC_MAX_OUTPUT_TOKENS = old_values['max_output_tokens']

    def test_translator_config_empty_include_lists_clear_existing_filters(self):
        old_values = {
            'include_files': runtime.INCLUDE_FILES,
            'include_prefixes': runtime.INCLUDE_PREFIXES,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp)
                base = workspace / 'work'
                base.mkdir()
                config_path = workspace / 'translator_config.json'
                config_path.write_text(
                    json.dumps({
                        'game_root': str(base),
                        'include_files': [],
                        'include_prefixes': [],
                    }),
                    encoding='utf-8',
                )

                with (
                    mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                    mock.patch.dict(os.environ, {}, clear=True),
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    runtime.INCLUDE_FILES = {'game/tl/schinese/old.rpy'}
                    runtime.INCLUDE_PREFIXES = {'game/tl/schinese/old'}
                    runtime.load_translator_settings()

                self.assertEqual(runtime.INCLUDE_FILES, set())
                self.assertEqual(runtime.INCLUDE_PREFIXES, set())
        finally:
            runtime.INCLUDE_FILES = old_values['include_files']
            runtime.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_legacy_api_config_empty_include_lists_clear_existing_filters(self):
        old_values = {
            'api_keys': runtime.API_KEYS,
            'include_files': runtime.INCLUDE_FILES,
            'include_prefixes': runtime.INCLUDE_PREFIXES,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / 'api_keys.json'
                config_path.write_text(
                    json.dumps({
                        'api_keys': ['test-key'],
                        'include_files': [],
                        'include_prefixes': [],
                    }),
                    encoding='utf-8',
                )

                with (
                    mock.patch.object(runtime, 'CONFIG_FILE', str(config_path)),
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    runtime.API_KEYS = []
                    runtime.INCLUDE_FILES = {'game/tl/schinese/old.rpy'}
                    runtime.INCLUDE_PREFIXES = {'game/tl/schinese/old'}
                    runtime.load_config()

                self.assertEqual(runtime.INCLUDE_FILES, set())
                self.assertEqual(runtime.INCLUDE_PREFIXES, set())
        finally:
            runtime.API_KEYS = old_values['api_keys']
            runtime.INCLUDE_FILES = old_values['include_files']
            runtime.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_prepare_does_not_discover_sdk_when_base_dir_is_filesystem_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            sdk = workspace / 'renpy-8.5.2-sdk'
            sdk.mkdir()
            (sdk / 'renpy.py').write_text('import renpy.bootstrap\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', os.path.abspath(os.sep)),
                mock.patch.object(runtime, 'ROOT_DIR', str(workspace)),
                mock.patch.object(runtime, 'TOOL_DIR', str(workspace / 'renpy-translation-lab')),
            ):
                self.assertEqual(runtime._discover_renpy_sdk_dir(), '')

    def test_prepare_template_command_uses_sdk_with_project_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            sdk = root / 'renpy-sdk'
            python_dir = sdk / 'lib' / 'py3-windows-x86_64'
            python_dir.mkdir(parents=True)
            python_exe = python_dir / 'python.exe'
            python_exe.write_text('', encoding='utf-8')
            launcher = sdk / 'renpy.py'
            launcher.write_text('import renpy.bootstrap\n', encoding='utf-8')

            with (
                mock.patch.object(runtime.sys, 'platform', 'win32'),
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', str(sdk)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_PYTHON_EXE', ''),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch.object(runtime, 'PREP_LANGUAGE', 'schinese'),
            ):
                info = runtime.get_prepare_template_command_info(str(base / 'game'))

            self.assertTrue(info['available'])
            self.assertEqual(info['kind'], 'sdk')
            self.assertEqual(info['command'], [str(python_exe), str(launcher), str(base), 'translate', 'schinese'])

    def test_prepare_template_command_uses_renpy_sh_on_non_windows_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            sdk = root / 'renpy-sdk'
            windows_python = sdk / 'lib' / 'py3-windows-x86_64' / 'python.exe'
            windows_python.parent.mkdir(parents=True)
            windows_python.write_text('', encoding='utf-8')
            launcher = sdk / 'renpy.py'
            launcher.write_text('import renpy.bootstrap\n', encoding='utf-8')
            shell_launcher = sdk / 'renpy.sh'
            shell_launcher.write_text('#!/bin/sh\n', encoding='utf-8')

            with (
                mock.patch.object(runtime.sys, 'platform', 'linux'),
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', str(sdk)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_PYTHON_EXE', ''),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch.object(runtime, 'PREP_LANGUAGE', 'schinese'),
            ):
                info = runtime.get_prepare_template_command_info(str(base / 'game'))

            self.assertTrue(info['available'])
            self.assertEqual(info['kind'], 'sdk')
            self.assertEqual(info['command'], [str(shell_launcher), str(base), 'translate', 'schinese'])
            self.assertEqual(info['cwd'], str(sdk))
            self.assertEqual(info['python_exe'], '')

    def test_find_bundled_python_ignores_windows_python_on_linux(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = Path(tmp) / 'renpy-sdk'
            windows_python = sdk / 'lib' / 'py3-windows-x86_64' / 'python.exe'
            windows_python.parent.mkdir(parents=True)
            windows_python.write_text('', encoding='utf-8')

            with mock.patch.object(runtime.sys, 'platform', 'linux'):
                self.assertEqual(runtime._find_bundled_python(str(sdk)), '')

    def test_prepare_existing_tl_without_launcher_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'script.rpy').write_text('translate schinese strings:\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(tl_dir)),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch.object(runtime, '_run_prepare_command') as run_mock,
                mock.patch('sys.stdout', io.StringIO()),
            ):
                runtime.run_prepare_steps()

            run_mock.assert_not_called()

    def test_prepare_tl_files_do_not_skip_rpa_unpack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            source_game = root / 'original' / 'game'
            work_tl = base / 'game' / 'tl' / 'schinese'
            work_tl.mkdir(parents=True)
            (work_tl / 'script.rpy').write_text('translate schinese strings:\n', encoding='utf-8')
            source_game.mkdir(parents=True)
            archive = source_game / 'scripts.rpa'
            archive.write_bytes(b'RPA-3.0 placeholder')

            base_dir = runtime.canonical_abs_path(base)
            work_game_dir = runtime.canonical_abs_path(base / 'game')
            with (
                mock.patch.object(runtime, 'BASE_DIR', base_dir),
                mock.patch.object(runtime, 'WORK_GAME_DIR', work_game_dir),
                mock.patch.object(runtime, 'TL_DIR', runtime.canonical_abs_path(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', True),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', False),
                mock.patch.object(runtime, '_extract_rpa_scripts', return_value=2) as extract_mock,
                mock.patch('sys.stdout', io.StringIO()),
            ):
                runtime.run_prepare_steps()

            self.assertTrue((base / 'game' / 'tl' / 'schinese' / 'script.rpy').is_file())
            extract_mock.assert_called_once_with(
                runtime.canonical_abs_path(archive),
                work_game_dir,
            )

    def test_prepare_missing_tl_without_launcher_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch('sys.stdout', io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    runtime.run_prepare_steps()

    def test_prepare_missing_tl_custom_template_command_error_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', ['{missing_placeholder}']),
                mock.patch('sys.stdout', io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    runtime.run_prepare_steps()

            self.assertIn('Custom template command error', str(raised.exception))
            self.assertIn('missing_placeholder', str(raised.exception))

    def test_prepare_first_time_template_failure_blocks_even_with_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'

            def fail_after_partial_file(*_args):
                tl_dir.mkdir(parents=True)
                (tl_dir / 'partial.rpy').write_text('translate schinese strings:\n', encoding='utf-8')
                return False

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(tl_dir)),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', ['generate-template']),
                mock.patch.object(runtime, '_run_prepare_command', side_effect=fail_after_partial_file),
                mock.patch('sys.stdout', io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    runtime.run_prepare_steps()

            self.assertIn('Translation template generation failed', str(raised.exception))

    def test_prepare_existing_tl_refresh_runs_template_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'script.rpy').write_text('translate schinese strings:\n', encoding='utf-8')
            sdk = root / 'renpy-sdk'
            sdk.mkdir()
            launcher = sdk / 'renpy.py'
            launcher.write_text('import renpy.bootstrap\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(tl_dir)),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', str(sdk)),
                mock.patch.object(runtime, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(runtime, 'PREP_PYTHON_EXE', sys.executable),
                mock.patch.object(runtime, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch.object(runtime, '_run_prepare_command', return_value=True) as run_mock,
                mock.patch('sys.stdout', io.StringIO()),
            ):
                runtime.run_prepare_steps()

            run_mock.assert_called_once()
            command = run_mock.call_args.args[0]
            self.assertEqual(command, [sys.executable, str(launcher), str(base), 'translate', 'schinese'])

    def test_prepare_existing_tl_refresh_disabled_skips_template_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'script.rpy').write_text('translate schinese strings:\n', encoding='utf-8')
            sdk = root / 'renpy-sdk'
            sdk.mkdir()
            (sdk / 'renpy.py').write_text('import renpy.bootstrap\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'BASE_DIR', str(base)),
                mock.patch.object(runtime, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(runtime, 'TL_DIR', str(tl_dir)),
                mock.patch.object(runtime, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(runtime, 'PREP_ENABLED', True),
                mock.patch.object(runtime, 'PREP_UNPACK_RPA', False),
                mock.patch.object(runtime, 'PREP_GENERATE_TEMPLATE', True),
                mock.patch.object(runtime, 'PREP_REFRESH_EXISTING_TEMPLATE', False),
                mock.patch.object(runtime, 'PREP_RENPY_SDK_DIR', str(sdk)),
                mock.patch.object(runtime, '_run_prepare_command') as run_mock,
                mock.patch('sys.stdout', io.StringIO()),
            ):
                runtime.run_prepare_steps()

            run_mock.assert_not_called()

    def test_process_batch_returns_only_successful_progress_entries(self):
        batch = [
            {
                'id': 'file:0:1',
                'text': 'Hello',
                'line': 0,
                'start': 1,
                'end': 8,
                'prefix': '',
                'quote': '"',
                'progress_entry': 'task:0:1',
            },
            {
                'id': 'file:0:10',
                'text': 'World',
                'line': 0,
                'start': 10,
                'end': 17,
                'prefix': '',
                'quote': '"',
                'progress_entry': 'task:0:10',
            },
        ]
        replacements = {}
        with mock.patch.object(runtime, 'call_gemini_sdk', return_value=[
            {'id': 'file:0:1', 'translation': '\u4f60\u597d'},
            {'id': 'file:0:10', 'translation': 'World'},
        ]):
            successful = runtime.process_batch(batch, replacements)

        self.assertEqual(successful, ['task:0:1'])
        self.assertEqual(replacements, {0: [(1, 8, '\u4f60\u597d', '', '"')]})

    def test_process_batch_stores_normalized_text_for_sync_rag(self):
        old_normalize_map = runtime.NORMALIZE_TRANSLATION_MAP
        old_use_memory = runtime.USE_TRANSLATION_MEMORY
        batch = [
            {
                'id': 'file:0:1',
                'text': 'Hello',
                'line': 0,
                'start': 1,
                'end': 8,
                'prefix': '',
                'quote': '"',
                'progress_entry': 'task:0:1',
            },
        ]
        try:
            runtime.NORMALIZE_TRANSLATION_MAP = {'\u65e7\u79f0': '\u65b0\u79f0'}
            runtime.USE_TRANSLATION_MEMORY = True
            replacements = {}
            with mock.patch.object(runtime, 'call_gemini_sdk', return_value=[
                {'id': 'file:0:1', 'translation': '\u65e7\u79f0\u4f60\u597d'},
            ]):
                runtime.process_batch(batch, replacements)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'script.rpy'
                lines = ['x"Hello"\n']
                runtime.commit_replacements(str(path), lines, replacements)
                committed_text = path.read_text(encoding='utf-8')
        finally:
            runtime.NORMALIZE_TRANSLATION_MAP = old_normalize_map
            runtime.USE_TRANSLATION_MEMORY = old_use_memory

        self.assertEqual(batch[0]['translated_text'], '\u65b0\u79f0\u4f60\u597d')
        self.assertEqual(replacements[0][0], (1, 8, '\u65e7\u79f0\u4f60\u597d', '', '"'))
        self.assertEqual(committed_text, 'x"\u65b0\u79f0\u4f60\u597d"\n')

    def test_sync_prompt_preserves_legacy_wrapper(self):
        prompt = runtime.build_prompt(
            [{'id': 'file:0:1', 'text': 'Hello Alice'}],
        )

        self.assertIn("You are translating a Ren'Py visual novel", prompt)
        self.assertIn('1.1 Keep all person names in English; do not translate names.', prompt)
        self.assertIn('Input JSON:', prompt)
        self.assertNotIn('CONTEXT BEFORE:', prompt)

    def test_sync_rag_prompt_includes_retrieved_memory_when_enabled(self):
        old_enabled = runtime.SYNC_RAG_ENABLED
        try:
            runtime.SYNC_RAG_ENABLED = True
            prompt = runtime.build_prompt(
                [{'id': 'file:0:1', 'text': 'Hello Alice'}],
                glossary_hits=[{'source': 'Alice', 'target': 'Alice'}],
                history_hits=[
                    {
                        'file_rel_path': 'script.rpy',
                        'line_start': 2,
                        'line_end': 2,
                        'translated_text': '\u4f60\u597d\uff0cAlice',
                        'quality_state': 'sync_applied',
                        'score': 0.91,
                    }
                ],
            )
        finally:
            runtime.SYNC_RAG_ENABLED = old_enabled

        self.assertIn('LOCKED TERMS', prompt)
        self.assertIn('RETRIEVED MEMORY', prompt)
        self.assertIn('\u4f60\u597d\uff0cAlice', prompt)

    def test_story_memory_prompt_blocks_are_optional(self):
        old_sync_rag_enabled = runtime.SYNC_RAG_ENABLED
        old_sync_story_enabled = runtime.SYNC_STORY_MEMORY_ENABLED
        old_batch_limit = batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS
        try:
            runtime.SYNC_RAG_ENABLED = False
            runtime.SYNC_STORY_MEMORY_ENABLED = False
            batch_prompt = batch_mod.build_user_prompt(
                [],
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
                [],
            )
            sync_prompt = runtime.build_prompt(
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
            )
            batch_empty_story_prompt = batch_mod.build_user_prompt(
                [],
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
                [],
                story_hits={'characters': [], 'relations': [], 'terms': [], 'scenes': []},
            )
            sync_empty_story_prompt = runtime.build_prompt(
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
                story_hits={'characters': [], 'relations': [], 'terms': [], 'scenes': []},
            )

            self.assertNotIn('STORY MEMORY', batch_prompt)
            self.assertNotIn('STORY MEMORY', sync_prompt)
            self.assertNotIn('STORY MEMORY', batch_empty_story_prompt)
            self.assertNotIn('STORY MEMORY', sync_empty_story_prompt)

            batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = 120
            prompt_with_story = batch_mod.build_user_prompt(
                [],
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
                [],
                story_hits={
                    'terms': [
                        {
                            'source': 'Void Gate',
                            'target': '\u865a\u7a7a\u95e8',
                            'note': '\u4e16\u754c\u89c2\u6838\u5fc3\u672f\u8bed',
                        },
                    ],
                },
            )
        finally:
            runtime.SYNC_RAG_ENABLED = old_sync_rag_enabled
            runtime.SYNC_STORY_MEMORY_ENABLED = old_sync_story_enabled
            batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = old_batch_limit

        self.assertIn('STORY MEMORY', prompt_with_story)
        self.assertIn('Void Gate -> \u865a\u7a7a\u95e8', prompt_with_story)

        term_only_block = story_memory.format_story_hits_block(
            {'terms': [{'source': 'Aether', 'target': '', 'note': 'Proper noun'}]},
            200,
        )
        self.assertIn('Term: Aether', term_only_block)
        self.assertNotIn('Keep unchanged', term_only_block)

    def test_sync_and_batch_reference_blocks_share_context_formatter(self):
        story_hits = {
            'terms': [
                {
                    'source': 'Void Gate',
                    'target': '\u865a\u7a7a\u95e8',
                    'note': '\u4e16\u754c\u89c2\u6838\u5fc3\u672f\u8bed',
                },
            ],
        }
        glossary_hits = [{'source': 'Alice', 'target': 'Alice'}]
        history_hits = [
            {
                'file_rel_path': 'script.rpy',
                'line_start': 2,
                'line_end': 2,
                'source_text': 'Open the Void Gate',
                'translated_text': '\u6253\u5f00\u865a\u7a7a\u95e8',
                'quality_state': 'sync_applied',
                'score': 0.91,
            }
        ]

        shared_block = prompt_context.build_reference_blocks(
            glossary_hits=glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
            history_char_limit=120,
            story_char_limit=160,
        )
        batch_prompt = batch_mod.build_user_prompt(
            [],
            [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
            [],
            glossary_hits=glossary_hits,
            history_hits=history_hits,
            story_hits=story_hits,
        )

        old_sync_rag_enabled = runtime.SYNC_RAG_ENABLED
        try:
            runtime.SYNC_RAG_ENABLED = True
            sync_prompt = runtime.build_prompt(
                [{'id': 'chapter1.rpy:0:1', 'text': 'Open the Void Gate'}],
                glossary_hits=glossary_hits,
                history_hits=history_hits,
                story_hits=story_hits,
            )
        finally:
            runtime.SYNC_RAG_ENABLED = old_sync_rag_enabled

        for marker in ('LOCKED TERMS', 'RETRIEVED MEMORY', 'STORY MEMORY'):
            self.assertIn(marker, shared_block)
            self.assertIn(marker, batch_prompt)
            self.assertIn(marker, sync_prompt)
        self.assertIn('- Keep unchanged: Alice', shared_block)
        self.assertIn('Void Gate -> \u865a\u7a7a\u95e8', batch_prompt)
        self.assertIn('\u6253\u5f00\u865a\u7a7a\u95e8', sync_prompt)

    def test_story_memory_hit_count_ignores_empty_categories(self):
        self.assertFalse(story_memory.has_story_hits({}))
        self.assertFalse(
            story_memory.has_story_hits(
                {'characters': [], 'relations': [], 'terms': [], 'scenes': []}
            )
        )
        self.assertTrue(story_memory.has_story_hits({'terms': [{'source': 'Void Gate'}]}))

    def test_story_memory_hit_counts_reports_categories(self):
        counts = story_memory.story_hit_counts(
            {
                'characters': [{'id': 'eileen'}],
                'relations': [{'left': 'eileen', 'right': 'noah'}],
                'terms': [{'source': 'Void Gate'}, {'source': 'Aether'}],
                'scenes': [{'file_rel_path': 'chapter1.rpy'}],
            }
        )

        self.assertEqual(
            counts,
            {
                'characters': 1,
                'relations': 1,
                'terms': 2,
                'scenes': 1,
            },
        )
        self.assertEqual(
            story_memory.story_hit_counts(None),
            {
                'characters': 0,
                'relations': 0,
                'terms': 0,
                'scenes': 0,
            },
        )

    def test_story_memory_normalize_has_fast_path_for_normalized_graphs(self):
        normalized = story_memory.normalize_story_graph({'terms': {'Void Gate': '\u865a\u7a7a\u95e8'}})

        self.assertIs(story_memory.normalize_story_graph(normalized), normalized)
        self.assertIn('Void Gate', normalized['terms'][0]['source'])
        self.assertNotIn('_story_memory_normalized', json.dumps(normalized))

    def test_story_graph_example_passes_lightweight_validation(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / 'docs' / 'story_graph.schema.json'
        example_path = repo_root / 'docs' / 'story_graph.example.json'

        schema = json.loads(schema_path.read_text(encoding='utf-8'))
        raw_graph = json.loads(example_path.read_text(encoding='utf-8'))

        self.assertEqual(
            schema['properties']['schema_version']['const'],
            story_memory.STORY_GRAPH_SCHEMA_VERSION,
        )
        self.assertEqual(story_memory.validate_story_graph(raw_graph), [])

        hits = story_memory.retrieve_story_hits(
            raw_graph,
            'chapter1.rpy',
            [
                {
                    'id': 'chapter1.rpy:129:0',
                    'text': 'Noah opens the Void Gate.',
                    'line': 129,
                    'speaker_id': 'e',
                },
            ],
        )

        self.assertTrue(story_memory.has_story_hits(hits))
        self.assertEqual({item['id'] for item in hits['characters']}, {'eileen', 'noah'})
        self.assertEqual(hits['relations'][0]['type'], 'close_friend')
        self.assertEqual(hits['terms'][0]['source'], 'Void Gate')
        self.assertEqual(hits['scenes'][0]['file_rel_path'], 'chapter1.rpy')

    def test_validate_story_graph_reports_non_fatal_schema_warnings(self):
        warnings = story_memory.validate_story_graph(
            {
                'schema_version': 2,
                'characters': [
                    'bad-character',
                    {'id': '', 'speaker_ids': {'bad': 'shape'}},
                    {'id': 'eileen', 'aliases': [{}]},
                    {'id': 'numeric-alias', 'speaker_ids': 1, 'aliases': ['ok', 2]},
                ],
                'relations': [
                    {'left': 'eileen', 'confidence': 'very'},
                    {'left': 'eileen', 'right': 'noah', 'confidence': 'nan'},
                    {'left': {'bad': 'shape'}, 'right': ['bad']},
                    'bad-relation',
                ],
                'terms': [
                    {'note': 'missing source'},
                    {'source': {'bad': 'shape'}},
                    {'target': {'bad': 'shape'}},
                    42,
                ],
                'scenes': [
                    {'line_start': 'ten', 'characters': {'bad': 'shape'}},
                    {'file_rel_path': {'bad': 'shape'}, 'summary': 'invalid path'},
                    {'file_rel_path': 'chapter1.rpy', 'line_start': 0, 'line_end': -1},
                ],
            }
        )

        self.assertTrue(any('schema_version' in warning for warning in warnings))
        self.assertTrue(any('characters[0]' in warning for warning in warnings))
        self.assertTrue(any('characters[1] is missing a usable id' in warning for warning in warnings))
        self.assertTrue(any('characters[3].speaker_ids' in warning for warning in warnings))
        self.assertTrue(any('characters[3].aliases[1]' in warning for warning in warnings))
        self.assertTrue(any('relations[0].right' in warning for warning in warnings))
        self.assertTrue(any('relations[1].confidence' in warning for warning in warnings))
        self.assertTrue(any('relations[2].left' in warning for warning in warnings))
        self.assertTrue(any('relations[2].right' in warning for warning in warnings))
        self.assertTrue(any('terms[0]' in warning for warning in warnings))
        self.assertTrue(any('terms[1].source' in warning for warning in warnings))
        self.assertTrue(any('terms[2].target' in warning for warning in warnings))
        self.assertTrue(any('scenes[0].line_start' in warning for warning in warnings))
        self.assertTrue(any('scenes[0].file_rel_path' in warning for warning in warnings))
        self.assertTrue(any('scenes[1].file_rel_path' in warning for warning in warnings))
        self.assertTrue(any('scenes[2].line_start should be >= 1' in warning for warning in warnings))
        self.assertTrue(any('scenes[2].line_end should be >= 1' in warning for warning in warnings))

    def test_validate_story_graph_accepts_legacy_term_shapes(self):
        self.assertEqual(
            story_memory.validate_story_graph(
                {
                    'schema_version': story_memory.STORY_GRAPH_SCHEMA_VERSION,
                    'terms': {
                        'Void Gate': '\u865a\u7a7a\u95e8',
                    },
                }
            ),
            [],
        )
        self.assertEqual(
            story_memory.validate_story_graph(
                {
                    'schema_version': story_memory.STORY_GRAPH_SCHEMA_VERSION,
                    'terms': [
                        {'term': 'Aether'},
                        {'translation': '\u4ee5\u592a\u95e8'},
                    ],
                }
            ),
            [],
        )

    def test_validate_story_graph_warns_when_schema_version_missing(self):
        warnings = story_memory.validate_story_graph({'terms': {'Void Gate': '\u865a\u7a7a\u95e8'}})

        self.assertTrue(any('schema_version is required' in warning for warning in warnings))

    def test_story_memory_rel_path_preserves_parent_segments(self):
        self.assertEqual(story_memory._normalize_rel_path('./chapter1.rpy'), 'chapter1.rpy')
        self.assertEqual(story_memory._normalize_rel_path('../chapter1.rpy'), '../chapter1.rpy')

    def test_story_memory_uses_speaker_id_case_insensitively(self):
        graph = {
            'characters': {
                'eileen': {
                    'speaker_ids': ['e'],
                    'zh_name': '\u827e\u7433',
                    'style': '\u8bed\u6c14\u8f7b\u5feb',
                },
                'noah': {
                    'speaker_ids': ['n'],
                    'zh_name': '\u8bfa\u4e9a',
                },
            },
            'relations': [
                {
                    'left': 'eileen',
                    'right': 'noah',
                    'type': 'close_friend',
                    'confidence': 0.85,
                },
            ],
        }
        hits = story_memory.retrieve_story_hits(
            graph,
            'chapter1.rpy',
            [{'id': 'chapter1.rpy:10:4', 'text': 'We should go.', 'speaker_id': 'E'}],
        )

        self.assertEqual([item['id'] for item in hits['characters']], ['eileen'])
        self.assertEqual(hits['relations'], [])

    def test_load_story_graph_warns_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_file = Path(tmp) / 'story_graph.json'
            graph_file.write_text('{bad json', encoding='utf-8')
            with mock.patch('builtins.print') as print_mock:
                graph = story_memory.load_story_graph(str(graph_file))

        self.assertEqual(graph, {'characters': {}, 'relations': [], 'terms': [], 'scenes': []})
        self.assertTrue(print_mock.called)
        self.assertIn('Failed to load story graph', print_mock.call_args[0][0])

    def test_load_story_graph_warns_on_schema_issues_but_keeps_valid_entries(self):
        graph = {
            'characters': {
                'eileen': {
                    'speaker_ids': 'e',
                    'zh_name': '\u827e\u7433',
                },
            },
            'relations': [
                {'left': 'eileen'},
            ],
            'terms': [
                {'source': 'Void Gate', 'target': '\u865a\u7a7a\u95e8'},
            ],
            'scenes': [
                {'file_rel_path': 'chapter1.rpy', 'line_start': 'bad'},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            graph_file = Path(tmp) / 'story_graph.json'
            graph_file.write_text(json.dumps(graph), encoding='utf-8')
            with mock.patch('builtins.print') as print_mock:
                loaded = story_memory.load_story_graph(str(graph_file))

        self.assertIn('eileen', loaded['characters'])
        self.assertEqual(loaded['characters']['eileen']['speaker_ids'], ['e'])
        self.assertEqual(loaded['terms'][0]['source'], 'Void Gate')
        self.assertTrue(print_mock.called)
        printed = '\n'.join(call.args[0] for call in print_mock.call_args_list)
        self.assertIn('Story graph', printed)
        self.assertIn('schema_version is required', printed)
        self.assertIn('relations[0].right', printed)
        self.assertIn('scenes[0].line_start', printed)

    def test_story_memory_graph_path_prefers_root_logs(self):
        old_values = {
            'root': runtime.ROOT_DIR,
            'base': runtime.BASE_DIR,
            'tool': runtime.TOOL_DIR,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                root_dir = tmp_path / 'repo'
                base_dir = tmp_path / 'game'
                tool_dir = tmp_path / 'tool'
                root_graph = root_dir / 'logs' / 'story_memory' / 'story_graph.json'
                base_graph = base_dir / 'logs' / 'story_memory' / 'story_graph.json'
                root_graph.parent.mkdir(parents=True)
                base_graph.parent.mkdir(parents=True)
                tool_dir.mkdir()
                root_graph.write_text('{}', encoding='utf-8')
                base_graph.write_text('{}', encoding='utf-8')
                runtime.ROOT_DIR = str(root_dir)
                runtime.BASE_DIR = str(base_dir)
                runtime.TOOL_DIR = str(tool_dir)

                resolved = runtime.resolve_story_memory_graph_path(
                    'logs/story_memory/story_graph.json'
                )
        finally:
            runtime.ROOT_DIR = old_values['root']
            runtime.BASE_DIR = old_values['base']
            runtime.TOOL_DIR = old_values['tool']

        self.assertEqual(Path(resolved), root_graph)

    def test_sync_story_memory_uses_local_graph_when_enabled(self):
        old_values = {
            'enabled': runtime.SYNC_STORY_MEMORY_ENABLED,
            'graph_file': runtime.SYNC_STORY_MEMORY_GRAPH_FILE,
            'max_context_chars': runtime.SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS,
            'top_k_relations': runtime.SYNC_STORY_MEMORY_TOP_K_RELATIONS,
            'top_k_terms': runtime.SYNC_STORY_MEMORY_TOP_K_TERMS,
            'include_scene_summary': runtime.SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY,
            'graph': runtime._SYNC_STORY_GRAPH,
            'graph_path': runtime._SYNC_STORY_GRAPH_PATH,
        }
        graph = {
            'characters': {
                'eileen': {
                    'zh_name': '\u827e\u7433',
                    'speaker_ids': ['eileen', 'eileen_side'],
                    'style': '\u8bed\u6c14\u8f7b\u5feb',
                },
                'noah': {
                    'zh_name': '\u8bfa\u4e9a',
                    'speaker_ids': ['noah'],
                },
            },
            'relations': [
                {
                    'left': 'eileen',
                    'right': 'noah',
                    'type': 'close_friend',
                    'note': '\u4e24\u4eba\u5173\u7cfb\u4eb2\u8fd1',
                    'confidence': 0.85,
                },
            ],
            'terms': [
                {
                    'source': 'Void Gate',
                    'target': '\u865a\u7a7a\u95e8',
                    'note': '\u5fc5\u987b\u7edf\u4e00',
                },
            ],
            'scenes': [
                {
                    'file_rel_path': 'chapter1.rpy',
                    'line_start': 120,
                    'line_end': 220,
                    'summary': '\u827e\u7433\u548c\u8bfa\u4e9a\u5728\u5929\u53f0\u8c08\u8bdd',
                    'characters': ['eileen', 'noah'],
                },
            ],
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                graph_file = Path(tmp) / 'story_graph.json'
                graph_file.write_text(json.dumps(graph), encoding='utf-8')
                runtime.SYNC_STORY_MEMORY_ENABLED = True
                runtime.SYNC_STORY_MEMORY_GRAPH_FILE = str(graph_file)
                runtime.SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = 500
                runtime.SYNC_STORY_MEMORY_TOP_K_RELATIONS = 4
                runtime.SYNC_STORY_MEMORY_TOP_K_TERMS = 8
                runtime.SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = True
                runtime._SYNC_STORY_GRAPH = None
                runtime._SYNC_STORY_GRAPH_PATH = ''

                items = [
                    {
                        'id': 'chapter1.rpy:129:0',
                        'text': 'Noah opens the Void Gate.',
                        'line': 129,
                        'file_rel_path': 'chapter1.rpy',
                    },
                ]
                hits = runtime.retrieve_sync_story_hits(items)
                prompt = runtime.build_prompt(items, story_hits=hits)
        finally:
            runtime.SYNC_STORY_MEMORY_ENABLED = old_values['enabled']
            runtime.SYNC_STORY_MEMORY_GRAPH_FILE = old_values['graph_file']
            runtime.SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS = old_values['max_context_chars']
            runtime.SYNC_STORY_MEMORY_TOP_K_RELATIONS = old_values['top_k_relations']
            runtime.SYNC_STORY_MEMORY_TOP_K_TERMS = old_values['top_k_terms']
            runtime.SYNC_STORY_MEMORY_INCLUDE_SCENE_SUMMARY = old_values['include_scene_summary']
            runtime._SYNC_STORY_GRAPH = old_values['graph']
            runtime._SYNC_STORY_GRAPH_PATH = old_values['graph_path']

        self.assertIn('STORY MEMORY', prompt)
        self.assertIn('Void Gate -> \u865a\u7a7a\u95e8', prompt)
        self.assertIn('eileen -> noah', prompt)
        self.assertLessEqual(len(story_memory.format_story_hits_block(hits, 80)), 80)

    def test_collect_translation_entries_does_not_skip_after_unmatched_source_markers(self):
        entries = runtime.collect_translation_entries_from_lines([
            'old "Dangling source"\n',
            '    # e "Real source"\n',
            '    e "\u771f\u5b9e\u8bd1\u6587"\n',
            '# "Comment source"\n',
            'old "Second source"\n',
            'new "\u7b2c\u4e8c\u6761"\n',
        ])

        self.assertEqual(
            [(entry['source'], entry['translation']) for entry in entries],
            [
                ('Real source', '\u771f\u5b9e\u8bd1\u6587'),
                ('Second source', '\u7b2c\u4e8c\u6761'),
            ],
        )

    def test_collect_translation_entries_decodes_source_literals(self):
        entries = runtime.collect_translation_entries_from_lines([
            'old "\\n"\n',
            'new "\\n"\n',
            '    # e "Say \\"hi\\""\n',
            '    e "Say \\"hi\\""\n',
        ])

        self.assertEqual(
            [(entry['source'], entry['translation']) for entry in entries],
            [('\n', '\n'), ('Say "hi"', 'Say "hi"')],
        )
        self.assertFalse(any(runtime.should_index_sync_rag_entry(entry) for entry in entries))

    def test_sync_rag_hash_key_uses_full_digest(self):
        self.assertEqual(runtime.sync_rag_hash_key('memory'), runtime.hash_text('memory'))
        self.assertEqual(len(runtime.sync_rag_hash_key('memory')), 40)

    def test_sync_rag_store_for_file_upserts_translation_entries(self):
        old_values = {
            'enabled': runtime.SYNC_RAG_ENABLED,
            'update_on_success': runtime.SYNC_RAG_UPDATE_ON_SUCCESS,
            'store_dir': runtime.SYNC_RAG_STORE_DIR,
            'output_dimensionality': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
            'segment_lines': runtime.SYNC_RAG_SEGMENT_LINES,
            'tl_dir': runtime.TL_DIR,
            'store': runtime._SYNC_RAG_STORE,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / 'tl'
            tl_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            target_file.write_text(
                'translate schinese start:\n'
                '    # e "Hello there"\n'
                '    e "\u4f60\u597d"\n',
                encoding='utf-8',
            )
            try:
                runtime.SYNC_RAG_ENABLED = True
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = True
                runtime.SYNC_RAG_STORE_DIR = str(Path(tmp) / 'store')
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = 3
                runtime.SYNC_RAG_SEGMENT_LINES = 4
                runtime.TL_DIR = str(tl_dir)
                runtime._SYNC_RAG_STORE = None

                with mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]):
                    summary = runtime.sync_rag_store_for_file(str(target_file))

                self.assertEqual(summary['upserted'], 1)
                records = list(runtime._SYNC_RAG_STORE.history.values())
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['source_text'], 'Hello there')
                self.assertEqual(records[0]['translated_text'], '\u4f60\u597d')
                self.assertEqual(records[0]['quality_state'], 'sync_applied')
            finally:
                runtime.SYNC_RAG_ENABLED = old_values['enabled']
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = old_values['update_on_success']
                runtime.SYNC_RAG_STORE_DIR = old_values['store_dir']
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = old_values['output_dimensionality']
                runtime.SYNC_RAG_SEGMENT_LINES = old_values['segment_lines']
                runtime.TL_DIR = old_values['tl_dir']
                runtime._SYNC_RAG_STORE = old_values['store']

    def test_sync_rag_store_for_tasks_updates_incrementally(self):
        old_values = {
            'enabled': runtime.SYNC_RAG_ENABLED,
            'update_on_success': runtime.SYNC_RAG_UPDATE_ON_SUCCESS,
            'store_dir': runtime.SYNC_RAG_STORE_DIR,
            'output_dimensionality': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
            'segment_lines': runtime.SYNC_RAG_SEGMENT_LINES,
            'tl_dir': runtime.TL_DIR,
            'store': runtime._SYNC_RAG_STORE,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / 'tl'
            tl_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    pass\n', encoding='utf-8')
            tasks = [
                {
                    'line': 1,
                    'start': 4,
                    'end': 11,
                    'text': 'Hello there',
                    'translated_text': '\u4f60\u597d',
                    'quote': '"',
                }
            ]
            try:
                runtime.SYNC_RAG_ENABLED = True
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = True
                runtime.SYNC_RAG_STORE_DIR = str(Path(tmp) / 'store')
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = 3
                runtime.SYNC_RAG_SEGMENT_LINES = 4
                runtime.TL_DIR = str(tl_dir)
                runtime._SYNC_RAG_STORE = None

                with (
                    mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]),
                    mock.patch.object(runtime, 'collect_sync_rag_records_for_file') as full_scan,
                ):
                    summary = runtime.sync_rag_store_for_tasks(str(target_file), tasks)

                full_scan.assert_not_called()
                self.assertEqual(summary['upserted'], 1)
                records = list(runtime._SYNC_RAG_STORE.history.values())
                self.assertEqual(records[0]['source_text'], 'Hello there')
                self.assertEqual(records[0]['translated_text'], '\u4f60\u597d')
                self.assertEqual(records[0]['record_scope'], 'task')
            finally:
                runtime.SYNC_RAG_ENABLED = old_values['enabled']
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = old_values['update_on_success']
                runtime.SYNC_RAG_STORE_DIR = old_values['store_dir']
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = old_values['output_dimensionality']
                runtime.SYNC_RAG_SEGMENT_LINES = old_values['segment_lines']
                runtime.TL_DIR = old_values['tl_dir']
                runtime._SYNC_RAG_STORE = old_values['store']

    def test_sync_rag_file_refresh_keeps_task_scoped_records(self):
        old_values = {
            'enabled': runtime.SYNC_RAG_ENABLED,
            'update_on_success': runtime.SYNC_RAG_UPDATE_ON_SUCCESS,
            'store_dir': runtime.SYNC_RAG_STORE_DIR,
            'output_dimensionality': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
            'segment_lines': runtime.SYNC_RAG_SEGMENT_LINES,
            'tl_dir': runtime.TL_DIR,
            'store': runtime._SYNC_RAG_STORE,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / 'tl'
            tl_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    e "\u4f60\u597d"\n', encoding='utf-8')
            tasks = [
                {
                    'line': 1,
                    'start': 6,
                    'end': 13,
                    'text': 'Hello there',
                    'translated_text': '\u4f60\u597d',
                    'quote': '"',
                }
            ]
            try:
                runtime.SYNC_RAG_ENABLED = True
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = True
                runtime.SYNC_RAG_STORE_DIR = str(Path(tmp) / 'store')
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = 3
                runtime.SYNC_RAG_SEGMENT_LINES = 4
                runtime.TL_DIR = str(tl_dir)
                runtime._SYNC_RAG_STORE = None

                with mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]):
                    task_summary = runtime.sync_rag_store_for_tasks(str(target_file), tasks)
                    refresh_summary = runtime.sync_rag_store_for_file(str(target_file))

                self.assertEqual(task_summary['upserted'], 1)
                self.assertEqual(refresh_summary['pruned'], 0)
                records = list(runtime._SYNC_RAG_STORE.history.values())
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['record_scope'], 'task')
                self.assertEqual(records[0]['source_text'], 'Hello there')
            finally:
                runtime.SYNC_RAG_ENABLED = old_values['enabled']
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = old_values['update_on_success']
                runtime.SYNC_RAG_STORE_DIR = old_values['store_dir']
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = old_values['output_dimensionality']
                runtime.SYNC_RAG_SEGMENT_LINES = old_values['segment_lines']
                runtime.TL_DIR = old_values['tl_dir']
                runtime._SYNC_RAG_STORE = old_values['store']

    def test_sync_rag_store_for_file_prunes_obsolete_chunks(self):
        old_values = {
            'enabled': runtime.SYNC_RAG_ENABLED,
            'update_on_success': runtime.SYNC_RAG_UPDATE_ON_SUCCESS,
            'store_dir': runtime.SYNC_RAG_STORE_DIR,
            'output_dimensionality': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
            'segment_lines': runtime.SYNC_RAG_SEGMENT_LINES,
            'tl_dir': runtime.TL_DIR,
            'store': runtime._SYNC_RAG_STORE,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / 'tl'
            tl_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            target_file.write_text(
                'translate schinese start:\n'
                '    # e "Hello there"\n'
                '    e "\u4f60\u597d"\n',
                encoding='utf-8',
            )
            try:
                runtime.SYNC_RAG_ENABLED = True
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = True
                runtime.SYNC_RAG_STORE_DIR = str(Path(tmp) / 'store')
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = 3
                runtime.SYNC_RAG_SEGMENT_LINES = 4
                runtime.TL_DIR = str(tl_dir)
                runtime._SYNC_RAG_STORE = None

                with mock.patch.object(runtime, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]):
                    first_summary = runtime.sync_rag_store_for_file(str(target_file))
                    target_file.write_text(
                        'translate schinese start:\n'
                        '    # e "Hello there"\n'
                        '    e "\u4f60\u597d"\n'
                        '    # e "Second line"\n'
                        '    e "\u7b2c\u4e8c\u884c"\n',
                        encoding='utf-8',
                    )
                    second_summary = runtime.sync_rag_store_for_file(str(target_file))

                self.assertEqual(first_summary['upserted'], 1)
                self.assertEqual(second_summary['pruned'], 1)
                self.assertEqual(second_summary['upserted'], 1)
                records = list(runtime._SYNC_RAG_STORE.history.values())
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['source_text'], 'Hello there\nSecond line')
                self.assertEqual(records[0]['translated_text'], '\u4f60\u597d\n\u7b2c\u4e8c\u884c')
            finally:
                runtime.SYNC_RAG_ENABLED = old_values['enabled']
                runtime.SYNC_RAG_UPDATE_ON_SUCCESS = old_values['update_on_success']
                runtime.SYNC_RAG_STORE_DIR = old_values['store_dir']
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = old_values['output_dimensionality']
                runtime.SYNC_RAG_SEGMENT_LINES = old_values['segment_lines']
                runtime.TL_DIR = old_values['tl_dir']
                runtime._SYNC_RAG_STORE = old_values['store']



class BootstrapWorkTests(unittest.TestCase):
    def test_canonical_abs_path_expands_windows_short_paths(self):
        if os.name != 'nt':
            self.skipTest('Windows-only short-path normalization')

        import ctypes

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / 'Game_Example'
            work = project / 'work'
            work.mkdir(parents=True)
            long_path = str(work.resolve())

            get_short = ctypes.windll.kernel32.GetShortPathNameW
            buffer = ctypes.create_unicode_buffer(260)
            if get_short(long_path, buffer, len(buffer)) == 0:
                self.skipTest('Could not resolve Windows short path for temp directory')

            short_path = buffer.value
            if os.path.normcase(short_path) == os.path.normcase(long_path):
                self.skipTest('Short-path generation disabled on this system')
            self.assertEqual(runtime._canonical_abs_path(short_path), long_path)

    def test_resolve_effective_game_root_prefers_nested_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            work = project / 'work'
            original_game = project / 'original' / 'game'
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)

            self.assertEqual(
                runtime.resolve_effective_game_root(str(project)),
                str(work.resolve()),
            )
            self.assertEqual(
                runtime.resolve_effective_game_root(str(work)),
                str(work.resolve()),
            )

    def test_resolve_effective_game_root_ignores_unrelated_work_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'SomeRepo'
            work = project / 'work'
            project.mkdir()
            work.mkdir()

            self.assertEqual(
                runtime.resolve_effective_game_root(str(project)),
                str(project.resolve()),
            )

    def test_context_storage_game_defaults_use_project_root(self):
        old_values = {
            'base': runtime.BASE_DIR,
            'log': runtime.LOG_DIR,
            'location': runtime.CONTEXT_STORAGE_LOCATION,
            'dir_name': runtime.CONTEXT_STORAGE_GAME_DIR_NAME,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / 'Game_Example'
                work = project / 'work'
                work.mkdir(parents=True)
                runtime.BASE_DIR = str(work)
                runtime.LOG_DIR = str(Path(tmp) / 'tool_logs')
                runtime.load_context_storage_settings({
                    'context_storage': {
                        'location': 'game',
                        'game_dir_name': 'translation_context',
                    }
                })

                self.assertEqual(
                    runtime.get_context_storage_root(),
                    str((project / 'translation_context').resolve()),
                )
                self.assertEqual(
                    runtime.get_default_batch_rag_store_dir(),
                    str((project / 'translation_context' / 'rag_store').resolve()),
                )
                self.assertEqual(
                    runtime.get_default_source_index_store_dir(),
                    str((project / 'translation_context' / 'source_index_store').resolve()),
                )
                self.assertEqual(
                    runtime.get_default_story_memory_graph_path(),
                    str((project / 'translation_context' / 'story_memory' / 'story_graph.json').resolve()),
                )
        finally:
            runtime.BASE_DIR = old_values['base']
            runtime.LOG_DIR = old_values['log']
            runtime.CONTEXT_STORAGE_LOCATION = old_values['location']
            runtime.CONTEXT_STORAGE_GAME_DIR_NAME = old_values['dir_name']

    def test_normalize_context_storage_dir_name_rejects_absolute_paths(self):
        self.assertEqual(
            runtime._normalize_context_storage_dir_name('C:\\ctx'),
            'translation_context',
        )
        self.assertEqual(
            runtime._normalize_context_storage_dir_name('C:/ctx'),
            'translation_context',
        )
        self.assertEqual(
            runtime._normalize_context_storage_dir_name('/translation_context'),
            'translation_context',
        )
        self.assertEqual(
            runtime._normalize_context_storage_dir_name('custom_context'),
            'custom_context',
        )

    def test_context_storage_tool_defaults_use_base_dir_slug(self):
        old_values = {
            'base': runtime.BASE_DIR,
            'log': runtime.LOG_DIR,
            'location': runtime.CONTEXT_STORAGE_LOCATION,
            'dir_name': runtime.CONTEXT_STORAGE_GAME_DIR_NAME,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project_a = Path(tmp) / 'Game_Example'
                project_b = Path(tmp) / 'Other_Game'
                (project_a / 'work').mkdir(parents=True)
                (project_b / 'work').mkdir(parents=True)
                log_dir = Path(tmp) / 'tool_logs'
                runtime.BASE_DIR = str(project_a / 'work')
                runtime.LOG_DIR = str(log_dir)
                runtime.load_context_storage_settings({'context_storage': {'location': 'tool'}})

                self.assertEqual(
                    runtime.get_default_context_store_dir('rag_store', str(project_b / 'work')),
                    str(log_dir / 'rag_store' / 'Other_Game'),
                )
        finally:
            runtime.BASE_DIR = old_values['base']
            runtime.LOG_DIR = old_values['log']
            runtime.CONTEXT_STORAGE_LOCATION = old_values['location']
            runtime.CONTEXT_STORAGE_GAME_DIR_NAME = old_values['dir_name']

    def test_context_storage_tool_defaults_keep_log_slug_layout(self):
        old_values = {
            'base': runtime.BASE_DIR,
            'log': runtime.LOG_DIR,
            'location': runtime.CONTEXT_STORAGE_LOCATION,
            'dir_name': runtime.CONTEXT_STORAGE_GAME_DIR_NAME,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp) / 'Game_Example'
                work = project / 'work'
                work.mkdir(parents=True)
                log_dir = Path(tmp) / 'tool_logs'
                runtime.BASE_DIR = str(work)
                runtime.LOG_DIR = str(log_dir)
                runtime.load_context_storage_settings({'context_storage': {'location': 'tool'}})

                self.assertEqual(runtime.get_context_storage_root(), str(log_dir))
                self.assertEqual(
                    runtime.get_default_batch_rag_store_dir(),
                    str(log_dir / 'rag_store' / 'Game_Example'),
                )
                self.assertEqual(
                    runtime.get_default_source_index_store_dir(),
                    str(log_dir / 'source_index_store' / 'Game_Example'),
                )
                self.assertEqual(
                    runtime.get_default_story_memory_graph_path(),
                    str(log_dir / 'story_memory' / 'story_graph.json'),
                )
        finally:
            runtime.BASE_DIR = old_values['base']
            runtime.LOG_DIR = old_values['log']
            runtime.CONTEXT_STORAGE_LOCATION = old_values['location']
            runtime.CONTEXT_STORAGE_GAME_DIR_NAME = old_values['dir_name']

    def test_resolve_project_root_from_original_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original = project / 'original'
            original.mkdir(parents=True)

            self.assertEqual(
                runtime.resolve_project_root(str(original)),
                str(project.resolve()),
            )
            self.assertEqual(
                runtime.resolve_work_dir(str(original)),
                str((project / 'work').resolve()),
            )

    def test_resolve_effective_game_root_keeps_original_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original = project / 'original'
            work = project / 'work'
            original.mkdir(parents=True)
            work.mkdir(parents=True)

            self.assertEqual(
                runtime.resolve_effective_game_root(str(original)),
                str(original.resolve()),
            )

    def test_resolve_original_game_dir_when_game_root_is_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')

            self.assertEqual(
                runtime.resolve_original_game_dir(str(project / 'original')),
                str(original_game.resolve()),
            )

    def test_load_translator_settings_redirects_project_root_to_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            work = project / 'work'
            original_game = project / 'original' / 'game'
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)
            config_path = root / 'translator_config.json'
            config_path.write_text(
                json.dumps({'game_root': str(project)}),
                encoding='utf-8',
            )

            with mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)):
                runtime.load_translator_settings()

            self.assertEqual(runtime.BASE_DIR, runtime.canonical_abs_path(work))
            saved = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertEqual(
                runtime.canonical_abs_path(saved['game_root']),
                runtime.canonical_abs_path(work),
            )

    def test_load_translator_settings_applies_runtime_when_persist_game_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            work = project / 'work'
            original_game = project / 'original' / 'game'
            work.mkdir(parents=True)
            original_game.mkdir(parents=True)
            config_path = root / 'translator_config.json'
            config_path.write_text(
                json.dumps({'game_root': str(project)}),
                encoding='utf-8',
            )

            with (
                mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                mock.patch.object(runtime, 'persist_game_root', side_effect=OSError('denied')),
            ):
                runtime.load_translator_settings()

            self.assertEqual(runtime.BASE_DIR, runtime.canonical_abs_path(work))
            saved = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertEqual(
                runtime.canonical_abs_path(saved['game_root']),
                runtime.canonical_abs_path(project),
            )

    def test_resolve_original_game_dir_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')

            self.assertEqual(
                runtime.resolve_original_game_dir(str(project)),
                str(original_game.resolve()),
            )

    def test_work_dir_bootstrap_allowed_when_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            project.mkdir()
            allowed, work_dir, reason = runtime.work_dir_bootstrap_allowed(str(project))
            self.assertTrue(allowed)
            self.assertEqual(work_dir, str((project / 'work').resolve()))
            self.assertEqual(reason, '')

            (project / 'work').mkdir()
            allowed, _, reason = runtime.work_dir_bootstrap_allowed(str(project))
            self.assertTrue(allowed)
            self.assertEqual(reason, '')

    def test_work_dir_bootstrap_not_allowed_when_work_has_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            work = project / 'work'
            work.mkdir(parents=True)
            (work / 'marker.txt').write_text('keep', encoding='utf-8')

            allowed, _, reason = runtime.work_dir_bootstrap_allowed(str(project))
            self.assertFalse(allowed)
            self.assertIn('not empty', reason)

    def test_bootstrap_work_from_original_creates_work_game_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')
            (original_game / 'images').mkdir()
            (original_game / 'images' / 'logo.png').write_bytes(b'png')

            result = runtime.bootstrap_work_from_original(base_dir=str(project))
            self.assertEqual(result['status'], 'created')
            self.assertTrue((project / 'work' / 'game' / 'script.rpy').is_file())
            self.assertTrue((project / 'work' / 'game' / 'images' / 'logo.png').is_file())
            self.assertEqual(result['files_copied'], 2)

    def test_copy_game_directory_throttles_bootstrap_progress_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            target = Path(tmp) / 'target'
            source.mkdir()
            for index in range(60):
                (source / f'file_{index:03d}.txt').write_text('x', encoding='utf-8')

            with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
                copied = runtime._copy_game_directory(str(source), str(target))

            lines = [
                line.strip()
                for line in stdout.getvalue().splitlines()
                if line.startswith('Work bootstrap copy progress:')
            ]

            self.assertEqual(copied, 60)
            self.assertEqual(lines[0], 'Work bootstrap copy progress: 0/60 files.')
            progress_counts = []
            for line in lines[1:]:
                match = re.fullmatch(
                    r'Work bootstrap copy progress: (\d+)/60 files, file=(.+)\.',
                    line,
                )
                self.assertIsNotNone(match, line)
                progress_counts.append(int(match.group(1)))
                self.assertRegex(match.group(2), r'^file_\d{3}\.txt$')
            self.assertEqual(progress_counts, [1, 25, 50, 60])
            self.assertLess(len(lines), 60)

    def test_bootstrap_work_from_original_skips_non_empty_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')
            work_game = project / 'work' / 'game'
            work_game.mkdir(parents=True)
            (work_game / 'existing.rpy').write_text('keep', encoding='utf-8')

            result = runtime.bootstrap_work_from_original(base_dir=str(project))
            self.assertEqual(result['status'], 'skipped')
            self.assertEqual(result['files_copied'], 0)

    def test_bootstrap_work_rejects_source_containing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            project.mkdir(parents=True)

            with mock.patch.object(runtime, 'SOURCE_GAME_DIR', str(project)):
                result = runtime.bootstrap_work_from_original(base_dir=str(project))

            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['files_copied'], 0)
            self.assertIn('must not contain work/game', result['message'])

    def test_bootstrap_work_applies_runtime_when_persist_game_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')

            with (
                mock.patch.object(runtime, 'persist_game_root', side_effect=OSError('denied')),
                mock.patch.object(runtime, 'TL_SUBDIR', 'game/tl/schinese'),
            ):
                result = runtime.bootstrap_work_from_original(
                    base_dir=str(project),
                    save_game_root=True,
                    refresh_runtime_paths=True,
                )

            self.assertEqual(result['status'], 'created')
            self.assertFalse(result['game_root_updated'])
            self.assertIn('Failed to update game_root', result['message'])
            self.assertEqual(
                runtime.BASE_DIR,
                runtime.canonical_abs_path(project / 'work'),
            )

    def test_bootstrap_work_persists_game_root_when_pointing_at_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')
            config_path = root / 'translator_config.json'
            config_path.write_text(
                json.dumps({'game_root': str(project)}),
                encoding='utf-8',
            )

            with (
                mock.patch.object(runtime, 'TRANSLATOR_CONFIG', str(config_path)),
                mock.patch.object(runtime, 'TL_SUBDIR', 'game/tl/schinese'),
            ):
                result = runtime.bootstrap_work_from_original(
                    base_dir=str(project),
                    save_game_root=True,
                    refresh_runtime_paths=True,
                )

            self.assertEqual(result['status'], 'created')
            self.assertTrue(result['game_root_updated'])
            saved = json.loads(config_path.read_text(encoding='utf-8'))
            self.assertEqual(
                runtime.canonical_abs_path(saved['game_root']),
                runtime.canonical_abs_path(project / 'work'),
            )
            self.assertEqual(
                runtime.BASE_DIR,
                runtime.canonical_abs_path(project / 'work'),
            )

if __name__ == '__main__':
    unittest.main()
