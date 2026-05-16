import ast
import importlib
import io
import json
import os
import pickle
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import rag_memory
import story_memory
import translator_runtime as runtime


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
            with mock.patch.object(runtime, 'call_gemini_sdk', return_value=[
                {'id': 'file:0:1', 'translation': '\u65e7\u79f0\u4f60\u597d'},
            ]):
                runtime.process_batch(batch, {})
        finally:
            runtime.NORMALIZE_TRANSLATION_MAP = old_normalize_map
            runtime.USE_TRANSLATION_MEMORY = old_use_memory

        self.assertEqual(batch[0]['translated_text'], '\u65b0\u79f0\u4f60\u597d')

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


class RagMemoryStoreTests(unittest.TestCase):
    def make_record(self, memory_id, file_rel_path='script.rpy', source='Hello', translation='\u4f60\u597d'):
        return {
            'memory_id': memory_id,
            'file_rel_path': file_rel_path,
            'source_text': source,
            'translated_text': translation,
            'embedding': [1.0, 0.0, 0.0],
            'quality_state': 'seed',
        }

    def test_json_rag_store_recovers_valid_rows_and_warns_on_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            metadata_path = store_dir / 'metadata.json'
            history_path = store_dir / 'history.jsonl'
            metadata_path.write_text('{bad json', encoding='utf-8')
            history_path.write_text(
                json.dumps(self.make_record('m1'), ensure_ascii=False) + '\n'
                '{bad row\n'
                '[]\n',
                encoding='utf-8',
            )

            store = rag_memory.JsonRagStore(str(store_dir))
            with mock.patch('builtins.print') as print_mock:
                count = store.count_history()

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertEqual(count, 1)
        self.assertIn('Failed to load RAG metadata', warnings)
        self.assertIn('Skipping invalid RAG history row', warnings)
        self.assertIn('Skipping non-object RAG history row', warnings)

    def test_json_rag_store_writes_history_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            history_path = store_dir / 'history.jsonl'
            original_record = self.make_record('old')
            history_path.write_text(
                json.dumps(original_record, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            store = rag_memory.JsonRagStore(str(store_dir))

            with mock.patch.object(rag_memory.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    store.upsert_history([self.make_record('new')])

            persisted = history_path.read_text(encoding='utf-8')
            temp_files = list(store_dir.glob('*.tmp.*'))

        self.assertEqual(persisted, json.dumps(original_record, ensure_ascii=False) + '\n')
        self.assertEqual(temp_files, [])

    def test_json_rag_store_warns_when_temp_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            store = rag_memory.JsonRagStore(str(store_dir))
            original_remove = rag_memory.os.remove

            def remove_tmp(path):
                if '.tmp.' in str(path):
                    raise OSError('tmp remove denied')
                return original_remove(path)

            with (
                mock.patch.object(rag_memory.os, 'replace', side_effect=OSError('replace failed')),
                mock.patch.object(rag_memory.os, 'remove', side_effect=remove_tmp),
                mock.patch('builtins.print') as print_mock,
            ):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    store.upsert_history([self.make_record('m1')])

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertIn('Failed to remove temporary RAG store file', warnings)
        self.assertIn('tmp remove denied', warnings)

    def test_json_rag_store_lock_conflict_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.write_text(
                json.dumps({'operation': 'upsert_history', 'owner': 'test-host', 'pid': 123}),
                encoding='utf-8',
            )
            store = rag_memory.JsonRagStore(str(store_dir))

            with self.assertRaisesRegex(rag_memory.JsonRagStoreLockError, 'test-host'):
                store.upsert_history([self.make_record('m1')])

            self.assertFalse((store_dir / 'history.jsonl').exists())

    def test_json_rag_store_creates_lock_with_restrictive_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            store = rag_memory.JsonRagStore(str(store_dir))
            original_open = rag_memory.os.open
            open_modes = []

            def tracking_open(path, flags, mode=0o777):
                open_modes.append(mode)
                return original_open(path, flags, mode)

            with mock.patch.object(rag_memory.os, 'open', side_effect=tracking_open):
                store.upsert_history([self.make_record('m1')])

        self.assertEqual(open_modes[0], 0o600)

    def test_json_rag_store_recovers_stale_lock_from_dead_local_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.write_text(
                json.dumps({
                    'operation': 'upsert_history',
                    'owner': rag_memory.socket.gethostname(),
                    'pid': 987654,
                    'created_at': rag_memory.now_iso(),
                }),
                encoding='utf-8',
            )
            store = rag_memory.JsonRagStore(str(store_dir))

            with (
                mock.patch.object(rag_memory.JsonRagStore, '_is_lock_owner_alive', return_value=False),
                mock.patch('builtins.print') as print_mock,
            ):
                store.upsert_history([self.make_record('m1')])

            reloaded = rag_memory.JsonRagStore(str(store_dir))
            history_ids = reloaded.history_ids_for_file('script.rpy')
            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertEqual(history_ids, ['m1'])
        self.assertIn('Recovered stale RAG store lock', warnings)

    def test_json_rag_store_recovers_stale_unreadable_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.write_text('{bad json', encoding='utf-8')
            stale_time = time.time() - rag_memory.LOCK_STALE_AFTER_SECONDS - 5
            os.utime(lock_path, (stale_time, stale_time))
            store = rag_memory.JsonRagStore(str(store_dir))

            with mock.patch('builtins.print') as print_mock:
                store.upsert_history([self.make_record('m1')])

            reloaded = rag_memory.JsonRagStore(str(store_dir))
            history_ids = reloaded.history_ids_for_file('script.rpy')
            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertEqual(history_ids, ['m1'])
        self.assertIn('Recovered stale RAG store lock', warnings)
        self.assertIn('unknown owner', warnings)

    def test_json_rag_store_keeps_fresh_unreadable_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.write_text('{bad json', encoding='utf-8')
            store = rag_memory.JsonRagStore(str(store_dir))

            with self.assertRaisesRegex(rag_memory.JsonRagStoreLockError, 'unknown owner'):
                store.upsert_history([self.make_record('m1')])

            self.assertFalse((store_dir / 'history.jsonl').exists())

    def test_json_rag_store_lock_cleanup_failure_fails_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            store = rag_memory.JsonRagStore(str(store_dir))
            with (
                mock.patch.object(rag_memory.os, 'remove', side_effect=OSError('remove denied')),
                mock.patch('builtins.print') as print_mock,
            ):
                with self.assertRaisesRegex(rag_memory.JsonRagStoreLockError, 'remove denied') as raised:
                    store.upsert_history([self.make_record('m1')])

                released_lock = json.loads(lock_path.read_text(encoding='utf-8'))
                warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

            with mock.patch('builtins.print'):
                store.upsert_history([self.make_record('m2')])
            reloaded = rag_memory.JsonRagStore(str(store_dir))
            history_ids = set(reloaded.history_ids_for_file('script.rpy'))

        self.assertIn('released_at', released_lock)
        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(history_ids, {'m1', 'm2'})
        self.assertIn('Failed to remove RAG store lock', warnings)
        self.assertIn('remove denied', warnings)

    def test_json_rag_store_lock_cleanup_failure_fails_inside_outer_except(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            store = rag_memory.JsonRagStore(str(store_dir))

            with (
                mock.patch.object(rag_memory.os, 'remove', side_effect=OSError('remove denied')),
                mock.patch('builtins.print'),
            ):
                try:
                    raise RuntimeError('outer error')
                except RuntimeError:
                    with self.assertRaisesRegex(rag_memory.JsonRagStoreLockError, 'remove denied'):
                        store.upsert_history([self.make_record('m1')])

    def test_json_rag_store_mark_released_closes_fdopen_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.write_text('{}', encoding='utf-8')
            store = rag_memory.JsonRagStore(str(store_dir))
            fake_fd = 777

            with (
                mock.patch.object(rag_memory.os, 'open', return_value=fake_fd),
                mock.patch.object(rag_memory.os, 'fdopen', side_effect=OSError('fdopen denied')),
                mock.patch.object(rag_memory.os, 'close') as close_mock,
                mock.patch('builtins.print') as print_mock,
            ):
                marked = store._mark_lock_released({'operation': 'upsert_history'})

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertFalse(marked)
        close_mock.assert_called_once_with(fake_fd)
        self.assertIn('Failed to mark RAG store lock released', warnings)
        self.assertIn('fdopen denied', warnings)

    def test_json_rag_store_mark_released_preserves_lock_on_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            original_lock = json.dumps({'operation': 'upsert_history', 'owner': 'test-host'})
            lock_path.write_text(original_lock, encoding='utf-8')
            store = rag_memory.JsonRagStore(str(store_dir))

            with (
                mock.patch.object(rag_memory.json, 'dump', side_effect=OSError('dump denied')),
                mock.patch('builtins.print') as print_mock,
            ):
                marked = store._mark_lock_released({'operation': 'upsert_history'})

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)
            persisted_lock = lock_path.read_text(encoding='utf-8')
            temp_files = list(store_dir.glob('*.released.tmp.*'))

        self.assertFalse(marked)
        self.assertEqual(persisted_lock, original_lock)
        self.assertEqual(temp_files, [])
        self.assertIn('Failed to mark RAG store lock released', warnings)
        self.assertIn('dump denied', warnings)

    def test_json_rag_store_mark_released_rejects_non_regular_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.rag_store.lock'
            lock_path.mkdir()
            store = rag_memory.JsonRagStore(str(store_dir))

            with mock.patch('builtins.print') as print_mock:
                marked = store._mark_lock_released({'operation': 'upsert_history'})

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)
            lock_is_dir = lock_path.is_dir()

        self.assertFalse(marked)
        self.assertTrue(lock_is_dir)
        self.assertIn('not a regular file', warnings)

    def test_json_rag_store_reload_under_lock_prevents_stale_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            stale_store = rag_memory.JsonRagStore(str(store_dir))
            fresh_store = rag_memory.JsonRagStore(str(store_dir))
            self.assertEqual(stale_store.count_history(), 0)

            fresh_store.upsert_history([self.make_record('fresh')])
            stale_store.upsert_history([self.make_record('stale')])

            reloaded = rag_memory.JsonRagStore(str(store_dir))
            history_ids = set(reloaded.history_ids_for_file('script.rpy'))

        self.assertEqual(history_ids, {'fresh', 'stale'})

    def test_json_rag_store_skips_reload_when_disk_snapshot_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            store = rag_memory.JsonRagStore(str(store_dir))

            store.upsert_history([self.make_record('m1')])
            with mock.patch.object(store, '_load_from_disk', wraps=store._load_from_disk) as load_mock:
                store.upsert_history([self.make_record('m2')])

            reloaded = rag_memory.JsonRagStore(str(store_dir))
            history_ids = set(reloaded.history_ids_for_file('script.rpy'))

        load_mock.assert_not_called()
        self.assertEqual(history_ids, {'m1', 'm2'})

    def test_json_rag_store_metadata_records_write_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            store = rag_memory.JsonRagStore(str(store_dir))

            store.upsert_history([self.make_record('m1')])

            metadata = json.loads((store_dir / 'metadata.json').read_text(encoding='utf-8'))

        self.assertEqual(metadata['history_count'], 1)
        self.assertEqual(metadata['last_write']['operation'], 'upsert_history')
        self.assertEqual(metadata['last_write']['pid'], os.getpid())


class BatchRagRegressionTests(unittest.TestCase):
    def test_format_history_hits_block_shows_source_translation_pair(self):
        block = batch_mod.format_history_hits_block([
            {
                'file_rel_path': 'script.rpy',
                'line_start': 10,
                'line_end': 12,
                'score': 0.8123,
                'quality_state': 'seed',
                'source_text': 'Aether Gate',
                'translated_text': '\u4ee5\u592a\u95e8',
            }
        ])

        self.assertIn('Source: Aether Gate -> Translation: \u4ee5\u592a\u95e8', block)
        self.assertIn('score=0.812', block)

    def test_format_history_hits_block_compares_untruncated_text(self):
        shared_prefix = 'A' * batch_mod.RAG_HISTORY_CHAR_LIMIT
        block = batch_mod.format_history_hits_block([
            {
                'file_rel_path': 'script.rpy',
                'line_start': 10,
                'line_end': 12,
                'score': 0.8123,
                'quality_state': 'seed',
                'source_text': shared_prefix + 'S',
                'translated_text': shared_prefix + 'T',
            }
        ])

        self.assertIn(' Source: ', block)
        self.assertIn(' -> Translation: ', block)
        self.assertNotIn('] Translation: ', block)

    def test_embed_history_records_uses_source_text_only(self):
        record = batch_mod.build_rag_record(
            'script.rpy',
            [{'line_number': 3, 'source': 'Aether Gate', 'translation': '\u4ee5\u592a\u95e8'}],
            'seed',
        )

        with mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as embed_mock:
            embedded = batch_mod.embed_history_records([record])

        embed_mock.assert_called_once_with(['Aether Gate'], batch_mod.RAG_DOCUMENT_TASK_TYPE)
        self.assertEqual(embedded[0]['embedding_text_kind'], 'source_text')
        self.assertEqual(embedded[0]['embedding_text_checksum'], batch_mod.hash_text('Aether Gate'))

    def test_prepare_rag_store_bootstraps_all_allowed_tl_files(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_bootstrap': batch_mod.RAG_BOOTSTRAP_ON_BUILD,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'rag_segment_lines': batch_mod.RAG_SEGMENT_LINES,
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'game' / 'tl' / 'schinese'
                tl_dir.mkdir(parents=True)
                pending_file = tl_dir / 'pending.rpy'
                memory_file = tl_dir / 'memory.rpy'
                pending_file.write_text('old "Needs translation"\nnew ""\n', encoding='utf-8')
                memory_file.write_text('old "Aether Gate"\nnew "\u4ee5\u592a\u95e8"\n', encoding='utf-8')

                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_BOOTSTRAP_ON_BUILD = True
                batch_mod.RAG_STORE_DIR = str(root / 'rag_store')
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod.RAG_SEGMENT_LINES = 4
                batch_mod._RAG_STORE = None
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.legacy.INCLUDE_FILES = set()
                batch_mod.legacy.INCLUDE_PREFIXES = set()

                embedded_inputs = []

                def fake_embed(contents, task_type):
                    embedded_inputs.extend(contents)
                    return [[1.0, 0.0, 0.0] for _ in contents]

                file_jobs = [{
                    'file_rel_path': 'pending.rpy',
                    'file_path': str(pending_file),
                    'task_count': 1,
                    'tasks': [],
                }]
                with mock.patch.object(batch_mod, 'embed_texts', side_effect=fake_embed):
                    summary = batch_mod.prepare_rag_store(file_jobs)

                store = batch_mod.get_rag_store()
                records = list(store.history.values())

            self.assertEqual(summary['scan_scope'], 'all_files')
            self.assertEqual(summary['files_scanned'], 2)
            self.assertEqual(summary['scanned'], 1)
            self.assertEqual(summary['upserted'], 1)
            self.assertEqual(embedded_inputs, ['Aether Gate'])
            self.assertEqual(records[0]['source_text'], 'Aether Gate')
            self.assertEqual(records[0]['translated_text'], '\u4ee5\u592a\u95e8')
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_BOOTSTRAP_ON_BUILD = old_values['rag_bootstrap']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.RAG_SEGMENT_LINES = old_values['rag_segment_lines']
            batch_mod.legacy.BASE_DIR = old_values['base_dir']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
            batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_bootstrap_rag_store_scans_all_allowed_tl_files_explicitly(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_bootstrap': batch_mod.RAG_BOOTSTRAP_ON_BUILD,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'rag_segment_lines': batch_mod.RAG_SEGMENT_LINES,
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'game' / 'tl' / 'schinese'
                tl_dir.mkdir(parents=True)
                memory_file = tl_dir / 'memory.rpy'
                memory_file.write_text('old "Aether Gate"\nnew "\u4ee5\u592a\u95e8"\n', encoding='utf-8')

                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_BOOTSTRAP_ON_BUILD = False
                batch_mod.RAG_STORE_DIR = str(root / 'rag_store')
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod.RAG_SEGMENT_LINES = 4
                batch_mod._RAG_STORE = None
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.legacy.INCLUDE_FILES = set()
                batch_mod.legacy.INCLUDE_PREFIXES = set()

                stdout = io.StringIO()
                with (
                    mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as embed_mock,
                    mock.patch('sys.stdout', stdout),
                ):
                    summary = batch_mod.bootstrap_rag_store(skip_prepare=True)

                store = batch_mod.get_rag_store()
                output = stdout.getvalue()

            embed_mock.assert_called_once_with(['Aether Gate'], batch_mod.RAG_DOCUMENT_TASK_TYPE)
            self.assertIn('RAG bootstrap summary:', output)
            self.assertIn('- scan_scope: all_files', output)
            self.assertIn('- embedded: 1', output)
            self.assertEqual(summary['scan_scope'], 'all_files')
            self.assertEqual(summary['files_scanned'], 1)
            self.assertEqual(summary['scanned'], 1)
            self.assertEqual(summary['embedded'], 1)
            self.assertEqual(summary['upserted'], 1)
            self.assertEqual(store.count_history(), 1)
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_BOOTSTRAP_ON_BUILD = old_values['rag_bootstrap']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.RAG_SEGMENT_LINES = old_values['rag_segment_lines']
            batch_mod.legacy.BASE_DIR = old_values['base_dir']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
            batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_bootstrap_rag_store_disabled_does_not_require_tl_dir(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'tl_dir': batch_mod.legacy.TL_DIR,
        }
        try:
            batch_mod.RAG_ENABLED = False
            batch_mod.legacy.TL_DIR = str(Path('missing') / 'tl')

            stdout = io.StringIO()
            with (
                mock.patch.object(batch_mod.legacy, 'run_prepare_steps') as prepare_mock,
                mock.patch('sys.stdout', stdout),
            ):
                summary = batch_mod.bootstrap_rag_store(skip_prepare=False)

            prepare_mock.assert_not_called()
            self.assertEqual(summary, {'enabled': False})
            self.assertIn('RAG is disabled', stdout.getvalue())
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']

    def test_bootstrap_rag_store_imports_external_jsonl_seed(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'game' / 'tl' / 'schinese'
                tl_dir.mkdir(parents=True)
                seed_path = root / 'parallel.jsonl'
                seed_path.write_text(
                    '\n'.join([
                        json.dumps({
                            'file_rel_path': 'external/memory.txt',
                            'line_start': None,
                            'line': 7,
                            'source': 'Aether Gate',
                            'translation': '\u4ee5\u592a\u95e8',
                        }, ensure_ascii=False),
                        '{"source": ',
                        json.dumps({'source': 'Same', 'translation': 'Same'}, ensure_ascii=False),
                    ]),
                    encoding='utf-8',
                )

                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_STORE_DIR = str(root / 'rag_store')
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod._RAG_STORE = None
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.legacy.INCLUDE_FILES = set()
                batch_mod.legacy.INCLUDE_PREFIXES = set()

                stdout = io.StringIO()
                with (
                    mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as embed_mock,
                    mock.patch('sys.stdout', stdout),
                ):
                    summary = batch_mod.bootstrap_rag_store(
                        skip_prepare=True,
                        seed_jsonl_paths=[str(seed_path)],
                    )

                store = batch_mod.get_rag_store()
                records = list(store.history.values())
                output = stdout.getvalue()

            embed_mock.assert_called_once_with(['Aether Gate'], batch_mod.RAG_DOCUMENT_TASK_TYPE)
            self.assertIn('- external_seed_files: 1', output)
            self.assertIn('- external_seed_records: 1', output)
            self.assertIn('- external_seed_invalid_json: 1', output)
            self.assertIn('- external_seed_filtered: 1', output)
            self.assertIn('- external_seed_skipped: 2', output)
            self.assertEqual(summary['external_seed_files'], 1)
            self.assertEqual(summary['external_seed_records'], 1)
            self.assertEqual(summary['external_seed_invalid_json'], 1)
            self.assertEqual(summary['external_seed_filtered'], 1)
            self.assertEqual(summary['external_seed_skipped'], 2)
            self.assertEqual(summary['files_scanned'], 0)
            self.assertEqual(summary['scanned'], 1)
            self.assertEqual(summary['embedded'], 1)
            self.assertEqual(records[0]['file_rel_path'], 'external/memory.txt')
            self.assertEqual(records[0]['line_start'], 7)
            self.assertEqual(records[0]['quality_state'], 'external_seed')
            self.assertEqual(records[0]['source_text'], 'Aether Gate')
            self.assertEqual(records[0]['translated_text'], '\u4ee5\u592a\u95e8')
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.legacy.BASE_DIR = old_values['base_dir']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
            batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_external_jsonl_seed_default_source_name_preserves_path_uniqueness(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'game' / 'tl' / 'schinese'
                tl_dir.mkdir(parents=True)
                left_seed = root / 'left' / 'parallel.jsonl'
                right_seed = root / 'right' / 'parallel.jsonl'
                left_seed.parent.mkdir()
                right_seed.parent.mkdir()
                seed_row = json.dumps({
                    'line': 1,
                    'source': 'Aether Gate',
                    'translation': '\u4ee5\u592a\u95e8',
                }, ensure_ascii=False)
                left_seed.write_text(
                    '\n'.join([
                        seed_row,
                        json.dumps({'source': 'left noop', 'translation': 'left noop'}, ensure_ascii=False),
                    ]),
                    encoding='utf-8',
                )
                right_seed.write_text(
                    '\n'.join([
                        seed_row,
                        json.dumps({'source': 'right noop', 'translation': 'right noop'}, ensure_ascii=False),
                    ]),
                    encoding='utf-8',
                )

                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_STORE_DIR = str(root / 'rag_store')
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod._RAG_STORE = None
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.legacy.INCLUDE_FILES = set()
                batch_mod.legacy.INCLUDE_PREFIXES = set()

                with (
                    mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    summary = batch_mod.bootstrap_rag_store(
                        skip_prepare=True,
                        seed_jsonl_paths=[str(left_seed), str(right_seed)],
                    )

                store = batch_mod.get_rag_store()
                records = list(store.history.values())
                file_rel_paths = {record['file_rel_path'] for record in records}
                memory_ids = {record['memory_id'] for record in records}

            self.assertEqual(summary['external_seed_files'], 2)
            self.assertEqual(summary['external_seed_records'], 2)
            self.assertEqual(summary['upserted'], 2)
            self.assertEqual(store.count_history(), 2)
            self.assertEqual(len(file_rel_paths), 2)
            self.assertEqual(len(memory_ids), 2)
            self.assertTrue(all(path.startswith('external/') for path in file_rel_paths))
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.legacy.BASE_DIR = old_values['base_dir']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
            batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_external_jsonl_seed_default_source_name_is_content_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_seed = root / 'left' / 'parallel.jsonl'
            right_seed = root / 'right' / 'renamed.jsonl'
            left_seed.parent.mkdir()
            right_seed.parent.mkdir()
            seed_content = json.dumps({
                'line': 1,
                'source': 'Aether Gate',
                'translation': '\u4ee5\u592a\u95e8',
            }, ensure_ascii=False)
            left_seed.write_text(seed_content, encoding='utf-8')
            right_seed.write_text(seed_content, encoding='utf-8')

            self.assertEqual(
                batch_mod.external_seed_source_name(str(left_seed)),
                batch_mod.external_seed_source_name(str(right_seed)),
            )

    def test_bootstrap_rag_store_imports_seed_when_tl_dir_is_missing(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                seed_path = root / 'parallel.jsonl'
                seed_path.write_text(
                    json.dumps({
                        'source': 'Aether Gate',
                        'translation': '\u4ee5\u592a\u95e8',
                    }, ensure_ascii=False),
                    encoding='utf-8',
                )

                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_STORE_DIR = str(root / 'rag_store')
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod._RAG_STORE = None
                batch_mod.legacy.BASE_DIR = str(root)
                batch_mod.legacy.TL_DIR = str(root / 'missing_tl')

                with (
                    mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as embed_mock,
                    mock.patch('sys.stdout', io.StringIO()),
                ):
                    summary = batch_mod.bootstrap_rag_store(
                        skip_prepare=True,
                        seed_jsonl_paths=[str(seed_path)],
                    )

                store = batch_mod.get_rag_store()

            embed_mock.assert_called_once_with(['Aether Gate'], batch_mod.RAG_DOCUMENT_TASK_TYPE)
            self.assertEqual(summary['files_scanned'], 0)
            self.assertEqual(summary['external_seed_records'], 1)
            self.assertEqual(summary['upserted'], 1)
            self.assertEqual(store.count_history(), 1)
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.legacy.BASE_DIR = old_values['base_dir']
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
            batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']

    def test_build_arg_parser_accepts_bootstrap_rag_command(self):
        args = batch_mod.build_arg_parser().parse_args([
            'bootstrap-rag',
            '--skip-prepare',
            '--seed-jsonl',
            'parallel.jsonl',
        ])

        self.assertEqual(args.command, 'bootstrap-rag')
        self.assertTrue(args.skip_prepare)
        self.assertEqual(args.seed_jsonl, ['parallel.jsonl'])

        parser = batch_mod.build_arg_parser()
        parser.parse_args(['bootstrap-rag', '--seed-jsonl', 'first.jsonl'])
        parsed_without_seed = parser.parse_args(['bootstrap-rag'])
        self.assertIsNone(parsed_without_seed.seed_jsonl)

    def test_summarize_batch_rag_reports_hit_count_rate_and_errors(self):
        summary = batch_mod.summarize_batch_rag(
            [
                {
                    'glossary_hits': [{'source': 'Aether', 'target': '\u4ee5\u592a'}],
                    'history_hits': [{'memory_id': 'm1'}, {'memory_id': 'm2'}],
                    'rag_stats': {'hit_count': 2},
                },
                {
                    'history_hits': [],
                    'rag_stats': {'error': 'embedding failed'},
                },
            ],
            {'upserted': 1},
        )

        self.assertEqual(summary['prepare'], {'upserted': 1})
        self.assertEqual(summary['chunks_with_glossary_hits'], 1)
        self.assertEqual(summary['chunks_with_history_hits'], 1)
        self.assertEqual(summary['history_hit_count'], 2)
        self.assertEqual(summary['history_hit_rate'], 0.5)
        self.assertEqual(summary['history_retrieval_errors'], 1)

    def test_summarize_batch_story_memory_reports_diagnostics(self):
        summary = batch_mod.summarize_batch_story_memory(
            [
                {
                    'story_hits': {
                        'characters': [{'id': 'eileen'}],
                        'relations': [{'left': 'eileen', 'right': 'noah'}],
                        'terms': [{'source': 'Void Gate', 'target': '\u865a\u7a7a\u95e8'}],
                        'scenes': [{'file_rel_path': 'chapter1.rpy'}],
                    },
                },
                {
                    'story_hits': {
                        'terms': [{'source': 'Aether', 'note': 'x' * 80}],
                    },
                },
                {},
            ],
            graph_file='logs/story_memory/story_graph.json',
            max_context_chars=30,
        )

        self.assertEqual(summary['graph_file'], 'logs/story_memory/story_graph.json')
        self.assertEqual(summary['chunks_with_story_hits'], 2)
        self.assertEqual(summary['story_hit_rate'], 2 / 3)
        self.assertEqual(
            summary['hit_counts'],
            {
                'characters': 1,
                'relations': 1,
                'terms': 2,
                'scenes': 1,
            },
        )
        self.assertEqual(summary['total_hit_count'], 5)
        self.assertGreaterEqual(summary['truncated_story_blocks'], 1)
        self.assertGreater(summary['formatted_char_count'], 0)
        self.assertLessEqual(summary['formatted_char_count'], 60)

    def test_summarize_batch_story_memory_uses_default_limit_for_bad_context_chars(self):
        old_limit = batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS
        try:
            batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = 200
            summary = batch_mod.summarize_batch_story_memory(
                [
                    {
                        'story_hits': {
                            'terms': [{'source': 'Aether', 'target': '\u4ee5\u592a'}],
                        },
                    },
                ],
                max_context_chars={'bad': 'shape'},
            )
        finally:
            batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = old_limit

        self.assertEqual(summary['chunks_with_story_hits'], 1)
        self.assertEqual(summary['truncated_story_blocks'], 0)
        self.assertGreater(summary['formatted_char_count'], 1)

    def test_sync_rag_store_reuses_embedding_when_only_translation_changes(self):
        old_values = {
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store_dir': batch_mod.RAG_STORE_DIR,
            'rag_store': batch_mod._RAG_STORE,
            'rag_dim': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'rag_segment_lines': batch_mod.RAG_SEGMENT_LINES,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                store_dir = root / 'rag_store'
                target_file = root / 'script.rpy'
                target_file.write_text('old "Aether Gate"\nnew "\u65e7\u8bd1"\n', encoding='utf-8')
                batch_mod.RAG_ENABLED = True
                batch_mod.RAG_STORE_DIR = str(store_dir)
                batch_mod.RAG_OUTPUT_DIMENSIONALITY = 3
                batch_mod.RAG_SEGMENT_LINES = 4
                batch_mod._RAG_STORE = None
                file_jobs = [{'file_rel_path': 'script.rpy', 'file_path': str(target_file)}]

                with mock.patch.object(batch_mod, 'embed_texts', return_value=[[1.0, 0.0, 0.0]]) as embed_mock:
                    first_summary = batch_mod.sync_rag_store_for_jobs(file_jobs)

                target_file.write_text('old "Aether Gate"\nnew "\u65b0\u8bd1"\n', encoding='utf-8')
                with mock.patch.object(batch_mod, 'embed_texts') as embed_mock_second:
                    second_summary = batch_mod.sync_rag_store_for_jobs(file_jobs)

                with mock.patch.object(batch_mod, 'embed_texts') as embed_mock_third:
                    third_summary = batch_mod.sync_rag_store_for_jobs(file_jobs)

                store = batch_mod.get_rag_store()
                records = list(store.history.values())

            embed_mock.assert_called_once()
            embed_mock_second.assert_not_called()
            embed_mock_third.assert_not_called()
            self.assertEqual(first_summary['embedded'], 1)
            self.assertEqual(second_summary['embedding_pending'], 0)
            self.assertEqual(second_summary['reused_embeddings'], 1)
            self.assertEqual(second_summary['upserted'], 1)
            self.assertEqual(third_summary['pending'], 0)
            self.assertEqual(third_summary['embedded'], 0)
            self.assertEqual(records[0]['translated_text'], '\u65b0\u8bd1')
            self.assertEqual(records[0]['embedding'], [1.0, 0.0, 0.0])
        finally:
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.RAG_STORE_DIR = old_values['rag_store_dir']
            batch_mod._RAG_STORE = old_values['rag_store']
            batch_mod.RAG_OUTPUT_DIMENSIONALITY = old_values['rag_dim']
            batch_mod.RAG_SEGMENT_LINES = old_values['rag_segment_lines']


class BatchRepairRegressionTests(unittest.TestCase):
    def test_split_manifest_keeps_first_child_latest_and_context_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            input_path = package_dir / 'requests.jsonl'
            manifest_path = package_dir / 'manifest.json'
            latest_path = root / 'latest_manifest.txt'
            chunks = [
                {
                    'key': 'chunk-1',
                    'file_rel_path': 'script.rpy',
                    'file_path': str(root / 'script.rpy'),
                    'items': [{'id': 'script.rpy:0:4', 'text': 'Hello'}],
                    'glossary_hits': [{'source': 'Hello', 'target': '\u4f60\u597d'}],
                    'history_hits': [
                        {'source_text': 'Hello', 'translated_text': '\u4f60\u597d'},
                        {'source_text': 'Hi', 'translated_text': '\u55e8'},
                    ],
                    'story_hits': {'terms': [{'source': 'Void Gate', 'target': '\u865a\u7a7a\u95e8'}]},
                },
                {
                    'key': 'chunk-2',
                    'file_rel_path': 'script.rpy',
                    'file_path': str(root / 'script.rpy'),
                    'items': [{'id': 'script.rpy:1:4', 'text': 'World'}],
                    'history_hits': [{'source_text': 'World', 'translated_text': '\u4e16\u754c'}],
                    'rag_stats': {'error': 'embedding failed'},
                },
            ]
            input_path.write_text(
                json.dumps({'key': 'chunk-1', 'request': {}}, ensure_ascii=False) + '\n' +
                json.dumps({'key': 'chunk-2', 'request': {}}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'display_name': 'demo',
                        'batch_model': 'gemini-test',
                        'input_jsonl_path': str(input_path),
                        'settings': {'target_size': 1},
                        'rag_enabled': True,
                        'rag_store_path': str(root / 'rag_store'),
                        'rag_settings': {'top_k_history': 4},
                        'rag_summary': {
                            'prepare': {'upserted': 3},
                            'chunks_with_history_hits': 2,
                            'history_hit_count': 3,
                            'history_hit_rate': 1.0,
                            'history_retrieval_errors': 1,
                        },
                        'story_memory_enabled': True,
                        'story_memory_graph_file': str(root / 'story_graph.json'),
                        'story_memory_settings': {'top_k_terms': 8},
                        'story_memory_summary': {'chunks_with_story_hits': 1},
                        'chunks': chunks,
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)):
                created = batch_mod.split_manifest(str(manifest_path), max_chunks=1)

            self.assertEqual(latest_path.read_text(encoding='utf-8'), created[0])
            source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(source_manifest['job_state'], 'LOCAL_SPLIT_SOURCE')
            first_child = json.loads(Path(created[0]).read_text(encoding='utf-8'))
            self.assertTrue(first_child['rag_enabled'])
            self.assertEqual(first_child['rag_settings'], {'top_k_history': 4})
            self.assertTrue(first_child['story_memory_enabled'])
            self.assertEqual(first_child['story_memory_settings'], {'top_k_terms': 8})
            self.assertEqual(first_child['rag_summary']['chunks_with_history_hits'], 1)
            self.assertEqual(first_child['rag_summary']['chunks_with_glossary_hits'], 1)
            self.assertEqual(first_child['rag_summary']['history_hit_count'], 2)
            self.assertEqual(first_child['rag_summary']['history_hit_rate'], 1.0)
            self.assertEqual(first_child['rag_summary']['history_retrieval_errors'], 0)
            self.assertEqual(first_child['rag_summary']['prepare'], {'upserted': 3})
            self.assertEqual(first_child['story_memory_summary']['chunks_with_story_hits'], 1)
            self.assertEqual(first_child['story_memory_summary']['graph_file'], str(root / 'story_graph.json'))
            self.assertEqual(first_child['story_memory_summary']['hit_counts']['terms'], 1)
            self.assertEqual(first_child['story_memory_summary']['total_hit_count'], 1)
            self.assertEqual(first_child['story_memory_summary']['story_hit_rate'], 1.0)
            self.assertEqual(first_child['story_memory_summary']['truncated_story_blocks'], 0)
            self.assertGreater(first_child['story_memory_summary']['formatted_char_count'], 0)

    def test_collect_result_actions_ignores_duplicate_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            response_text = json.dumps(
                [
                    {'id': 'script.rpy:0:4', 'translation': '\u4f60\u597d'},
                    {'id': 'script.rpy:0:4', 'translation': '\u518d\u89c1'},
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(result_path),
                'chunks': [
                    {
                        'key': 'chunk-1',
                        'file_rel_path': 'script.rpy',
                        'items': [
                            {
                                'id': 'script.rpy:0:4',
                                'line': 0,
                                'start': 4,
                                'end': 11,
                                'text': 'Hello',
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    }
                ],
            }

            replacements, _translated, failures, summary = batch_mod.collect_result_actions(manifest)

        self.assertEqual(summary['valid_items'], 1)
        self.assertEqual(summary['reason_counts']['duplicate_result_id'], 1)
        self.assertEqual(len(replacements['script.rpy'][0]), 1)
        self.assertEqual(failures, [])

    def test_collect_result_actions_skips_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            line = '    e "Hallo"\n'
            start = line.index('"Hallo"')
            end = start + len('"Hallo"')
            target_file.write_text(line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(result_path),
                'files': {'script.rpy': {'path': str(target_file)}},
                'chunks': [
                    {
                        'key': 'chunk-1',
                        'file_rel_path': 'script.rpy',
                        'items': [
                            {
                                'id': f'script.rpy:0:{start}',
                                'line': 0,
                                'start': start,
                                'end': end,
                                'text': 'Hello',
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    }
                ],
            }

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                replacements, translated, failures, summary = batch_mod.collect_result_actions(
                    manifest,
                    validate_sources=True,
                )

        self.assertEqual(replacements, {})
        self.assertEqual(translated, {})
        self.assertEqual(summary['candidate_valid_items'], 1)
        self.assertEqual(summary['valid_items'], 0)
        self.assertEqual(summary['skipped_items'], 1)
        self.assertEqual(summary['source_mismatch_items'], 1)
        self.assertEqual(summary['pending_files'], 0)
        self.assertEqual(summary['pending_lines'], 0)
        self.assertEqual(summary['reason_counts']['source_text_mismatch'], 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['error'], 'Source text mismatch during source validation')
        self.assertEqual(failures[0]['current_text'], 'Hallo')

    def test_apply_results_handles_multiple_replacements_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            dialogue_line = '    call screen test("Hello", "World")\n'
            hello_start = dialogue_line.index('"Hello"')
            hello_end = hello_start + len('"Hello"')
            world_start = dialogue_line.index('"World"')
            world_end = world_start + len('"World"')
            target_file.write_text('label test:\n' + dialogue_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [
                    {'id': f'script.rpy:1:{hello_start}', 'translation': '\u4f60\u597d'},
                    {'id': f'script.rpy:1:{world_start}', 'translation': '\u4e16\u754c'},
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:1:{hello_start}',
                                        'line': 1,
                                        'start': hello_start,
                                        'end': hello_end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                    {
                                        'id': f'script.rpy:1:{world_start}',
                                        'line': 1,
                                        'start': world_start,
                                        'end': world_end,
                                        'text': 'World',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                manifest = batch_mod.apply_results(str(manifest_path))

            updated_script = target_file.read_text(encoding='utf-8')
            saved_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        self.assertIn('call screen test("\u4f60\u597d", "\u4e16\u754c")', updated_script)
        update_progress.assert_called_once_with('script.rpy', [1])
        self.assertEqual(manifest['apply_summary']['applied_files'], 1)
        self.assertEqual(manifest['apply_summary']['applied_lines'], 1)
        self.assertEqual(manifest['apply_summary']['recoverable_items'], 2)
        self.assertEqual(manifest['apply_summary']['skipped_items'], 0)
        self.assertIn('applied_at', saved_manifest)

    def test_apply_results_revalidates_snapshot_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            original_line = '    e "Hello"\n'
            changed_line = '    e "Hallo"\n'
            start = original_line.index('"Hello"')
            end = start + len('"Hello"')
            target_file.write_text(original_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{start}',
                                        'line': 0,
                                        'start': start,
                                        'end': end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            collect_result_actions = batch_mod.collect_result_actions

            def mutate_after_initial_validation(*args, **kwargs):
                result = collect_result_actions(*args, **kwargs)
                target_file.write_text(changed_line, encoding='utf-8')
                return result

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'collect_result_actions', side_effect=mutate_after_initial_validation),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
                mock.patch.object(batch_mod, 'append_failure_entries') as append_failures,
            ):
                manifest = batch_mod.apply_results(str(manifest_path))

            final_script = target_file.read_text(encoding='utf-8')

        self.assertEqual(final_script, changed_line)
        update_progress.assert_not_called()
        append_failures.assert_called_once()
        self.assertEqual(manifest['apply_summary']['applied_files'], 0)
        self.assertEqual(manifest['apply_summary']['recoverable_items'], 0)
        self.assertEqual(manifest['apply_summary']['skipped_items'], 1)
        self.assertEqual(manifest['apply_summary']['source_mismatch_items'], 1)

    def test_apply_results_rejects_already_applied_manifest_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest_path = package_dir / 'manifest.json'
            manifest_path.write_text(
                json.dumps({'applied_at': '2026-05-12T12:00:00'}, ensure_ascii=False),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(SystemExit, 'already applied'):
                batch_mod.apply_results(str(manifest_path))

    def test_apply_results_force_keeps_source_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            line = '    e "Hallo"\n'
            start = line.index('"Hallo"')
            end = start + len('"Hallo"')
            target_file.write_text(line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'applied_at': '2026-05-12T12:00:00',
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{start}',
                                        'line': 0,
                                        'start': start,
                                        'end': end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'append_failure_entries') as append_failures,
            ):
                manifest = batch_mod.apply_results(str(manifest_path), force=True)

            unchanged_script = target_file.read_text(encoding='utf-8')

        self.assertEqual(unchanged_script, line)
        self.assertEqual(manifest['apply_summary']['applied_files'], 0)
        self.assertEqual(manifest['apply_summary']['recoverable_items'], 0)
        self.assertEqual(manifest['apply_summary']['skipped_items'], 1)
        self.assertEqual(manifest['apply_summary']['source_mismatch_items'], 1)
        append_failures.assert_called_once()

    def test_manifest_result_path_must_stay_in_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(root / 'outside-results.jsonl'),
            }

            with self.assertRaisesRegex(SystemExit, 'escapes'):
                batch_mod.resolve_manifest_result_path(manifest)

    def test_manifest_result_path_rejects_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / 'package'
            package_dir.mkdir()
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': '../outside-results.jsonl',
            }

            with self.assertRaisesRegex(SystemExit, 'parent directory'):
                batch_mod.resolve_manifest_result_path(manifest)

    def test_apply_results_rejects_manifest_file_path_outside_tl_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': 'script.rpy:0:4', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(root / 'outside.rpy')}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': 'script.rpy:0:4',
                                        'line': 0,
                                        'start': 4,
                                        'end': 11,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'escapes'):
                    batch_mod.apply_results(str(manifest_path))

    def test_load_repair_report_items_accepts_batch_failure_log_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    pass\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 0,
                    'text': 'Hello',
                    'error': 'Validation failed',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['file'], str(target_file.resolve()))
        self.assertEqual(items[0]['source'], 'Hello')
        self.assertEqual(items[0]['line'], 1)

    def test_load_repair_report_items_rejects_file_outside_tl_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            tl_dir.mkdir()
            outside_file = root / 'outside.rpy'
            outside_file.write_text('label outside:\n    pass\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file': str(outside_file),
                    'line': 1,
                    'source': 'Hello',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'escapes'):
                    batch_mod.load_repair_report_items(str(report_path))

    def test_load_repair_report_items_rejects_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': '../outside.rpy',
                    'line': 0,
                    'text': 'Hello',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'parent directory'):
                    batch_mod.load_repair_report_items(str(report_path))

    def test_load_repair_report_items_distinguishes_start_zero_from_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    "Menu"\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Menu',
                }, ensure_ascii=False) + '\n' +
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Menu',
                    'start': 0,
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].get('start'), None)
        self.assertEqual(items[1]['start'], 0)

    def test_repair_jobs_keep_multiple_items_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text(
                'label test:\n'
                '    call screen test("Hello", "World")\n',
                encoding='utf-8',
            )
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Hello',
                    'id': 'script.rpy:1:21',
                }, ensure_ascii=False) + '\n' +
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'World',
                    'id': 'script.rpy:1:30',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))
                jobs, unresolved = batch_mod.build_repair_jobs(items, batch_size=2)

        self.assertEqual([item['source'] for item in items], ['Hello', 'World'])
        self.assertEqual(unresolved, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual([item['text'] for item in jobs[0]['items']], ['Hello', 'World'])
        self.assertEqual(len({item['id'] for item in jobs[0]['items']}), 2)

    def test_repair_jobs_parse_line_start_end_repair_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            duplicate_line = '    call screen test("Menu", "Menu")\n'
            second_start = duplicate_line.rindex('"Menu"')
            second_end = second_start + len('"Menu"')
            target_file.write_text(
                'label test:\n' + duplicate_line,
                encoding='utf-8',
            )
            report_path = tl_dir / 'repair_failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file': str(target_file.resolve()),
                    'line': 2,
                    'source': 'Menu',
                    'id': f'{target_file.resolve()}:2:{second_start}:{second_end}',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))
                jobs, unresolved = batch_mod.build_repair_jobs(items, batch_size=2)

        self.assertEqual(unresolved, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['items'][0]['start'], second_start)


if __name__ == '__main__':
    unittest.main()
