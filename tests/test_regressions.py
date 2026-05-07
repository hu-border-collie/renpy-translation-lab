import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import story_memory
import translator_runtime as runtime


class TranslatorRuntimeRegressionTests(unittest.TestCase):
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

    def test_story_memory_normalize_has_fast_path_for_normalized_graphs(self):
        normalized = story_memory.normalize_story_graph({'terms': {'Void Gate': '\u865a\u7a7a\u95e8'}})

        self.assertIs(story_memory.normalize_story_graph(normalized), normalized)
        self.assertIn('Void Gate', normalized['terms'][0]['source'])
        self.assertNotIn('_story_memory_normalized', json.dumps(normalized))

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


class BatchRepairRegressionTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
