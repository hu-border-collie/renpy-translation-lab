import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import translator_runtime as runtime


class TranslatorRuntimeRegressionTests(unittest.TestCase):
    def test_collect_tasks_keeps_distinct_entries_on_same_line(self):
        tasks = runtime.collect_tasks(['call screen test("Hello", "World")\n'])
        self.assertEqual(len(tasks), 2)
        self.assertNotEqual(tasks[0]['start'], tasks[1]['start'])
        self.assertNotEqual(tasks[0]['progress_entry'], tasks[1]['progress_entry'])

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
