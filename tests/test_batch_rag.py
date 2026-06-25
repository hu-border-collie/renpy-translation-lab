import ast
import hashlib
import importlib
import io
import json
import os
import pickle
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



class BatchRagRegressionTests(unittest.TestCase):
    def test_empty_thinking_level_omits_thinking_config_and_warning(self):
        old_values = {
            'model': batch_mod.BATCH_MODEL,
            'thinking_level': batch_mod.BATCH_THINKING_LEVEL,
        }
        try:
            batch_mod.BATCH_MODEL = 'gemini-3.1-flash-lite'
            batch_mod.BATCH_THINKING_LEVEL = ''

            config = batch_mod.build_generation_config([
                {
                    'id': 'script.rpy:1:0',
                    'text': 'Hello',
                }
            ])
            warnings = batch_mod.get_batch_risk_warnings()
        finally:
            batch_mod.BATCH_MODEL = old_values['model']
            batch_mod.BATCH_THINKING_LEVEL = old_values['thinking_level']

        self.assertNotIn('thinking_config', config)
        self.assertFalse(any('thinking_level' in warning for warning in warnings))

    def test_build_chunks_keeps_context_task_dicts(self):
        old_values = {
            'target_size': batch_mod.BATCH_TARGET_SIZE,
            'target_chars': batch_mod.BATCH_TARGET_CHARS,
            'context_before': batch_mod.BATCH_CONTEXT_BEFORE,
            'context_after': batch_mod.BATCH_CONTEXT_AFTER,
            'rag_enabled': batch_mod.RAG_ENABLED,
            'story_enabled': batch_mod.STORY_MEMORY_ENABLED,
        }
        try:
            batch_mod.BATCH_TARGET_SIZE = 1
            batch_mod.BATCH_CONTEXT_BEFORE = 1
            batch_mod.BATCH_CONTEXT_AFTER = 1
            batch_mod.RAG_ENABLED = False
            batch_mod.STORY_MEMORY_ENABLED = False

            chunks = batch_mod.build_chunks([
                {
                    'file_rel_path': 'script.rpy',
                    'file_path': 'script.rpy',
                    'tasks': [
                        {
                            'id': 'script.rpy:1:0',
                            'text': 'Before line',
                            'line': 1,
                            'start': 0,
                            'end': 11,
                            'speaker_id': 'e',
                            'speaker_name': 'Eileen',
                            'progress_entry': 'task:1:0',
                        },
                        {
                            'id': 'script.rpy:2:0',
                            'text': 'Target line',
                            'line': 2,
                            'start': 0,
                            'end': 11,
                            'speaker_id': 'n',
                            'speaker_name': 'Noah',
                            'progress_entry': 'task:2:0',
                        },
                        {
                            'id': 'script.rpy:3:0',
                            'text': 'After line',
                            'line': 3,
                            'start': 0,
                            'end': 10,
                            'speaker_id': 'e',
                            'speaker_name': 'Eileen',
                            'progress_entry': 'task:3:0',
                        },
                    ],
                }
            ])
        finally:
            batch_mod.BATCH_TARGET_SIZE = old_values['target_size']
            batch_mod.BATCH_TARGET_CHARS = old_values['target_chars']
            batch_mod.BATCH_CONTEXT_BEFORE = old_values['context_before']
            batch_mod.BATCH_CONTEXT_AFTER = old_values['context_after']
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.STORY_MEMORY_ENABLED = old_values['story_enabled']

        self.assertEqual(chunks[1]['context_past'][0]['speaker_name'], 'Eileen')
        self.assertEqual(chunks[1]['context_past'][0]['progress_entry'], 'task:1:0')
        self.assertEqual(chunks[1]['context_future'][0]['speaker_name'], 'Eileen')

    def test_build_chunks_respects_source_char_limit(self):
        old_values = {
            'target_size': batch_mod.BATCH_TARGET_SIZE,
            'target_chars': batch_mod.BATCH_TARGET_CHARS,
            'context_before': batch_mod.BATCH_CONTEXT_BEFORE,
            'context_after': batch_mod.BATCH_CONTEXT_AFTER,
            'rag_enabled': batch_mod.RAG_ENABLED,
            'story_enabled': batch_mod.STORY_MEMORY_ENABLED,
        }
        try:
            batch_mod.BATCH_TARGET_SIZE = 20
            batch_mod.BATCH_TARGET_CHARS = 10
            batch_mod.BATCH_CONTEXT_BEFORE = 1
            batch_mod.BATCH_CONTEXT_AFTER = 1
            batch_mod.RAG_ENABLED = False
            batch_mod.STORY_MEMORY_ENABLED = False

            tasks = []
            for index, text in enumerate(['x' * 12, 'short', 'small', 'tiny']):
                tasks.append({
                    'id': f'script.rpy:{index}:0',
                    'text': text,
                    'line': index,
                    'start': 0,
                    'end': len(text),
                })

            chunks = batch_mod.build_chunks([{
                'file_rel_path': 'script.rpy',
                'file_path': 'script.rpy',
                'tasks': tasks,
            }])
        finally:
            batch_mod.BATCH_TARGET_SIZE = old_values['target_size']
            batch_mod.BATCH_TARGET_CHARS = old_values['target_chars']
            batch_mod.BATCH_CONTEXT_BEFORE = old_values['context_before']
            batch_mod.BATCH_CONTEXT_AFTER = old_values['context_after']
            batch_mod.RAG_ENABLED = old_values['rag_enabled']
            batch_mod.STORY_MEMORY_ENABLED = old_values['story_enabled']

        self.assertEqual([len(chunk['items']) for chunk in chunks], [1, 2, 1])
        self.assertEqual([chunk['source_char_count'] for chunk in chunks], [12, 10, 4])
        self.assertEqual(chunks[0]['items'][0]['text'], 'x' * 12)

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

    def test_build_arg_parser_accepts_doctor_command(self):
        args = batch_mod.build_arg_parser().parse_args(['doctor'])

        self.assertEqual(args.command, 'doctor')

    def test_build_arg_parser_accepts_bootstrap_work_command(self):
        args = batch_mod.build_arg_parser().parse_args(['bootstrap-work'])

        self.assertEqual(args.command, 'bootstrap-work')
        self.assertFalse(args.no_update_game_root)

    def test_doctor_report_recommends_bootstrap_work_when_work_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'Game_Example'
            original_game = project / 'original' / 'game'
            original_game.mkdir(parents=True)
            (original_game / 'script.rpy').write_text('label start:\n', encoding='utf-8')

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(project)),
                mock.patch.object(batch_mod.legacy, 'WORK_GAME_DIR', str(project / 'game')),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(project / 'game' / 'tl' / 'schinese')),
                mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/schinese'),
                mock.patch.object(batch_mod.legacy, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'schinese'),
                mock.patch.object(batch_mod.legacy, 'PREP_ENABLED', True),
                mock.patch.object(batch_mod.legacy, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', None),
                mock.patch.object(batch_mod.legacy, 'resolve_original_game_dir', return_value=str(original_game)),
                mock.patch.object(batch_mod.legacy, 'work_dir_bootstrap_allowed', return_value=(True, str(project / 'work'), '')),
            ):
                report = batch_mod.collect_doctor_report()

        joined = ' '.join(report.get('recommendations', []))
        self.assertIn('bootstrap-work', joined)

    def test_doctor_report_classifies_existing_tl_without_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'script.rpy').write_text(
                'translate schinese start_123:\n'
                '    # e "Hello"\n'
                '    e "\u4f60\u597d"\n'
                '\n'
                'translate schinese strings:\n'
                '    old "Start"\n'
                '    new "\u5f00\u59cb"\n',
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(base)),
                mock.patch.object(batch_mod.legacy, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod.legacy, 'TL_SUBDIR', 'game/tl/schinese'),
                mock.patch.object(batch_mod.legacy, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LANGUAGE', 'schinese'),
                mock.patch.object(batch_mod.legacy, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', None),
            ):
                report = batch_mod.collect_doctor_report()

        self.assertEqual(report['mode'], 'existing_tl_only')
        self.assertFalse(report['can_generate_template'])
        self.assertEqual(report['counts']['rpy_files'], 1)
        self.assertEqual(report['counts']['translate_blocks'], 1)
        self.assertEqual(report['counts']['string_sections'], 1)
        self.assertEqual(report['counts']['old_lines'], 1)
        self.assertEqual(report['counts']['new_lines'], 1)
        self.assertEqual(report['counts']['commented_original_lines'], 1)

    def test_doctor_report_warns_when_dialogue_comments_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'script.rpy').write_text(
                'translate schinese start_123:\n'
                '    e "\u4f60\u597d"\n',
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(base)),
                mock.patch.object(batch_mod.legacy, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod.legacy, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', None),
            ):
                report = batch_mod.collect_doctor_report()

        self.assertIn('Dialogue translation blocks do not include source comments', ' '.join(report['warnings']))

    def test_doctor_report_does_not_treat_translate_python_as_dialogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'
            tl_dir = base / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            (tl_dir / 'style.rpy').write_text(
                'translate schinese python:\n'
                '    gui.text_font = "SourceHanSansLite.ttf"\n',
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(base)),
                mock.patch.object(batch_mod.legacy, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod.legacy, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', None),
            ):
                report = batch_mod.collect_doctor_report()

        self.assertEqual(report['counts']['translate_blocks'], 0)
        self.assertNotIn('Dialogue translation blocks do not include source comments', ' '.join(report['warnings']))

    def test_doctor_command_does_not_require_api_keys(self):
        with (
            mock.patch.object(batch_mod, 'initialize_batch_logging') as logging_mock,
            mock.patch.object(batch_mod.legacy, 'load_translator_settings'),
            mock.patch.object(batch_mod.legacy, 'load_glossary'),
            mock.patch.object(batch_mod, 'load_batch_settings'),
            mock.patch.object(batch_mod, 'print_banner'),
            mock.patch.object(batch_mod, 'collect_doctor_report', return_value={'counts': {}, 'warnings': []}),
            mock.patch.object(batch_mod, 'print_doctor_report'),
            mock.patch.object(batch_mod.legacy, 'load_config') as load_config_mock,
        ):
            batch_mod.main(['doctor'])

        load_config_mock.assert_not_called()
        logging_mock.assert_not_called()

    def test_doctor_report_explains_custom_template_command_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'work'

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(base)),
                mock.patch.object(batch_mod.legacy, 'WORK_GAME_DIR', str(base / 'game')),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(base / 'game' / 'tl' / 'schinese')),
                mock.patch.object(batch_mod.legacy, 'SOURCE_GAME_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_RENPY_SDK_DIR', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_LAUNCHER_PY', ''),
                mock.patch.object(batch_mod.legacy, 'PREP_TEMPLATE_COMMAND', ['{missing_placeholder}']),
            ):
                report = batch_mod.collect_doctor_report()

        joined_warnings = ' '.join(report['warnings'])
        self.assertFalse(report['can_generate_template'])
        self.assertEqual(report['template_command_kind'], 'custom')
        self.assertIn('missing_placeholder', report['template_reason'])
        self.assertIn('Custom template command cannot be rendered', joined_warnings)

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



if __name__ == '__main__':
    unittest.main()
