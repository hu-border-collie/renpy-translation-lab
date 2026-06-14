# -*- coding: utf-8 -*-
import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
from rag_memory import JsonSourceIndexStore, JsonSourceIndexStoreLockError, now_iso


class SourceIndexStoreTests(unittest.TestCase):
    def make_segment(self, source_id, file_rel_path='script.rpy', source='Hello', start=10, end=13):
        return {
            'source_id': source_id,
            'file_rel_path': file_rel_path,
            'line_start': start,
            'line_end': end,
            'line_span': [start, end],
            'source_text': source,
            'source_checksum': batch_mod.hash_text(source),
            'embedding': [0.1, 0.2, 0.3],
            'embedding_metadata': {
                'embedding_model': 'gemini-embedding-001',
                'embedding_task_type': 'RETRIEVAL_DOCUMENT',
                'embedding_dim': 3,
                'embedding_text_checksum': batch_mod.hash_text(source),
            },
            'created_at': '2026-06-14T09:00:00',
            'updated_at': '2026-06-14T09:00:00',
        }

    def test_json_store_lifecycle_and_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            metadata_path = store_dir / 'source_metadata.json'
            segments_path = store_dir / 'source_segments.jsonl'
            metadata_path.write_text('{bad json', encoding='utf-8')
            segments_path.write_text(
                json.dumps(self.make_segment('s1'), ensure_ascii=False) + '\n'
                '{bad row\n'
                '[]\n',
                encoding='utf-8',
            )

            store = JsonSourceIndexStore(str(store_dir))
            with mock.patch('builtins.print') as print_mock:
                count = store.count_segments()

            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertEqual(count, 1)
        self.assertIn('Failed to load source metadata', warnings)
        self.assertIn('Skipping invalid source segment row', warnings)
        self.assertIn('Skipping non-object source segment row', warnings)

    def test_json_store_atomic_write_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            segments_path = store_dir / 'source_segments.jsonl'
            original = self.make_segment('s1')
            segments_path.write_text(
                json.dumps(original, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            store = JsonSourceIndexStore(str(store_dir))

            with mock.patch.object(batch_mod.os, 'replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    store.upsert_segments([self.make_segment('s2')])

            persisted = segments_path.read_text(encoding='utf-8')
            temp_files = list(store_dir.glob('*.tmp.*'))

        self.assertEqual(persisted, json.dumps(original, ensure_ascii=False) + '\n')
        self.assertEqual(temp_files, [])

    def test_json_store_lock_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.source_index.lock'
            lock_path.write_text(
                json.dumps({'operation': 'upsert_segments', 'owner': 'test-host', 'pid': 123}),
                encoding='utf-8',
            )
            store = JsonSourceIndexStore(str(store_dir))

            with self.assertRaisesRegex(JsonSourceIndexStoreLockError, 'test-host'):
                store.upsert_segments([self.make_segment('s1')])

            self.assertFalse((store_dir / 'source_segments.jsonl').exists())

    def test_json_store_recovers_stale_lock_dead_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            lock_path = store_dir / '.source_index.lock'
            lock_path.write_text(
                json.dumps({
                    'operation': 'upsert_segments',
                    'owner': socket.gethostname(),
                    'pid': 987654,
                    'created_at': now_iso(),
                }),
                encoding='utf-8',
            )
            store = JsonSourceIndexStore(str(store_dir))

            with (
                mock.patch.object(JsonSourceIndexStore, '_is_lock_owner_alive', return_value=False),
                mock.patch('builtins.print') as print_mock,
            ):
                store.upsert_segments([self.make_segment('s1')])

            reloaded = JsonSourceIndexStore(str(store_dir))
            segment_ids = reloaded.segment_ids_for_file('script.rpy')
            warnings = '\n'.join(str(call.args[0]) for call in print_mock.call_args_list)

        self.assertEqual(segment_ids, ['s1'])
        self.assertIn('Recovered stale source store lock', warnings)

    def test_collect_source_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp) / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            rpy_path = tl_dir / 'script.rpy'
            rpy_content = (
                "# game/script.rpy:10\n"
                "translate schinese start_f3396d11:\n"
                "    # \"Hello world\"\n"
                "    \"你好世界\"\n"
                "\n"
                "# game/script.rpy:15\n"
                "translate schinese start_12345678:\n"
                "    # \"Goodbye world\"\n"
                "    \"再见世界\"\n"
            )
            rpy_path.write_text(rpy_content, encoding='utf-8')

            file_jobs = [{'file_rel_path': 'script.rpy', 'file_path': str(rpy_path)}]
            with mock.patch('gemini_translate_batch.RAG_SEGMENT_LINES', 1):
                segments = batch_mod.collect_source_segments_for_jobs(file_jobs)

            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]['source_text'], 'Hello world')
            self.assertEqual(segments[1]['source_text'], 'Goodbye world')
            self.assertEqual(segments[0]['line_start'], 4)
            self.assertEqual(segments[0]['line_end'], 4)
            self.assertEqual(segments[0]['line_span'], [4, 4])

    def test_bootstrap_source_index_sync_and_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / 'store'
            tl_dir = Path(tmp) / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)

            old_store_dir = batch_mod.SOURCE_INDEX_STORE_DIR
            old_tl_dir = batch_mod.legacy.TL_DIR
            try:
                batch_mod.SOURCE_INDEX_STORE_DIR = str(store_dir)
                batch_mod.legacy.TL_DIR = str(tl_dir)

                rpy_path = tl_dir / 'script.rpy'
                rpy_content = (
                    "# game/script.rpy:10\n"
                    "translate schinese start_f3396d11:\n"
                    "    # \"Hello world\"\n"
                    "    \"你好世界\"\n"
                )
                rpy_path.write_text(rpy_content, encoding='utf-8')

                store = JsonSourceIndexStore(str(store_dir))
                stale_record = self.make_segment('stale_id', file_rel_path='script.rpy', source='Old stale text', start=1, end=2)
                unchanged_id = batch_mod.hash_key('script.rpy:4:4')
                unchanged_record = self.make_segment(unchanged_id, file_rel_path='script.rpy', source='Hello world', start=4, end=4)
                unchanged_record['embedding'] = [0.9, 0.9, 0.9]
                unchanged_record['embedding_metadata']['embedding_model'] = batch_mod.RAG_EMBEDDING_MODEL

                store.upsert_segments([stale_record, unchanged_record])

                mock_embeddings = [[0.9, 0.9, 0.9]]
                with (
                    mock.patch('gemini_translate_batch.embed_texts', return_value=mock_embeddings) as embed_mock,
                    mock.patch('gemini_translate_batch.all_rag_file_jobs', return_value=[{'file_rel_path': 'script.rpy', 'file_path': str(rpy_path)}]),
                    mock.patch('gemini_translate_batch.RAG_OUTPUT_DIMENSIONALITY', 3),
                    mock.patch('gemini_translate_batch.RAG_SEGMENT_LINES', 1),
                ):
                    summary = batch_mod.bootstrap_source_index(skip_prepare=True, prune=True)

                embed_mock.assert_not_called()
                self.assertEqual(summary['scanned'], 1)
                self.assertEqual(summary['reused_embeddings'], 1)
                self.assertEqual(summary['embedding_pending'], 0)
                self.assertEqual(summary['stale_count'], 1)
                self.assertEqual(summary['pruned'], 1)

                reloaded = JsonSourceIndexStore(str(store_dir))
                reloaded.load()
                self.assertEqual(reloaded.count_segments(), 1)
                self.assertIn(unchanged_id, reloaded.segments)
                self.assertNotIn('stale_id', reloaded.segments)
            finally:
                batch_mod.SOURCE_INDEX_STORE_DIR = old_store_dir
                batch_mod.legacy.TL_DIR = old_tl_dir

    def test_bootstrap_source_index_reembeds_when_embedding_metadata_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / 'store'
            tl_dir = Path(tmp) / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)

            old_store_dir = batch_mod.SOURCE_INDEX_STORE_DIR
            old_tl_dir = batch_mod.legacy.TL_DIR
            try:
                batch_mod.SOURCE_INDEX_STORE_DIR = str(store_dir)
                batch_mod.legacy.TL_DIR = str(tl_dir)

                rpy_path = tl_dir / 'script.rpy'
                rpy_content = (
                    "# game/script.rpy:10\n"
                    "translate schinese start_f3396d11:\n"
                    "    # \"Hello world\"\n"
                    "    \"你好世界\"\n"
                )
                rpy_path.write_text(rpy_content, encoding='utf-8')

                source_id = batch_mod.hash_key('script.rpy:4:4')
                outdated_record = self.make_segment(source_id, file_rel_path='script.rpy', source='Hello world', start=4, end=4)
                outdated_record['embedding'] = [0.4, 0.4]
                outdated_record['embedding_metadata']['embedding_dim'] = 2
                outdated_record['embedding_metadata']['embedding_task_type'] = 'SEMANTIC_SIMILARITY'
                JsonSourceIndexStore(str(store_dir)).upsert_segments([outdated_record])

                mock_embeddings = [[0.8, 0.8, 0.8]]
                with (
                    mock.patch('gemini_translate_batch.embed_texts', return_value=mock_embeddings) as embed_mock,
                    mock.patch('gemini_translate_batch.all_rag_file_jobs', return_value=[{'file_rel_path': 'script.rpy', 'file_path': str(rpy_path)}]),
                    mock.patch('gemini_translate_batch.RAG_OUTPUT_DIMENSIONALITY', 3),
                    mock.patch('gemini_translate_batch.RAG_SEGMENT_LINES', 1),
                ):
                    summary = batch_mod.bootstrap_source_index(skip_prepare=True, prune=True)

                embed_mock.assert_called_once()
                self.assertEqual(summary['reused_embeddings'], 0)
                self.assertEqual(summary['embedding_pending'], 1)
                self.assertEqual(summary['embedded'], 1)

                reloaded = JsonSourceIndexStore(str(store_dir))
                reloaded.load()
                updated = reloaded.get_segment(source_id)
                self.assertEqual(updated['embedding'], mock_embeddings[0])
                self.assertEqual(updated['embedding_metadata']['embedding_dim'], 3)
                self.assertEqual(updated['embedding_metadata']['embedding_task_type'], batch_mod.RAG_DOCUMENT_TASK_TYPE)
            finally:
                batch_mod.SOURCE_INDEX_STORE_DIR = old_store_dir
                batch_mod.legacy.TL_DIR = old_tl_dir

    def test_bootstrap_source_index_without_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / 'store'
            tl_dir = Path(tmp) / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)

            old_store_dir = batch_mod.SOURCE_INDEX_STORE_DIR
            old_tl_dir = batch_mod.legacy.TL_DIR
            try:
                batch_mod.SOURCE_INDEX_STORE_DIR = str(store_dir)
                batch_mod.legacy.TL_DIR = str(tl_dir)

                rpy_path = tl_dir / 'script.rpy'
                rpy_content = (
                    "# game/script.rpy:10\n"
                    "translate schinese start_f3396d11:\n"
                    "    # \"Hello world\"\n"
                    "    \"你好世界\"\n"
                )
                rpy_path.write_text(rpy_content, encoding='utf-8')

                store = JsonSourceIndexStore(str(store_dir))
                stale_record = self.make_segment('stale_id', file_rel_path='script.rpy', source='Old stale text', start=1, end=2)
                store.upsert_segments([stale_record])

                mock_embeddings = [[0.1, 0.2, 0.3]]
                with (
                    mock.patch('gemini_translate_batch.embed_texts', return_value=mock_embeddings),
                    mock.patch('gemini_translate_batch.all_rag_file_jobs', return_value=[{'file_rel_path': 'script.rpy', 'file_path': str(rpy_path)}]),
                    mock.patch('gemini_translate_batch.RAG_OUTPUT_DIMENSIONALITY', 3),
                    mock.patch('gemini_translate_batch.RAG_SEGMENT_LINES', 1),
                ):
                    summary = batch_mod.bootstrap_source_index(skip_prepare=True, prune=False)

                self.assertEqual(summary['stale_count'], 1)
                self.assertEqual(summary['pruned'], 0)

                reloaded = JsonSourceIndexStore(str(store_dir))
                reloaded.load()
                self.assertEqual(reloaded.count_segments(), 2)
                self.assertIn('stale_id', reloaded.segments)
            finally:
                batch_mod.SOURCE_INDEX_STORE_DIR = old_store_dir
                batch_mod.legacy.TL_DIR = old_tl_dir


if __name__ == '__main__':
    unittest.main()
