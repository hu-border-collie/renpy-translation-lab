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

    def test_json_rag_store_ranks_revision_applied_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = rag_memory.JsonRagStore(tmp)
            unknown_record = self.make_record('unknown')
            unknown_record['quality_state'] = 'unknown'
            revision_record = self.make_record('revision')
            revision_record['quality_state'] = 'revision_applied'

            store.upsert_history([unknown_record, revision_record])
            results = store.search_history([1.0, 0.0, 0.0], top_k=2, min_similarity=0.0)

        self.assertEqual(results[0]['quality_state'], 'revision_applied')

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



if __name__ == '__main__':
    unittest.main()
