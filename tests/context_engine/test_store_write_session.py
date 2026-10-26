"""Preparation keeps the shared JSON-store writer exclusion across steps."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from context_engine.rag_memory import (
    JsonRagStore, JsonRagStoreLockError, JsonSourceIndexStore, JsonSourceIndexStoreLockError,
)


class StoreWriteSessionTests(unittest.TestCase):
    def test_nested_writes_exclude_other_instances_and_threads_and_release_on_failure(self):
        for store_type, lock_error in ((JsonRagStore, JsonRagStoreLockError),
                                       (JsonSourceIndexStore, JsonSourceIndexStoreLockError)):
            with self.subTest(store=store_type.__name__), tempfile.TemporaryDirectory() as root:
                store = store_type(root)
                with self.assertRaisesRegex(RuntimeError, 'original interrupted preparation'):
                    with store.write_session('original_fixture'):
                        store.set_metadata(mv_preparation={'status': 'building'})
                        with store.write_session('nested_fixture'):
                            store.set_metadata(original_counter=1)
                        with self.assertRaises(lock_error):
                            store_type(root).set_metadata(original_counter=2)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            with self.assertRaises(lock_error):
                                executor.submit(store.set_metadata, original_counter=3).result()
                        self.assertTrue(Path(store.lock_path).is_file())
                        raise RuntimeError('original interrupted preparation')
                self.assertFalse(Path(store.lock_path).exists())
                store_type(root).set_metadata(original_counter=4)
                reopened = store_type(root)
                reopened.load()
                self.assertEqual(reopened.metadata['original_counter'], 4)
