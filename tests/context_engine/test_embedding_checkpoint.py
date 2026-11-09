"""Recovery of durable batches without a provider or a published store."""
import json
from pathlib import Path
import tempfile
import unittest

from context_engine.embedding_checkpoint import EmbeddingCheckpoint


class EmbeddingCheckpointTests(unittest.TestCase):
    def test_incomplete_or_corrupted_suffix_keeps_only_verified_batches(self):
        for suffix in (b'{"entries":', b'{}\n', b'{"entries":[],"sha256":"bad"}\n'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'checkpoint.jsonl'
                checkpoint = EmbeddingCheckpoint(path, 'source-and-model', 3)
                checkpoint.save([{'id': 'one', 'fingerprint': 'current', 'embedding': [1., .5, .25]}])
                committed = path.read_bytes()
                with path.open('ab') as handle:
                    handle.write(suffix)
                recovered = EmbeddingCheckpoint(path, 'source-and-model', 3)
                self.assertEqual(recovered.get('one', 'current'), [1., .5, .25])
                self.assertIsNone(recovered.get('one', 'changed-record'))
                self.assertEqual(path.read_bytes(), committed)
                recovered.save([{'id': 'two', 'fingerprint': 'current', 'embedding': [.5, .5, 1.]}])
                self.assertEqual(len(EmbeddingCheckpoint(path, 'source-and-model', 3).entries), 2)

    def test_mismatched_inputs_or_dimension_discard_old_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.jsonl'
            for binding, dimension in (('new-source', 3), ('source-and-model', 4)):
                checkpoint = EmbeddingCheckpoint(path, 'source-and-model', 3)
                checkpoint.save([{'id': 'one', 'fingerprint': 'current', 'embedding': [1., .5, .25]}])
                recovered = EmbeddingCheckpoint(path, binding, dimension)
                self.assertFalse(recovered.entries)
                self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 1)

    def test_invalid_vectors_and_tampered_batches_are_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.jsonl'
            checkpoint = EmbeddingCheckpoint(path, 'source-and-model', 3)
            for vector in ([1., 2.], [True, 1., 2.], [float('nan'), 1., 2.], ['1', 1., 2.]):
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    checkpoint.save([{'id': 'one', 'fingerprint': 'current', 'embedding': vector}])
            checkpoint.save([{'id': 'one', 'fingerprint': 'current', 'embedding': [1., .5, .25]}])
            lines = path.read_text(encoding='utf-8').splitlines()
            payload = json.loads(lines[1])
            payload['entries'][0]['embedding'][0] = 2.
            path.write_text(lines[0] + '\n' + json.dumps(payload) + '\n', encoding='utf-8')
            self.assertFalse(EmbeddingCheckpoint(path, 'source-and-model', 3).entries)
