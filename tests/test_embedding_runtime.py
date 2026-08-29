# -*- coding: utf-8 -*-
"""Unit tests for production embedding runtime wiring (issue #341)."""

import tempfile
import unittest
from types import SimpleNamespace

from embedding_backend import (
    EmbeddingBackendError,
    EmbeddingContractError,
    EmbeddingErrorCategory,
    EmbeddingTaskType,
)
from embedding_runtime import (
    BACKEND_GEMINI,
    BACKEND_OPENAI_COMPATIBLE,
    EmbeddingRuntimeError,
    build_embedding_adapter,
    embed_texts,
    ensure_store_document_identity,
    parse_embedding_runtime_settings,
    public_error_diagnostics,
    semantic_task_type,
)
from rag_memory import EmbeddingStoreIdentityError, JsonRagStore, JsonSourceIndexStore


class _GeminiModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class ParseSettingsTests(unittest.TestCase):
    def test_gemini_defaults_keep_historical_model(self):
        settings = parse_embedding_runtime_settings({})
        self.assertEqual(settings.backend, BACKEND_GEMINI)
        self.assertEqual(settings.provider, 'google_ai')
        self.assertEqual(settings.model, 'gemini-embedding-001')
        self.assertEqual(settings.output_dimension, 768)
        identity = settings.document_identity()
        self.assertEqual(identity.backend, 'google_genai')
        self.assertEqual(identity.task_type, EmbeddingTaskType.DOCUMENT)
        public = settings.public_dict()
        self.assertNotIn('api_key', public)
        self.assertNotIn('endpoint', public)
        self.assertIn('fingerprint', public)

    def test_openai_compatible_requires_explicit_model_and_provider(self):
        with self.assertRaises(EmbeddingContractError):
            parse_embedding_runtime_settings({'embedding_backend': 'openai_compatible'})
        with self.assertRaises(EmbeddingContractError):
            parse_embedding_runtime_settings(
                {
                    'embedding_backend': 'litellm',
                    'embedding_model': 'text-embedding-3-small',
                }
            )
        settings = parse_embedding_runtime_settings(
            {
                'embedding_backend': 'openai_compatible',
                'embedding_provider': 'openai',
                'embedding_model': 'text-embedding-3-small',
                'output_dimensionality': 1536,
            }
        )
        self.assertEqual(settings.backend, BACKEND_OPENAI_COMPATIBLE)
        self.assertEqual(settings.model, 'text-embedding-3-small')
        self.assertEqual(settings.output_dimension, 1536)

    def test_does_not_read_generation_model_fields(self):
        settings = parse_embedding_runtime_settings(
            {
                'model': 'gemini-3.1-flash-lite',
                'sync_model': 'should-not-win',
            }
        )
        self.assertEqual(settings.model, 'gemini-embedding-001')

    def test_rejects_unknown_backend_and_gemini_api_key_env(self):
        with self.assertRaises(EmbeddingContractError):
            parse_embedding_runtime_settings({'embedding_backend': 'claude'})
        with self.assertRaises(EmbeddingContractError):
            parse_embedding_runtime_settings(
                {
                    'embedding_backend': 'gemini',
                    'embedding_api_key_env': 'GEMINI_API_KEY',
                }
            )

    def test_semantic_task_type_maps_native_names(self):
        self.assertEqual(
            semantic_task_type('RETRIEVAL_DOCUMENT'),
            EmbeddingTaskType.DOCUMENT,
        )
        self.assertEqual(semantic_task_type('query'), EmbeddingTaskType.QUERY)


class AdapterEmbedTests(unittest.TestCase):
    def test_gemini_embed_uses_injected_client(self):
        settings = parse_embedding_runtime_settings(
            {'embedding_model': 'gemini-embedding-001', 'output_dimensionality': 2}
        )
        models = _GeminiModels(
            response=SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 0.0])],
            )
        )
        configs = []

        def config_factory(**kwargs):
            configs.append(kwargs)
            return SimpleNamespace(**kwargs)

        adapter = build_embedding_adapter(
            settings,
            gemini_client=SimpleNamespace(models=models),
            gemini_config_factory=config_factory,
        )
        vectors = embed_texts(adapter, ['hello'], 'RETRIEVAL_QUERY', timeout_seconds=5)
        self.assertEqual(vectors, [[1.0, 0.0]])
        self.assertEqual(models.calls[0]['model'], 'gemini-embedding-001')
        self.assertEqual(configs[0]['task_type'], 'RETRIEVAL_QUERY')
        self.assertEqual(configs[0]['output_dimensionality'], 2)

    def test_openai_compatible_embed_uses_injected_transport(self):
        settings = parse_embedding_runtime_settings(
            {
                'embedding_backend': 'openai_compatible',
                'embedding_provider': 'openai',
                'embedding_model': 'text-embedding-3-small',
                'output_dimensionality': 2,
            }
        )
        calls = []

        def transport(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=[0.0, 1.0])],
                usage=SimpleNamespace(prompt_tokens=3, total_tokens=3),
            )

        adapter = build_embedding_adapter(settings, openai_transport=transport, api_key='secret')
        vectors = embed_texts(adapter, ['hello'], EmbeddingTaskType.QUERY, timeout_seconds=9)
        self.assertEqual(vectors, [[0.0, 1.0]])
        self.assertEqual(calls[0]['api_base'], 'https://api.openai.com/v1')
        public = settings.public_dict()
        self.assertNotIn('secret', str(public))
        self.assertNotIn('api.openai.com', str(public))

    def test_public_error_diagnostics_hide_provider_text(self):
        error = EmbeddingBackendError(EmbeddingErrorCategory.AUTHENTICATION, retryable=False)
        payload = public_error_diagnostics(error)
        self.assertEqual(payload['error_category'], 'authentication')
        self.assertNotIn('secret', str(payload))
        leaked = public_error_diagnostics(RuntimeError('https://example.test/key=secret'))
        self.assertEqual(leaked['failure_reason'], 'retrieval_error')
        self.assertNotIn('secret', str(leaked))
        self.assertNotIn('example.test', str(leaked))


class StoreIdentityTests(unittest.TestCase):
    def settings(self):
        return parse_embedding_runtime_settings(
            {'embedding_model': 'gemini-embedding-001', 'output_dimensionality': 2}
        )

    def test_empty_store_accepts_identity_and_compatible_search(self):
        settings = self.settings()
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRagStore(tmp)
            result = ensure_store_document_identity(store, settings.document_identity())
            self.assertTrue(result['ready'])
            self.assertFalse(result['rebuilt'])
            store.upsert_history(
                [
                    {
                        'memory_id': 'm1',
                        'file_rel_path': 'a.rpy',
                        'source_text': 'Hello',
                        'translated_text': '你好',
                        'embedding': [1.0, 0.0],
                    }
                ]
            )
            hits, diagnostics = store.search_history_compatible(
                [1.0, 0.0],
                settings.query_identity(),
                top_k=1,
                min_similarity=0.1,
            )
            self.assertEqual(len(hits), 1)
            self.assertTrue(diagnostics['embedding_compatibility']['compatible'])

    def test_existing_vectors_without_identity_require_rebuild(self):
        settings = self.settings()
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonRagStore(tmp)
            store.upsert_history(
                [{'memory_id': 'm1', 'embedding': [1.0, 0.0], 'source_text': 'Hello'}]
            )
            blocked = ensure_store_document_identity(store, settings.document_identity())
            self.assertFalse(blocked['ready'])
            self.assertEqual(blocked['action'], 'rebuild_store')
            with self.assertRaises(EmbeddingStoreIdentityError):
                store.set_embedding_identity(settings.document_identity())
            rebuilt = ensure_store_document_identity(
                store,
                settings.document_identity(),
                rebuild=True,
            )
            self.assertTrue(rebuilt['rebuilt'])
            self.assertEqual(store.count_history(), 0)
            hits, diagnostics = store.search_history_compatible(
                [1.0, 0.0],
                settings.query_identity(),
            )
            self.assertEqual(hits, [])
            self.assertTrue(diagnostics['embedding_compatibility']['compatible'])

    def test_source_index_rejects_mismatched_query_identity(self):
        settings = self.settings()
        other = parse_embedding_runtime_settings(
            {
                'embedding_backend': 'openai_compatible',
                'embedding_provider': 'openai',
                'embedding_model': 'text-embedding-3-small',
                'output_dimensionality': 2,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonSourceIndexStore(tmp)
            ensure_store_document_identity(store, settings.document_identity())
            store.upsert_segments(
                [
                    {
                        'source_id': 's1',
                        'file_rel_path': 'a.rpy',
                        'source_text': 'Hello',
                        'embedding': [1.0, 0.0],
                    }
                ]
            )
            hits, diagnostics = store.search_segments_compatible(
                [1.0, 0.0],
                other.query_identity(),
                top_k=1,
                min_similarity=0.0,
            )
            self.assertEqual(hits, [])
            report = diagnostics['embedding_compatibility']
            self.assertFalse(report['compatible'])
            self.assertEqual(report['action'], 'rebuild_store')
            self.assertIn('backend_mismatch', report['codes'])


class RuntimeErrorTests(unittest.TestCase):
    def test_missing_transport_is_runtime_error(self):
        settings = parse_embedding_runtime_settings({})
        with self.assertRaises(EmbeddingRuntimeError):
            build_embedding_adapter(settings)
        openai_settings = parse_embedding_runtime_settings(
            {
                'embedding_backend': 'openai_compatible',
                'embedding_provider': 'openai',
                'embedding_model': 'text-embedding-3-small',
            }
        )
        with self.assertRaises(EmbeddingRuntimeError):
            build_embedding_adapter(openai_settings)


if __name__ == '__main__':
    unittest.main()
