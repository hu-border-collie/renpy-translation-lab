# -*- coding: utf-8 -*-
"""Sync/Batch Source Index and Published PA provider contract (issue #341)."""

import io
import tempfile
import unittest
from unittest import mock

import advanced_context
import gemini_translate_batch as batch_mod
import translator_runtime as runtime
from embedding_runtime import parse_embedding_runtime_settings
from rag_memory import JsonSourceIndexStore


class QueryAndRenderTests(unittest.TestCase):
    def test_source_only_query_excludes_local_past(self):
        target = [{'text': 'Hello world'}]
        past = [{'text': 'Previous line'}]
        source_query = advanced_context.build_source_only_query_text(target)
        history_query = advanced_context.build_history_query_text(target, past)
        self.assertEqual(source_query, 'Target:\n- Hello world')
        self.assertIn('Context before:', history_query)
        self.assertIn('Previous line', history_query)

    def test_sync_and_batch_render_the_same_source_and_analysis_partitions(self):
        source_hits = [
            {
                'source_id': 's1',
                'file_rel_path': 'script.rpy',
                'line_start': 10,
                'line_end': 12,
                'source_text': 'A distant scene',
                'score': 0.91,
            }
        ]
        analysis = {
            'text': 'Published brief body',
            'injectable': True,
            'diagnostics': 'status=published fingerprint=abc',
            'labels': [],
            'routes': [],
            'local_diagnostics': '',
        }
        retrieval = advanced_context.render_retrieval_reference_text(
            None,
            None,
            source_hits,
            history_char_limit=220,
            story_char_limit=1200,
        )
        analysis_text = advanced_context.render_analysis_reference_text(analysis)
        self.assertEqual(
            retrieval,
            runtime._render_sync_retrieval_reference_text(None, None, source_hits),
        )
        self.assertEqual(
            retrieval,
            batch_mod._render_batch_retrieval_reference_text(None, None, source_hits),
        )
        self.assertIn('RELATED PROJECT CONTEXT:', retrieval)
        self.assertIn('Source excerpt: A distant scene', retrieval)
        self.assertNotIn('RETRIEVED MEMORY:', retrieval)
        self.assertEqual(
            analysis_text,
            batch_mod._render_batch_analysis_reference_text(analysis),
        )
        self.assertIn('PROJECT BRIEF:', analysis_text)
        self.assertIn('Published brief body', analysis_text)

    def test_disabled_source_index_does_not_add_related_project_context(self):
        text = advanced_context.render_retrieval_reference_text(
            [{'source_text': 'Hello', 'translated_text': '你好', 'score': 1.0, 'file_rel_path': 'a.rpy', 'line_start': 1, 'line_end': 1, 'quality_state': 'seed'}],
            None,
            None,
            history_char_limit=220,
            story_char_limit=1200,
        )
        self.assertIn('RETRIEVED MEMORY:', text)
        self.assertNotIn('RELATED PROJECT CONTEXT:', text)


class PublishedAnalysisSkipTests(unittest.TestCase):
    def test_skip_diagnostics_cover_draft_stale_missing_and_disabled(self):
        cases = (
            {'reason': 'injection_disabled', 'injectable': False},
            {'reason': 'brief_not_published:draft', 'injectable': False, 'status': {'brief_status': 'draft'}},
            {'reason': 'brief_not_fresh:stale', 'injectable': False, 'status': {'brief_status': 'stale'}},
            {'reason': 'brief_not_published:missing', 'injectable': False, 'status': {'brief_status': 'missing'}},
        )
        for payload in cases:
            diagnostics = advanced_context.analysis_skip_diagnostics(payload)
            self.assertFalse(diagnostics['injectable'])
            self.assertTrue(diagnostics['reason'])
            self.assertEqual(advanced_context.render_analysis_reference_text(payload), '')

    def test_sync_inject_disabled_keeps_empty_analysis(self):
        old = runtime.SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF
        try:
            runtime.SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = False
            payload = runtime.load_sync_injectable_project_context('script.rpy', [1])
            self.assertFalse(payload['injectable'])
            self.assertEqual(payload['reason'], 'injection_disabled')
            self.assertEqual(advanced_context.render_analysis_reference_text(payload), '')
        finally:
            runtime.SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF = old


class SyncSourceIndexRetrievalTests(unittest.TestCase):
    def test_disabled_source_index_does_not_open_store(self):
        old = runtime.SYNC_SOURCE_INDEX_ENABLED
        try:
            runtime.SYNC_SOURCE_INDEX_ENABLED = False
            with mock.patch.object(runtime, 'get_sync_source_index_store') as store_mock:
                hits, stats = runtime.retrieve_sync_source_hits([{'text': 'Hello'}])
            self.assertEqual(hits, [])
            self.assertEqual(stats, {'enabled': False})
            store_mock.assert_not_called()
        finally:
            runtime.SYNC_SOURCE_INDEX_ENABLED = old

    def test_compatible_store_returns_source_hits_for_sync(self):
        old = {
            'enabled': runtime.SYNC_SOURCE_INDEX_ENABLED,
            'store_dir': runtime.SYNC_SOURCE_INDEX_STORE_DIR,
            'store': runtime._SYNC_SOURCE_INDEX_STORE,
            'dim': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
            'model': runtime.SYNC_RAG_EMBEDDING_MODEL,
            'top_k': runtime.SYNC_SOURCE_INDEX_TOP_K,
            'min_sim': runtime.SYNC_SOURCE_INDEX_MIN_SIMILARITY,
            'char_limit': runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT,
        }
        with tempfile.TemporaryDirectory() as tmp:
            try:
                runtime.SYNC_SOURCE_INDEX_ENABLED = True
                runtime.SYNC_SOURCE_INDEX_STORE_DIR = tmp
                runtime._SYNC_SOURCE_INDEX_STORE = None
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = 2
                runtime.SYNC_SOURCE_INDEX_TOP_K = 2
                runtime.SYNC_SOURCE_INDEX_MIN_SIMILARITY = 0.1
                runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT = 80
                store = JsonSourceIndexStore(tmp)
                settings = parse_embedding_runtime_settings(
                    {
                        'embedding_model': runtime.SYNC_RAG_EMBEDDING_MODEL,
                        'output_dimensionality': 2,
                    }
                )
                store.set_embedding_identity(settings.document_identity())
                store.upsert_segments(
                    [
                        {
                            'source_id': 's1',
                            'file_rel_path': 'script.rpy',
                            'line_start': 1,
                            'line_end': 1,
                            'source_text': 'Hello there friend',
                            'embedding': [1.0, 0.0],
                            'embedding_metadata': {
                                'embedding_model': runtime.SYNC_RAG_EMBEDDING_MODEL,
                                'embedding_task_type': runtime.SYNC_RAG_DOCUMENT_TASK_TYPE,
                                'embedding_dim': 2,
                            },
                        }
                    ]
                )
                with mock.patch.object(runtime, 'embed_sync_query_text', return_value=[1.0, 0.0]):
                    hits, stats = runtime.retrieve_sync_source_hits([{'text': 'Hello there'}])
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]['source_id'], 's1')
                self.assertTrue(stats['enabled'])
                self.assertNotIn('translated_text', hits[0])
            finally:
                runtime.SYNC_SOURCE_INDEX_ENABLED = old['enabled']
                runtime.SYNC_SOURCE_INDEX_STORE_DIR = old['store_dir']
                runtime._SYNC_SOURCE_INDEX_STORE = old['store']
                runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY = old['dim']
                runtime.SYNC_RAG_EMBEDDING_MODEL = old['model']
                runtime.SYNC_SOURCE_INDEX_TOP_K = old['top_k']
                runtime.SYNC_SOURCE_INDEX_MIN_SIMILARITY = old['min_sim']
                runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT = old['char_limit']


class EmbeddingLoaderRegressionTests(unittest.TestCase):
    def _batch_embedding_globals(self):
        return {
            'RAG_ENABLED': batch_mod.RAG_ENABLED,
            'RAG_EMBEDDING_MODEL': batch_mod.RAG_EMBEDDING_MODEL,
            'RAG_EMBEDDING_BACKEND': batch_mod.RAG_EMBEDDING_BACKEND,
            'RAG_EMBEDDING_PROVIDER': batch_mod.RAG_EMBEDDING_PROVIDER,
            'RAG_EMBEDDING_ENDPOINT': batch_mod.RAG_EMBEDDING_ENDPOINT,
            'RAG_EMBEDDING_TIMEOUT_SECONDS': batch_mod.RAG_EMBEDDING_TIMEOUT_SECONDS,
            'RAG_EMBEDDING_API_KEY_ENV': batch_mod.RAG_EMBEDDING_API_KEY_ENV,
            'RAG_EMBEDDING_LOAD_ERROR': batch_mod.RAG_EMBEDDING_LOAD_ERROR,
            'RAG_QUERY_TASK_TYPE': batch_mod.RAG_QUERY_TASK_TYPE,
            'RAG_DOCUMENT_TASK_TYPE': batch_mod.RAG_DOCUMENT_TASK_TYPE,
            'RAG_OUTPUT_DIMENSIONALITY': batch_mod.RAG_OUTPUT_DIMENSIONALITY,
            'SYNC_BACKEND': batch_mod.SYNC_BACKEND,
            'SYNC_MODEL': batch_mod.SYNC_MODEL,
            'SYNC_TIMEOUT_SECONDS': batch_mod.SYNC_TIMEOUT_SECONDS,
        }

    def _restore_batch_embedding_globals(self, snapshot):
        for key, value in snapshot.items():
            setattr(batch_mod, key, value)

    def _sync_embedding_globals(self):
        return {
            'SYNC_RAG_ENABLED': runtime.SYNC_RAG_ENABLED,
            'SYNC_RAG_EMBEDDING_MODEL': runtime.SYNC_RAG_EMBEDDING_MODEL,
            'SYNC_RAG_EMBEDDING_BACKEND': runtime.SYNC_RAG_EMBEDDING_BACKEND,
            'SYNC_RAG_EMBEDDING_PROVIDER': runtime.SYNC_RAG_EMBEDDING_PROVIDER,
            'SYNC_RAG_EMBEDDING_ENDPOINT': runtime.SYNC_RAG_EMBEDDING_ENDPOINT,
            'SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS': runtime.SYNC_RAG_EMBEDDING_TIMEOUT_SECONDS,
            'SYNC_RAG_EMBEDDING_API_KEY_ENV': runtime.SYNC_RAG_EMBEDDING_API_KEY_ENV,
            'SYNC_RAG_EMBEDDING_LOAD_ERROR': runtime.SYNC_RAG_EMBEDDING_LOAD_ERROR,
            'SYNC_RAG_QUERY_TASK_TYPE': runtime.SYNC_RAG_QUERY_TASK_TYPE,
            'SYNC_RAG_DOCUMENT_TASK_TYPE': runtime.SYNC_RAG_DOCUMENT_TASK_TYPE,
            'SYNC_RAG_OUTPUT_DIMENSIONALITY': runtime.SYNC_RAG_OUTPUT_DIMENSIONALITY,
        }

    def _restore_sync_embedding_globals(self, snapshot):
        for key, value in snapshot.items():
            setattr(runtime, key, value)

    def test_load_batch_settings_persists_openai_compatible_backend_globals(self):
        snapshot = self._batch_embedding_globals()
        try:
            with mock.patch.object(
                batch_mod,
                'load_json_file',
                side_effect=[
                    {},
                    {
                        'batch': {
                            'rag': {
                                'embedding_backend': 'openai_compatible',
                                'embedding_provider': 'openai',
                                'embedding_model': 'text-embedding-3-small',
                                'embedding_timeout_seconds': 12.5,
                                'output_dimensionality': 1536,
                            }
                        }
                    },
                ],
            ), mock.patch(
                'project_context_settings.apply_project_context_settings_to_config',
                side_effect=lambda config, root: config,
            ):
                batch_mod.load_batch_settings()
            self.assertEqual(batch_mod.RAG_EMBEDDING_BACKEND, 'openai_compatible')
            self.assertEqual(batch_mod.RAG_EMBEDDING_PROVIDER, 'openai')
            self.assertEqual(batch_mod.RAG_EMBEDDING_MODEL, 'text-embedding-3-small')
            self.assertEqual(batch_mod.RAG_EMBEDDING_TIMEOUT_SECONDS, 12.5)
            self.assertEqual(batch_mod.RAG_EMBEDDING_LOAD_ERROR, '')
            settings = batch_mod.current_batch_embedding_settings()
            self.assertEqual(settings.backend, 'openai_compatible')
            self.assertEqual(settings.model, 'text-embedding-3-small')
            self.assertEqual(settings.timeout_seconds, 12.5)
            self.assertEqual(settings.output_dimension, 1536)
        finally:
            self._restore_batch_embedding_globals(snapshot)

    def test_load_batch_settings_fails_loud_for_explicit_non_gemini_misconfig(self):
        snapshot = self._batch_embedding_globals()
        try:
            with mock.patch.object(
                batch_mod,
                'load_json_file',
                side_effect=[
                    {},
                    {
                        'batch': {
                            'rag': {
                                'embedding_backend': 'openai_compatible',
                                'embedding_model': 'text-embedding-3-small',
                            }
                        }
                    },
                ],
            ), mock.patch(
                'project_context_settings.apply_project_context_settings_to_config',
                side_effect=lambda config, root: config,
            ):
                with self.assertRaises(SystemExit) as captured:
                    batch_mod.load_batch_settings()
            self.assertIn('invalid batch.rag embedding settings', str(captured.exception))
            self.assertEqual(
                batch_mod.RAG_EMBEDDING_BACKEND,
                snapshot['RAG_EMBEDDING_BACKEND'],
            )
        finally:
            self._restore_batch_embedding_globals(snapshot)

    def test_load_batch_settings_records_gemini_task_type_fallback(self):
        snapshot = self._batch_embedding_globals()
        try:
            with mock.patch.object(
                batch_mod,
                'load_json_file',
                side_effect=[
                    {},
                    {
                        'batch': {
                            'rag': {
                                'query_task_type': 'SEMANTIC_SIMILARITY',
                                'document_task_type': 'document',
                            }
                        }
                    },
                ],
            ), mock.patch(
                'project_context_settings.apply_project_context_settings_to_config',
                side_effect=lambda config, root: config,
            ), mock.patch('sys.stdout', io.StringIO()):
                batch_mod.load_batch_settings()
            self.assertTrue(batch_mod.RAG_EMBEDDING_LOAD_ERROR)
            self.assertIn('not supported', batch_mod.RAG_EMBEDDING_LOAD_ERROR)
            self.assertEqual(batch_mod.RAG_EMBEDDING_BACKEND, 'gemini')
            self.assertEqual(batch_mod.RAG_QUERY_TASK_TYPE, 'RETRIEVAL_QUERY')
            self.assertEqual(batch_mod.RAG_DOCUMENT_TASK_TYPE, 'RETRIEVAL_DOCUMENT')
        finally:
            self._restore_batch_embedding_globals(snapshot)

    def test_load_sync_rag_settings_persists_openai_compatible_backend(self):
        snapshot = self._sync_embedding_globals()
        try:
            runtime.load_sync_rag_settings(
                {
                    'sync': {
                        'rag': {
                            'embedding_backend': 'openai_compatible',
                            'embedding_provider': 'openai',
                            'embedding_model': 'text-embedding-3-small',
                            'embedding_timeout_seconds': 9,
                            'output_dimensionality': 1536,
                            'query_task_type': 'query',
                        }
                    }
                }
            )
            self.assertEqual(runtime.SYNC_RAG_EMBEDDING_BACKEND, 'openai_compatible')
            self.assertEqual(runtime.SYNC_RAG_EMBEDDING_MODEL, 'text-embedding-3-small')
            self.assertEqual(runtime.SYNC_RAG_QUERY_TASK_TYPE, 'RETRIEVAL_QUERY')
            settings = runtime.current_sync_embedding_settings()
            self.assertEqual(settings.backend, 'openai_compatible')
            self.assertEqual(settings.timeout_seconds, 9)
        finally:
            self._restore_sync_embedding_globals(snapshot)

    def test_load_sync_rag_settings_fails_loud_for_explicit_non_gemini_misconfig(self):
        snapshot = self._sync_embedding_globals()
        try:
            with self.assertRaises(SystemExit) as captured:
                runtime.load_sync_rag_settings(
                    {
                        'sync': {
                            'rag': {
                                'embedding_backend': 'openai_compatible',
                                'embedding_model': 'text-embedding-3-small',
                            }
                        }
                    }
                )
            self.assertIn('invalid sync.rag embedding settings', str(captured.exception))
        finally:
            self._restore_sync_embedding_globals(snapshot)


class ContextPolicyBudgetTests(unittest.TestCase):
    def test_sync_plan_context_policy_keeps_source_index_budget_separate(self):
        old = {
            'history': runtime.SYNC_RAG_HISTORY_CHAR_LIMIT,
            'enabled': runtime.SYNC_SOURCE_INDEX_ENABLED,
            'top_k': runtime.SYNC_SOURCE_INDEX_TOP_K,
            'char_limit': runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT,
        }
        try:
            runtime.SYNC_RAG_HISTORY_CHAR_LIMIT = 220
            runtime.SYNC_SOURCE_INDEX_ENABLED = True
            runtime.SYNC_SOURCE_INDEX_TOP_K = 4
            runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT = 220
            policy = runtime._sync_plan_context_policy()
            self.assertEqual(policy.history_char_limit, 220)
            self.assertEqual(policy.source_index_char_limit, 880)
        finally:
            runtime.SYNC_RAG_HISTORY_CHAR_LIMIT = old['history']
            runtime.SYNC_SOURCE_INDEX_ENABLED = old['enabled']
            runtime.SYNC_SOURCE_INDEX_TOP_K = old['top_k']
            runtime.SYNC_SOURCE_INDEX_CHAR_LIMIT = old['char_limit']


if __name__ == '__main__':
    unittest.main()
