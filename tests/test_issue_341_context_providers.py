# -*- coding: utf-8 -*-
"""Sync/Batch Source Index and Published PA provider contract (issue #341)."""

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


if __name__ == '__main__':
    unittest.main()
