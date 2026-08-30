# -*- coding: utf-8 -*-
"""Issue #346 P5 golden contracts for #341 context providers."""

import hashlib
import types
import unittest
from contextlib import ExitStack
from unittest import mock

import advanced_context
import gemini_translate_batch as batch_mod
import model_profile
import translation_plan
import translator_runtime as runtime
from embedding_runtime import parse_embedding_runtime_settings
from engine_adapters.contracts import SourceDocument


def _fixture_jobs():
    content = b'label start:\n    e "Hello tonight."\n    "Keep [Name!t]."\n'
    document = SourceDocument(
        file_rel_path='script.rpy',
        file_path='C:/fixture/script.rpy',
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    project = types.SimpleNamespace(
        engine='renpy',
        adapter_version='fixture-adapter',
        project_snapshot_fingerprint='fixture-project',
        source_documents=(document,),
    )
    snapshot = types.SimpleNamespace(project=project)
    jobs = [{
        'file_rel_path': 'script.rpy',
        'file_path': document.file_path,
        'tasks': [
            {
                'id': 'script:1:4',
                'text': 'Hello tonight.',
                'line': 1,
                'speaker_id': 'e',
                'speaker_name': 'Eileen',
                'block_name': 'start',
            },
            {
                'id': 'script:2:4',
                'text': 'Keep [Name!t].',
                'line': 2,
                'block_name': 'start',
            },
        ],
    }]
    return jobs, snapshot, document


def _published_context():
    return {
        'text': 'Published project brief for the current story.',
        'injectable': True,
        'reason': '',
        'brief_status': 'published',
        'source_fingerprint': 'fixture-source-fingerprint',
        'diagnostics': 'status=published fingerprint=fixture-source-fingerprint',
        'labels': [{'label_id': 'tone', 'summary': 'Keep the dialogue warm.'}],
        'routes': [{'route_id': 'start', 'summary': 'Opening scene route.'}],
        'local_diagnostics': 'selected=1',
    }


class P5ProductionGoldenTests(unittest.TestCase):
    def test_sync_and_batch_source_pa_embedding_requests_are_equivalent(self):
        jobs, snapshot, _document = _fixture_jobs()
        batch_jobs = batch_mod.TranslationFileJobs(
            jobs,
            adapter_snapshot=snapshot,
        )
        routing = model_profile.resolve_routing_plan_from_runtime(
            sync_backend='gemini',
            sync_model='gemini-2.5-flash',
            sync_models=('gemini-2.5-flash',),
        )
        source_hits = [{
            'source_id': 'source-1',
            'file_rel_path': 'other/script.rpy',
            'line_start': 10,
            'line_end': 11,
            'source_text': 'A related source excerpt.',
            'score': 0.93,
        }]
        source_stats = {
            'enabled': True,
            'hit_count': 1,
            'matched_count': 1,
            'filtered_count': 0,
            'stale_hits_skipped': 0,
            'below_similarity_count': 0,
            'truncated_count': 0,
            'source_context_chars': len(source_hits[0]['source_text']),
            'source_context_char_budget': 80,
        }
        project_context = _published_context()
        glossary = {'Hello': '你好'}
        preserve = ['[Name!t]']
        non_translatable = {'Eileen'}

        patches = (
            mock.patch.object(runtime, 'MAX_ITEMS', 60),
            mock.patch.object(runtime, 'MAX_CHARS', 18000),
            mock.patch.object(runtime, 'SYNC_CONTEXT_BEFORE', 30),
            mock.patch.object(runtime, 'SYNC_CONTEXT_AFTER', 10),
            mock.patch.object(runtime, 'SYNC_RAG_ENABLED', False),
            mock.patch.object(runtime, 'SYNC_RAG_HISTORY_CHAR_LIMIT', 220),
            mock.patch.object(runtime, 'SYNC_STORY_MEMORY_ENABLED', False),
            mock.patch.object(runtime, 'SYNC_STORY_MEMORY_MAX_CONTEXT_CHARS', 1200),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_ENABLED', True),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_TOP_K', 2),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_MIN_SIMILARITY', 0.7),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_CHAR_LIMIT', 40),
            mock.patch.object(runtime, 'SYNC_PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF', True),
            mock.patch.object(runtime, 'SYNC_MACRO_SETTING', 'Use a warm stage tone.'),
            mock.patch.object(runtime, 'NORMALIZE_TRANSLATION_MAP', glossary),
            mock.patch.object(runtime, 'PRESERVE_TERMS', preserve),
            mock.patch.object(runtime, 'NON_TRANSLATABLE_EXACT', non_translatable),
            mock.patch.object(batch_mod, 'BATCH_TARGET_SIZE', 60),
            mock.patch.object(batch_mod, 'BATCH_TARGET_CHARS', 18000),
            mock.patch.object(batch_mod, 'BATCH_CONTEXT_BEFORE', 30),
            mock.patch.object(batch_mod, 'BATCH_CONTEXT_AFTER', 10),
            mock.patch.object(batch_mod, 'RAG_ENABLED', False),
            mock.patch.object(batch_mod, 'RAG_HISTORY_CHAR_LIMIT', 220),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_ENABLED', True),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_TOP_K', 2),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_MIN_SIMILARITY', 0.7),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_CHAR_LIMIT', 40),
            mock.patch.object(batch_mod, 'STORY_MEMORY_ENABLED', False),
            mock.patch.object(batch_mod, 'STORY_MEMORY_MAX_CONTEXT_CHARS', 1200),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_ENABLED', True),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF', True),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_MAX_BRIEF_CHARS', 4000),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_MAX_LABEL_SUMMARY_CHARS', 800),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_MAX_ROUTE_SUMMARY_CHARS', 1200),
            mock.patch.object(batch_mod, 'BATCH_MACRO_SETTING', 'Use a warm stage tone.'),
            mock.patch.object(batch_mod.legacy, 'NORMALIZE_TRANSLATION_MAP', glossary),
            mock.patch.object(batch_mod.legacy, 'PRESERVE_TERMS', preserve),
            mock.patch.object(batch_mod.legacy, 'NON_TRANSLATABLE_EXACT', non_translatable),
            mock.patch.object(runtime, 'retrieve_sync_source_hits', return_value=(source_hits, source_stats)),
            mock.patch.object(runtime, 'load_sync_injectable_project_context', return_value=project_context),
            mock.patch.object(batch_mod, 'retrieve_source_hits', return_value=(source_hits, source_stats)),
            mock.patch.object(batch_mod, 'load_injectable_project_context_for_prompts', return_value=project_context),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            sync_build, _sync_captures = runtime.build_sync_translation_plan(
                jobs,
                snapshot,
                routing,
                run_id='p5-sync',
            )
            batch_build = batch_mod._build_batch_translation_plan(
                batch_jobs,
                routing_plan=routing,
            )['plan_build']

        self.assertTrue(
            translation_plan.plan_diff(
                sync_build.requests,
                batch_build.requests,
            )['equivalent']
        )
        sync_request = sync_build.requests[0]
        batch_request = batch_build.requests[0]
        self.assertEqual(
            translation_plan.canonical_semantic_request(sync_request),
            translation_plan.canonical_semantic_request(batch_request),
        )
        self.assertEqual(sync_request.prompt_fingerprint, batch_request.prompt_fingerprint)
        self.assertEqual(sync_request.context_assembly, batch_request.context_assembly)
        self.assertIn('RELATED PROJECT CONTEXT:', sync_request.user_prompt)
        self.assertIn('PROJECT BRIEF:', sync_request.user_prompt)
        self.assertIn('Published project brief', sync_request.user_prompt)
        diagnostics = translation_plan.summarize_request_diagnostics(
            sync_build.plan.request_summaries
        )
        self.assertEqual(diagnostics['context_provider_diagnostic_requests'], 1)
        self.assertEqual(diagnostics['context_provider_downgrade_count'], 0)
        self.assertIn('source_index:available', diagnostics['context_provider_status_counts'])
        self.assertIn(
            'published_project_analysis:available',
            diagnostics['context_provider_status_counts'],
        )
        self.assertIn(
            'query_task_type',
            runtime.current_sync_embedding_settings().public_dict(),
        )


class P5ProviderDiagnosticGoldenTests(unittest.TestCase):
    def _build(self, strategy, source_diagnostic, analysis_diagnostic):
        jobs, _snapshot, _document = _fixture_jobs()
        settings = parse_embedding_runtime_settings({
            'embedding_model': 'gemini-embedding-001',
            'output_dimensionality': 768,
        })
        source_text = (
            'RELATED PROJECT CONTEXT:\n- related source'
            if source_diagnostic.get('enabled') is not False
            and not source_diagnostic.get('reason')
            else ''
        )
        analysis_text = (
            'PROJECT BRIEF:\nPublished brief'
            if analysis_diagnostic.get('injectable')
            else ''
        )
        return translation_plan.build_translation_plan(
            jobs,
            execution_strategy=strategy,
            source_identity=translation_plan.SourceIdentity(
                engine='renpy',
                adapter_version='fixture',
                project_identity_digest='project',
                source_snapshot_fingerprint='source',
            ),
            config_snapshot={'fixture': True},
            model_profile_snapshot={'model': 'fixture'},
            chunk_policy=translation_plan.ChunkPolicy(60, 18000),
            context_policy=translation_plan.ContextPolicy(
                local_context_before=30,
                local_context_after=10,
                history_char_limit=220,
                story_char_limit=1200,
                source_index_char_limit=80,
                analysis_char_limit=6000,
            ),
            preserve_terms=['[Name!t]'],
            normalize_map={'Hello': '你好'},
            non_translatable_exact={'Eileen'},
            macro_setting='Use a warm stage tone.',
            retrieval_blocks_provider=lambda _chunk: {
                'text': source_text,
                'diagnostics': {
                    'source_index': source_diagnostic,
                    'embedding_provider': settings.public_dict(),
                },
            },
            analysis_blocks_provider=lambda _chunk: {
                'text': analysis_text,
                'diagnostics': {
                    'published_project_analysis': analysis_diagnostic,
                },
            },
        )

    def test_all_disabled_missing_incompatible_rebuild_draft_stale_cases_match(self):
        source_cases = (
            ('disabled', {'enabled': False}),
            ('missing', {'enabled': True, 'reason': 'empty_source_store'}),
            ('incompatible', {
                'enabled': True,
                'reason': 'rebuild_store',
                'action': 'rebuild_store',
                'embedding_compatibility': {
                    'compatible': False,
                    'action': 'rebuild_store',
                    'codes': ['model_mismatch'],
                },
            }),
            ('rebuild', {
                'enabled': True,
                'reason': 'rebuild_store',
                'action': 'rebuild_store',
            }),
        )
        analysis_cases = (
            ('disabled', {'injectable': False, 'reason': 'injection_disabled'}),
            ('missing', {
                'injectable': False,
                'reason': 'brief_not_published:missing',
                'brief_status': 'missing',
            }),
            ('draft', {
                'injectable': False,
                'reason': 'brief_not_published:draft',
                'brief_status': 'draft',
            }),
            ('stale', {
                'injectable': False,
                'reason': 'brief_not_fresh:stale',
                'brief_status': 'stale',
            }),
        )
        for source_name, source_diagnostic in source_cases:
            for analysis_name, analysis_diagnostic in analysis_cases:
                with self.subTest(source=source_name, analysis=analysis_name):
                    sync = self._build(
                        model_profile.ExecutionStrategy.SYNC.value,
                        source_diagnostic,
                        analysis_diagnostic,
                    )
                    batch = self._build(
                        model_profile.ExecutionStrategy.GEMINI_BATCH.value,
                        source_diagnostic,
                        analysis_diagnostic,
                    )
                    report = translation_plan.plan_diff(sync.requests, batch.requests)
                    self.assertTrue(report['equivalent'], translation_plan.format_plan_diff(report))
                    self.assertEqual(
                        sync.requests[0].context_assembly,
                        batch.requests[0].context_assembly,
                    )
                    summary = translation_plan.summarize_request_diagnostics(
                        sync.plan.request_summaries
                    )
                    self.assertGreaterEqual(summary['context_provider_downgrade_count'], 2)
                    self.assertIn(
                        f'source_index:{source_diagnostic.get("reason", "disabled") if source_diagnostic.get("enabled") is not False else "disabled"}',
                        summary['context_provider_downgrade_reasons'],
                    )
                    self.assertIn(
                        f'published_project_analysis:{analysis_diagnostic["reason"]}',
                        summary['context_provider_downgrade_reasons'],
                    )

    def test_batch_project_analysis_keeps_published_identity_for_diagnostics(self):
        payload = {
            'text': '',
            'injectable': False,
            'reason': 'brief_not_fresh:stale',
            'status': {
                'brief_status': 'stale',
                'artifacts': {
                    'project_brief': {
                        'lineage': {'source_fingerprint': 'old-source-fingerprint'},
                    },
                },
            },
            'diagnostics': '',
        }
        with (
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_ENABLED', True),
            mock.patch.object(batch_mod, 'PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF', True),
            mock.patch.object(batch_mod, '_PROJECT_BRIEF_CACHE', None),
            mock.patch.object(batch_mod, '_PROJECT_BRIEF_CACHE_KEY', None),
            mock.patch.object(
                batch_mod,
                'compute_current_project_analysis_fingerprint',
                return_value='current-source-fingerprint',
            ),
            mock.patch(
                'project_analysis.load_injectable_project_context',
                return_value=payload,
            ),
        ):
            result = batch_mod.load_injectable_project_context_for_prompts(
                'script.rpy', [2]
            )

        diagnostics = advanced_context.analysis_skip_diagnostics(result)
        self.assertEqual(diagnostics['brief_status'], 'stale')
        self.assertEqual(diagnostics['source_fingerprint'], 'old-source-fingerprint')
        self.assertEqual(diagnostics['reason'], 'brief_not_fresh:stale')


class P5BudgetAndEmbeddingTests(unittest.TestCase):
    def test_source_index_aggregate_budget_crops_ranked_hits(self):
        matches = [
            {'source_id': 'one', 'source_text': 'abcdefghij', 'score': 0.9},
            {'source_id': 'two', 'source_text': 'klmnopqrst', 'score': 0.8},
            {'source_id': 'three', 'source_text': 'uvwxyz', 'score': 0.7},
        ]
        hits, truncated_count, source_chars = advanced_context.shape_source_hits(
            matches,
            char_limit=8,
            char_budget=10,
        )
        self.assertEqual([hit['source_text'] for hit in hits], ['abcde...', 'kl'])
        self.assertEqual(source_chars, 10)
        self.assertEqual(truncated_count, 2)

        class Store:
            store_dir = 'fixture-source-index'
            metadata = {'schema_version': 1}

            def count_segments(self):
                return len(matches)

            def search_segments_compatible(self, *_args, **_kwargs):
                return matches, {
                    'embedding_compatibility': {'compatible': True},
                    'matched_before_top_k': len(matches),
                }

        shaped, stats = advanced_context.retrieve_source_hits_compatible(
            Store(),
            [1.0],
            object(),
            top_k=3,
            min_similarity=0.0,
            char_limit=8,
            char_budget=10,
            query_text='Target:\n- query',
        )
        self.assertEqual(len(shaped), 2)
        self.assertEqual(stats['source_context_chars'], 10)
        self.assertEqual(stats['source_context_budget_dropped_count'], 1)
        self.assertTrue(stats['source_context_budget_exhausted'])

    def test_sync_and_batch_source_queries_use_same_embedding_contract(self):
        settings = parse_embedding_runtime_settings({
            'embedding_model': 'gemini-embedding-001',
            'output_dimensionality': 2,
        })
        calls = {'sync': [], 'batch': []}
        matches = [{
            'source_id': 'source-1',
            'file_rel_path': 'script.rpy',
            'line_start': 1,
            'line_end': 1,
            'source_text': 'Related excerpt',
            'score': 0.95,
        }]

        class Compatibility:
            compatible = True
            action = ''

            def to_dict(self):
                return {'compatible': True, 'action': '', 'codes': []}

        class Store:
            store_dir = 'fixture-source-index'
            metadata = {'schema_version': 1}

            def count_segments(self):
                return 1

            def embedding_compatibility(self, _identity):
                return Compatibility()

            def search_segments_compatible(self, *_args, **_kwargs):
                return matches, {
                    'embedding_compatibility': {'compatible': True},
                    'matched_before_top_k': 1,
                    'returned_count': 1,
                }

        store = Store()
        patches = (
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_ENABLED', True),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_TOP_K', 1),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_CHAR_LIMIT', 80),
            mock.patch.object(runtime, 'SYNC_SOURCE_INDEX_MIN_SIMILARITY', 0.0),
            mock.patch.object(runtime, 'SYNC_RAG_OUTPUT_DIMENSIONALITY', 2),
            mock.patch.object(runtime, 'SYNC_RAG_EMBEDDING_MODEL', settings.model),
            mock.patch.object(runtime, 'SYNC_RAG_DOCUMENT_TASK_TYPE', settings.native_document_task_type),
            mock.patch.object(runtime, 'current_sync_embedding_settings', return_value=settings),
            mock.patch.object(runtime, 'get_sync_source_index_store', return_value=store),
            mock.patch.object(
                runtime,
                'embed_sync_query_text',
                side_effect=lambda text: calls['sync'].append(text) or [1.0, 0.0],
            ),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_ENABLED', True),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_TOP_K', 1),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_CHAR_LIMIT', 80),
            mock.patch.object(batch_mod, 'SOURCE_INDEX_MIN_SIMILARITY', 0.0),
            mock.patch.object(batch_mod, 'RAG_OUTPUT_DIMENSIONALITY', 2),
            mock.patch.object(batch_mod, 'RAG_EMBEDDING_MODEL', settings.model),
            mock.patch.object(batch_mod, 'RAG_DOCUMENT_TASK_TYPE', settings.native_document_task_type),
            mock.patch.object(batch_mod, 'current_batch_embedding_settings', return_value=settings),
            mock.patch.object(batch_mod, 'get_source_index_store', return_value=store),
            mock.patch.object(
                batch_mod,
                'embed_query_text',
                side_effect=lambda text: calls['batch'].append(text) or [1.0, 0.0],
            ),
        )
        with ExitStack() as stack:
            for patcher in patches:
                stack.enter_context(patcher)
            sync_hits, sync_stats = runtime.retrieve_sync_source_hits(
                [{'text': 'Hello world'}]
            )
            batch_hits, batch_stats = batch_mod.retrieve_source_hits(
                [{'text': 'Hello world'}],
                [{'text': 'Previous line must not enter Source Index query'}],
            )

        self.assertEqual(calls['sync'], ['Target:\n- Hello world'])
        self.assertEqual(calls['batch'], calls['sync'])
        self.assertEqual(sync_hits, batch_hits)
        self.assertEqual(sync_stats['embedding_provider'], batch_stats['embedding_provider'])
        self.assertEqual(sync_stats['source_context_chars'], batch_stats['source_context_chars'])
        self.assertEqual(sync_stats['source_context_char_budget'], 80)


if __name__ == '__main__':
    unittest.main()
