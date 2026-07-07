import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch_mod
import translation_ab_experiment as ab_mod


FIXTURE_MANIFEST = (
    Path(__file__).parent / 'fixtures' / 'golden_batch_minimal' / 'expected' / 'manifest_snapshot.json'
)
SAMPLE_ITEM_ID = 'chapter01/dialogue.rpy:chapter01_start:1:8dfe92d9'
SAMPLE_ITEM_IDS = [
    'chapter01/dialogue.rpy:chapter01_start:1:8dfe92d9',
    'chapter01/dialogue.rpy:chapter01_start:2:3fbd4128',
    'chapter01/dialogue.rpy:chapter01_start:3:2ecaab1c',
    'chapter01/dialogue.rpy:chapter01_start:4:8c8bd881',
]
FIXTURE_VARIANTS = Path(__file__).parent / 'fixtures' / 'ab_variants_minimal.json'


def _full_chunk_translations() -> str:
    return json.dumps(
        [{'id': item_id, 'translation': '你好'} for item_id in SAMPLE_ITEM_IDS],
    )


class TranslationAbExperimentTests(unittest.TestCase):
    def test_deep_merge_dict_merges_nested_batch_settings(self):
        merged = ab_mod.deep_merge_dict(
            {'batch': {'story_memory': {'enabled': False, 'max_context_chars': 800}}},
            {'batch': {'story_memory': {'enabled': True}}},
        )
        self.assertTrue(merged['batch']['story_memory']['enabled'])
        self.assertEqual(merged['batch']['story_memory']['max_context_chars'], 800)

    def test_load_variants_file_requires_unique_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'variants.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump([{'name': 'a'}, {'name': 'a'}], handle)
            with self.assertRaises(SystemExit):
                ab_mod.load_variants_file(path)

    def test_load_variants_file_requires_at_least_two_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'variants.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump([{'name': 'only_one'}], handle)
            with self.assertRaises(SystemExit) as ctx:
                ab_mod.load_variants_file(path)
            self.assertIn('at least two', str(ctx.exception))

    def test_extract_translation_map_detects_missing_and_extra_ids(self):
        items = [{'id': 'line-1', 'text': 'Hello'}, {'id': 'line-2', 'text': 'World'}]
        response_text = json.dumps(
            [
                {'id': 'line-1', 'translation': '你好'},
                {'id': 'line-3', 'translation': '多余'},
            ],
        )
        translations, error = ab_mod.extract_translation_map(response_text, items)
        self.assertEqual(translations['line-1'], '你好')
        self.assertIn('Missing translations for ids: line-2', error)
        self.assertIn('Unexpected result ids: line-3', error)

    def test_variant_batch_settings_restores_baseline_globals(self):
        baseline_rag = batch_mod.RAG_ENABLED
        with ab_mod.variant_batch_settings({'batch': {'rag': {'enabled': True}}}):
            self.assertTrue(batch_mod.RAG_ENABLED)
        self.assertEqual(batch_mod.RAG_ENABLED, baseline_rag)

    def test_dry_run_skips_retrieval_calls(self):
        chunk = {
            'items': [{'id': 'line-1', 'text': 'Hello'}],
            'context_past': [],
            'context_future': [],
            'file_rel_path': 'script.rpy',
        }
        with mock.patch.object(batch_mod, 'retrieve_glossary_hits') as glossary_mock, \
             mock.patch.object(batch_mod, 'retrieve_history_hits') as history_mock, \
             mock.patch.object(batch_mod, 'retrieve_source_hits') as source_mock, \
             mock.patch.object(batch_mod, 'retrieve_batch_story_hits') as story_mock:
            enriched = ab_mod.enrich_chunk_for_current_settings(chunk, dry_run=True)
        glossary_mock.assert_not_called()
        history_mock.assert_not_called()
        source_mock.assert_not_called()
        story_mock.assert_not_called()
        self.assertEqual(enriched['glossary_hits'], [])
        self.assertEqual(enriched['history_hits'], [])
        self.assertEqual(enriched['source_hits'], [])
        self.assertNotIn('story_hits', enriched)

    def test_render_markdown_report_includes_variant_columns(self):
        chunk_results = [
            ab_mod.ChunkExperimentResult(
                chunk_key='chunk-1',
                file_rel_path='script.rpy',
                items=[{'id': 'line-1', 'text': 'Hello'}],
                variant_results=[
                    ab_mod.VariantRunResult(
                        variant_name='baseline',
                        settings={
                            'model': 'test-model',
                            'story_memory_enabled': False,
                            'rag_enabled': False,
                            'source_index_enabled': False,
                            'macro_setting_preview': 'Base',
                        },
                        translations={'line-1': '你好'},
                    ),
                    ab_mod.VariantRunResult(
                        variant_name='story_memory',
                        settings={
                            'model': 'test-model',
                            'story_memory_enabled': True,
                            'rag_enabled': False,
                            'source_index_enabled': False,
                            'macro_setting_preview': 'Base',
                        },
                        translations={'line-1': '你好啊'},
                    ),
                ],
            )
        ]
        variants = [{'name': 'baseline', 'overrides': {}}, {'name': 'story_memory', 'overrides': {}}]
        report = ab_mod.render_markdown_report(
            manifest_path='manifest.json',
            variants=variants,
            chunk_results=chunk_results,
            experiment_settings={'dry_run': True},
        )
        self.assertIn('baseline', report)
        self.assertIn('story_memory', report)
        self.assertIn('| line-1 | Hello | 你好 | 你好啊 |', report)

    def test_run_translation_ab_experiment_dry_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, 'manifest.json')
            with open(FIXTURE_MANIFEST, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            manifest['mode'] = batch_mod.MANIFEST_MODE_TRANSLATION
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            loaded = batch_mod.load_manifest(manifest_path)
            variants = ab_mod.load_variants_file(str(FIXTURE_VARIANTS))
            output_dir = os.path.join(tmp, 'experiment')
            summary = ab_mod.run_translation_ab_experiment(
                loaded,
                variants,
                limit=1,
                offset=0,
                output_dir=output_dir,
                dry_run=True,
            )
            self.assertTrue(os.path.isfile(summary['report_path']))
            self.assertTrue(os.path.isfile(summary['results_path']))
            self.assertTrue(os.path.isfile(summary['settings_path']))
            with open(summary['results_path'], 'r', encoding='utf-8') as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]['variants']), 2)
            self.assertTrue(all(variant['dry_run'] for variant in rows[0]['variants']))

    def test_run_translation_ab_experiment_calls_sync_runner_per_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, 'manifest.json')
            with open(FIXTURE_MANIFEST, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            manifest['mode'] = batch_mod.MANIFEST_MODE_TRANSLATION
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            loaded = batch_mod.load_manifest(manifest_path)
            variants = ab_mod.load_variants_file(str(FIXTURE_VARIANTS))
            calls = []

            def fake_sync_runner(request_payload, model_name, api_key_index=None):
                calls.append(model_name)
                return {
                    'response_text': _full_chunk_translations(),
                    'finish_reason': 'STOP',
                    'usage_metadata': {'total_tokens': 12},
                }

            summary = ab_mod.run_translation_ab_experiment(
                loaded,
                variants,
                limit=1,
                offset=0,
                output_dir=os.path.join(tmp, 'experiment'),
                dry_run=False,
                sync_runner=fake_sync_runner,
            )
            self.assertEqual(len(calls), 2)
            with open(summary['results_path'], 'r', encoding='utf-8') as handle:
                row = json.loads(handle.readline())
            translations = {variant['name']: variant['translations'] for variant in row['variants']}
            self.assertEqual(translations['baseline'][SAMPLE_ITEM_ID], '你好')
            self.assertEqual(translations['story_memory'][SAMPLE_ITEM_ID], '你好')

    def test_run_translation_ab_experiment_records_variant_sync_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, 'manifest.json')
            with open(FIXTURE_MANIFEST, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            manifest['mode'] = batch_mod.MANIFEST_MODE_TRANSLATION
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            loaded = batch_mod.load_manifest(manifest_path)
            variants = ab_mod.load_variants_file(str(FIXTURE_VARIANTS))
            call_count = {'value': 0}

            def flaky_sync_runner(request_payload, model_name, api_key_index=None):
                call_count['value'] += 1
                if call_count['value'] == 1:
                    return {
                        'response_text': _full_chunk_translations(),
                        'finish_reason': 'STOP',
                        'usage_metadata': {},
                    }
                raise RuntimeError('sync failed')

            summary = ab_mod.run_translation_ab_experiment(
                loaded,
                variants,
                limit=1,
                offset=0,
                output_dir=os.path.join(tmp, 'experiment'),
                dry_run=False,
                sync_runner=flaky_sync_runner,
            )
            self.assertTrue(os.path.isfile(summary['report_path']))
            with open(summary['results_path'], 'r', encoding='utf-8') as handle:
                row = json.loads(handle.readline())
            self.assertEqual(len(row['variants']), 2)
            errors = {variant['name']: variant.get('error', '') for variant in row['variants']}
            self.assertEqual(errors['baseline'], '')
            self.assertIn('sync failed', errors['story_memory'])

    def test_run_translation_ab_experiment_writes_partial_outputs_on_experiment_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, 'manifest.json')
            with open(FIXTURE_MANIFEST, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            manifest['mode'] = batch_mod.MANIFEST_MODE_TRANSLATION
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            loaded = batch_mod.load_manifest(manifest_path)
            variants = ab_mod.load_variants_file(str(FIXTURE_VARIANTS))
            original_context = ab_mod.variant_batch_settings
            entered = {'value': 0}

            @contextlib.contextmanager
            def flaky_context(overrides):
                entered['value'] += 1
                if entered['value'] == 2:
                    raise RuntimeError('variant settings failed')
                with original_context(overrides) as settings:
                    yield settings

            with mock.patch.object(ab_mod, 'variant_batch_settings', side_effect=flaky_context):
                summary = ab_mod.run_translation_ab_experiment(
                    loaded,
                    variants,
                    limit=1,
                    offset=0,
                    output_dir=os.path.join(tmp, 'experiment'),
                    dry_run=True,
                )
            self.assertTrue(os.path.isfile(summary['report_path']))
            self.assertIn('experiment_error', summary)
            with open(summary['settings_path'], 'r', encoding='utf-8') as handle:
                settings_payload = json.load(handle)
            self.assertIn('experiment_error', settings_payload)
            with open(summary['results_path'], 'r', encoding='utf-8') as handle:
                row = json.loads(handle.readline())
            self.assertEqual(len(row['variants']), 1)

    def test_compare_variants_cli_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, 'manifest.json')
            with open(FIXTURE_MANIFEST, 'r', encoding='utf-8') as source:
                manifest = json.load(source)
            manifest['mode'] = batch_mod.MANIFEST_MODE_TRANSLATION
            with open(manifest_path, 'w', encoding='utf-8') as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)

            output_dir = os.path.join(tmp, 'out')
            argv = [
                'compare-variants',
                manifest_path,
                '--variants-file',
                str(FIXTURE_VARIANTS),
                '--limit',
                '1',
                '--dry-run',
                '--output-dir',
                output_dir,
            ]
            with mock.patch.object(batch_mod, 'initialize_batch_logging'), \
                 mock.patch.object(batch_mod.legacy, 'load_config'), \
                 mock.patch.object(batch_mod.legacy, 'load_translator_settings'), \
                 mock.patch.object(batch_mod.legacy, 'load_glossary'), \
                 mock.patch.object(batch_mod, 'load_batch_settings'), \
                 mock.patch.object(batch_mod, 'print_banner'):
                batch_mod.main(argv)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, 'ab_report.md')))


if __name__ == '__main__':
    unittest.main()