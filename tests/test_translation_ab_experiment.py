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
FIXTURE_VARIANTS = Path(__file__).parent / 'fixtures' / 'ab_variants_minimal.json'


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
                    'response_text': json.dumps([{'id': SAMPLE_ITEM_ID, 'translation': '你好'}]),
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