import json
import os
import tempfile
import unittest
from unittest import mock

import batch_cost_estimate
import gemini_translate_batch as batch_mod


class BatchCostEstimateTests(unittest.TestCase):
    def test_estimate_manifest_cost_uses_jsonl_and_pricing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = os.path.join(tmp_dir, 'requests.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'key': 'chunk-00001',
                            'request': {
                                'system_instruction': {'parts': [{'text': 'abcd'}]},
                                'contents': [
                                    {
                                        'role': 'user',
                                        'parts': [{'text': 'efgh'}],
                                    }
                                ],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )

            manifest = {
                'batch_model': 'gemini-3.1-flash-lite',
                'input_jsonl_path': jsonl_path,
                'settings': {'max_output_tokens': 1000},
                'summary': {'chunk_count': 2},
            }
            estimate = batch_cost_estimate.estimate_manifest_cost(manifest)

            self.assertEqual(estimate['request_count'], 1)
            self.assertEqual(estimate['estimated_input_tokens'], 2)
            self.assertEqual(estimate['estimated_output_tokens_max'], 2000)
            self.assertGreater(estimate['estimated_cost_max'], estimate['estimated_cost_min'])

    def test_cost_estimate_exceeds_max(self):
        estimate = {'estimated_cost_max': 12.5}
        self.assertTrue(batch_cost_estimate.cost_estimate_exceeds_max(estimate, 10))
        self.assertFalse(batch_cost_estimate.cost_estimate_exceeds_max(estimate, 12.5))

    def test_estimate_usage_cost_reuses_shared_pricing(self):
        pricing = {
            'currency': 'USD',
            'models': {
                'gemini-test': {
                    'input_per_million': 1.0,
                    'output_per_million': 2.0,
                }
            },
        }
        self.assertEqual(
            batch_cost_estimate.estimate_usage_cost(
                'gemini-test',
                prompt_tokens=1_000_000,
                output_tokens=500_000,
                pricing_config=pricing,
            ),
            2.0,
        )
        self.assertIsNone(
            batch_cost_estimate.estimate_usage_cost(
                'unknown-model',
                prompt_tokens=10,
                output_tokens=5,
                pricing_config=pricing,
            )
        )
        self.assertIsNone(
            batch_cost_estimate.estimate_usage_cost(
                'gemini-test',
                prompt_tokens=None,
                output_tokens=5,
                pricing_config=pricing,
            )
        )

    def test_estimate_final_review_uses_unit_or_request_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = os.path.join(tmp_dir, 'requests.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    json.dumps(
                        {
                            'key': 'fr-u1',
                            'request': {
                                'contents': [
                                    {'role': 'user', 'parts': [{'text': 'abcd'}]},
                                ],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
                handle.write(
                    json.dumps(
                        {
                            'key': 'fr-u2',
                            'request': {
                                'contents': [
                                    {'role': 'user', 'parts': [{'text': 'efgh'}]},
                                ],
                            },
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )

            manifest = {
                'batch_model': 'gemini-3.1-flash-lite',
                'input_jsonl_path': jsonl_path,
                'settings': {'max_output_tokens': 1000},
                'mode': 'final_review',
                'summary': {'unit_count': 2, 'request_count': 2},
            }
            tokens = batch_cost_estimate.estimate_manifest_tokens(manifest)
            self.assertEqual(tokens['request_count'], 2)
            self.assertEqual(tokens['chunk_count'], 2)
            self.assertEqual(tokens['estimated_output_tokens_max'], 2000)

    def test_ensure_manifest_cost_estimate_exits_when_jsonl_missing(self):
        manifest = {
            'input_jsonl_path': 'missing.jsonl',
            'batch_model': 'gemini-3.1-flash-lite',
            'settings': {'max_output_tokens': 1000},
            'summary': {'chunk_count': 1},
        }
        with self.assertRaises(SystemExit) as ctx:
            batch_mod.ensure_manifest_cost_estimate(manifest)
        self.assertIn('Batch input JSONL not found', str(ctx.exception))

    def test_submit_manifest_blocks_when_max_cost_exceeded(self):
        manifest = {
            '_manifest_path': '/tmp/manifest.json',
            'job_name': '',
            'input_jsonl_path': 'missing.jsonl',
            'batch_model': 'gemini-3.1-flash-lite',
            'cost_estimate': {
                'estimated_cost_max': 9.0,
                'currency': 'USD',
            },
        }
        with mock.patch.object(batch_mod, 'load_manifest', return_value=manifest):
            with self.assertRaises(SystemExit) as ctx:
                batch_mod.submit_manifest(target='pkg', max_cost=5.0)
        self.assertIn('Submit blocked by --max-cost', str(ctx.exception))


class DoctorGlossaryStoryGraphTests(unittest.TestCase):
    def test_collect_glossary_story_graph_conflicts_reports_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            glossary_path = os.path.join(tmp_dir, 'glossary.json')
            story_graph_path = os.path.join(tmp_dir, 'story_graph.json')
            with open(glossary_path, 'w', encoding='utf-8') as handle:
                json.dump({'normalize_map': {'Void Gate': '虚空门'}}, handle, ensure_ascii=False)
            with open(story_graph_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'terms': [
                            {'source': 'Void Gate', 'target': '虚空之门'},
                        ]
                    },
                    handle,
                    ensure_ascii=False,
                )

            conflicts = batch_mod.collect_glossary_story_graph_conflicts(
                glossary_path=glossary_path,
                story_graph_path=story_graph_path,
            )
            self.assertEqual(len(conflicts), 1)
            self.assertIn('Void Gate', conflicts[0])
            self.assertIn('虚空门', conflicts[0])
            self.assertIn('虚空之门', conflicts[0])

    def test_collect_glossary_story_graph_conflicts_ignores_whitespace_and_case(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            glossary_path = os.path.join(tmp_dir, 'glossary.json')
            story_graph_path = os.path.join(tmp_dir, 'story_graph.json')
            with open(glossary_path, 'w', encoding='utf-8') as handle:
                json.dump({'normalize_map': {'void gate': '虚空门'}}, handle, ensure_ascii=False)
            with open(story_graph_path, 'w', encoding='utf-8') as handle:
                json.dump(
                    {
                        'terms': [
                            {'source': '  Void   Gate ', 'target': '虚空门'},
                        ]
                    },
                    handle,
                    ensure_ascii=False,
                )

            conflicts = batch_mod.collect_glossary_story_graph_conflicts(
                glossary_path=glossary_path,
                story_graph_path=story_graph_path,
            )
            self.assertEqual(conflicts, [])

    def test_collect_glossary_story_graph_conflicts_missing_files(self):
        self.assertEqual(
            batch_mod.collect_glossary_story_graph_conflicts(
                glossary_path='',
                story_graph_path='',
            ),
            [],
        )


if __name__ == '__main__':
    unittest.main()