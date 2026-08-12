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


class BatchRepairRegressionTests(unittest.TestCase):
    def test_apply_revisions_rejects_mismatched_active_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                'mode': batch_mod.MANIFEST_MODE_REVISION,
                'base_dir': str(root / 'project-a'),
                'tl_dir': str(root / 'project-a' / 'game' / 'tl' / 'schinese'),
            }
            with (
                mock.patch.object(batch_mod, 'load_manifest', return_value=manifest),
                mock.patch.object(
                    batch_mod.legacy,
                    'BASE_DIR',
                    str(root / 'project-b'),
                ),
                mock.patch.object(
                    batch_mod.legacy,
                    'TL_DIR',
                    str(root / 'project-b' / 'game' / 'tl' / 'schinese'),
                ),
                mock.patch.object(batch_mod, 'collect_revision_actions') as collect_mock,
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    'manifest project does not match the active project',
                ):
                    batch_mod.apply_revisions('manifest.json')

            collect_mock.assert_not_called()

    def test_parse_json_payload_recovers_prefixed_keyword_object(self):
        payload = batch_mod.parse_json_payload(
            'Earlier attempt: []\n'
            'Here is the JSON: {"candidates":[],"chunk_summary":"片段概要","summary_evidence_item_ids":["line-1"]}\n'
            'Done.'
        )

        self.assertEqual(payload['chunk_summary'], '片段概要')
        self.assertEqual(payload['summary_evidence_item_ids'], ['line-1'])

    def test_parse_json_payload_preserves_partial_array_salvage(self):
        payload = batch_mod.parse_json_payload(
            '[{"id":"line-1","translation":"第一行"},{"id":"line-2","translation":"第二行"'
        )

        self.assertIsInstance(payload, list)
        self.assertEqual(payload, [{'id': 'line-1', 'translation': '第一行'}])

    def test_parse_json_payload_preserves_fenced_json_compatibility(self):
        payload = batch_mod.parse_json_payload(
            '```json\n'
            '{"translations":[{"id":"line-1","translation":"第一行"}]}\n'
            '```'
        )

        self.assertEqual(
            payload,
            {'translations': [{'id': 'line-1', 'translation': '第一行'}]},
        )

    def test_split_manifest_keeps_first_child_latest_and_context_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            input_path = package_dir / 'requests.jsonl'
            manifest_path = package_dir / 'manifest.json'
            latest_path = root / 'latest_manifest.txt'
            chunks = [
                {
                    'key': 'chunk-1',
                    'file_rel_path': 'script.rpy',
                    'file_path': str(root / 'script.rpy'),
                    'items': [{'id': 'script.rpy:0:4', 'text': 'Hello'}],
                    'glossary_hits': [{'source': 'Hello', 'target': '\u4f60\u597d'}],
                    'history_hits': [
                        {'source_text': 'Hello', 'translated_text': '\u4f60\u597d'},
                        {'source_text': 'Hi', 'translated_text': '\u55e8'},
                    ],
                    'story_hits': {'terms': [{'source': 'Void Gate', 'target': '\u865a\u7a7a\u95e8'}]},
                },
                {
                    'key': 'chunk-2',
                    'file_rel_path': 'script.rpy',
                    'file_path': str(root / 'script.rpy'),
                    'items': [{'id': 'script.rpy:1:4', 'text': 'World'}],
                    'history_hits': [{'source_text': 'World', 'translated_text': '\u4e16\u754c'}],
                    'rag_stats': {'error': 'embedding failed'},
                },
            ]
            input_path.write_text(
                json.dumps({'key': 'chunk-1', 'request': {}}, ensure_ascii=False) + '\n' +
                json.dumps({'key': 'chunk-2', 'request': {}}, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'display_name': 'demo',
                        'batch_model': 'gemini-test',
                        'input_jsonl_path': str(input_path),
                        'settings': {'target_size': 1},
                        'rag_enabled': True,
                        'rag_store_path': str(root / 'rag_store'),
                        'rag_settings': {'top_k_history': 4},
                        'rag_summary': {
                            'prepare': {'upserted': 3},
                            'chunks_with_history_hits': 2,
                            'history_hit_count': 3,
                            'history_hit_rate': 1.0,
                            'history_retrieval_errors': 1,
                        },
                        'story_memory_enabled': True,
                        'story_memory_graph_file': str(root / 'story_graph.json'),
                        'story_memory_settings': {'top_k_terms': 8},
                        'story_memory_summary': {'chunks_with_story_hits': 1},
                        'chunks': chunks,
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)):
                created = batch_mod.split_manifest(str(manifest_path), max_chunks=1)

            self.assertEqual(latest_path.read_text(encoding='utf-8'), created[0])
            source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(source_manifest['job_state'], 'LOCAL_SPLIT_SOURCE')
            first_child = json.loads(Path(created[0]).read_text(encoding='utf-8'))
            self.assertTrue(first_child['rag_enabled'])
            self.assertEqual(first_child['rag_settings'], {'top_k_history': 4})
            self.assertTrue(first_child['story_memory_enabled'])
            self.assertEqual(first_child['story_memory_settings'], {'top_k_terms': 8})
            self.assertEqual(first_child['rag_summary']['chunks_with_history_hits'], 1)
            self.assertEqual(first_child['rag_summary']['chunks_with_glossary_hits'], 1)
            self.assertEqual(first_child['rag_summary']['history_hit_count'], 2)
            self.assertEqual(first_child['rag_summary']['history_hit_rate'], 1.0)
            self.assertEqual(first_child['rag_summary']['history_retrieval_errors'], 0)
            self.assertEqual(first_child['rag_summary']['prepare'], {'upserted': 3})
            self.assertEqual(first_child['story_memory_summary']['chunks_with_story_hits'], 1)
            self.assertEqual(first_child['story_memory_summary']['graph_file'], str(root / 'story_graph.json'))
            self.assertEqual(first_child['story_memory_summary']['hit_counts']['terms'], 1)
            self.assertEqual(first_child['story_memory_summary']['total_hit_count'], 1)
            self.assertEqual(first_child['story_memory_summary']['story_hit_rate'], 1.0)
            self.assertEqual(first_child['story_memory_summary']['truncated_story_blocks'], 0)
            self.assertGreater(first_child['story_memory_summary']['formatted_char_count'], 0)
            first_child['_manifest_path'] = created[0]
            first_child['_package_dir'] = str(Path(created[0]).parent)
            next_manifest = batch_mod._canonical_abs_path(created[1])
            self.assertEqual(
                batch_mod.next_split_manifest_path(first_child),
                next_manifest,
            )
            self.assertEqual(
                batch_mod.mark_next_split_after_apply(first_child),
                next_manifest,
            )
            self.assertEqual(first_child['next_split_manifest_path'], next_manifest)

    def test_submit_quota_failure_records_split_recommendation(self):
        class QuotaError(Exception):
            status_code = 429

        class UploadedFile:
            name = 'files/uploaded'

        class FakeFiles:
            def upload(self, **_kwargs):
                return UploadedFile()

        class FakeBatches:
            def create(self, **_kwargs):
                raise QuotaError('429 RESOURCE_EXHAUSTED')

        class FakeClient:
            files = FakeFiles()
            batches = FakeBatches()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            input_path = package_dir / 'requests.jsonl'
            manifest_path = package_dir / 'manifest.json'
            latest_path = root / 'latest_manifest.txt'
            input_path.write_text('{}\n', encoding='utf-8')
            chunks = [
                {'key': f'chunk-{index}', 'items': [{'id': str(index)}]}
                for index in range(401)
            ]
            manifest_path.write_text(
                json.dumps(
                    {
                        'display_name': 'demo large package',
                        'batch_model': 'gemini-test',
                        'input_jsonl_path': str(input_path),
                        'job_name': '',
                        'summary': {'chunk_count': 401, 'item_count': 401},
                        'chunks': chunks,
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            fake_types = mock.Mock(UploadFileConfig=lambda **kwargs: kwargs)
            stdout = io.StringIO()

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)), \
                 mock.patch.object(batch_mod.legacy, 'API_KEYS', ['key']), \
                 mock.patch.object(batch_mod, 'genai_types', fake_types), \
                 mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()), \
                 mock.patch('sys.stdout', stdout):
                with self.assertRaises(QuotaError):
                    batch_mod.submit_manifest(str(manifest_path))

            saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            recommendation = saved['last_submit_quota_recommendation']
            self.assertEqual(saved['job_state'], 'SUBMIT_FAILED')
            self.assertEqual(saved['last_submit_error_type'], 'quota_or_resource_exhausted')
            self.assertTrue(saved['split_recommended'])
            self.assertIn('--max-chunks 400', recommendation['command'])
            self.assertIn('--max-items 12000', recommendation['command'])
            self.assertIn('Suggested split command:', stdout.getvalue())

    def test_submit_manifest_clears_split_recommendation_after_quota_retry_success(self):
        class QuotaError(Exception):
            status_code = 429

        class UploadedFile:
            name = 'files/uploaded'

        class BatchJob:
            name = 'batches/job-1'
            state = 'JOB_STATE_PENDING'

        class FakeFiles:
            def upload(self, **_kwargs):
                return UploadedFile()

        class FakeBatches:
            def __init__(self):
                self.calls = 0

            def create(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise QuotaError('429 RESOURCE_EXHAUSTED')
                return BatchJob()

        class FakeClient:
            def __init__(self):
                self.batches = FakeBatches()

            files = FakeFiles()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            input_path = package_dir / 'requests.jsonl'
            manifest_path = package_dir / 'manifest.json'
            latest_path = root / 'latest_manifest.txt'
            input_path.write_text('{}\n', encoding='utf-8')
            chunks = [
                {'key': f'chunk-{index}', 'items': [{'id': str(index)}]}
                for index in range(401)
            ]
            manifest_path.write_text(
                json.dumps(
                    {
                        'display_name': 'demo large package',
                        'batch_model': 'gemini-test',
                        'input_jsonl_path': str(input_path),
                        'job_name': '',
                        'summary': {'chunk_count': 401, 'item_count': 401},
                        'chunks': chunks,
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            fake_types = mock.Mock(UploadFileConfig=lambda **kwargs: kwargs)

            with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)), \
                 mock.patch.object(batch_mod.legacy, 'API_KEYS', ['key-a', 'key-b']), \
                 mock.patch.object(batch_mod, 'genai_types', fake_types), \
                 mock.patch.object(batch_mod, 'create_batch_client', return_value=FakeClient()), \
                 mock.patch.object(batch_mod.legacy, 'rotate_api_key', return_value=True):
                result = batch_mod.submit_manifest(str(manifest_path))

            saved = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(result, str(manifest_path))
            self.assertEqual(saved['job_name'], 'batches/job-1')
            self.assertEqual(saved['last_submit_error'], '')
            self.assertNotIn('last_submit_error_type', saved)
            self.assertNotIn('split_recommended', saved)
            self.assertNotIn('last_submit_quota_recommendation', saved)

    def test_retry_package_and_merge_replace_only_failed_chunks(self):
        old_tl_dir = batch_mod.legacy.TL_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'tl'
                package_dir = root / 'package'
                latest_path = root / 'latest_manifest.txt'
                tl_dir.mkdir()
                package_dir.mkdir()
                target_file = tl_dir / 'script.rpy'
                target_file.write_text(
                    '    "Hello"\n'
                    '    "World"\n'
                    '    "Again"\n',
                    encoding='utf-8',
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)

                chunks = [
                    {
                        'key': 'chunk-ok',
                        'file_rel_path': 'script.rpy',
                        'file_path': str(target_file),
                        'chunk_index': 1,
                        'context_past': [],
                        'context_future': [],
                        'items': [
                            {
                                'id': 'script.rpy:0:4:11:hello',
                                'text': 'Hello',
                                'line': 0,
                                'start': 4,
                                'end': 11,
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    },
                    {
                        'key': 'chunk-bad',
                        'file_rel_path': 'script.rpy',
                        'file_path': str(target_file),
                        'chunk_index': 2,
                        'context_past': [],
                        'context_future': [],
                        'items': [
                            {
                                'id': 'script.rpy:1:4:11:world',
                                'text': 'World',
                                'line': 1,
                                'start': 4,
                                'end': 11,
                                'prefix': '',
                                'quote': '"',
                            },
                            {
                                'id': 'script.rpy:2:4:11:again',
                                'text': 'Again',
                                'line': 2,
                                'start': 4,
                                'end': 11,
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    },
                ]

                parent_result_path = package_dir / 'results.jsonl'
                batch_mod.write_jsonl_file(
                    str(parent_result_path),
                    [
                        {
                            'key': 'chunk-ok',
                            'response': {
                                'candidates': [
                                    {
                                        'content': {
                                            'parts': [
                                                {
                                                    'text': json.dumps(
                                                        [{'id': 'script.rpy:0:4:11:hello', 'translation': '你好'}],
                                                        ensure_ascii=False,
                                                    )
                                                }
                                            ]
                                        },
                                        'finishReason': 'STOP',
                                    }
                                ]
                            },
                        },
                        {
                            'key': 'chunk-bad',
                            'response': {
                                'candidates': [
                                    {
                                        'content': {
                                            'parts': [
                                                {
                                                    'text': json.dumps(
                                                        [
                                                            {
                                                                'id': 'script.rpy:1:4:11:world',
                                                                'translation': '世界',
                                                            }
                                                        ],
                                                        ensure_ascii=False,
                                                    )
                                                }
                                            ]
                                        },
                                        'finishReason': 'STOP',
                                    }
                                ]
                            },
                        },
                    ],
                )

                manifest_path = package_dir / 'manifest.json'
                manifest_path.write_text(
                    json.dumps(
                        {
                            'version': 1,
                            'core_schema_version': translation_core.CORE_SCHEMA_VERSION,
                            'mode': batch_mod.MANIFEST_MODE_TRANSLATION,
                            'display_name': 'demo',
                            'batch_model': 'gemini-test',
                            'input_jsonl_path': str(package_dir / 'requests.jsonl'),
                            'result_jsonl_path': 'results.jsonl',
                            'settings': {'target_size': 2},
                            'files': {'script.rpy': {'path': str(target_file), 'task_count': 3}},
                            'summary': {'file_count': 1, 'chunk_count': 2, 'item_count': 3},
                            'source_index_enabled': True,
                            'source_index_store_path': str(root / 'source_index'),
                            'source_index_settings': {'top_k': 4},
                            'chunks': chunks,
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )
                batch_mod.write_jsonl_file(
                    str(package_dir / 'requests.jsonl'),
                    [batch_mod.build_batch_request(chunk) for chunk in chunks],
                )

                with mock.patch.object(batch_mod, 'LATEST_MANIFEST_FILE', str(latest_path)):
                    checked = batch_mod.check_results(str(manifest_path))
                    self.assertEqual(checked['last_check_summary']['safety_level'], batch_mod.CHECK_SAFETY_WARN)

                    retry_manifest_path = batch_mod.build_retry_package(str(manifest_path))
                    retry_manifest = json.loads(Path(retry_manifest_path).read_text(encoding='utf-8'))
                    self.assertEqual(
                        [chunk['key'] for chunk in retry_manifest['chunks']],
                        ['chunk-bad-retry-001'],
                    )
                    self.assertEqual(retry_manifest['summary']['item_count'], 1)
                    self.assertEqual(
                        retry_manifest['chunks'][0]['retry_item_ids'],
                        ['script.rpy:2:4:11:again'],
                    )
                    self.assertTrue(retry_manifest['source_index_enabled'])
                    self.assertEqual(latest_path.read_text(encoding='utf-8'), retry_manifest_path)
                    retry_request = json.loads(Path(retry_manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()[0])
                    self.assertEqual(retry_request['key'], 'chunk-bad-retry-001')
                    self.assertIn(
                        'copy that exact source substring verbatim',
                        retry_request['request']['system_instruction']['parts'][0]['text'],
                    )

                    retry_result_path = Path(retry_manifest_path).parent / 'results.jsonl'
                    batch_mod.write_jsonl_file(
                        str(retry_result_path),
                        [
                            {
                                'key': 'chunk-bad-retry-001',
                                'response': {
                                    'candidates': [
                                        {
                                            'content': {
                                                'parts': [
                                                    {
                                                        'text': json.dumps(
                                                            [
                                                                {
                                                                    'id': 'script.rpy:2:4:11:again',
                                                                    'translation': '再次',
                                                                }
                                                            ],
                                                            ensure_ascii=False,
                                                        )
                                                    }
                                                ]
                                            },
                                            'finishReason': 'STOP',
                                        }
                                    ]
                                },
                            }
                        ],
                    )
                    retry_manifest['result_jsonl_path'] = 'results.jsonl'
                    Path(retry_manifest_path).write_text(json.dumps(retry_manifest, ensure_ascii=False), encoding='utf-8')

                    merged_parent_path = batch_mod.merge_retry_results(str(manifest_path), retry_manifest_path)
                    merged_manifest = json.loads(Path(merged_parent_path).read_text(encoding='utf-8'))
                    self.assertNotEqual(merged_manifest['result_jsonl_path'], 'results.jsonl')
                    self.assertNotIn('last_check_summary', merged_manifest)

                    merged_rows = [
                        json.loads(line)
                        for line in (package_dir / merged_manifest['result_jsonl_path']).read_text(encoding='utf-8').splitlines()
                    ]
                    self.assertEqual([row['key'] for row in merged_rows], ['chunk-ok', 'chunk-bad'])
                    self.assertIn('世界', json.dumps(merged_rows[1], ensure_ascii=False))
                    self.assertIn('再次', json.dumps(merged_rows[1], ensure_ascii=False))
                    merged_text = batch_mod.extract_text_from_response_payload(
                        merged_rows[1]['response']
                    )
                    self.assertIsInstance(json.loads(merged_text), list)
                    self.assertIn('translations', merged_rows[1]['normalized_response'])

                    rechecked = batch_mod.check_results(str(manifest_path))
                    self.assertEqual(rechecked['last_check_summary']['safety_level'], batch_mod.CHECK_SAFETY_SAFE)
        finally:
            batch_mod.legacy.TL_DIR = old_tl_dir

    def test_partial_retry_merge_preserves_response_and_updates_normalized(self):
        parent_chunk = {
            'key': 'parent',
            'items': [
                {'id': 'a', 'text': 'Hello'},
                {'id': 'b', 'text': 'World'},
            ],
        }
        retry_chunk = {
            'key': 'retry-b',
            'retry_parent_key': 'parent',
            'items': [{'id': 'b', 'text': 'World'}],
        }
        parent_response = batch_mod.response_payload_with_text(
            {},
            json.dumps(
                {'translations': [{'id': 'a', 'translation': '你好'}]},
                ensure_ascii=False,
            ),
        )
        retry_response = batch_mod.response_payload_with_text(
            {},
            json.dumps(
                {
                    'translations': [
                        {'id': 'a', 'translation': '不应覆盖'},
                        {'id': 'b', 'translation': '世界'},
                    ]
                },
                ensure_ascii=False,
            ),
        )
        parent_row = {
            'key': 'parent',
            'response': parent_response,
            'normalized_response': {
                'translations': [{'id': 'a', 'translation': '你好'}],
            },
        }

        merged, replaced = batch_mod.merge_parent_row_with_retry_item_rows(
            parent_row,
            parent_chunk,
            [retry_chunk],
            {'retry-b': {'key': 'retry-b', 'response': retry_response}},
        )

        self.assertEqual(replaced, 1)
        self.assertEqual(merged['response'], parent_response)
        self.assertEqual(
            {
                item['id']: item['translation']
                for item in merged['normalized_response']['translations']
            },
            {'a': '你好', 'b': '世界'},
        )
        self.assertTrue(merged['contract_diagnostics']['complete'])
        self.assertEqual(
            [
                item['id']
                for item in batch_mod.result_items_from_row(
                    merged,
                    'merged',
                    parent_chunk['items'],
                )
            ],
            ['a', 'b'],
        )

    def test_partial_retry_merge_audits_empty_or_invalid_results(self):
        parent_chunk = {
            'key': 'parent',
            'items': [
                {'id': 'a', 'text': 'Hello'},
                {'id': 'b', 'text': 'World'},
            ],
        }
        retry_chunk = {
            'key': 'retry-b',
            'retry_parent_key': 'parent',
            'items': [{'id': 'b', 'text': 'World'}],
        }
        parent_response = batch_mod.response_payload_with_text(
            {},
            json.dumps(
                {'translations': [{'id': 'a', 'translation': '你好'}]},
                ensure_ascii=False,
            ),
        )
        parent_row = {'key': 'parent', 'response': parent_response}
        retry_payloads = (
            {'translations': []},
            {'translations': [{'id': 'unknown', 'translation': '错误'}]},
        )

        for retry_payload in retry_payloads:
            with self.subTest(retry_payload=retry_payload):
                retry_response = batch_mod.response_payload_with_text(
                    {},
                    json.dumps(retry_payload, ensure_ascii=False),
                )
                merged, replaced = batch_mod.merge_parent_row_with_retry_item_rows(
                    parent_row,
                    parent_chunk,
                    [retry_chunk],
                    {'retry-b': {'key': 'retry-b', 'response': retry_response}},
                )

                self.assertEqual(replaced, 0)
                self.assertEqual(merged['response'], parent_response)
                self.assertEqual(
                    merged['normalized_response'],
                    {'translations': [{'id': 'a', 'translation': '你好'}]},
                )
                diagnostics = merged['contract_diagnostics']
                self.assertFalse(diagnostics['complete'])
                self.assertEqual(diagnostics['retry_ids'], ['b'])
                self.assertEqual(
                    diagnostics['reason_counts']['response_missing_expected_id'],
                    1,
                )

    def test_direct_retry_canonicalization_preserves_complete_normalized_response(self):
        chunk = {
            'key': 'chunk-1',
            'items': [
                {'id': 'a', 'text': 'Hello'},
                {'id': 'b', 'text': 'World'},
            ],
        }
        first_pass_response = batch_mod.response_payload_with_text(
            {},
            json.dumps(
                {'translations': [{'id': 'a', 'translation': '你好'}]},
                ensure_ascii=False,
            ),
        )
        row = {
            'key': 'chunk-1',
            'response': first_pass_response,
            'normalized_response': {
                'translations': [
                    {'id': 'a', 'translation': '你好'},
                    {'id': 'b', 'translation': '世界'},
                ],
            },
            'contract_diagnostics': {'complete': True, 'custom': 'stale'},
            'response_semantics': {
                'response': 'first_pass_provider_payload',
                'normalized_response': 'final_merged_contract',
            },
        }

        canonical = batch_mod.canonical_translation_result_row(row, chunk)

        self.assertEqual(canonical['response'], first_pass_response)
        self.assertEqual(
            [
                item['id']
                for item in canonical['normalized_response']['translations']
            ],
            ['a', 'b'],
        )
        self.assertTrue(canonical['contract_diagnostics']['complete'])
        self.assertNotIn('custom', canonical['contract_diagnostics'])
        self.assertEqual(
            canonical['response_semantics'],
            row['response_semantics'],
        )

    def test_create_keyword_package_uses_keyword_mode_manifest(self):
        old_values = {
            'tl_dir': batch_mod.legacy.TL_DIR,
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'tl'
                jobs_dir = root / 'batch_jobs'
                tl_dir.mkdir()
                target_file = tl_dir / 'script.rpy'
                target_file.write_text(
                    'translate schinese start:\n'
                    '    old "Void Gate"\n'
                    '    new "虚空门"\n'
                    'label demo:\n'
                    '    e "Aether Compass"\n',
                    encoding='utf-8',
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.LOG_DIR = str(root / 'logs')
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / 'repair_runs')
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')

                with mock.patch.object(batch_mod.legacy, 'run_prepare_steps') as prepare_mock:
                    manifest_path = batch_mod.create_keyword_package(
                        chunk_size=1,
                        max_candidates_per_chunk=3,
                    )
                manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
                request_rows = [
                    json.loads(line)
                    for line in Path(manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()
                ]
        finally:
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.LOG_DIR = old_values['log_dir']
            batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
            batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
            batch_mod.LATEST_MANIFEST_FILE = old_values['latest']

        self.assertEqual(manifest['mode'], batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION)
        self.assertEqual(manifest['summary']['item_count'], 2)
        self.assertEqual(manifest['summary']['chunk_count'], 2)
        self.assertEqual(manifest['chunks'][0]['items'][0]['line_number'], 2)
        prepare_mock.assert_not_called()
        schema = request_rows[0]['request']['generation_config']['response_json_schema']
        candidate_schema = schema['properties']['candidates']
        self.assertEqual(schema['type'], 'object')
        self.assertIn('candidates', schema['required'])
        self.assertIn('chunk_summary', schema['required'])
        self.assertIn('summary_evidence_item_ids', schema['required'])
        self.assertEqual(candidate_schema['maxItems'], 3)
        self.assertIn('source', candidate_schema['items']['required'])
        self.assertIn('source_item_ids', candidate_schema['items']['required'])
        self.assertIn('source_item_ids', candidate_schema['items']['properties'])
        system_text = request_rows[0]['request']['system_instruction']['parts'][0]['text']
        self.assertIn('Existing glossary entries', system_text)
        self.assertIn('chunk_summary', system_text)

    def test_export_keyword_candidates_dedupes_jsonl_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                {
                    'candidates': [
                        {
                            'source': 'Void Gate',
                            'suggested_target': '虚空门',
                            'category': 'term',
                            'confidence': 0.7,
                            'evidence': 'script.rpy:2',
                            'source_item_ids': ['script.rpy:2:keyword:0'],
                        },
                        {
                            'source': 'Void Gate',
                            'suggested_target': '虚空门',
                            'category': 'term',
                            'confidence': 0.9,
                            'evidence': 'script.rpy:3',
                            'source_item_ids': ['script.rpy:3:keyword:1'],
                        },
                    ],
                    'chunk_summary': '一行提到虚空门，另一行提到其他术语。',
                    'summary_evidence_item_ids': ['script.rpy:2:keyword:0', 'script.rpy:3:keyword:1'],
                },
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'kw-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'mode': batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION,
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'kw-1',
                                'file_rel_path': 'script.rpy',
                                'line_numbers': [2, 3],
                                'items': [
                                    {
                                        'id': 'script.rpy:2:keyword:0',
                                        'line_number': 2,
                                        'text': 'Void Gate',
                                    },
                                    {
                                        'id': 'script.rpy:3:keyword:1',
                                        'line_number': 3,
                                        'text': 'Other Term',
                                    },
                                ],
                            },
                            {
                                'key': 'kw-2',
                                'file_rel_path': 'script.rpy',
                                'line_numbers': [4],
                                'items': [
                                    {
                                        'id': 'script.rpy:4:keyword:2',
                                        'line_number': 4,
                                        'text': 'Missing Term',
                                    },
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            export = batch_mod.export_keyword_candidates(str(manifest_path))
            jsonl_path = Path(export['jsonl_path'])
            markdown_path = Path(export['markdown_path'])
            summary_jsonl_path = Path(export['summary_jsonl_path'])
            summary_markdown_path = Path(export['summary_markdown_path'])
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding='utf-8').splitlines()
            ]
            summary_rows = [
                json.loads(line)
                for line in summary_jsonl_path.read_text(encoding='utf-8').splitlines()
            ]
            markdown_text = markdown_path.read_text(encoding='utf-8')
            summary_markdown_text = summary_markdown_path.read_text(encoding='utf-8')

        self.assertEqual(export['summary']['candidate_count_raw'], 2)
        self.assertEqual(export['summary']['candidate_count_deduped'], 1)
        self.assertEqual(export['summary']['chunk_summary_count'], 1)
        self.assertEqual(rows[0]['source'], 'Void Gate')
        self.assertEqual(rows[0]['confidence'], 0.9)
        self.assertEqual(rows[0]['occurrences'], 2)
        self.assertEqual(rows[0]['source_lines'], [2, 3])
        self.assertEqual(
            rows[0]['source_item_ids'],
            ['script.rpy:2:keyword:0', 'script.rpy:3:keyword:1'],
        )
        self.assertEqual(summary_rows[0]['chunk_summary'], '一行提到虚空门，另一行提到其他术语。')
        self.assertEqual(summary_rows[0]['source_lines'], [2, 3])
        self.assertEqual(export['summary']['missing_chunk_rows'], 1)
        self.assertIn('Void Gate', markdown_text)
        self.assertIn('虚空门', summary_markdown_text)
        self.assertIn('Chunk lines', summary_markdown_text)
        self.assertIn('Evidence lines', summary_markdown_text)

    def test_export_keyword_candidates_accepts_legacy_candidate_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [
                    {
                        'source': 'Legacy Term',
                        'suggested_target': '旧术语',
                        'category': 'term',
                        'confidence': 0.8,
                        'evidence': 'script.rpy:2',
                        'source_item_ids': ['script.rpy:2:keyword:0'],
                    }
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'kw-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'mode': batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION,
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'kw-1',
                                'file_rel_path': 'script.rpy',
                                'line_numbers': [2],
                                'items': [
                                    {
                                        'id': 'script.rpy:2:keyword:0',
                                        'line_number': 2,
                                        'text': 'Legacy Term',
                                    },
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            export = batch_mod.export_keyword_candidates(str(manifest_path))
            rows = [
                json.loads(line)
                for line in Path(export['jsonl_path']).read_text(encoding='utf-8').splitlines()
            ]
            summary_jsonl_text = Path(export['summary_jsonl_path']).read_text(encoding='utf-8')

        self.assertEqual(rows[0]['source'], 'Legacy Term')
        self.assertEqual(export['summary']['candidate_count_deduped'], 1)
        self.assertEqual(export['summary']['chunk_summary_count'], 0)
        self.assertEqual(summary_jsonl_text, '')

    def test_export_keyword_candidates_rejects_reserved_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'kw-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': '[]'}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'mode': batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION,
                        'input_jsonl_path': str(package_dir / 'requests.jsonl'),
                        'result_jsonl_path': str(result_path),
                        'chunks': [{'key': 'kw-1', 'file_rel_path': 'script.rpy', 'items': []}],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(SystemExit, 'reserved package file'):
                batch_mod.export_keyword_candidates(str(manifest_path), output_jsonl='results.jsonl')
            with self.assertRaisesRegex(SystemExit, 'must be different files'):
                batch_mod.export_keyword_candidates(
                    str(manifest_path),
                    output_jsonl='same.jsonl',
                    output_markdown='same.jsonl',
                )

    def test_create_batch_package_dir_avoids_existing_directory(self):
        old_jobs_dir = batch_mod.BATCH_JOBS_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                batch_mod.BATCH_JOBS_DIR = tmp
                first = batch_mod.create_batch_package_dir('same_package')
                second = batch_mod.create_batch_package_dir('same_package')
                self.assertNotEqual(first, second)
                self.assertTrue(Path(first).is_dir())
                self.assertTrue(Path(second).is_dir())
        finally:
            batch_mod.BATCH_JOBS_DIR = old_jobs_dir

    def test_check_and_apply_reject_non_translation_manifests(self):
        for mode in (batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION, batch_mod.MANIFEST_MODE_REVISION):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                manifest_path = Path(tmp) / 'manifest.json'
                manifest_path.write_text(
                    json.dumps({'mode': mode}),
                    encoding='utf-8',
                )

                with self.assertRaisesRegex(SystemExit, 'check only supports translation manifests'):
                    batch_mod.check_results(str(manifest_path))
                with self.assertRaisesRegex(SystemExit, 'apply only supports translation manifests'):
                    batch_mod.apply_results(str(manifest_path))

    def test_create_revision_package_uses_revision_mode_manifest(self):
        old_values = {
            'tl_dir': batch_mod.legacy.TL_DIR,
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'tl'
                jobs_dir = root / 'batch_jobs'
                tl_dir.mkdir()
                target_file = tl_dir / 'script.rpy'
                target_file.write_text(
                    'translate schinese start:\n'
                    '    old "Void Gate"\n'
                    '    new "虚空门"\n',
                    encoding='utf-8',
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.LOG_DIR = str(root / 'logs')
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / 'repair_runs')
                batch_mod.SYNC_RUNS_DIR = str(root / 'sync_runs')
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')

                manifest_path = batch_mod.create_revision_package(skip_prepare=True, chunk_size=1)
                manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
                request_rows = [
                    json.loads(line)
                    for line in Path(manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()
                ]
        finally:
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.LOG_DIR = old_values['log_dir']
            batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
            batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
            batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
            batch_mod.LATEST_MANIFEST_FILE = old_values['latest']

        schema = request_rows[0]['request']['generation_config']['response_json_schema']
        target_text = request_rows[0]['request']['contents'][0]['parts'][0]['text']
        self.assertEqual(manifest['mode'], batch_mod.MANIFEST_MODE_REVISION)
        self.assertEqual(manifest['summary']['item_count'], 1)
        self.assertIn('build_warnings', manifest)
        self.assertNotIn('warnings', manifest)
        self.assertIn(
            'should_update',
            schema['properties']['revisions']['items']['required'],
        )
        self.assertIn('current_translation', target_text)

    def test_sync_keyword_candidates_runs_requests_and_exports_reports(self):
        old_values = {
            'tl_dir': batch_mod.legacy.TL_DIR,
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'tl'
                jobs_dir = root / 'batch_jobs'
                tl_dir.mkdir()
                jobs_dir.mkdir()
                previous_latest = root / 'previous_manifest.json'
                target_file = tl_dir / 'script.rpy'
                target_file.write_text(
                    'translate schinese start:\n'
                    '    old "Void Gate"\n'
                    '    new "虚空门"\n',
                    encoding='utf-8',
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.LOG_DIR = str(root / 'logs')
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / 'repair_runs')
                batch_mod.SYNC_RUNS_DIR = str(root / 'sync_runs')
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')
                Path(batch_mod.LATEST_MANIFEST_FILE).write_text(str(previous_latest), encoding='utf-8')
                response_text = json.dumps(
                    {
                        'candidates': [
                            {
                                'source': 'Void Gate',
                                'suggested_target': '虚空门',
                                'category': 'term',
                                'confidence': 0.9,
                                'evidence': 'script.rpy:2:keyword:0',
                                'source_item_ids': ['script.rpy:2:keyword:0'],
                            }
                        ],
                        'chunk_summary': '这里提到了虚空门。',
                        'summary_evidence_item_ids': ['script.rpy:2:keyword:0'],
                    },
                    ensure_ascii=False,
                )

                with mock.patch.object(
                    batch_mod,
                    'run_sync_request',
                    return_value={
                        'response_payload': {
                            'candidates': [{'content': {'parts': [{'text': response_text}]}}],
                        },
                        'response_text': response_text,
                        'finish_reason': 'STOP',
                        'usage_metadata': {},
                    },
                ) as sync_request:
                    export = batch_mod.sync_keyword_candidates(skip_prepare=True, chunk_size=1, limit=1)

                jsonl_path = Path(export['jsonl_path'])
                rows = [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding='utf-8').splitlines()
                ]
                summary_rows = [
                    json.loads(line)
                    for line in Path(export['summary_jsonl_path']).read_text(encoding='utf-8').splitlines()
                ]
                latest_after = Path(batch_mod.LATEST_MANIFEST_FILE).read_text(encoding='utf-8')
        finally:
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.LOG_DIR = old_values['log_dir']
            batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
            batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
            batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
            batch_mod.LATEST_MANIFEST_FILE = old_values['latest']

        sync_request.assert_called_once()
        self.assertEqual(export['summary']['candidate_count_deduped'], 1)
        self.assertEqual(export['summary']['chunk_summary_count'], 1)
        self.assertEqual(rows[0]['source'], 'Void Gate')
        self.assertEqual(rows[0]['suggested_target'], '虚空门')
        self.assertEqual(summary_rows[0]['chunk_summary'], '这里提到了虚空门。')
        self.assertEqual(latest_after, str(previous_latest))

    def test_execute_sync_rows_targeted_retry_merges_only_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / 'sync-run'
            package_dir.mkdir()
            chunk = {
                'key': 'chunk-1',
                'file_path': str(package_dir / 'script.rpy'),
                'file_rel_path': 'script.rpy',
                'chunk_index': 1,
                'context_past': [],
                'context_future': [],
                'items': [
                    {
                        'id': 'a',
                        'text': 'Hello',
                        'file_rel_path': 'script.rpy',
                        'line': 0,
                        'line_number': 1,
                        'start': 4,
                        'end': 11,
                        'prefix': '',
                        'quote': '"',
                    },
                    {
                        'id': 'b',
                        'text': 'World',
                        'file_rel_path': 'script.rpy',
                        'line': 1,
                        'line_number': 2,
                        'start': 4,
                        'end': 11,
                        'prefix': '',
                        'quote': '"',
                    },
                ],
            }
            request_rows = [batch_mod.build_batch_request(chunk, model='gemini-test')]
            unselected_chunk = {
                'key': 'chunk-2',
                'file_path': str(package_dir / 'extra.rpy'),
                'file_rel_path': 'extra.rpy',
                'chunk_index': 2,
                'context_past': [],
                'context_future': [],
                'items': [
                    {
                        'id': 'c',
                        'text': 'Unused',
                        'file_rel_path': 'extra.rpy',
                        'line': 0,
                        'line_number': 1,
                        'start': 4,
                        'end': 12,
                        'prefix': '',
                        'quote': '"',
                    },
                ],
            }
            manifest_path = batch_mod.make_sync_manifest(
                package_dir=str(package_dir),
                mode=batch_mod.MANIFEST_MODE_TRANSLATION,
                display_name='targeted-retry-test',
                chunks=[chunk, unselected_chunk],
                request_rows=request_rows,
                settings={},
            )
            responses = [
                {'translations': [{'id': 'a', 'translation': '你好'}]},
                {'translations': [{'id': 'b', 'translation': '世界'}]},
            ]
            seen_requests = []
            seen_models = []

            def fake_sync(request, model_name, **_kwargs):
                seen_requests.append(request)
                seen_models.append(model_name)
                payload = responses[len(seen_requests) - 1]
                text = json.dumps(payload, ensure_ascii=False)
                return {
                    'response_payload': {
                        'candidates': [{'content': {'parts': [{'text': text}]}}],
                    },
                    'response_text': text,
                    'finish_reason': 'STOP',
                    'usage_metadata': {'totalTokenCount': 1},
                    'provider': 'gemini',
                    'model': 'gemini-test',
                    'execution_mode': 'sync',
                }

            with (
                mock.patch.object(batch_mod, 'SYNC_MODEL', 'sync-override'),
                mock.patch.object(
                    batch_mod,
                    'run_sync_request',
                    side_effect=fake_sync,
                ),
            ):
                manifest = batch_mod.execute_sync_request_rows(
                    manifest_path,
                    request_rows,
                )

            result_row = json.loads(
                Path(manifest['result_jsonl_path']).read_text(encoding='utf-8').strip()
            )
            retry_prompt = seen_requests[1]['contents'][0]['parts'][0]['text']

        self.assertEqual(len(seen_requests), 2)
        self.assertEqual(seen_models, ['sync-override', 'sync-override'])
        self.assertNotIn('"id":"a"', retry_prompt)
        self.assertIn('"id":"b"', retry_prompt)
        self.assertEqual(
            [item['id'] for item in result_row['normalized_response']['translations']],
            ['a', 'b'],
        )
        self.assertTrue(result_row['contract_diagnostics']['complete'])
        self.assertEqual(
            result_row['response_semantics'],
            {
                'response': 'first_pass_provider_payload',
                'normalized_response': 'final_merged_contract',
            },
        )
        first_pass_payload = batch_mod.parse_json_payload(
            batch_mod.extract_text_from_response_payload(result_row['response'])
        )
        self.assertEqual(
            [item['id'] for item in first_pass_payload['translations']],
            ['a'],
        )
        authoritative_payload = batch_mod.result_row_contract_payload(result_row)
        self.assertEqual(
            [item['id'] for item in authoritative_payload['translations']],
            ['a', 'b'],
        )
        self.assertNotIn('response', result_row['provider_response_attempts'][0])
        self.assertIn('response', result_row['provider_response_attempts'][1])
        self.assertEqual(manifest['sync_summary']['contract_expected_items'], 2)
        self.assertEqual(manifest['sync_summary']['targeted_retry_requests'], 1)
        self.assertEqual(manifest['sync_summary']['contract_final_completeness'], 1.0)
        self.assertEqual(manifest['job_state'], 'SYNC_COMPLETED')

    def test_keyword_targeted_retry_preserves_valid_first_pass_candidates(self):
        items = [
            {'id': 'a', 'text': 'Void Gate'},
            {'id': 'b', 'text': 'Moon Key'},
        ]
        first = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'place',
                        'confidence': 0.9,
                        'evidence': 'a',
                        'source_item_ids': ['a'],
                    },
                    {
                        'source': 'Broken',
                        'suggested_target': '错误',
                        'category': 'term',
                        'confidence': 0.5,
                        'evidence': 'unknown',
                        'source_item_ids': ['unknown'],
                    },
                ],
                'chunk_summary': 'first',
                'summary_evidence_item_ids': ['a'],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )
        retry = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Moon Key',
                        'suggested_target': '月之钥',
                        'category': 'item',
                        'confidence': 0.8,
                        'evidence': 'b',
                        'source_item_ids': ['b'],
                    },
                ],
                'chunk_summary': 'retry',
                'summary_evidence_item_ids': ['b'],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )

        merged = batch_mod._merge_sync_contract_reports(
            first,
            retry,
            {'items': items},
            translation_core.MODE_KEYWORD_EXTRACTION,
        )

        self.assertTrue(merged.complete)
        self.assertEqual(
            [item['source'] for item in merged.items],
            ['Void Gate', 'Moon Key'],
        )
        self.assertEqual(merged.metadata['chunk_summary'], 'retry')

    def test_keyword_merge_revalidates_combined_candidate_provenance(self):
        items = [
            {'id': 'a', 'text': 'Void Gate'},
            {'id': 'b', 'text': 'Moon Key'},
        ]
        first = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'place',
                        'confidence': 0.9,
                        'evidence': 'a',
                        'source_item_ids': ['a'],
                    },
                ],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )
        first.items.append({
            'source': 'Injected invalid candidate',
            'suggested_target': '非法候选',
            'category': 'term',
            'confidence': 0.5,
            'evidence': 'outside',
            'source_item_ids': ['outside'],
        })
        retry = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Moon Key',
                        'suggested_target': '月之钥',
                        'category': 'item',
                        'confidence': 0.8,
                        'evidence': 'b',
                        'source_item_ids': ['b'],
                    },
                ],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )

        merged = batch_mod._merge_sync_contract_reports(
            first,
            retry,
            {'items': items},
            translation_core.MODE_KEYWORD_EXTRACTION,
        )

        self.assertFalse(merged.complete)
        self.assertEqual(
            [item['source'] for item in merged.items],
            ['Void Gate', 'Moon Key'],
        )
        self.assertEqual(
            merged.reason_counts()['result_unknown_source_id'],
            1,
        )
        self.assertEqual(merged.retry_ids, ['a', 'b'])

    def test_keyword_empty_retry_preserves_first_pass_issue_and_candidates(self):
        items = [{'id': 'a', 'text': 'Void Gate'}]
        first = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'term',
                        'confidence': 0.9,
                        'evidence': 'a',
                        'source_item_ids': ['a'],
                    },
                    {
                        'source': 'Broken',
                        'suggested_target': '错误',
                        'category': 'term',
                        'confidence': 0.5,
                        'evidence': 'unknown',
                        'source_item_ids': ['unknown'],
                    },
                ],
                'chunk_summary': 'first summary',
                'summary_evidence_item_ids': ['a'],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )
        retry = translation_core.validate_model_response(
            {
                'candidates': [],
                'chunk_summary': '',
                'summary_evidence_item_ids': [],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )

        merged = batch_mod._merge_sync_contract_reports(
            first,
            retry,
            {'items': items},
            translation_core.MODE_KEYWORD_EXTRACTION,
        )

        self.assertFalse(merged.complete)
        self.assertEqual([item['source'] for item in merged.items], ['Void Gate'])
        self.assertEqual(merged.metadata['chunk_summary'], 'first summary')
        self.assertEqual(merged.retry_ids, ['a'])
        self.assertEqual(
            merged.reason_counts()['result_unknown_source_id'],
            1,
        )

    def test_translation_retry_cannot_replace_non_retry_first_pass_ids(self):
        items = [
            {'id': 'a', 'text': 'Hello'},
            {'id': 'b', 'text': 'World'},
        ]
        first = translation_core.validate_model_response(
            {
                'translations': [
                    {'id': 'a', 'translation': '首轮译文'},
                ],
            },
            expected_units=items,
        )
        retry = translation_core.validate_model_response(
            {
                'translations': [
                    {'id': 'a', 'translation': '不应覆盖'},
                    {'id': 'b', 'translation': '重试译文'},
                ],
            },
            expected_units=items,
        )

        merged = batch_mod._merge_sync_contract_reports(
            first,
            retry,
            {'items': items},
            translation_core.MODE_TRANSLATION,
        )

        self.assertTrue(merged.complete)
        self.assertEqual(
            {item['id']: item['translation'] for item in merged.items},
            {'a': '首轮译文', 'b': '重试译文'},
        )

    def test_keyword_retry_progress_drops_superseded_first_pass_issues(self):
        items = [
            {'id': 'a', 'text': 'Void Gate'},
            {'id': 'b', 'text': 'Moon Key'},
        ]
        first = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'place',
                        'confidence': 0.9,
                        'evidence': 'a',
                        'source_item_ids': ['a'],
                    },
                    {
                        'source': 'Broken first',
                        'suggested_target': '错误',
                        'category': 'term',
                        'confidence': 0.5,
                        'evidence': 'unknown',
                        'source_item_ids': ['unknown'],
                    },
                ],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )
        retry = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Moon Key',
                        'suggested_target': '月之钥',
                        'category': 'item',
                        'confidence': 0.8,
                        'evidence': 'b',
                        'source_item_ids': ['b'],
                    },
                    {
                        'source': 'Broken retry',
                        'suggested_target': '错误',
                        'confidence': 0.5,
                        'evidence': 'b',
                        'source_item_ids': ['b'],
                    },
                ],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=items,
        )

        merged = batch_mod._merge_sync_contract_reports(
            first,
            retry,
            {'items': items},
            translation_core.MODE_KEYWORD_EXTRACTION,
        )

        self.assertFalse(merged.complete)
        self.assertEqual(
            [item['source'] for item in merged.items],
            ['Void Gate', 'Moon Key'],
        )
        self.assertNotIn('result_unknown_source_id', merged.reason_counts())
        self.assertEqual(merged.reason_counts()['result_missing_field'], 1)
        self.assertEqual(merged.retry_ids, ['a', 'b'])

    def test_sync_keyword_summary_uses_chunk_completeness(self):
        chunk = {
            'key': 'keyword-1',
            'file_path': 'script.rpy',
            'file_rel_path': 'script.rpy',
            'items': [{'id': 'a', 'text': 'Void Gate'}],
        }
        first_payload = {
            'candidates': [
                {
                    'source': 'Void Gate',
                    'suggested_target': '虚空门',
                    'category': 'place',
                    'confidence': 0.9,
                    'evidence': 'a',
                    'source_item_ids': ['a'],
                },
                {
                    'source': 'Broken',
                    'suggested_target': '错误',
                    'category': 'term',
                    'confidence': 0.5,
                    'evidence': 'unknown',
                    'source_item_ids': ['unknown'],
                },
            ],
        }
        retry_payload = {'candidates': []}
        payloads = [first_payload, retry_payload]

        def fake_sync(_request, _model, **_kwargs):
            payload = payloads.pop(0)
            text = json.dumps(payload, ensure_ascii=False)
            return {
                'response_payload': {
                    'candidates': [{'content': {'parts': [{'text': text}]}}],
                },
                'response_text': text,
                'finish_reason': 'STOP',
                'usage_metadata': {},
                'provider': 'gemini',
                'model': 'gemini-test',
                'execution_mode': 'sync',
            }

        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            request_rows = [
                batch_mod.build_keyword_request(
                    chunk,
                    max_candidates_per_chunk=5,
                    model='gemini-test',
                )
            ]
            manifest_path = batch_mod.make_sync_manifest(
                package_dir=str(package_dir),
                mode=batch_mod.MANIFEST_MODE_KEYWORD_EXTRACTION,
                display_name='keyword-completeness-test',
                chunks=[chunk],
                request_rows=request_rows,
                settings={},
            )
            with mock.patch.object(
                batch_mod,
                'run_sync_request',
                side_effect=fake_sync,
            ):
                manifest = batch_mod.execute_sync_request_rows(
                    manifest_path,
                    request_rows,
                )

        summary = manifest['sync_summary']
        self.assertEqual(summary['contract_expected_chunks'], 1)
        self.assertEqual(summary['contract_first_pass_complete_chunks'], 0)
        self.assertEqual(summary['contract_final_complete_chunks'], 0)
        self.assertEqual(summary['contract_final_chunk_completeness'], 0.0)
        self.assertNotIn('contract_expected_items', summary)
        self.assertNotIn('contract_final_valid_items', summary)
        self.assertEqual(summary['contract_partial_requests'], 1)

    def test_sync_contract_parse_failures_use_stable_reason_codes(self):
        cases = (
            ('', translation_core.CONTRACT_EMPTY_RESPONSE_TEXT),
            ('not-json', translation_core.CONTRACT_INVALID_JSON),
        )
        for response_text, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                chunk = {
                    'key': 'chunk-1',
                    'file_path': 'script.rpy',
                    'file_rel_path': 'script.rpy',
                    'context_past': [],
                    'context_future': [],
                    'items': [{'id': 'a', 'text': 'Hello'}],
                }
                result = {
                    'response_payload': batch_mod.response_payload_with_text(
                        {},
                        response_text,
                    ) if response_text else {},
                    'response_text': response_text,
                    'finish_reason': 'STOP',
                    'usage_metadata': {},
                    'provider': 'gemini',
                    'model': 'gemini-test',
                    'execution_mode': 'sync',
                }
                with tempfile.TemporaryDirectory() as tmp:
                    package_dir = Path(tmp)
                    manifest_path = package_dir / 'manifest.json'
                    result_path = package_dir / 'results.jsonl'
                    manifest_path.write_text(
                        json.dumps({
                            'version': 2,
                            'mode': batch_mod.MANIFEST_MODE_TRANSLATION,
                            'batch_model': 'gemini-test',
                            'result_jsonl_path': str(result_path),
                            'chunks': [chunk],
                        }),
                        encoding='utf-8',
                    )
                    with mock.patch.object(
                        batch_mod,
                        'run_sync_request',
                        return_value=result,
                    ):
                        manifest = batch_mod.execute_sync_request_rows(
                            str(manifest_path),
                            [{'key': 'chunk-1', 'request': {}}],
                        )
                    row = json.loads(result_path.read_text(encoding='utf-8'))

                reasons = manifest['sync_summary']['reason_counts']
                self.assertEqual(reasons[expected_reason], 2)
                self.assertNotIn('response_contract_error', reasons)
                self.assertNotIn('targeted_retry_contract_error', reasons)
                self.assertEqual(
                    row['contract_diagnostics']['reason_counts'],
                    {expected_reason: 1},
                )

    def test_integrity_scan_uses_stable_contract_parse_reasons(self):
        cases = (
            ({}, translation_core.CONTRACT_EMPTY_RESPONSE_TEXT),
            (
                batch_mod.response_payload_with_text({}, 'not-json'),
                translation_core.CONTRACT_INVALID_JSON,
            ),
        )
        for response, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                with tempfile.TemporaryDirectory() as tmp:
                    result_path = Path(tmp) / 'results.jsonl'
                    batch_mod.write_jsonl_file(
                        str(result_path),
                        [{'key': 'chunk-1', 'response': response}],
                    )
                    manifest = {
                        '_package_dir': tmp,
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'items': [{'id': 'a', 'text': 'Hello'}],
                            }
                        ],
                    }
                    issue_keys, reasons = (
                        batch_mod.collect_result_integrity_issue_keys(manifest)
                    )

                self.assertEqual(issue_keys, {'chunk-1'})
                self.assertEqual(reasons[expected_reason], 1)
                self.assertNotIn('failed_to_parse_model_json', reasons)

    def test_sync_unknown_extra_id_is_audited_without_retranslating_valid_ids(self):
        chunk = {
            'key': 'chunk-1',
            'items': [{'id': 'a', 'text': 'Hello'}, {'id': 'b', 'text': 'World'}],
        }
        payload = {
            'translations': [
                {'id': 'a', 'translation': '你好'},
                {'id': 'b', 'translation': '世界'},
                {'id': 'unknown', 'translation': '多余'},
            ]
        }
        text = json.dumps(payload, ensure_ascii=False)
        result = {
            'response_payload': {
                'candidates': [{'content': {'parts': [{'text': text}]}}],
            },
            'response_text': text,
            'finish_reason': 'STOP',
            'usage_metadata': {},
            'provider': 'gemini',
            'model': 'gemini-test',
            'execution_mode': 'sync',
        }
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest_path = package_dir / 'manifest.json'
            result_path = package_dir / 'results.jsonl'
            manifest_path.write_text(
                json.dumps({
                    'version': 2,
                    'mode': batch_mod.MANIFEST_MODE_TRANSLATION,
                    'batch_model': 'gemini-test',
                    'result_jsonl_path': str(result_path),
                    'chunks': [chunk],
                }),
                encoding='utf-8',
            )
            request_rows = [{'key': 'chunk-1', 'request': {}}]
            with mock.patch.object(
                batch_mod,
                'run_sync_request',
                return_value=result,
            ) as run_sync:
                manifest = batch_mod.execute_sync_request_rows(
                    str(manifest_path),
                    request_rows,
                )

        run_sync.assert_called_once()
        self.assertEqual(manifest['job_state'], 'SYNC_PARTIAL')
        self.assertEqual(manifest['sync_summary']['contract_final_valid_items'], 2)
        self.assertEqual(manifest['sync_summary']['targeted_retry_requests'], 0)
        self.assertEqual(manifest['sync_summary']['reason_counts']['result_unknown_id'], 1)

    def test_sync_revisions_previews_and_optionally_applies(self):
        old_values = {
            'tl_dir': batch_mod.legacy.TL_DIR,
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tl_dir = root / 'tl'
                jobs_dir = root / 'batch_jobs'
                tl_dir.mkdir()
                jobs_dir.mkdir()
                previous_latest = root / 'previous_manifest.json'
                target_file = tl_dir / 'script.rpy'
                new_line = '    new "虚空门"\n'
                target_file.write_text(
                    'translate schinese start:\n'
                    '    old "Void Gate"\n'
                    + new_line,
                    encoding='utf-8',
                )
                batch_mod.legacy.TL_DIR = str(tl_dir)
                batch_mod.LOG_DIR = str(root / 'logs')
                batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
                batch_mod.REPAIR_RUNS_DIR = str(root / 'repair_runs')
                batch_mod.SYNC_RUNS_DIR = str(root / 'sync_runs')
                batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')
                Path(batch_mod.LATEST_MANIFEST_FILE).write_text(str(previous_latest), encoding='utf-8')

                def run_sync_revision_response(request, *_args, **_kwargs):
                    prompt_text = request['contents'][0]['parts'][0]['text']
                    target_text = prompt_text.split('TARGET:\n', 1)[1].split('\n\nCONTEXT AFTER:', 1)[0]
                    target_id = json.loads(target_text)[0]['id']
                    response_text = json.dumps(
                        [
                            {
                                'id': target_id,
                                'should_update': True,
                                'revised_translation': '虚空之门',
                                'reason': '统一术语',
                            }
                        ],
                        ensure_ascii=False,
                    )
                    return {
                        'response_payload': {
                            'candidates': [{'content': {'parts': [{'text': response_text}]}}],
                        },
                        'response_text': response_text,
                        'finish_reason': 'STOP',
                        'usage_metadata': {},
                    }

                with (
                    mock.patch.object(
                        batch_mod,
                        'run_sync_request',
                        side_effect=run_sync_revision_response,
                    ) as sync_request,
                    mock.patch.object(batch_mod, 'update_progress') as update_progress,
                ):
                    manifest = batch_mod.sync_revisions(
                        skip_prepare=True,
                        chunk_size=1,
                        limit=1,
                        apply=True,
                    )

                updated_script = target_file.read_text(encoding='utf-8')
                latest_after = Path(batch_mod.LATEST_MANIFEST_FILE).read_text(encoding='utf-8')
        finally:
            batch_mod.legacy.TL_DIR = old_values['tl_dir']
            batch_mod.LOG_DIR = old_values['log_dir']
            batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
            batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
            batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
            batch_mod.LATEST_MANIFEST_FILE = old_values['latest']

        sync_request.assert_called_once()
        update_progress.assert_called_once_with('script.rpy', [2])
        self.assertIn('new "虚空之门"', updated_script)
        self.assertEqual(manifest['revision_apply_summary']['applied_files'], 1)
        self.assertEqual(latest_after, str(previous_latest))

    def test_preview_and_apply_revisions_updates_existing_new_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            new_line = '    new "虚空门"\n'
            start = new_line.index('"虚空门"')
            end = start + len('"虚空门"')
            target_file.write_text(
                'translate schinese start:\n'
                '    old "Void Gate"\n'
                + new_line,
                encoding='utf-8',
            )
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            item_id = f'script.rpy:2:{start}:revision:0'
            response_text = json.dumps(
                [
                    {
                        'id': item_id,
                        'should_update': True,
                        'revised_translation': '虚空之门',
                        'reason': '统一术语',
                    }
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'rv-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'mode': batch_mod.MANIFEST_MODE_REVISION,
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'rv-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': item_id,
                                        'line': 2,
                                        'line_number': 3,
                                        'start': start,
                                        'end': end,
                                        'text': 'Void Gate',
                                        'source': 'Void Gate',
                                        'current_translation': '虚空门',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                preview_manifest = batch_mod.preview_revisions(str(manifest_path))
                before_apply = target_file.read_text(encoding='utf-8')
                preview_jsonl = Path(preview_manifest['last_revision_preview']['jsonl_path'])
                preview_jsonl_exists = preview_jsonl.is_file()
                applied_manifest = batch_mod.apply_revisions(str(manifest_path))

            updated_script = target_file.read_text(encoding='utf-8')

        self.assertIn('new "虚空门"', before_apply)
        self.assertIn('new "虚空之门"', updated_script)
        self.assertTrue(preview_jsonl_exists)
        self.assertEqual(preview_manifest['last_revision_preview']['summary']['valid_items'], 1)
        self.assertEqual(applied_manifest['revision_apply_summary']['applied_files'], 1)
        self.assertEqual(applied_manifest['revision_apply_summary']['recoverable_items'], 1)
        update_progress.assert_called_once_with('script.rpy', [2])

    def test_preview_revisions_validates_output_paths_and_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            result_path.write_text('', encoding='utf-8')
            manifest_path.write_text(
                json.dumps(
                    {
                        'mode': batch_mod.MANIFEST_MODE_REVISION,
                        'base_dir': str(package_dir),
                        'tl_dir': str(package_dir),
                        'input_jsonl_path': str(package_dir / 'requests.jsonl'),
                        'result_jsonl_path': str(result_path),
                        'chunks': [],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(SystemExit, 'reserved package file'):
                batch_mod.preview_revisions(str(manifest_path), output_jsonl='results.jsonl')
            with self.assertRaisesRegex(SystemExit, 'must be different files'):
                batch_mod.preview_revisions(
                    str(manifest_path),
                    output_jsonl='same.jsonl',
                    output_markdown='same.jsonl',
                )

            manifest = batch_mod.preview_revisions(
                str(manifest_path),
                output_jsonl='reports/revision_preview.jsonl',
                output_markdown='reports/revision_preview.md',
            )
            jsonl_exists = Path(manifest['last_revision_preview']['jsonl_path']).is_file()
            markdown_exists = Path(manifest['last_revision_preview']['markdown_path']).is_file()

        self.assertTrue(jsonl_exists)
        self.assertTrue(markdown_exists)

    def test_apply_revisions_revalidates_current_translation_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            original_new_line = '    new "虚空门"\n'
            changed_new_line = '    new "星门"\n'
            start = original_new_line.index('"虚空门"')
            end = start + len('"虚空门"')
            target_file.write_text(
                'translate schinese start:\n'
                '    old "Void Gate"\n'
                + changed_new_line,
                encoding='utf-8',
            )
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            item_id = f'script.rpy:2:{start}:revision:0'
            response_text = json.dumps(
                [
                    {
                        'id': item_id,
                        'should_update': True,
                        'revised_translation': '虚空之门',
                        'reason': '统一术语',
                    }
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'rv-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'mode': batch_mod.MANIFEST_MODE_REVISION,
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'rv-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': item_id,
                                        'line': 2,
                                        'line_number': 3,
                                        'start': start,
                                        'end': end,
                                        'text': 'Void Gate',
                                        'source': 'Void Gate',
                                        'current_translation': '虚空门',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'append_failure_entries') as append_failures,
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                preview_manifest = batch_mod.preview_revisions(str(manifest_path))
                preview_jsonl = Path(preview_manifest['last_revision_preview']['jsonl_path'])
                preview_rows = [
                    json.loads(line)
                    for line in preview_jsonl.read_text(encoding='utf-8').splitlines()
                ]
                manifest = batch_mod.apply_revisions(str(manifest_path))

            final_script = target_file.read_text(encoding='utf-8')

        self.assertIn('new "星门"', final_script)
        self.assertEqual(preview_manifest['last_revision_preview']['summary']['valid_items'], 0)
        self.assertEqual(preview_rows[0]['status'], 'source_mismatch')
        self.assertIn('Source text mismatch', preview_rows[0]['error'])
        self.assertEqual(manifest['revision_apply_summary']['applied_files'], 0)
        self.assertEqual(manifest['revision_apply_summary']['recoverable_items'], 0)
        self.assertEqual(manifest['revision_apply_summary']['skipped_items'], 1)
        self.assertEqual(manifest['revision_apply_summary']['source_mismatch_items'], 1)
        append_failures.assert_called_once()
        update_progress.assert_not_called()

    def test_collect_result_actions_rejects_duplicate_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            result_path = package_dir / 'results.jsonl'
            response_text = json.dumps(
                [
                    {'id': 'script.rpy:0:4', 'translation': '\u4f60\u597d'},
                    {'id': 'script.rpy:0:4', 'translation': '\u518d\u89c1'},
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(result_path),
                'chunks': [
                    {
                        'key': 'chunk-1',
                        'file_rel_path': 'script.rpy',
                        'items': [
                            {
                                'id': 'script.rpy:0:4',
                                'line': 0,
                                'start': 4,
                                'end': 11,
                                'text': 'Hello',
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    }
                ],
            }

            replacements, _translated, failures, summary = batch_mod.collect_result_actions(manifest)

        self.assertEqual(summary['valid_items'], 0)
        self.assertEqual(summary['reason_counts']['result_duplicate_id'], 1)
        self.assertEqual(replacements, {})
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['id'], 'script.rpy:0:4')

    def test_collect_result_actions_skips_source_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            line = '    e "Hallo"\n'
            start = line.index('"Hallo"')
            end = start + len('"Hallo"')
            target_file.write_text(line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(result_path),
                'files': {'script.rpy': {'path': str(target_file)}},
                'chunks': [
                    {
                        'key': 'chunk-1',
                        'file_rel_path': 'script.rpy',
                        'items': [
                            {
                                'id': f'script.rpy:0:{start}',
                                'line': 0,
                                'start': start,
                                'end': end,
                                'text': 'Hello',
                                'prefix': '',
                                'quote': '"',
                            }
                        ],
                    }
                ],
            }

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                replacements, translated, failures, summary = batch_mod.collect_result_actions(
                    manifest,
                    validate_sources=True,
                )

        self.assertEqual(replacements, {})
        self.assertEqual(translated, {})
        self.assertEqual(summary['candidate_valid_items'], 1)
        self.assertEqual(summary['valid_items'], 0)
        self.assertEqual(summary['skipped_items'], 1)
        self.assertEqual(summary['source_mismatch_items'], 1)
        self.assertEqual(summary['pending_files'], 0)
        self.assertEqual(summary['pending_lines'], 0)
        self.assertEqual(summary['reason_counts']['source_text_mismatch'], 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['error'], 'Source text mismatch during source validation')
        self.assertEqual(failures[0]['current_text'], 'Hallo')

    def test_apply_results_handles_multiple_replacements_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            dialogue_line = '    call screen test("Hello", "World")\n'
            hello_start = dialogue_line.index('"Hello"')
            hello_end = hello_start + len('"Hello"')
            world_start = dialogue_line.index('"World"')
            world_end = world_start + len('"World"')
            target_file.write_text('label test:\n' + dialogue_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [
                    {'id': f'script.rpy:1:{hello_start}', 'translation': '\u4f60\u597d'},
                    {'id': f'script.rpy:1:{world_start}', 'translation': '\u4e16\u754c'},
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:1:{hello_start}',
                                        'line': 1,
                                        'start': hello_start,
                                        'end': hello_end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                    {
                                        'id': f'script.rpy:1:{world_start}',
                                        'line': 1,
                                        'start': world_start,
                                        'end': world_end,
                                        'text': 'World',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                batch_mod.check_results(str(manifest_path))
                manifest = batch_mod.apply_results(str(manifest_path))

            updated_script = target_file.read_text(encoding='utf-8')
            saved_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        self.assertIn('call screen test("\u4f60\u597d", "\u4e16\u754c")', updated_script)
        update_progress.assert_called_once_with('script.rpy', [1])
        self.assertEqual(manifest['apply_summary']['applied_files'], 1)
        self.assertEqual(manifest['apply_summary']['applied_lines'], 1)
        self.assertEqual(manifest['apply_summary']['recoverable_items'], 2)
        self.assertEqual(manifest['apply_summary']['skipped_items'], 0)
        self.assertIn('applied_at', saved_manifest)

    def test_apply_results_resumes_already_written_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            first_line = '    e "Hello"\n'
            second_line = '    e "World"\n'
            first_start = first_line.index('"Hello"')
            first_end = first_start + len('"Hello"')
            second_start = second_line.index('"World"')
            second_end = second_start + len('"World"')
            target_file.write_text(first_line + second_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [
                    {'id': f'script.rpy:0:{first_start}', 'translation': '\u4f60\u597d'},
                    {'id': f'script.rpy:1:{second_start}', 'translation': '\u4e16\u754c'},
                ],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'execution': 'sync',
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{first_start}',
                                        'line': 0,
                                        'start': first_start,
                                        'end': first_end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                    {
                                        'id': f'script.rpy:1:{second_start}',
                                        'line': 1,
                                        'start': second_start,
                                        'end': second_end,
                                        'text': 'World',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                batch_mod.check_results(str(manifest_path))
                target_file.write_text('    e "\u4f60\u597d"\n' + second_line, encoding='utf-8')
                manifest = batch_mod.apply_results(str(manifest_path))

            updated_script = target_file.read_text(encoding='utf-8')

        self.assertEqual(updated_script, '    e "\u4f60\u597d"\n    e "\u4e16\u754c"\n')
        update_progress.assert_called_once_with('script.rpy', [0, 1])
        self.assertEqual(manifest['apply_summary']['applied_files'], 1)
        self.assertEqual(manifest['apply_summary']['applied_lines'], 2)
        self.assertEqual(manifest['apply_summary']['recoverable_items'], 2)
        self.assertEqual(manifest['apply_summary']['skipped_items'], 0)
    def test_apply_results_excludes_progress_only_files_from_adapter_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            first_file = tl_dir / 'first.rpy'
            second_file = tl_dir / 'second.rpy'
            first_line = '    e "Hello"\n'
            second_line = '    e "World"\n'
            first_start = first_line.index('"Hello"')
            second_start = second_line.index('"World"')
            first_file.write_text(first_line, encoding='utf-8')
            second_file.write_text(second_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            result_rows = (
                {
                    'key': 'chunk-first',
                    'response': {
                        'candidates': [
                            {
                                'content': {
                                    'parts': [
                                        {
                                            'text': json.dumps(
                                                [
                                                    {
                                                        'id': f'first.rpy:0:{first_start}',
                                                        'translation': '你好',
                                                    }
                                                ],
                                                ensure_ascii=False,
                                            )
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
                {
                    'key': 'chunk-second',
                    'response': {
                        'candidates': [
                            {
                                'content': {
                                    'parts': [
                                        {
                                            'text': json.dumps(
                                                [
                                                    {
                                                        'id': f'second.rpy:0:{second_start}',
                                                        'translation': '世界',
                                                    }
                                                ],
                                                ensure_ascii=False,
                                            )
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                },
            )
            result_path.write_text(
                ''.join(
                    json.dumps(row, ensure_ascii=False) + '\n'
                    for row in result_rows
                ),
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'execution': 'sync',
                        'files': {
                            'first.rpy': {'path': str(first_file)},
                            'second.rpy': {'path': str(second_file)},
                        },
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-first',
                                'file_rel_path': 'first.rpy',
                                'items': [
                                    {
                                        'id': f'first.rpy:0:{first_start}',
                                        'line': 0,
                                        'start': first_start,
                                        'end': first_start + len('"Hello"'),
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            },
                            {
                                'key': 'chunk-second',
                                'file_rel_path': 'second.rpy',
                                'items': [
                                    {
                                        'id': f'second.rpy:0:{second_start}',
                                        'line': 0,
                                        'start': second_start,
                                        'end': second_start + len('"World"'),
                                        'text': 'World',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                batch_mod.check_results(str(manifest_path))
                first_file.write_text('    e "你好"\n', encoding='utf-8')
                manifest = batch_mod.apply_results(str(manifest_path))

            self.assertEqual(first_file.read_text(encoding='utf-8'), '    e "你好"\n')
            self.assertEqual(second_file.read_text(encoding='utf-8'), '    e "世界"\n')

        update_progress.assert_has_calls(
            [
                mock.call('first.rpy', [0]),
                mock.call('second.rpy', [0]),
            ],
            any_order=True,
        )
        self.assertEqual(manifest['apply_summary']['applied_files'], 2)
        self.assertEqual(manifest['apply_summary']['applied_lines'], 2)


    def test_apply_results_revalidates_snapshot_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            original_line = '    e "Hello"\n'
            changed_line = '    e "Hallo"\n'
            start = original_line.index('"Hello"')
            end = start + len('"Hello"')
            target_file.write_text(original_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{start}',
                                        'line': 0,
                                        'start': start,
                                        'end': end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            collect_result_actions = batch_mod.collect_result_actions

            def mutate_after_initial_validation(*args, **kwargs):
                result = collect_result_actions(*args, **kwargs)
                target_file.write_text(changed_line, encoding='utf-8')
                return result

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.check_results(str(manifest_path))

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'collect_result_actions', side_effect=mutate_after_initial_validation),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
                mock.patch.object(batch_mod, 'append_failure_entries') as append_failures,
            ):
                with self.assertRaisesRegex(SystemExit, 'source revalidation is not safe'):
                    batch_mod.apply_results(str(manifest_path))

            final_script = target_file.read_text(encoding='utf-8')
            saved_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.check_results(str(manifest_path))
            rechecked_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        self.assertEqual(final_script, changed_line)
        update_progress.assert_not_called()
        append_failures.assert_called_once()
        self.assertIn('last_apply_failure_report_path', saved_manifest)
        self.assertNotIn('last_apply_failure_report_path', rechecked_manifest)

    def test_apply_results_rejects_already_applied_manifest_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest_path = package_dir / 'manifest.json'
            manifest_path.write_text(
                json.dumps({'applied_at': '2026-05-12T12:00:00'}, ensure_ascii=False),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(SystemExit, 'already applied'):
                batch_mod.apply_results(str(manifest_path))

    def test_apply_results_force_keeps_source_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            line = '    e "Hallo"\n'
            start = line.index('"Hallo"')
            end = start + len('"Hallo"')
            target_file.write_text(line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'applied_at': '2026-05-12T12:00:00',
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{start}',
                                        'line': 0,
                                        'start': start,
                                        'end': end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'append_failure_entries') as append_failures,
            ):
                batch_mod.check_results(str(manifest_path))
                with self.assertRaisesRegex(SystemExit, 'not safe'):
                    batch_mod.apply_results(str(manifest_path), force=True)

            unchanged_script = target_file.read_text(encoding='utf-8')

        self.assertEqual(unchanged_script, line)
        append_failures.assert_not_called()
    def test_apply_results_force_refuses_adapter_block_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            target_file = root / 'script.rpy'
            original_line = '    e "Hello"\n'
            target_file.write_text(original_line, encoding='utf-8')
            start = original_line.index('"Hello"')
            replacements = {
                0: [
                    (
                        start,
                        start + len('"Hello"'),
                        '你好',
                        '',
                        '"',
                        'Hello',
                        'item-1',
                        'chunk-1',
                    )
                ]
            }
            manifest = {
                'applied_at': '2026-05-12T12:00:00',
                '_package_dir': str(package_dir),
                '_manifest_path': str(package_dir / 'manifest.json'),
                'execution': 'sync',
                'files': {'script.rpy': {'path': str(target_file)}},
            }
            summary = {'reason_counts': {}, 'valid_items': 1, 'failure_items': 0}
            failures = []

            def block_adapter_plan(
                _manifest,
                _replacements_by_file,
                live_summary,
                failure_entries,
                live_sources=None,
            ):
                del live_sources
                batch_mod.bump_counter(
                    live_summary['reason_counts'],
                    'adapter_writeback_block',
                )
                live_summary['adapter_writeback_status'] = 'block'
                failure_entries.append({'reason_code': 'adapter_writeback_block'})
                return None, None

            def attach_block(_manifest, live_summary):
                live_summary['safety_level'] = batch_mod.CHECK_SAFETY_BLOCK
                return live_summary

            with (
                mock.patch.object(batch_mod, 'load_manifest', return_value=manifest),
                mock.patch.object(batch_mod, 'require_manifest_mode'),
                mock.patch.object(batch_mod, 'require_manifest_project_match'),
                mock.patch.object(batch_mod, 'recover_atomic_write_transaction'),
                mock.patch.object(batch_mod, 'require_safe_check_for_apply'),
                mock.patch.object(
                    batch_mod,
                    'collect_result_actions',
                    return_value=(
                        {'script.rpy': replacements},
                        {'script.rpy': {0}},
                        failures,
                        summary,
                    ),
                ),
                mock.patch.object(
                    batch_mod,
                    'resolve_manifest_file_path',
                    return_value=str(target_file),
                ),
                mock.patch.object(
                    batch_mod,
                    'validate_replacements_for_lines',
                    return_value=(replacements, {0}, [], 0, 0),
                ),
                mock.patch.object(
                    batch_mod,
                    '_validate_adapter_writeback_plan',
                    side_effect=block_adapter_plan,
                ),
                mock.patch.object(
                    batch_mod,
                    'attach_check_contract',
                    side_effect=attach_block,
                ),
                mock.patch.object(batch_mod, 'append_failure_entries'),
                mock.patch.object(
                    batch_mod,
                    'write_apply_failure_report',
                    return_value=str(package_dir / 'apply_failure_report.json'),
                ),
                mock.patch.object(batch_mod, 'save_manifest'),
                mock.patch.object(batch_mod, 'atomic_write_many_lines') as atomic_write,
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                with self.assertRaisesRegex(SystemExit, 'not safe'):
                    batch_mod.apply_results('manifest.json', force=True)

            self.assertEqual(target_file.read_text(encoding='utf-8'), original_line)
            atomic_write.assert_not_called()
            update_progress.assert_not_called()

    def test_apply_results_rejects_warn_check_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            target_file = tl_dir / 'script.rpy'
            first_line = '    e "Hello"\n'
            second_line = '    e "World"\n'
            first_start = first_line.index('"Hello"')
            first_end = first_start + len('"Hello"')
            second_start = second_line.index('"World"')
            second_end = second_start + len('"World"')
            target_file.write_text(first_line + second_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{first_start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(target_file)}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{first_start}',
                                        'line': 0,
                                        'start': first_start,
                                        'end': first_end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                    {
                                        'id': f'script.rpy:1:{second_start}',
                                        'line': 1,
                                        'start': second_start,
                                        'end': second_end,
                                        'text': 'World',
                                        'prefix': '',
                                        'quote': '"',
                                    },
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                checked = batch_mod.check_results(str(manifest_path))
                with self.assertRaisesRegex(SystemExit, 'not safe'):
                    batch_mod.apply_results(str(manifest_path))

            final_script = target_file.read_text(encoding='utf-8')
            check_failures = [
                json.loads(line)
                for line in (package_dir / 'check_failures.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()
            ]

        self.assertEqual(checked['last_check_summary']['safety_level'], 'warn')
        self.assertEqual(
            checked['last_check_summary']['safety_reasons']['warn'][
                'response_missing_expected_id'
            ],
            1,
        )
        self.assertNotIn(
            'response_missing_item_id',
            checked['last_check_summary']['safety_reasons']['warn'],
        )
        self.assertEqual(final_script, first_line + second_line)
        update_progress.assert_not_called()
        self.assertEqual(check_failures[0]['status'], 'warn')
        self.assertEqual(
            check_failures[0]['reason_code'],
            'response_missing_expected_id',
        )

    def test_manifest_result_path_must_stay_in_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = root / 'package'
            package_dir.mkdir()
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': str(root / 'outside-results.jsonl'),
            }

            with self.assertRaisesRegex(SystemExit, 'escapes'):
                batch_mod.resolve_manifest_result_path(manifest)

    def test_manifest_result_path_rejects_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / 'package'
            package_dir.mkdir()
            manifest = {
                '_package_dir': str(package_dir),
                'result_jsonl_path': '../outside-results.jsonl',
            }

            with self.assertRaisesRegex(SystemExit, 'parent directory'):
                batch_mod.resolve_manifest_result_path(manifest)

    def test_apply_results_rejects_manifest_from_different_active_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_a = root / 'project-a'
            project_b = root / 'project-b'
            tl_dir_a = project_a / 'game' / 'tl' / 'schinese'
            tl_dir_b = project_b / 'game' / 'tl' / 'schinese'
            package_dir = root / 'package'
            tl_dir_a.mkdir(parents=True)
            tl_dir_b.mkdir(parents=True)
            package_dir.mkdir()
            source_line = '    e "Hello"\n'
            target_a = tl_dir_a / 'script.rpy'
            target_b = tl_dir_b / 'script.rpy'
            target_a.write_text(source_line, encoding='utf-8')
            target_b.write_text(source_line, encoding='utf-8')
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            start = source_line.index('"Hello"')
            end = start + len('"Hello"')
            response_text = json.dumps(
                [{'id': f'script.rpy:0:{start}', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'version': 1,
                        'manifest_version': 1,
                        'base_dir': str(project_a),
                        'tl_dir': str(tl_dir_a),
                        'files': {'script.rpy': {'path': 'script.rpy'}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': f'script.rpy:0:{start}',
                                        'line': 0,
                                        'start': start,
                                        'end': end,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(project_a)),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir_a)),
            ):
                checked = batch_mod.check_results(str(manifest_path))

            fingerprint = checked['last_check_summary']['check_fingerprint']
            self.assertEqual(fingerprint['project']['base_dir'], str(project_a.resolve()))
            self.assertEqual(fingerprint['project']['tl_dir'], str(tl_dir_a.resolve()))

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(project_b)),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir_b)),
            ):
                with self.assertRaisesRegex(SystemExit, 'does not match the active project'):
                    batch_mod.check_results(str(manifest_path))
                with self.assertRaisesRegex(SystemExit, 'does not match the active project'):
                    batch_mod.apply_results(str(manifest_path))

            self.assertEqual(target_a.read_text(encoding='utf-8'), source_line)
            self.assertEqual(target_b.read_text(encoding='utf-8'), source_line)

            with (
                mock.patch.object(batch_mod.legacy, 'BASE_DIR', str(project_a)),
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir_a)),
                mock.patch.object(batch_mod, 'update_progress'),
            ):
                batch_mod.apply_results(str(manifest_path))

            self.assertIn('\u4f60\u597d', target_a.read_text(encoding='utf-8'))
            self.assertEqual(target_b.read_text(encoding='utf-8'), source_line)

    def test_check_results_rejects_unbound_legacy_relative_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'game' / 'tl' / 'schinese'
            package_dir = root / 'package'
            tl_dir.mkdir(parents=True)
            package_dir.mkdir()
            manifest_path = package_dir / 'manifest.json'
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': 'script.rpy'}},
                        'result_jsonl_path': 'results.jsonl',
                        'chunks': [],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'project identity is missing'):
                    batch_mod.check_results(str(manifest_path))
                with self.assertRaisesRegex(SystemExit, 'project identity is missing'):
                    batch_mod.apply_results(str(manifest_path))

    def test_apply_results_rejects_manifest_file_path_outside_tl_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            package_dir = root / 'package'
            tl_dir.mkdir()
            package_dir.mkdir()
            result_path = package_dir / 'results.jsonl'
            manifest_path = package_dir / 'manifest.json'
            response_text = json.dumps(
                [{'id': 'script.rpy:0:4', 'translation': '\u4f60\u597d'}],
                ensure_ascii=False,
            )
            result_path.write_text(
                json.dumps(
                    {
                        'key': 'chunk-1',
                        'response': {
                            'candidates': [
                                {'content': {'parts': [{'text': response_text}]}}
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        'files': {'script.rpy': {'path': str(root / 'outside.rpy')}},
                        'result_jsonl_path': str(result_path),
                        'chunks': [
                            {
                                'key': 'chunk-1',
                                'file_rel_path': 'script.rpy',
                                'items': [
                                    {
                                        'id': 'script.rpy:0:4',
                                        'line': 0,
                                        'start': 4,
                                        'end': 11,
                                        'text': 'Hello',
                                        'prefix': '',
                                        'quote': '"',
                                    }
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'escapes'):
                    batch_mod.check_results(str(manifest_path))

    def test_load_repair_report_items_accepts_batch_failure_log_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    pass\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 0,
                    'text': 'Hello',
                    'error': 'Validation failed',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['file'], str(target_file.resolve()))
        self.assertEqual(items[0]['source'], 'Hello')
        self.assertEqual(items[0]['line'], 1)

    @unittest.skipUnless(os.name == 'nt', 'Windows path alias regression')
    def test_resolve_path_under_dir_accepts_windows_short_path_alias(self):
        short_base = r'C:\Users\RUNNER~1\AppData\Local\Temp\case'
        long_base = r'C:\Users\runneradmin\AppData\Local\Temp\case'
        short_file = short_base + r'\script.rpy'
        long_file = long_base + r'\script.rpy'

        def canonical(path):
            normalized = os.path.normcase(os.path.abspath(path))
            if normalized == os.path.normcase(os.path.abspath(short_base)):
                return long_base
            if normalized == os.path.normcase(os.path.abspath(short_file)):
                return long_file
            return os.path.abspath(path)

        with mock.patch.object(batch_mod, '_canonical_abs_path', side_effect=canonical):
            self.assertEqual(
                batch_mod.resolve_path_under_dir(short_base, 'script.rpy', 'repair file'),
                long_file,
            )
            self.assertEqual(
                batch_mod.resolve_path_under_dir(short_base, long_file, 'repair file'),
                long_file,
            )

    def test_load_repair_report_items_rejects_file_outside_tl_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'tl'
            tl_dir.mkdir()
            outside_file = root / 'outside.rpy'
            outside_file.write_text('label outside:\n    pass\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file': str(outside_file),
                    'line': 1,
                    'source': 'Hello',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'escapes'):
                    batch_mod.load_repair_report_items(str(report_path))

    def test_load_repair_report_items_rejects_parent_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': '../outside.rpy',
                    'line': 0,
                    'text': 'Hello',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'parent directory'):
                    batch_mod.load_repair_report_items(str(report_path))

    def test_load_repair_report_items_distinguishes_start_zero_from_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text('label test:\n    "Menu"\n', encoding='utf-8')
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Menu',
                }, ensure_ascii=False) + '\n' +
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Menu',
                    'start': 0,
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].get('start'), None)
        self.assertEqual(items[1]['start'], 0)

    def test_repair_jobs_keep_multiple_items_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            target_file.write_text(
                'label test:\n'
                '    call screen test("Hello", "World")\n',
                encoding='utf-8',
            )
            report_path = tl_dir / 'failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'Hello',
                    'id': 'script.rpy:1:21',
                }, ensure_ascii=False) + '\n' +
                json.dumps({
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'text': 'World',
                    'id': 'script.rpy:1:30',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))
                jobs, unresolved = batch_mod.build_repair_jobs(items, batch_size=2)

        self.assertEqual([item['source'] for item in items], ['Hello', 'World'])
        self.assertEqual(unresolved, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual([item['text'] for item in jobs[0]['items']], ['Hello', 'World'])
        self.assertEqual(len({item['id'] for item in jobs[0]['items']}), 2)

    def test_repair_jobs_parse_line_start_end_repair_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = Path(tmp)
            target_file = tl_dir / 'script.rpy'
            duplicate_line = '    call screen test("Menu", "Menu")\n'
            second_start = duplicate_line.rindex('"Menu"')
            second_end = second_start + len('"Menu"')
            target_file.write_text(
                'label test:\n' + duplicate_line,
                encoding='utf-8',
            )
            report_path = tl_dir / 'repair_failures.jsonl'
            report_path.write_text(
                json.dumps({
                    'file': str(target_file.resolve()),
                    'line': 2,
                    'source': 'Menu',
                    'id': f'{target_file.resolve()}:2:{second_start}:{second_end}',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                items = batch_mod.load_repair_report_items(str(report_path))
                jobs, unresolved = batch_mod.build_repair_jobs(items, batch_size=2)

        self.assertEqual(unresolved, [])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['items'][0]['start'], second_start)

    def test_repair_jobs_include_story_memory_when_enabled(self):
        old_values = {
            'enabled': batch_mod.STORY_MEMORY_ENABLED,
            'graph_file': batch_mod.STORY_MEMORY_GRAPH_FILE,
            'max_context_chars': batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS,
            'graph': batch_mod._STORY_GRAPH,
            'graph_path': batch_mod._STORY_GRAPH_PATH,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tl_dir = Path(tmp)
                target_file = tl_dir / 'script.rpy'
                graph_file = tl_dir / 'story_graph.json'
                target_file.write_text(
                    'label test:\n'
                    '    e "Open the Void Gate"\n',
                    encoding='utf-8',
                )
                graph_file.write_text(
                    json.dumps(
                        {
                            'schema_version': 1,
                            'characters': {
                                'eileen': {
                                    'zh_name': '艾琳',
                                    'speaker_ids': ['e'],
                                    'style': '语气轻快',
                                },
                            },
                            'relations': [],
                            'terms': [
                                {
                                    'source': 'Void Gate',
                                    'target': '虚空门',
                                    'note': '世界观核心术语',
                                },
                            ],
                            'scenes': [
                                {
                                    'file_rel_path': 'script.rpy',
                                    'line_start': 3,
                                    'line_end': 3,
                                    'summary': '偏后一行的场景。',
                                    'characters': ['eileen'],
                                },
                                {
                                    'file_rel_path': 'script.rpy',
                                    'line_start': 2,
                                    'line_end': 2,
                                    'summary': '正确边界场景。',
                                    'characters': ['eileen'],
                                },
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding='utf-8',
                )

                batch_mod.STORY_MEMORY_ENABLED = True
                batch_mod.STORY_MEMORY_GRAPH_FILE = str(graph_file)
                batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = 500
                batch_mod._STORY_GRAPH = None
                batch_mod._STORY_GRAPH_PATH = ''
                with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                    jobs, unresolved = batch_mod.build_repair_jobs(
                        [
                            {
                                'file': str(target_file),
                                'line': 2,
                                'source': 'Open the Void Gate',
                            },
                        ],
                        batch_size=1,
                    )
                request = batch_mod.build_repair_request(jobs[0])
                prompt = request['request']['contents'][0]['parts'][0]['text']
                summary = batch_mod.summarize_batch_story_memory(
                    jobs,
                    graph_file=str(graph_file),
                    max_context_chars=500,
                )
        finally:
            batch_mod.STORY_MEMORY_ENABLED = old_values['enabled']
            batch_mod.STORY_MEMORY_GRAPH_FILE = old_values['graph_file']
            batch_mod.STORY_MEMORY_MAX_CONTEXT_CHARS = old_values['max_context_chars']
            batch_mod._STORY_GRAPH = old_values['graph']
            batch_mod._STORY_GRAPH_PATH = old_values['graph_path']

        self.assertEqual(unresolved, [])
        self.assertEqual(jobs[0]['file_rel_path'], 'script.rpy')
        self.assertEqual(jobs[0]['items'][0]['speaker_id'], 'e')
        self.assertEqual(jobs[0]['items'][0]['line_number'], 2)
        self.assertIn('story_hits', jobs[0])
        self.assertEqual(jobs[0]['story_hits']['scenes'][0]['summary'], '正确边界场景。')
        self.assertIn('STORY MEMORY', prompt)
        self.assertIn('Void Gate -> 虚空门', prompt)
        self.assertEqual(summary['chunks_with_story_hits'], 1)


class RevisionApplyPreviewContractTests(unittest.TestCase):
    """Issue #294: apply must bind to a valid preview and report true outcome."""

    def _make_package(
        self,
        root,
        *,
        current_new='虚空门',
        revised='虚空之门',
        should_update=True,
        file_new=None,
    ):
        tl_dir = root / 'tl'
        package_dir = root / 'package'
        tl_dir.mkdir()
        package_dir.mkdir()
        target_file = tl_dir / 'script.rpy'
        file_new = file_new if file_new is not None else current_new
        new_line = f'    new "{file_new}"\n'
        start = new_line.index(f'"{file_new}"')
        end = start + len(file_new) + 2
        target_file.write_text(
            'translate schinese start:\n'
            '    old "Void Gate"\n'
            + new_line,
            encoding='utf-8',
        )
        result_path = package_dir / 'results.jsonl'
        manifest_path = package_dir / 'manifest.json'
        item_id = f'script.rpy:2:{start}:revision:0'
        response_text = json.dumps(
            [
                {
                    'id': item_id,
                    'should_update': should_update,
                    'revised_translation': revised,
                    'reason': '统一术语',
                }
            ],
            ensure_ascii=False,
        )
        result_path.write_text(
            json.dumps(
                {
                    'key': 'rv-1',
                    'response': {
                        'candidates': [
                            {'content': {'parts': [{'text': response_text}]}}
                        ]
                    },
                },
                ensure_ascii=False,
            )
            + '\n',
            encoding='utf-8',
        )
        manifest_path.write_text(
            json.dumps(
                {
                    'mode': batch_mod.MANIFEST_MODE_REVISION,
                    'files': {'script.rpy': {'path': str(target_file)}},
                    'result_jsonl_path': str(result_path),
                    'chunks': [
                        {
                            'key': 'rv-1',
                            'file_rel_path': 'script.rpy',
                            'items': [
                                {
                                    'id': item_id,
                                    'line': 2,
                                    'line_number': 3,
                                    'start': start,
                                    'end': end,
                                    'text': 'Void Gate',
                                    'source': 'Void Gate',
                                    'current_translation': current_new,
                                    'prefix': '',
                                    'quote': '"',
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        return manifest_path, target_file, result_path

    def _load_manifest(self, manifest_path):
        return json.loads(Path(manifest_path).read_text(encoding='utf-8'))

    def test_apply_without_preview_blocks_and_keeps_manifest_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(root)
            original = target_file.read_text(encoding='utf-8')
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                with self.assertRaisesRegex(SystemExit, 'run preview-revisions'):
                    batch_mod.apply_revisions(str(manifest_path))

            manifest = self._load_manifest(manifest_path)
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'missing_preview')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(target_file.read_text(encoding='utf-8'), original)

    def test_apply_rejects_replaced_results_since_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, result_path = self._make_package(root)
            original = target_file.read_text(encoding='utf-8')
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                result_path.write_text(
                    result_path.read_text(encoding='utf-8').rstrip()
                    + '\n{"replacement": true}\n',
                    encoding='utf-8',
                )
                with self.assertRaisesRegex(SystemExit, 'result JSONL changed since preview'):
                    batch_mod.apply_revisions(str(manifest_path))

            manifest = self._load_manifest(manifest_path)
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'results_changed')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(target_file.read_text(encoding='utf-8'), original)

    def test_apply_rejects_source_changed_since_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(root)
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                target_file.write_text(
                    target_file.read_text(encoding='utf-8').replace('虚空门', '星门'),
                    encoding='utf-8',
                )
                changed = target_file.read_text(encoding='utf-8')
                with self.assertRaisesRegex(SystemExit, 'source files changed since preview'):
                    batch_mod.apply_revisions(str(manifest_path))

            manifest = self._load_manifest(manifest_path)
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'source_changed')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(target_file.read_text(encoding='utf-8'), changed)

    def test_apply_rejects_manifest_changed_since_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(root)
            original = target_file.read_text(encoding='utf-8')
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                manifest = self._load_manifest(manifest_path)
                manifest['summary'] = {'item_count': 999}
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
                with self.assertRaisesRegex(SystemExit, 'manifest changed since preview'):
                    batch_mod.apply_revisions(str(manifest_path))

            manifest = self._load_manifest(manifest_path)
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'manifest_changed')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(target_file.read_text(encoding='utf-8'), original)

    def test_apply_rejects_project_identity_changed_since_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(root)
            original = target_file.read_text(encoding='utf-8')
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                manifest = self._load_manifest(manifest_path)
                manifest['last_revision_preview']['project_identity']['tl_dir'] = (
                    str(root / 'other' / 'tl')
                )
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
                with self.assertRaisesRegex(SystemExit, 'project identity changed since preview'):
                    batch_mod.apply_revisions(str(manifest_path))

            manifest = self._load_manifest(manifest_path)
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'project_changed')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(target_file.read_text(encoding='utf-8'), original)

    def test_all_mismatch_apply_records_blocked_reason_and_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(
                root,
                current_new='虚空门',
                file_new='星门',
            )
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                manifest = batch_mod.apply_revisions(str(manifest_path))

            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(manifest['revision_apply_blocked_reason'], 'all_items_blocked')
            self.assertIn(
                'No revisions could be written back',
                manifest['revision_apply_message'],
            )
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(manifest['revision_apply_summary']['applied_files'], 0)

    def test_apply_unchanged_only_reports_no_op_without_applied_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(
                root,
                revised='虚空门',
                should_update=True,
            )
            original = target_file.read_text(encoding='utf-8')
            tl_dir = target_file.parent
            with (
                mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)),
                mock.patch.object(batch_mod, 'update_progress') as update_progress,
            ):
                batch_mod.preview_revisions(str(manifest_path))
                manifest = batch_mod.apply_revisions(str(manifest_path))

            self.assertEqual(manifest['revision_apply_state'], 'no_op')
            self.assertNotIn('revision_applied_at', manifest)
            self.assertEqual(manifest['revision_apply_summary']['applied_files'], 0)
            self.assertEqual(manifest['revision_apply_summary']['applied_lines'], 0)
            self.assertEqual(manifest['revision_apply_summary']['unchanged_items'], 1)
            update_progress.assert_not_called()
            self.assertEqual(target_file.read_text(encoding='utf-8'), original)

    def test_fresh_preview_clears_stale_apply_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, result_path = self._make_package(root)
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                result_path.write_text(
                    result_path.read_text(encoding='utf-8').rstrip()
                    + '\n{"replacement": true}\n',
                    encoding='utf-8',
                )
                with self.assertRaisesRegex(SystemExit, 'result JSONL changed since preview'):
                    batch_mod.apply_revisions(str(manifest_path))
                manifest = self._load_manifest(manifest_path)
                self.assertEqual(manifest['revision_apply_state'], 'blocked')
                self.assertIn('revision_apply_blocked_reason', manifest)

                result_path.write_text(
                    '\n'.join(
                        line
                        for line in result_path.read_text(encoding='utf-8').splitlines()
                        if '"replacement": true' not in line
                    )
                    + '\n',
                    encoding='utf-8',
                )
                batch_mod.preview_revisions(str(manifest_path))
                manifest = self._load_manifest(manifest_path)
                self.assertNotIn('revision_apply_state', manifest)
                self.assertNotIn('revision_apply_blocked_reason', manifest)
                self.assertNotIn('revision_apply_message', manifest)
                self.assertNotIn('revision_apply_summary', manifest)
                self.assertNotIn('last_revision_apply_summary', manifest)
                self.assertEqual(manifest['last_revision_preview']['summary']['valid_items'], 1)

                applied = batch_mod.apply_revisions(str(manifest_path))
                self.assertEqual(applied['revision_apply_state'], 'applied')

    def test_fresh_preview_after_applied_moves_history_and_reopens_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, target_file, _ = self._make_package(root)
            tl_dir = target_file.parent
            with mock.patch.object(batch_mod.legacy, 'TL_DIR', str(tl_dir)):
                batch_mod.preview_revisions(str(manifest_path))
                first = batch_mod.apply_revisions(str(manifest_path))
                self.assertEqual(first['revision_apply_state'], 'applied')
                self.assertIn('revision_applied_at', first)
                first_applied_at = first['revision_applied_at']

                batch_mod.preview_revisions(str(manifest_path))
                manifest = self._load_manifest(manifest_path)
                self.assertNotIn('revision_applied_at', manifest)
                self.assertNotIn('revision_apply_state', manifest)
                self.assertNotIn('revision_apply_summary', manifest)
                history = manifest.get('revision_apply_history') or []
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]['applied_at'], first_applied_at)

                second = batch_mod.apply_revisions(str(manifest_path))
                # The same revision was already written back; the reopened gate
                # now reports no_op instead of being blocked by a stale guard.
                self.assertEqual(second['revision_apply_state'], 'no_op')
                self.assertNotIn('revision_applied_at', second)
                manifest = self._load_manifest(manifest_path)
                self.assertEqual(len(manifest.get('revision_apply_history') or []), 1)



if __name__ == '__main__':
    unittest.main()
