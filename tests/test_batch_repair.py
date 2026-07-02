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
                    '    "World"\n',
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
                                        'content': {'parts': [{'text': '[]'}]},
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
                            'files': {'script.rpy': {'path': str(target_file), 'task_count': 2}},
                            'summary': {'file_count': 1, 'chunk_count': 2, 'item_count': 2},
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
                    self.assertEqual([chunk['key'] for chunk in retry_manifest['chunks']], ['chunk-bad'])
                    self.assertEqual(retry_manifest['summary']['item_count'], 1)
                    self.assertTrue(retry_manifest['source_index_enabled'])
                    self.assertEqual(latest_path.read_text(encoding='utf-8'), retry_manifest_path)
                    retry_request = json.loads(Path(retry_manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()[0])
                    self.assertEqual(retry_request['key'], 'chunk-bad')
                    self.assertIn(
                        'copy that exact source substring verbatim',
                        retry_request['request']['system_instruction']['parts'][0]['text'],
                    )

                    retry_result_path = Path(retry_manifest_path).parent / 'results.jsonl'
                    batch_mod.write_jsonl_file(
                        str(retry_result_path),
                        [
                            {
                                'key': 'chunk-bad',
                                'response': {
                                    'candidates': [
                                        {
                                            'content': {
                                                'parts': [
                                                    {
                                                        'text': json.dumps(
                                                            [{'id': 'script.rpy:1:4:11:world', 'translation': '世界'}],
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

                    rechecked = batch_mod.check_results(str(manifest_path))
                    self.assertEqual(rechecked['last_check_summary']['safety_level'], batch_mod.CHECK_SAFETY_SAFE)
        finally:
            batch_mod.legacy.TL_DIR = old_tl_dir

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
        self.assertEqual(rows[0]['source_lines'], [2])
        self.assertEqual(rows[0]['source_item_ids'], ['script.rpy:2:keyword:0'])
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
        self.assertIn('should_update', schema['items']['required'])
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

    def test_collect_result_actions_ignores_duplicate_result_ids(self):
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

        self.assertEqual(summary['valid_items'], 1)
        self.assertEqual(summary['reason_counts']['duplicate_result_id'], 1)
        self.assertEqual(len(replacements['script.rpy'][0]), 1)
        self.assertEqual(failures, [])

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
        self.assertEqual(checked['last_check_summary']['safety_reasons']['warn']['response_missing_item_id'], 1)
        self.assertEqual(final_script, first_line + second_line)
        update_progress.assert_not_called()
        self.assertEqual(check_failures[0]['status'], 'warn')
        self.assertEqual(check_failures[0]['reason_code'], 'response_missing_item_id')

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



if __name__ == '__main__':
    unittest.main()
