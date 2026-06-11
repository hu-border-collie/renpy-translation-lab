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



class BatchGoldenCorpusTests(unittest.TestCase):
    def _copy_fixture_tl(self, root):
        tl_dir = root / 'game' / 'tl' / 'schinese'
        tl_dir.parent.mkdir(parents=True)
        shutil.copytree(GOLDEN_BATCH_FIXTURE_DIR / 'tl', tl_dir)
        return tl_dir

    def _patch_batch_environment(self, root, tl_dir):
        old_values = {
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
            'progress': batch_mod.PROGRESS_LOG,
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store': batch_mod._RAG_STORE,
            'story_enabled': batch_mod.STORY_MEMORY_ENABLED,
            'story_graph': batch_mod._STORY_GRAPH,
            'story_graph_path': batch_mod._STORY_GRAPH_PATH,
        }
        log_dir = root / 'logs'
        jobs_dir = log_dir / 'batch_jobs'
        repair_dir = log_dir / 'repair_runs'
        sync_dir = log_dir / 'sync_runs'
        batch_mod.legacy.BASE_DIR = str(root)
        batch_mod.legacy.TL_DIR = str(tl_dir)
        batch_mod.legacy.INCLUDE_FILES = set()
        batch_mod.legacy.INCLUDE_PREFIXES = set()
        batch_mod.LOG_DIR = str(log_dir)
        batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
        batch_mod.REPAIR_RUNS_DIR = str(repair_dir)
        batch_mod.SYNC_RUNS_DIR = str(sync_dir)
        batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')
        batch_mod.PROGRESS_LOG = str(log_dir / 'translation_progress_batch.json')
        batch_mod.RAG_ENABLED = False
        batch_mod._RAG_STORE = None
        batch_mod.STORY_MEMORY_ENABLED = False
        batch_mod._STORY_GRAPH = None
        batch_mod._STORY_GRAPH_PATH = ''
        return old_values

    def _restore_batch_environment(self, old_values):
        batch_mod.legacy.BASE_DIR = old_values['base_dir']
        batch_mod.legacy.TL_DIR = old_values['tl_dir']
        batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
        batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']
        batch_mod.LOG_DIR = old_values['log_dir']
        batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
        batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
        batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
        batch_mod.LATEST_MANIFEST_FILE = old_values['latest']
        batch_mod.PROGRESS_LOG = old_values['progress']
        batch_mod.RAG_ENABLED = old_values['rag_enabled']
        batch_mod._RAG_STORE = old_values['rag_store']
        batch_mod.STORY_MEMORY_ENABLED = old_values['story_enabled']
        batch_mod._STORY_GRAPH = old_values['story_graph']
        batch_mod._STORY_GRAPH_PATH = old_values['story_graph_path']

    def _load_manifest(self, manifest_path):
        return json.loads(Path(manifest_path).read_text(encoding='utf-8'))

    def _manifest_snapshot(self, manifest):
        return {
            'mode': manifest['mode'],
            'core_schema_version': manifest['core_schema_version'],
            'summary': manifest['summary'],
            'settings': manifest['settings'],
            'files': {
                rel_path: {'task_count': info['task_count']}
                for rel_path, info in manifest['files'].items()
            },
            'chunks': [
                {
                    'key': chunk['key'],
                    'file_rel_path': chunk['file_rel_path'],
                    'chunk_index': chunk['chunk_index'],
                    'line_numbers': chunk['line_numbers'],
                    'source_char_count': chunk['source_char_count'],
                    'context_past': [
                        {'line': item['line'], 'text': item['text']}
                        for item in chunk['context_past']
                    ],
                    'context_future': [
                        {'line': item['line'], 'text': item['text']}
                        for item in chunk['context_future']
                    ],
                    'items': [
                        {
                            key: item[key]
                            for key in (
                                'id',
                                'text',
                                'line',
                                'line_number',
                                'start',
                                'end',
                                'prefix',
                                'quote',
                                'speaker_id',
                                'speaker',
                                'speaker_name',
                            )
                            if key in item
                        }
                        for item in chunk['items']
                    ],
                }
                for chunk in manifest['chunks']
            ],
        }

    def _request_snapshot(self, manifest):
        chunk_by_key = {chunk['key']: chunk for chunk in manifest['chunks']}
        request_rows = [
            json.loads(line)
            for line in Path(manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        rows = []
        for row in request_rows:
            request = row['request']
            chunk = chunk_by_key[row['key']]
            config = request['generation_config']
            user_prompt = request['contents'][0]['parts'][0]['text']
            system_text = request['system_instruction']['parts'][0]['text']
            rows.append(
                {
                    'key': row['key'],
                    'file_rel_path': chunk['file_rel_path'],
                    'request_keys': sorted(request.keys()),
                    'content_roles': [content.get('role') for content in request['contents']],
                    'target_item_ids': [item['id'] for item in chunk['items']],
                    'system_instruction_sha256': hashlib.sha256(system_text.encode('utf-8')).hexdigest(),
                    'user_prompt_sha256': hashlib.sha256(user_prompt.encode('utf-8')).hexdigest(),
                    'generation_config': {
                        'keys': sorted(config.keys()),
                        'temperature': config['temperature'],
                        'max_output_tokens': config['max_output_tokens'],
                        'response_mime_type': config['response_mime_type'],
                        'thinking_config': config.get('thinking_config', {}),
                        'response_json_schema': config['response_json_schema'],
                    },
                }
            )
        return {'rows': rows}

    def _stable_summary(self, summary):
        return {
            'expected_chunks': summary['expected_chunks'],
            'result_rows': summary['result_rows'],
            'processed_chunks': summary['processed_chunks'],
            'expected_items': summary['expected_items'],
            'candidate_valid_items': summary['candidate_valid_items'],
            'valid_items': summary['valid_items'],
            'pending_files': summary['pending_files'],
            'pending_lines': summary['pending_lines'],
            'skipped_items': summary['skipped_items'],
            'source_mismatch_items': summary['source_mismatch_items'],
            'failure_items': summary['failure_items'],
            'chunk_row_errors': summary['chunk_row_errors'],
            'missing_response_chunks': summary['missing_response_chunks'],
            'partial_chunks': summary['partial_chunks'],
            'max_tokens_chunks': summary['max_tokens_chunks'],
            'reason_counts': summary['reason_counts'],
        }

    def _assert_or_update_json(self, relative_path, actual):
        expected_path = GOLDEN_BATCH_FIXTURE_DIR / relative_path
        text = json.dumps(actual, ensure_ascii=False, indent=2) + '\n'
        if os.environ.get(UPDATE_GOLDEN_BATCH_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(text, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        self.assertEqual(actual, expected)

    def _assert_or_update_text(self, relative_path, actual):
        expected_path = GOLDEN_BATCH_FIXTURE_DIR / relative_path
        if os.environ.get(UPDATE_GOLDEN_BATCH_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(actual, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        self.assertEqual(actual, expected_path.read_text(encoding='utf-8'))

    def _write_mock_results(self, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = self._load_manifest(manifest_path)
        translations = json.loads(
            (GOLDEN_BATCH_FIXTURE_DIR / 'model_results.json').read_text(encoding='utf-8')
        )
        result_path = manifest_path.parent / 'results.jsonl'
        rows = []
        for chunk in manifest['chunks']:
            result_items = [
                {'id': item['id'], 'translation': translations[item['text']]}
                for item in chunk['items']
            ]
            response_text = json.dumps(result_items, ensure_ascii=False)
            rows.append(
                {
                    'key': chunk['key'],
                    'response': {
                        'candidates': [
                            {
                                'content': {'parts': [{'text': response_text}]},
                                'finishReason': 'STOP',
                            }
                        ],
                        'usageMetadata': {
                            'promptTokenCount': 100,
                            'candidatesTokenCount': 40,
                            'totalTokenCount': 140,
                        },
                    },
                }
            )
        result_path.write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
            encoding='utf-8',
        )
        manifest['result_jsonl_path'] = 'results.jsonl'
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def test_golden_batch_build_check_apply_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-minimal',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                manifest = self._load_manifest(manifest_path)

                self._assert_or_update_json(
                    'expected/manifest_snapshot.json',
                    self._manifest_snapshot(manifest),
                )
                self._assert_or_update_json(
                    'expected/request_snapshot.json',
                    self._request_snapshot(manifest),
                )

                checked_manifest = batch_mod.check_results(str(manifest_path))
                self.assertEqual(checked_manifest['last_check_summary']['safety_level'], 'safe')
                applied_manifest = batch_mod.apply_results(str(manifest_path))
                progress = json.loads(Path(batch_mod.PROGRESS_LOG).read_text(encoding='utf-8'))

                check_apply_snapshot = {
                    'last_check_summary': self._stable_summary(checked_manifest['last_check_summary']),
                    'apply_summary': applied_manifest['apply_summary'],
                    'progress': progress,
                }
                self._assert_or_update_json(
                    'expected/check_apply_snapshot.json',
                    check_apply_snapshot,
                )
                self._assert_or_update_text(
                    'expected/applied/chapter01/dialogue.rpy',
                    (tl_dir / 'chapter01' / 'dialogue.rpy').read_text(encoding='utf-8'),
                )
                self._assert_or_update_text(
                    'expected/applied/chapter02/strings.rpy',
                    (tl_dir / 'chapter02' / 'strings.rpy').read_text(encoding='utf-8'),
                )

                with self.assertRaisesRegex(SystemExit, 'already applied'):
                    batch_mod.apply_results(str(manifest_path))
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_rejects_changed_source_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-minimal',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                with self.assertRaisesRegex(SystemExit, 'no valid check summary'):
                    batch_mod.apply_results(str(manifest_path))

                batch_mod.check_results(str(manifest_path))
                results_path = manifest_path.parent / 'results.jsonl'
                results_path.write_text(
                    results_path.read_text(encoding='utf-8') + '\n',
                    encoding='utf-8',
                )
                with self.assertRaisesRegex(SystemExit, 'changed after the last check'):
                    batch_mod.apply_results(str(manifest_path))

                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                dialogue_path = tl_dir / 'chapter01' / 'dialogue.rpy'
                dialogue_path.write_text(
                    dialogue_path.read_text(encoding='utf-8').replace(
                        'e "Welcome back, traveler."',
                        'e "Welcome home, traveler."',
                    ),
                    encoding='utf-8',
                )

                with self.assertRaisesRegex(SystemExit, 'current results are not safe'):
                    batch_mod.apply_results(str(manifest_path))
                final_dialogue = dialogue_path.read_text(encoding='utf-8')
                saved_manifest = self._load_manifest(manifest_path)

                self.assertIn('e "Welcome home, traveler."', final_dialogue)
                self.assertNotIn('e "旅人，欢迎回来。"', final_dialogue)
                self.assertIn('e "Don\'t touch the crystal."', final_dialogue)
                self.assertNotIn('e "请勿触碰水晶。"', final_dialogue)
                self.assertIn('last_apply_failure_report_path', saved_manifest)
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_rejects_changed_manifest_items_after_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-minimal',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                checked_manifest = batch_mod.check_results(str(manifest_path))
                self.assertEqual(checked_manifest['last_check_summary']['safety_level'], 'safe')

                manifest = self._load_manifest(manifest_path)
                manifest['chunks'][0]['items'].append(
                    {
                        'id': 'chapter01/dialogue.rpy:999:0',
                        'file_rel_path': 'chapter01/dialogue.rpy',
                        'line': 999,
                        'start': 0,
                        'end': 5,
                        'text': 'extra',
                        'prefix': '',
                        'quote': '"',
                    }
                )
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )

                with self.assertRaisesRegex(SystemExit, 'changed after the last check'):
                    batch_mod.apply_results(str(manifest_path))
            finally:
                self._restore_batch_environment(old_values)


class RevisionGoldenCorpusTests(unittest.TestCase):
    def _copy_fixture_tl(self, root):
        tl_dir = root / 'game' / 'tl' / 'schinese'
        tl_dir.parent.mkdir(parents=True)
        shutil.copytree(GOLDEN_REVISION_FIXTURE_DIR / 'tl', tl_dir)
        return tl_dir

    def _patch_batch_environment(self, root, tl_dir):
        old_values = {
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
            'progress': batch_mod.PROGRESS_LOG,
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store': batch_mod._RAG_STORE,
            'story_enabled': batch_mod.STORY_MEMORY_ENABLED,
            'story_graph': batch_mod._STORY_GRAPH,
            'story_graph_path': batch_mod._STORY_GRAPH_PATH,
        }
        log_dir = root / 'logs'
        jobs_dir = log_dir / 'batch_jobs'
        batch_mod.legacy.BASE_DIR = str(root)
        batch_mod.legacy.TL_DIR = str(tl_dir)
        batch_mod.legacy.INCLUDE_FILES = set()
        batch_mod.legacy.INCLUDE_PREFIXES = set()
        batch_mod.LOG_DIR = str(log_dir)
        batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
        batch_mod.REPAIR_RUNS_DIR = str(log_dir / 'repair_runs')
        batch_mod.SYNC_RUNS_DIR = str(log_dir / 'sync_runs')
        batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')
        batch_mod.PROGRESS_LOG = str(log_dir / 'translation_progress_batch.json')
        batch_mod.RAG_ENABLED = False
        batch_mod._RAG_STORE = None
        batch_mod.STORY_MEMORY_ENABLED = False
        batch_mod._STORY_GRAPH = None
        batch_mod._STORY_GRAPH_PATH = ''
        return old_values

    def _restore_batch_environment(self, old_values):
        batch_mod.legacy.BASE_DIR = old_values['base_dir']
        batch_mod.legacy.TL_DIR = old_values['tl_dir']
        batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
        batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']
        batch_mod.LOG_DIR = old_values['log_dir']
        batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
        batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
        batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
        batch_mod.LATEST_MANIFEST_FILE = old_values['latest']
        batch_mod.PROGRESS_LOG = old_values['progress']
        batch_mod.RAG_ENABLED = old_values['rag_enabled']
        batch_mod._RAG_STORE = old_values['rag_store']
        batch_mod.STORY_MEMORY_ENABLED = old_values['story_enabled']
        batch_mod._STORY_GRAPH = old_values['story_graph']
        batch_mod._STORY_GRAPH_PATH = old_values['story_graph_path']

    def _load_manifest(self, manifest_path):
        return json.loads(Path(manifest_path).read_text(encoding='utf-8'))

    def _assert_or_update_json(self, relative_path, actual):
        expected_path = GOLDEN_REVISION_FIXTURE_DIR / relative_path
        text = json.dumps(actual, ensure_ascii=False, indent=2) + '\n'
        if os.environ.get(UPDATE_GOLDEN_REVISION_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(text, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        self.assertEqual(actual, expected)

    def _assert_or_update_text(self, relative_path, actual):
        expected_path = GOLDEN_REVISION_FIXTURE_DIR / relative_path
        if os.environ.get(UPDATE_GOLDEN_REVISION_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(actual, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        self.assertEqual(actual, expected_path.read_text(encoding='utf-8'))

    def _manifest_snapshot(self, manifest):
        return {
            'mode': manifest['mode'],
            'core_schema_version': manifest['core_schema_version'],
            'summary': manifest['summary'],
            'settings': manifest['settings'],
            'revision_settings': manifest['revision_settings'],
            'files': {
                rel_path: {'task_count': info['task_count']}
                for rel_path, info in manifest['files'].items()
            },
            'chunks': [
                {
                    'key': chunk['key'],
                    'mode': chunk['mode'],
                    'file_rel_path': chunk['file_rel_path'],
                    'chunk_index': chunk['chunk_index'],
                    'line_numbers': chunk['line_numbers'],
                    'context_past': chunk['context_past'],
                    'context_future': chunk['context_future'],
                    'items': [
                        {
                            key: item[key]
                            for key in (
                                'id',
                                'text',
                                'source',
                                'current_translation',
                                'line',
                                'line_number',
                                'start',
                                'end',
                                'prefix',
                                'quote',
                            )
                            if key in item
                        }
                        for item in chunk['items']
                    ],
                }
                for chunk in manifest['chunks']
            ],
        }

    def _request_snapshot(self, manifest):
        chunk_by_key = {chunk['key']: chunk for chunk in manifest['chunks']}
        request_rows = [
            json.loads(line)
            for line in Path(manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        rows = []
        for row in request_rows:
            request = row['request']
            chunk = chunk_by_key[row['key']]
            config = request['generation_config']
            user_prompt = request['contents'][0]['parts'][0]['text']
            system_text = request['system_instruction']['parts'][0]['text']
            rows.append(
                {
                    'key': row['key'],
                    'file_rel_path': chunk['file_rel_path'],
                    'request_keys': sorted(request.keys()),
                    'content_roles': [content.get('role') for content in request['contents']],
                    'target_item_ids': [item['id'] for item in chunk['items']],
                    'system_instruction_sha256': hashlib.sha256(system_text.encode('utf-8')).hexdigest(),
                    'user_prompt_sha256': hashlib.sha256(user_prompt.encode('utf-8')).hexdigest(),
                    'generation_config': {
                        'keys': sorted(config.keys()),
                        'temperature': config['temperature'],
                        'max_output_tokens': config['max_output_tokens'],
                        'response_mime_type': config['response_mime_type'],
                        'thinking_config': config.get('thinking_config', {}),
                        'response_json_schema': config['response_json_schema'],
                    },
                }
            )
        return {'rows': rows}

    def _stable_revision_summary(self, summary):
        return {
            'expected_chunks': summary['expected_chunks'],
            'result_rows': summary['result_rows'],
            'processed_chunks': summary['processed_chunks'],
            'expected_items': summary['expected_items'],
            'parsed_items': summary['parsed_items'],
            'candidate_valid_items': summary['candidate_valid_items'],
            'revision_candidate_items': summary['revision_candidate_items'],
            'valid_items': summary['valid_items'],
            'unchanged_items': summary['unchanged_items'],
            'pending_files': summary['pending_files'],
            'pending_lines': summary['pending_lines'],
            'skipped_items': summary['skipped_items'],
            'source_mismatch_items': summary['source_mismatch_items'],
            'failure_items': summary['failure_items'],
            'chunk_row_errors': summary['chunk_row_errors'],
            'missing_response_chunks': summary['missing_response_chunks'],
            'partial_chunks': summary['partial_chunks'],
            'max_tokens_chunks': summary['max_tokens_chunks'],
            'reason_counts': summary['reason_counts'],
        }

    def _write_mock_revision_results(self, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = self._load_manifest(manifest_path)
        revisions = json.loads(
            (GOLDEN_REVISION_FIXTURE_DIR / 'model_results.json').read_text(encoding='utf-8')
        )
        result_path = manifest_path.parent / 'results.jsonl'
        rows = []
        for chunk in manifest['chunks']:
            result_items = []
            for item in chunk['items']:
                revision = revisions[item['source']]
                result_items.append(
                    {
                        'id': item['id'],
                        'should_update': revision['should_update'],
                        'revised_translation': revision['revised_translation'],
                        'reason': revision['reason'],
                    }
                )
            response_text = json.dumps(result_items, ensure_ascii=False)
            rows.append(
                {
                    'key': chunk['key'],
                    'response': {
                        'candidates': [
                            {
                                'content': {'parts': [{'text': response_text}]},
                                'finishReason': 'STOP',
                            }
                        ],
                        'usageMetadata': {
                            'promptTokenCount': 100,
                            'candidatesTokenCount': 40,
                            'totalTokenCount': 140,
                        },
                    },
                }
            )
        result_path.write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
            encoding='utf-8',
        )
        manifest['result_jsonl_path'] = 'results.jsonl'
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _read_preview_rows(self, manifest):
        preview_jsonl = Path(manifest['last_revision_preview']['jsonl_path'])
        return [
            json.loads(line)
            for line in preview_jsonl.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def test_golden_revision_build_preview_apply_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_revision_package(
                        display_name_override='golden-revision-minimal',
                        skip_prepare=True,
                        chunk_size=3,
                    )
                )
                self._write_mock_revision_results(manifest_path)
                manifest = self._load_manifest(manifest_path)

                self._assert_or_update_json(
                    'expected/manifest_snapshot.json',
                    self._manifest_snapshot(manifest),
                )
                self._assert_or_update_json(
                    'expected/request_snapshot.json',
                    self._request_snapshot(manifest),
                )

                preview_manifest = batch_mod.preview_revisions(str(manifest_path))
                preview_rows = self._read_preview_rows(preview_manifest)
                preview_markdown = Path(preview_manifest['last_revision_preview']['markdown_path']).read_text(
                    encoding='utf-8'
                )
                applied_manifest = batch_mod.apply_revisions(str(manifest_path))
                progress = json.loads(Path(batch_mod.PROGRESS_LOG).read_text(encoding='utf-8'))

                preview_apply_snapshot = {
                    'last_revision_preview_summary': self._stable_revision_summary(
                        preview_manifest['last_revision_preview']['summary']
                    ),
                    'preview_rows': preview_rows,
                    'revision_apply_summary': applied_manifest['revision_apply_summary'],
                    'last_revision_apply_summary': self._stable_revision_summary(
                        applied_manifest['last_revision_apply_summary']
                    ),
                    'progress': progress,
                }
                self._assert_or_update_json(
                    'expected/preview_apply_snapshot.json',
                    preview_apply_snapshot,
                )
                self._assert_or_update_text(
                    'expected/revision_preview.md',
                    preview_markdown,
                )
                self._assert_or_update_text(
                    'expected/applied/chapter01/revisions.rpy',
                    (tl_dir / 'chapter01' / 'revisions.rpy').read_text(encoding='utf-8'),
                )

                with self.assertRaisesRegex(SystemExit, 'already applied'):
                    batch_mod.apply_revisions(str(manifest_path))
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_revision_apply_rejects_changed_current_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_revision_package(
                        display_name_override='golden-revision-minimal',
                        skip_prepare=True,
                        chunk_size=3,
                    )
                )
                self._write_mock_revision_results(manifest_path)
                revisions_path = tl_dir / 'chapter01' / 'revisions.rpy'
                revisions_path.write_text(
                    revisions_path.read_text(encoding='utf-8').replace(
                        'new "虚空门"',
                        'new "星门"',
                    ),
                    encoding='utf-8',
                )

                preview_manifest = batch_mod.preview_revisions(str(manifest_path))
                preview_rows = self._read_preview_rows(preview_manifest)
                applied_manifest = batch_mod.apply_revisions(str(manifest_path))
                final_revisions = revisions_path.read_text(encoding='utf-8')

                self.assertIn('old "Void Gate"', final_revisions)
                self.assertIn('new "星门"', final_revisions)
                self.assertNotIn('new "虚空之门"', final_revisions)
                self.assertIn('new "晶核钥匙"', final_revisions)
                self.assertEqual(preview_rows[0]['status'], 'source_mismatch')
                self.assertEqual(preview_rows[1]['status'], 'pending')
                self.assertEqual(preview_rows[2]['status'], 'unchanged')
                self.assertEqual(applied_manifest['revision_apply_summary']['candidate_items'], 2)
                self.assertEqual(applied_manifest['revision_apply_summary']['recoverable_items'], 1)
                self.assertEqual(applied_manifest['revision_apply_summary']['unchanged_items'], 1)
                self.assertEqual(applied_manifest['revision_apply_summary']['skipped_items'], 1)
                self.assertEqual(applied_manifest['revision_apply_summary']['source_mismatch_items'], 1)
                self.assertEqual(applied_manifest['revision_apply_summary']['failure_count'], 1)
            finally:
                self._restore_batch_environment(old_values)


class KeywordGoldenCorpusTests(unittest.TestCase):
    def _copy_fixture_tl(self, root):
        tl_dir = root / 'game' / 'tl' / 'schinese'
        tl_dir.parent.mkdir(parents=True)
        shutil.copytree(GOLDEN_KEYWORD_FIXTURE_DIR / 'tl', tl_dir)
        return tl_dir

    def _patch_batch_environment(self, root, tl_dir):
        old_values = {
            'base_dir': batch_mod.legacy.BASE_DIR,
            'tl_dir': batch_mod.legacy.TL_DIR,
            'include_files': set(batch_mod.legacy.INCLUDE_FILES),
            'include_prefixes': set(batch_mod.legacy.INCLUDE_PREFIXES),
            'log_dir': batch_mod.LOG_DIR,
            'jobs_dir': batch_mod.BATCH_JOBS_DIR,
            'repair_dir': batch_mod.REPAIR_RUNS_DIR,
            'sync_dir': batch_mod.SYNC_RUNS_DIR,
            'latest': batch_mod.LATEST_MANIFEST_FILE,
            'rag_enabled': batch_mod.RAG_ENABLED,
            'rag_store': batch_mod._RAG_STORE,
            'story_enabled': batch_mod.STORY_MEMORY_ENABLED,
            'story_graph': batch_mod._STORY_GRAPH,
            'story_graph_path': batch_mod._STORY_GRAPH_PATH,
        }
        log_dir = root / 'logs'
        jobs_dir = log_dir / 'batch_jobs'
        batch_mod.legacy.BASE_DIR = str(root)
        batch_mod.legacy.TL_DIR = str(tl_dir)
        batch_mod.legacy.INCLUDE_FILES = set()
        batch_mod.legacy.INCLUDE_PREFIXES = set()
        batch_mod.LOG_DIR = str(log_dir)
        batch_mod.BATCH_JOBS_DIR = str(jobs_dir)
        batch_mod.REPAIR_RUNS_DIR = str(log_dir / 'repair_runs')
        batch_mod.SYNC_RUNS_DIR = str(log_dir / 'sync_runs')
        batch_mod.LATEST_MANIFEST_FILE = str(jobs_dir / 'latest_manifest.txt')
        batch_mod.RAG_ENABLED = False
        batch_mod._RAG_STORE = None
        batch_mod.STORY_MEMORY_ENABLED = False
        batch_mod._STORY_GRAPH = None
        batch_mod._STORY_GRAPH_PATH = ''
        return old_values

    def _restore_batch_environment(self, old_values):
        batch_mod.legacy.BASE_DIR = old_values['base_dir']
        batch_mod.legacy.TL_DIR = old_values['tl_dir']
        batch_mod.legacy.INCLUDE_FILES = old_values['include_files']
        batch_mod.legacy.INCLUDE_PREFIXES = old_values['include_prefixes']
        batch_mod.LOG_DIR = old_values['log_dir']
        batch_mod.BATCH_JOBS_DIR = old_values['jobs_dir']
        batch_mod.REPAIR_RUNS_DIR = old_values['repair_dir']
        batch_mod.SYNC_RUNS_DIR = old_values['sync_dir']
        batch_mod.LATEST_MANIFEST_FILE = old_values['latest']
        batch_mod.RAG_ENABLED = old_values['rag_enabled']
        batch_mod._RAG_STORE = old_values['rag_store']
        batch_mod.STORY_MEMORY_ENABLED = old_values['story_enabled']
        batch_mod._STORY_GRAPH = old_values['story_graph']
        batch_mod._STORY_GRAPH_PATH = old_values['story_graph_path']

    def _load_manifest(self, manifest_path):
        return json.loads(Path(manifest_path).read_text(encoding='utf-8'))

    def _assert_or_update_json(self, relative_path, actual):
        expected_path = GOLDEN_KEYWORD_FIXTURE_DIR / relative_path
        text = json.dumps(actual, ensure_ascii=False, indent=2) + '\n'
        if os.environ.get(UPDATE_GOLDEN_KEYWORD_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(text, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        expected = json.loads(expected_path.read_text(encoding='utf-8'))
        self.assertEqual(actual, expected)

    def _assert_or_update_text(self, relative_path, actual):
        expected_path = GOLDEN_KEYWORD_FIXTURE_DIR / relative_path
        if os.environ.get(UPDATE_GOLDEN_KEYWORD_ENV):
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            expected_path.write_text(actual, encoding='utf-8')
            return
        self.assertTrue(expected_path.is_file(), f'Missing golden file: {expected_path}')
        self.assertEqual(actual, expected_path.read_text(encoding='utf-8'))

    def _manifest_snapshot(self, manifest):
        return {
            'mode': manifest['mode'],
            'core_schema_version': manifest['core_schema_version'],
            'summary': manifest['summary'],
            'settings': manifest['settings'],
            'keyword_settings': manifest['keyword_settings'],
            'files': {
                rel_path: {'task_count': info['task_count']}
                for rel_path, info in manifest['files'].items()
            },
            'chunks': [
                {
                    'key': chunk['key'],
                    'mode': chunk['mode'],
                    'file_rel_path': chunk['file_rel_path'],
                    'chunk_index': chunk['chunk_index'],
                    'line_numbers': chunk['line_numbers'],
                    'items': [
                        {
                            key: item[key]
                            for key in (
                                'id',
                                'text',
                                'file_rel_path',
                                'line_number',
                                'translation_line_number',
                                'speaker_id',
                                'speaker_name',
                            )
                            if key in item
                        }
                        for item in chunk['items']
                    ],
                }
                for chunk in manifest['chunks']
            ],
        }

    def _request_snapshot(self, manifest):
        chunk_by_key = {chunk['key']: chunk for chunk in manifest['chunks']}
        request_rows = [
            json.loads(line)
            for line in Path(manifest['input_jsonl_path']).read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        rows = []
        for row in request_rows:
            request = row['request']
            chunk = chunk_by_key[row['key']]
            config = request['generation_config']
            user_prompt = request['contents'][0]['parts'][0]['text']
            system_text = request['system_instruction']['parts'][0]['text']
            rows.append(
                {
                    'key': row['key'],
                    'file_rel_path': chunk['file_rel_path'],
                    'request_keys': sorted(request.keys()),
                    'content_roles': [content.get('role') for content in request['contents']],
                    'target_item_ids': [item['id'] for item in chunk['items']],
                    'system_instruction_sha256': hashlib.sha256(system_text.encode('utf-8')).hexdigest(),
                    'user_prompt_sha256': hashlib.sha256(user_prompt.encode('utf-8')).hexdigest(),
                    'generation_config': {
                        'keys': sorted(config.keys()),
                        'temperature': config['temperature'],
                        'max_output_tokens': config['max_output_tokens'],
                        'response_mime_type': config['response_mime_type'],
                        'thinking_config': config.get('thinking_config', {}),
                        'response_json_schema': config['response_json_schema'],
                    },
                }
            )
        return {'rows': rows}

    def _read_jsonl(self, path):
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def _item_matches_source(self, item, source):
        return str(source or '').lower() in str(item.get('text') or '').lower()

    def _items_matching_sources(self, chunk, sources):
        matched = []
        seen = set()
        for source in sources or []:
            for item in chunk.get('items') or []:
                item_id = item.get('id')
                if item_id in seen:
                    continue
                if self._item_matches_source(item, source):
                    seen.add(item_id)
                    matched.append(item)
        return matched

    def _write_mock_keyword_results(self, manifest_path):
        manifest_path = Path(manifest_path)
        manifest = self._load_manifest(manifest_path)
        model_results = json.loads(
            (GOLDEN_KEYWORD_FIXTURE_DIR / 'model_results.json').read_text(encoding='utf-8')
        )
        candidate_specs = model_results['candidates']
        summary_specs = model_results['chunk_summaries']
        result_path = manifest_path.parent / 'results.jsonl'
        rows = []
        for chunk in manifest['chunks']:
            candidates = []
            for spec in candidate_specs:
                matched_items = self._items_matching_sources(chunk, [spec['source']])
                if not matched_items:
                    continue
                candidate = {
                    'source': spec['source'],
                    'suggested_target': spec['suggested_target'],
                    'category': spec['category'],
                    'confidence': spec['confidence'],
                    'evidence': spec['evidence'],
                    'source_item_ids': [item['id'] for item in matched_items],
                }
                candidates.append(candidate)
            summary_spec = summary_specs[str(chunk['chunk_index'])]
            summary_items = self._items_matching_sources(chunk, summary_spec['sources'])
            response_text = json.dumps(
                {
                    'candidates': candidates,
                    'chunk_summary': summary_spec['chunk_summary'],
                    'summary_evidence_item_ids': [item['id'] for item in summary_items],
                },
                ensure_ascii=False,
            )
            rows.append(
                {
                    'key': chunk['key'],
                    'response': {
                        'candidates': [
                            {
                                'content': {'parts': [{'text': response_text}]},
                                'finishReason': 'STOP',
                            }
                        ],
                        'usageMetadata': {
                            'promptTokenCount': 120,
                            'candidatesTokenCount': 50,
                            'totalTokenCount': 170,
                        },
                    },
                }
            )
        result_path.write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
            encoding='utf-8',
        )
        manifest['result_jsonl_path'] = 'results.jsonl'
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def test_golden_keyword_build_export_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root)
            old_values = self._patch_batch_environment(root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_keyword_package(
                        display_name_override='golden-keyword-minimal',
                        skip_prepare=True,
                        chunk_size=2,
                        max_candidates_per_chunk=4,
                    )
                )
                self._write_mock_keyword_results(manifest_path)
                manifest = self._load_manifest(manifest_path)

                self._assert_or_update_json(
                    'expected/manifest_snapshot.json',
                    self._manifest_snapshot(manifest),
                )
                self._assert_or_update_json(
                    'expected/request_snapshot.json',
                    self._request_snapshot(manifest),
                )

                export = batch_mod.export_keyword_candidates(str(manifest_path))
                export_snapshot = {
                    'summary': export['summary'],
                    'candidate_rows': self._read_jsonl(export['jsonl_path']),
                    'summary_rows': self._read_jsonl(export['summary_jsonl_path']),
                }
                self._assert_or_update_json(
                    'expected/export_snapshot.json',
                    export_snapshot,
                )
                self._assert_or_update_text(
                    'expected/keyword_candidates.md',
                    Path(export['markdown_path']).read_text(encoding='utf-8'),
                )
                self._assert_or_update_text(
                    'expected/keyword_chunk_summaries.md',
                    Path(export['summary_markdown_path']).read_text(encoding='utf-8'),
                )

                self.assertEqual(export['summary']['candidate_count_raw'], 5)
                self.assertEqual(export['summary']['candidate_count_deduped'], 3)
                self.assertEqual(export['summary']['chunk_summary_count'], 2)
            finally:
                self._restore_batch_environment(old_values)



if __name__ == '__main__':
    unittest.main()
