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
from types import SimpleNamespace
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
            # Simulate a provider obeying the current protected TARGET view.
            from engine_adapters import structure_rules
            for result_item in result_items:
                protection = (chunk.get('transport_metadata') or {}).get('structure_protection')
                if protection is None:
                    continue
                mapping = protection['items'][result_item['id']]
                available = list(mapping['entries'])
                text = result_item['translation']
                parts, end = [], 0
                for start, stop, kind in structure_rules.spans(text, protection['engine']):
                    entry = next(entry for entry in available if entry['value'] == text[start:stop])
                    available.remove(entry)
                    parts.extend((text[end:start], entry['marker']))
                    end = stop
                parts.append(text[end:])
                result_item['translation'] = ''.join(parts)
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

    def test_golden_batch_export_only_does_not_apply_or_advance_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            game_root = workspace / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-export-only',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                checked_manifest = batch_mod.check_results(str(manifest_path))
                self.assertEqual(
                    checked_manifest['last_check_summary']['writeback_gate']['decision'],
                    'allow',
                )

                game_before = {
                    path.relative_to(game_root).as_posix(): path.read_bytes()
                    for path in tl_dir.rglob('*')
                    if path.is_file()
                }
                progress_path = Path(batch_mod.PROGRESS_LOG)
                progress_before = progress_path.read_bytes() if progress_path.exists() else None
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_before = latest_path.read_bytes() if latest_path.exists() else None
                export_root = workspace / 'exports'

                with mock.patch.object(batch_mod, 'update_progress') as update_progress:
                    exported_manifest = batch_mod.apply_results(
                        str(manifest_path),
                        export_only=str(export_root),
                    )

                self.assertNotIn('applied_at', exported_manifest)
                self.assertNotIn('apply_summary', exported_manifest)
                self.assertEqual(exported_manifest['export_summary']['status'], 'exported')
                self.assertEqual(exported_manifest['export_summary']['exported_files'], 2)
                update_progress.assert_not_called()
                self.assertEqual(
                    {
                        path.relative_to(game_root).as_posix(): path.read_bytes()
                        for path in tl_dir.rglob('*')
                        if path.is_file()
                    },
                    game_before,
                )
                self.assertEqual(
                    progress_path.read_bytes() if progress_path.exists() else None,
                    progress_before,
                )
                self.assertEqual(
                    latest_path.read_bytes() if latest_path.exists() else None,
                    latest_before,
                )
                self.assertEqual(
                    (export_root / 'game' / 'tl' / 'schinese' / 'chapter01' / 'dialogue.rpy').read_bytes(),
                    (GOLDEN_BATCH_FIXTURE_DIR / 'expected' / 'applied' / 'chapter01' / 'dialogue.rpy').read_bytes(),
                )
                self.assertEqual(
                    (export_root / 'game' / 'tl' / 'schinese' / 'chapter02' / 'strings.rpy').read_bytes(),
                    (GOLDEN_BATCH_FIXTURE_DIR / 'expected' / 'applied' / 'chapter02' / 'strings.rpy').read_bytes(),
                )

                second_manifest = batch_mod.apply_results(
                    str(manifest_path),
                    export_only=str(export_root),
                )
                self.assertTrue(second_manifest['export_summary']['idempotent'])
                self.assertNotIn('applied_at', second_manifest)

                applied_manifest = batch_mod.apply_results(str(manifest_path))
                self.assertIn('applied_at', applied_manifest)
                self.assertEqual(
                    (tl_dir / 'chapter01' / 'dialogue.rpy').read_bytes(),
                    (GOLDEN_BATCH_FIXTURE_DIR / 'expected' / 'applied' / 'chapter01' / 'dialogue.rpy').read_bytes(),
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_end_to_end_and_idempotent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                checked = batch_mod.check_results(str(manifest_path))
                self.assertEqual(
                    checked['last_check_summary']['writeback_gate']['decision'],
                    'allow',
                )
                export_root = root / 'exports'
                applied = batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )

                self.assertIn('applied_at', applied)
                self.assertEqual(applied['export_summary']['mode'], 'apply-export')
                self.assertEqual(
                    applied['export_summary']['status'],
                    'applied_and_exported',
                )
                self.assertEqual(applied['export_summary']['exported_files'], 2)
                self.assertEqual(applied['apply_summary']['applied_files'], 2)
                self.assertEqual(applied['apply_summary']['applied_lines'], 6)
                self.assertEqual(
                    applied['apply_state_advancement']['status'],
                    'complete',
                )
                for relative in (
                    'chapter01/dialogue.rpy',
                    'chapter02/strings.rpy',
                ):
                    expected = (
                        GOLDEN_BATCH_FIXTURE_DIR
                        / 'expected'
                        / 'applied'
                        / relative
                    ).read_bytes()
                    self.assertEqual((tl_dir / relative).read_bytes(), expected)
                    self.assertEqual(
                        (
                            export_root
                            / 'game'
                            / 'tl'
                            / 'schinese'
                            / relative
                        ).read_bytes(),
                        expected,
                    )

                progress = json.loads(
                    Path(batch_mod.PROGRESS_LOG).read_text(encoding='utf-8')
                )
                self.assertTrue(progress)
                self.assertTrue(Path(batch_mod.LATEST_MANIFEST_FILE).is_file())

                second = batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )
                self.assertEqual(
                    second['export_summary']['status'],
                    'applied_and_exported',
                )
                self.assertEqual(
                    second['export_summary']['state_advancement_status'],
                    'complete',
                )

                with self.assertRaisesRegex(SystemExit, 'already applied'):
                    batch_mod.apply_results(str(manifest_path))
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_state_recovery_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-recovery',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                export_root = root / 'exports'
                with mock.patch.object(
                    batch_mod,
                    '_finish_apply_export_state',
                    side_effect=RuntimeError('simulated state save failure'),
                ):
                    with self.assertRaisesRegex(RuntimeError, 'simulated state save'):
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )

                record_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'pending',
                )
                progress_path = Path(batch_mod.PROGRESS_LOG)
                progress_before = (
                    progress_path.read_bytes() if progress_path.exists() else None
                )

                resumed = batch_mod.apply_results(str(manifest_path))
                self.assertIn('applied_at', resumed)
                self.assertEqual(
                    resumed['export_summary']['state_advancement_status'],
                    'complete',
                )
                progress_after_once = Path(batch_mod.PROGRESS_LOG).read_bytes()
                self.assertNotEqual(progress_before, progress_after_once)

                deadline = batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )
                self.assertEqual(
                    deadline['export_summary']['state_advancement_status'],
                    'complete',
                )
                self.assertEqual(
                    Path(batch_mod.PROGRESS_LOG).read_bytes(),
                    progress_after_once,
                )
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'complete',
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_force_does_not_bypass_stale_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-stale',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                source = tl_dir / 'chapter01' / 'dialogue.rpy'
                source.write_bytes(source.read_bytes() + b'# external drift\n')
                export_root = root / 'exports'
                with self.assertRaises(SystemExit):
                    batch_mod.apply_results(
                        str(manifest_path),
                        force=True,
                        export_dir=str(export_root),
                    )
                self.assertFalse(export_root.exists())
                self.assertTrue(source.read_bytes().endswith(b'# external drift\n'))
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_rejects_non_empty_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-conflict',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                export_root = root / 'exports'
                export_root.mkdir()
                (export_root / 'unrelated.txt').write_bytes(b'keep')
                before = {
                    path.relative_to(tl_dir).as_posix(): path.read_bytes()
                    for path in tl_dir.rglob('*')
                    if path.is_file()
                }
                with self.assertRaises(
                    batch_mod.cli_contract.MachineContractError
                ) as raised:
                    batch_mod.apply_results(
                        str(manifest_path),
                        force=True,
                        export_dir=str(export_root),
                    )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_DESTINATION_CONFLICT',
                )
                self.assertEqual(
                    (export_root / 'unrelated.txt').read_bytes(),
                    b'keep',
                )
                self.assertEqual(
                    {
                        path.relative_to(tl_dir).as_posix(): path.read_bytes()
                        for path in tl_dir.rglob('*')
                        if path.is_file()
                    },
                    before,
                )
            finally:
                self._restore_batch_environment(old_values)


    def test_golden_batch_apply_export_progress_failure_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-progress',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                export_root = root / 'exports'
                real_update_progress = batch_mod.update_progress
                calls = {'count': 0}

                def fail_first_progress(*args, **kwargs):
                    calls['count'] += 1
                    if calls['count'] == 1:
                        raise OSError('progress disk full')
                    return real_update_progress(*args, **kwargs)

                with mock.patch.object(
                    batch_mod,
                    'update_progress',
                    side_effect=fail_first_progress,
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertNotIn('applied_at', raw_manifest)
                record_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'pending',
                )

                resumed = batch_mod.apply_results(str(manifest_path))
                self.assertIn('applied_at', resumed)
                self.assertEqual(
                    resumed['apply_state_advancement']['status'],
                    'complete',
                )
                progress = json.loads(
                    Path(batch_mod.PROGRESS_LOG).read_text(encoding='utf-8')
                )
                for line_numbers in progress.values():
                    self.assertEqual(len(line_numbers), len(set(line_numbers)))
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_manifest_save_failure_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-manifest',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                export_root = root / 'exports'

                def fail_first_manifest(*args, **kwargs):
                    raise OSError('manifest fsync failed')

                with mock.patch.object(
                    batch_mod,
                    'save_manifest',
                    side_effect=fail_first_manifest,
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertNotIn('applied_at', raw_manifest)

                resumed = batch_mod.apply_results(str(manifest_path))
                self.assertIn('applied_at', resumed)
                self.assertEqual(
                    resumed['apply_state_advancement']['status'],
                    'complete',
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_batch_apply_export_receipt_complete_failure_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game_root = root / 'project'
            tl_dir = self._copy_fixture_tl(game_root)
            old_values = self._patch_batch_environment(game_root, tl_dir)
            try:
                manifest_path = Path(
                    batch_mod.create_batch_package(
                        display_name_override='golden-batch-apply-export-receipt',
                        skip_prepare=True,
                    )
                )
                self._write_mock_results(manifest_path)
                batch_mod.check_results(str(manifest_path))
                export_root = root / 'exports'

                def fail_first_mark(*args, **kwargs):
                    raise OSError('receipt complete marker failed')

                with mock.patch.object(
                    batch_mod,
                    '_mark_apply_export_state_complete',
                    side_effect=fail_first_mark,
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertIn('applied_at', raw_manifest)
                progress_path = Path(batch_mod.PROGRESS_LOG)
                progress_after_failure = (
                    progress_path.read_bytes() if progress_path.exists() else None
                )

                resumed = batch_mod.apply_results(str(manifest_path))
                self.assertEqual(
                    resumed['apply_state_advancement']['status'],
                    'complete',
                )
                record_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'complete',
                )
                self.assertEqual(
                    progress_path.read_bytes(),
                    progress_after_failure,
                )
            finally:
                self._restore_batch_environment(old_values)


    def _prepare_apply_export_case(self, temporary, name):
        root = Path(temporary)
        game_root = root / 'project'
        tl_dir = self._copy_fixture_tl(game_root)
        old_values = self._patch_batch_environment(game_root, tl_dir)
        manifest_path = Path(
            batch_mod.create_batch_package(
                display_name_override=name,
                skip_prepare=True,
            )
        )
        self._write_mock_results(manifest_path)
        batch_mod.check_results(str(manifest_path))
        return root, manifest_path, tl_dir, old_values

    def test_golden_export_only_rejects_pending_apply_export_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-export-only-p2-journal'
            )
            try:
                package_dir = Path(manifest_path).parent
                journal = package_dir / '.apply_export_transaction.json'
                journal.write_text('{}', encoding='utf-8')
                source = tl_dir / 'chapter01' / 'dialogue.rpy'
                source.write_bytes(b'transaction-new\n')
                progress_path = Path(batch_mod.PROGRESS_LOG)
                progress_before = (
                    progress_path.read_bytes() if progress_path.exists() else None
                )
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_before = (
                    latest_path.read_bytes() if latest_path.exists() else None
                )
                target = root / 'export-only-target'
                with self.assertRaises(
                    batch_mod.cli_contract.MachineContractError
                ) as raised:
                    batch_mod.apply_results(
                        str(manifest_path),
                        export_only=str(target),
                    )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_RECOVERY_REQUIRED',
                )
                self.assertEqual(
                    raised.exception.details.get('recovery_state'),
                    'recovery_required',
                )
                self.assertEqual(source.read_bytes(), b'transaction-new\n')
                self.assertFalse(target.exists())
                self.assertEqual(
                    progress_path.read_bytes() if progress_path.exists() else None,
                    progress_before,
                )
                self.assertEqual(
                    latest_path.read_bytes() if latest_path.exists() else None,
                    latest_before,
                )
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertNotIn('applied_at', raw_manifest)
                self.assertTrue(journal.exists())
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_export_only_rejects_pending_apply_export_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-export-only-p2-pending'
            )
            try:
                export_root = root / 'exports'
                with mock.patch.object(
                    batch_mod,
                    '_finish_apply_export_state',
                    side_effect=RuntimeError('simulated state failure'),
                ):
                    with self.assertRaisesRegex(RuntimeError, 'simulated state'):
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                source = tl_dir / 'chapter01' / 'dialogue.rpy'
                committed_bytes = source.read_bytes()
                progress_path = Path(batch_mod.PROGRESS_LOG)
                progress_before = (
                    progress_path.read_bytes() if progress_path.exists() else None
                )
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_before = (
                    latest_path.read_bytes() if latest_path.exists() else None
                )
                target = root / 'export-only-after-pending'
                with self.assertRaises(
                    batch_mod.cli_contract.MachineContractError
                ) as raised:
                    batch_mod.apply_results(
                        str(manifest_path),
                        export_only=str(target),
                    )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_RECOVERY_REQUIRED',
                )
                self.assertEqual(
                    raised.exception.details.get('recovery_state'),
                    'state_pending',
                )
                self.assertEqual(source.read_bytes(), committed_bytes)
                self.assertFalse(target.exists())
                self.assertEqual(
                    progress_path.read_bytes() if progress_path.exists() else None,
                    progress_before,
                )
                self.assertEqual(
                    latest_path.read_bytes() if latest_path.exists() else None,
                    latest_before,
                )
                record_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                self.assertEqual(
                    json.loads(record_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'pending',
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_apply_export_complete_replay_revalidates_outputs(self):
        scenarios = ('missing_export', 'modified_workspace', 'extra_export_entry')
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root, manifest_path, tl_dir, old_values = (
                    self._prepare_apply_export_case(tmp, f'golden-replay-{scenario}')
                )
                try:
                    export_root = root / 'exports'
                    batch_mod.apply_results(
                        str(manifest_path),
                        export_dir=str(export_root),
                    )
                    exported = (
                        export_root
                        / 'game'
                        / 'tl'
                        / 'schinese'
                        / 'chapter01'
                        / 'dialogue.rpy'
                    )
                    workspace = tl_dir / 'chapter01' / 'dialogue.rpy'
                    if scenario == 'missing_export':
                        exported.unlink()
                    elif scenario == 'modified_workspace':
                        workspace.write_bytes(workspace.read_bytes() + b'# drift\n')
                    else:
                        (export_root / 'rogue').mkdir()
                        (export_root / 'rogue' / 'extra.txt').write_bytes(b'extra')

                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                    self.assertEqual(
                        raised.exception.code_name,
                        'APPLY_EXPORT_OUTPUT_CHANGED',
                    )
                    self.assertEqual(
                        raised.exception.details.get('recovery_state'),
                        'state_conflict',
                    )
                    self.assertIn(
                        'changed',
                        raised.exception.details,
                    )

                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as forced:
                        batch_mod.apply_results(str(manifest_path), force=True)
                    self.assertEqual(
                        forced.exception.code_name,
                        'APPLY_EXPORT_OUTPUT_CHANGED',
                    )
                finally:
                    self._restore_batch_environment(old_values)

    def test_golden_apply_export_rag_error_stays_pending_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-rag-pending'
            )
            try:
                export_root = root / 'exports'
                rag_calls = []

                def fake_rag(jobs, quality_state='batch_applied'):
                    rag_calls.append(list(jobs))
                    if len(rag_calls) == 1:
                        return {'enabled': True, 'error': 'store unavailable'}
                    return {'enabled': True, 'upserted': 0, 'pending': 0}

                with (
                    mock.patch.object(batch_mod, 'RAG_ENABLED', True),
                    mock.patch.object(
                        batch_mod,
                        'sync_rag_store_for_jobs',
                        side_effect=fake_rag,
                    ),
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                    self.assertEqual(
                        raised.exception.code_name,
                        'APPLY_EXPORT_STATE_PENDING',
                    )
                    self.assertEqual(
                        raised.exception.details.get('pending_steps'),
                        ['rag'],
                    )
                    self.assertTrue(
                        raised.exception.details.get('files_committed')
                    )
                    self.assertEqual(len(rag_calls), 1)

                    raw_manifest = json.loads(
                        Path(manifest_path).read_text(encoding='utf-8')
                    )
                    advancement = raw_manifest.get('apply_state_advancement') or {}
                    self.assertEqual(advancement.get('status'), 'pending')
                    self.assertEqual(advancement.get('pending_steps'), ['rag'])
                    self.assertNotIn('applied_at', raw_manifest)
                    record_path = (
                        Path(manifest_path).parent
                        / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                    )
                    self.assertEqual(
                        json.loads(record_path.read_text(encoding='utf-8'))[
                            'exports'
                        ][0]['state_advancement']['status'],
                        'pending',
                    )

                    resumed = batch_mod.apply_results(str(manifest_path))
                    self.assertEqual(len(rag_calls), 2)
                    self.assertIn('applied_at', resumed)
                    self.assertEqual(
                        resumed['apply_state_advancement']['status'],
                        'complete',
                    )
                    self.assertEqual(
                        json.loads(record_path.read_text(encoding='utf-8'))[
                            'exports'
                        ][0]['state_advancement']['status'],
                        'complete',
                    )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_apply_export_latest_cursor_replay_is_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-latest-normal'
            )
            try:
                export_root = root / 'exports'
                previous_manifest = (
                    Path(manifest_path).parent / 'previous' / 'manifest.json'
                )
                previous_manifest.parent.mkdir(parents=True, exist_ok=True)
                previous_manifest.write_text('{}', encoding='utf-8')
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_path.write_text(str(previous_manifest), encoding='utf-8')

                first = batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )
                latest_after_first = latest_path.read_text(encoding='utf-8').strip()
                self.assertEqual(
                    first['apply_state_advancement']['latest_cursor']['status'],
                    'advanced',
                )
                self.assertEqual(
                    os.path.normcase(os.path.realpath(latest_after_first)),
                    os.path.normcase(os.path.realpath(str(manifest_path))),
                )

                second = batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8').strip(),
                    latest_after_first,
                )
                self.assertIn(
                    second['apply_state_advancement']['latest_cursor']['status'],
                    {'advanced', 'already_advanced'},
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_apply_export_latest_cursor_retains_newer_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-latest-newer'
            )
            try:
                export_root = root / 'exports'
                with mock.patch.object(
                    batch_mod,
                    '_mark_apply_export_state_complete',
                    side_effect=OSError('receipt marker failed'),
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as raised:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                receipt_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                self.assertEqual(
                    json.loads(receipt_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'pending',
                )

                third_manifest = Path(manifest_path).parent / 'part_third' / 'manifest.json'
                third_manifest.parent.mkdir(parents=True, exist_ok=True)
                third_manifest.write_text('{}', encoding='utf-8')
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_path.write_text(str(third_manifest), encoding='utf-8')

                resumed = batch_mod.apply_results(str(manifest_path))
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8').strip(),
                    str(third_manifest),
                )
                latest_cursor = resumed['apply_state_advancement']['latest_cursor']
                self.assertEqual(latest_cursor['status'], 'retained_newer')
                self.assertEqual(
                    resumed['apply_summary']['latest_cursor']['status'],
                    'retained_newer',
                )
                self.assertIn('newer', latest_cursor['message'])
                self.assertEqual(
                    json.loads(receipt_path.read_text(encoding='utf-8'))['exports'][0][
                        'state_advancement'
                    ]['status'],
                    'complete',
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_apply_export_rag_pending_authoritative_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-rag-authority'
            )
            try:
                export_root = root / 'exports'
                rag_calls = []

                def fake_rag(jobs, quality_state='batch_applied'):
                    rag_calls.append(list(jobs))
                    if len(rag_calls) == 1:
                        return {'enabled': True, 'error': 'store unavailable'}
                    return {'enabled': True, 'upserted': 0, 'pending': 0}

                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_before = (
                    latest_path.read_text(encoding='utf-8') if latest_path.exists() else None
                )

                with (
                    mock.patch.object(batch_mod, 'RAG_ENABLED', True),
                    mock.patch.object(
                        batch_mod,
                        'sync_rag_store_for_jobs',
                        side_effect=fake_rag,
                    ),
                ):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as first_failure:
                        batch_mod.apply_results(
                            str(manifest_path),
                            export_dir=str(export_root),
                        )
                self.assertEqual(
                    first_failure.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                self.assertEqual(len(rag_calls), 1)
                record_path = (
                    Path(manifest_path).parent
                    / batch_mod.batch_export.APPLY_EXPORT_RECORD_FILE
                )
                record = json.loads(record_path.read_text(encoding='utf-8'))
                pending_state = record['exports'][0]['state_advancement']
                self.assertEqual(pending_state['pending_steps'], ['rag'])
                self.assertTrue(pending_state.get('pending_reason'))
                self.assertTrue(pending_state.get('last_error'))
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertNotIn('applied_at', raw_manifest)
                self.assertEqual(
                    raw_manifest['apply_state_advancement']['status'],
                    'pending',
                )
                self.assertEqual(
                    raw_manifest['apply_state_advancement']['pending_steps'],
                    ['rag'],
                )
                self.assertTrue(
                    raw_manifest['apply_state_advancement'].get('last_error')
                )
                self.assertEqual(
                    raw_manifest['apply_state_advancement'].get('rag_status'),
                    'error',
                )

                # RAG disabled must not erase a persisted pending RAG step.
                with mock.patch.object(batch_mod, 'RAG_ENABLED', False):
                    with self.assertRaises(
                        batch_mod.cli_contract.MachineContractError
                    ) as blocked:
                        batch_mod.apply_results(str(manifest_path))
                self.assertEqual(
                    blocked.exception.code_name,
                    'APPLY_EXPORT_STATE_PENDING',
                )
                self.assertEqual(
                    blocked.exception.details.get('rag_status'),
                    'blocked_disabled',
                )
                self.assertEqual(
                    blocked.exception.details.get('pending_steps'),
                    ['rag'],
                )
                self.assertEqual(len(rag_calls), 1)
                raw_manifest = json.loads(
                    Path(manifest_path).read_text(encoding='utf-8')
                )
                self.assertNotIn('applied_at', raw_manifest)
                self.assertEqual(
                    raw_manifest['apply_state_advancement']['pending_steps'],
                    ['rag'],
                )
                self.assertEqual(
                    raw_manifest['apply_state_advancement']['rag_status'],
                    'blocked_disabled',
                )
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8')
                    if latest_path.exists()
                    else None,
                    latest_before,
                )
                record = json.loads(record_path.read_text(encoding='utf-8'))
                pending_state = record['exports'][0]['state_advancement']
                self.assertEqual(pending_state['status'], 'pending')
                self.assertEqual(pending_state['pending_steps'], ['rag'])
                self.assertNotEqual(pending_state.get('rag_status'), 'complete')
                self.assertNotEqual(pending_state.get('rag_status'), 'waived')
                self.assertNotEqual(
                    raw_manifest['apply_state_advancement']['status'],
                    'complete',
                )

                # Re-enabling RAG retries the persisted step before complete.
                with (
                    mock.patch.object(batch_mod, 'RAG_ENABLED', True),
                    mock.patch.object(
                        batch_mod,
                        'sync_rag_store_for_jobs',
                        side_effect=fake_rag,
                    ),
                ):
                    resumed = batch_mod.apply_results(str(manifest_path))
                self.assertEqual(len(rag_calls), 2)
                self.assertIn('applied_at', resumed)
                self.assertEqual(
                    resumed['apply_state_advancement']['status'],
                    'complete',
                )
                self.assertEqual(
                    resumed['apply_state_advancement']['rag_status'],
                    'complete',
                )
                record = json.loads(record_path.read_text(encoding='utf-8'))
                completed_state = record['exports'][0]['state_advancement']
                self.assertEqual(completed_state['status'], 'complete')
                self.assertEqual(completed_state['pending_steps'], [])
                self.assertEqual(completed_state['rag_status'], 'complete')
                self.assertTrue(latest_path.exists())
            finally:
                self._restore_batch_environment(old_values)

    def test_golden_apply_export_rejects_workspace_parent_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, manifest_path, tl_dir, old_values = self._prepare_apply_export_case(
                tmp, 'golden-workspace-link'
            )
            link_created = False
            moved_parent = None
            parent = None
            try:
                export_root = root / 'exports'
                batch_mod.apply_results(
                    str(manifest_path),
                    export_dir=str(export_root),
                )
                parent = tl_dir / 'chapter01'
                moved_parent = tl_dir / 'moved_chapter01'
                os.rename(parent, moved_parent)
                try:
                    os.symlink(moved_parent, parent, target_is_directory=True)
                    link_created = True
                except (OSError, NotImplementedError):
                    if os.name == 'nt':
                        completed = subprocess.run(
                            [
                                'cmd',
                                '/c',
                                'mklink',
                                '/J',
                                str(parent),
                                str(moved_parent),
                            ],
                            capture_output=True,
                            text=True,
                        )
                        link_created = completed.returncode == 0 and os.path.isdir(parent)
                if not link_created:
                    os.rename(moved_parent, parent)
                    moved_parent = None
                    self.skipTest('directory symlink/junction creation unavailable')

                workspace = parent / 'dialogue.rpy'
                self.assertEqual(
                    workspace.read_bytes(),
                    (
                        GOLDEN_BATCH_FIXTURE_DIR
                        / 'expected'
                        / 'applied'
                        / 'chapter01'
                        / 'dialogue.rpy'
                    ).read_bytes(),
                )
                with self.assertRaises(
                    batch_mod.cli_contract.MachineContractError
                ) as raised:
                    batch_mod.apply_results(
                        str(manifest_path),
                        export_dir=str(export_root),
                    )
                self.assertEqual(
                    raised.exception.code_name,
                    'APPLY_EXPORT_OUTPUT_CHANGED',
                )
                self.assertEqual(
                    raised.exception.details.get('changed'),
                    'apply_export.workspace_conflict',
                )

                with self.assertRaises(
                    batch_mod.cli_contract.MachineContractError
                ) as forced:
                    batch_mod.apply_results(str(manifest_path), force=True)
                self.assertEqual(
                    forced.exception.code_name,
                    'APPLY_EXPORT_OUTPUT_CHANGED',
                )
            finally:
                if link_created and parent is not None:
                    try:
                        if os.path.isdir(parent) and not os.path.islink(parent):
                            os.rmdir(parent)
                        else:
                            os.unlink(parent)
                    except OSError:
                        pass
                if moved_parent is not None and moved_parent.exists() and parent is not None:
                    try:
                        os.rename(moved_parent, parent)
                    except OSError:
                        pass
                self._restore_batch_environment(old_values)

    def test_remember_latest_manifest_if_unchanged_retains_other_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tl_dir = self._copy_fixture_tl(Path(tmp) / 'project')
            old_values = self._patch_batch_environment(Path(tmp) / 'project', tl_dir)
            try:
                batch_mod.ensure_batch_dirs()
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)
                latest_path.write_text('writer-B', encoding='utf-8')
                retained = batch_mod.remember_latest_manifest_if_unchanged(
                    'writer-A',
                    'writer-C',
                )
                self.assertEqual(retained['status'], 'retained_newer')
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8'),
                    'writer-B',
                )

                advanced = batch_mod.remember_latest_manifest_if_unchanged(
                    'writer-B',
                    'writer-C',
                )
                self.assertEqual(advanced['status'], 'advanced')
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8'),
                    'writer-C',
                )

                already = batch_mod.remember_latest_manifest_if_unchanged(
                    'writer-A',
                    'writer-C',
                )
                self.assertEqual(already['status'], 'already_advanced')
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8'),
                    'writer-C',
                )
            finally:
                self._restore_batch_environment(old_values)

    def test_latest_manifest_writers_share_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = self._copy_fixture_tl(root / 'project')
            old_values = self._patch_batch_environment(root / 'project', tl_dir)
            try:
                lock_path = batch_mod._latest_manifest_lock_path()
                latest_path = Path(batch_mod.LATEST_MANIFEST_FILE)

                def start_locked_writer(marker_name, value, hold):
                    marker = root / marker_name
                    script = (
                        "import sys, time\n"
                        "from pathlib import Path\n"
                        "import atomic_io\n"
                        "lock, latest, marker, value = sys.argv[1:5]\n"
                        "with atomic_io.exclusive_file_lock(lock, timeout=5.0):\n"
                        "    Path(marker).write_text('held')\n"
                        "    time.sleep(float(sys.argv[5]))\n"
                        "    atomic_io.atomic_write_text(latest, value)\n"
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            '-c',
                            script,
                            str(lock_path),
                            str(latest_path),
                            str(marker),
                            value,
                            str(hold),
                        ],
                        cwd=str(Path(__file__).resolve().parents[1]),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    deadline = time.time() + 5.0
                    while time.time() < deadline:
                        if marker.exists():
                            return process, marker
                        if process.poll() is not None:
                            break
                        time.sleep(0.02)
                    process.kill()
                    _stdout, stderr = process.communicate()
                    self.fail(f'locked writer did not start: {stderr}')

                # Conditional helper must wait for the writer lock and then
                # preserve the newer value instead of overwriting it.
                process, _marker = start_locked_writer(
                    'marker-cas',
                    'writer-B',
                    0.6,
                )
                retained = batch_mod.remember_latest_manifest_if_unchanged(
                    'writer-A',
                    'writer-C',
                )
                process.communicate(timeout=5.0)
                self.assertEqual(retained['status'], 'retained_newer')
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8'),
                    'writer-B',
                )

                # Unconditional writer must also use the same lock: it waits
                # for the holder and therefore wins the final value.
                process, _marker = start_locked_writer(
                    'marker-unconditional',
                    'writer-B2',
                    0.6,
                )
                batch_mod.remember_latest_manifest('writer-D')
                process.communicate(timeout=5.0)
                self.assertEqual(
                    latest_path.read_text(encoding='utf-8'),
                    'writer-D',
                )
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
                machine_preview = batch_mod.build_machine_success_envelope(
                    'preview-revisions',
                    preview_manifest,
                    SimpleNamespace(target=str(manifest_path)),
                )
                self.assertTrue(machine_preview['ok'])
                self.assertEqual(
                    machine_preview['status'],
                    preview_manifest['last_revision_preview']['check_status'],
                )
                self.assertTrue(
                    machine_preview['artifacts']['revision_preview_jsonl'].endswith(
                        '.jsonl'
                    )
                )
                preview_rows = self._read_preview_rows(preview_manifest)
                preview_markdown = Path(preview_manifest['last_revision_preview']['markdown_path']).read_text(
                    encoding='utf-8'
                )
                applied_manifest = batch_mod.apply_revisions(str(manifest_path))
                progress = json.loads(Path(batch_mod.PROGRESS_LOG).read_text(encoding='utf-8'))

                self.assertEqual(applied_manifest['revision_apply_state'], 'applied')
                self.assertIn('revision_applied_at', applied_manifest)
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
                self.assertEqual(applied_manifest['revision_apply_state'], 'partial')
                self.assertIn('revision_applied_at', applied_manifest)
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
                source_path = tl_dir / 'chapter01' / 'keywords.rpy'
                source_before_export = source_path.read_text(encoding='utf-8')

                self._assert_or_update_json(
                    'expected/manifest_snapshot.json',
                    self._manifest_snapshot(manifest),
                )
                self._assert_or_update_json(
                    'expected/request_snapshot.json',
                    self._request_snapshot(manifest),
                )

                export = batch_mod.export_keyword_candidates(str(manifest_path))
                machine_export = batch_mod.build_machine_success_envelope(
                    'export-keywords',
                    dict(export),
                    SimpleNamespace(target=str(manifest_path)),
                )
                self.assertTrue(machine_export['ok'])
                self.assertEqual(machine_export['status'], 'completed')
                self.assertEqual(
                    machine_export['artifacts']['keyword_candidates'],
                    export['jsonl_path'],
                )
                self.assertEqual(source_path.read_text(encoding='utf-8'), source_before_export)
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
