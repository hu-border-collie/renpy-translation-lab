import copy
import json
import contextlib
import io
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import gemini_translate_batch as batch
import external_translation_work as work
import sync_translation_preview
import translation_quality
import revision_corpus
import atomic_io
from cli_contract import MachineContractError, strict_exit_code, success_envelope


FIXTURE = Path(__file__).parent / 'fixtures' / 'external_work' / 'chapter.rpy'
TRANSLATIONS = [
    '灯笼还是温的。', '这灯笼还热着呢。', '[name]，别跟着那道蓝光走。',
    '哦，{i}真棒{/i}。又是一扇锁着的门。', '等待信号', '回到灯笼旁',
]


class ExternalWorkTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.tl = self.root / 'game' / 'tl' / 'schinese'
        self.tl.mkdir(parents=True)
        self.rpy = self.tl / 'chapter.rpy'
        self.rpy.write_bytes(FIXTURE.read_bytes())
        for obj, values in (
            (batch.legacy, {'BASE_DIR': str(self.root), 'TL_DIR': str(self.tl),
                            'TL_SUBDIR': 'schinese', 'PREP_LANGUAGE': 'schinese',
                            'GLOSSARY_FILE': '',
                            'GENERATION_TARGET_LANGUAGE': 'schinese',
                            'INCLUDE_FILES': set(), 'INCLUDE_PREFIXES': set()}),
            (batch, {'BATCH_JOBS_DIR': str(self.root / 'packages'),
                     'LATEST_MANIFEST_FILE': str(self.root / 'latest.txt'),
                     'RAG_ENABLED': False, 'STORY_MEMORY_ENABLED': False,
                     'PROJECT_ANALYSIS_ENABLED': False, 'SOURCE_INDEX_ENABLED': False,
                     'BATCH_QUALITY_POLICY': translation_quality.normalize_policy({})}),
        ):
            for key, value in values.items():
                self.stack.enter_context(mock.patch.object(obj, key, value))
        for name in ('run_sync_request', 'prepare_rag_store', 'embed_source_segments',
                     'bootstrap_rag_store', 'create_client', 'build_batch_request'):
            if hasattr(batch, name):
                self.stack.enter_context(mock.patch.object(batch, name, side_effect=AssertionError('project model called')))
        self.stack.enter_context(mock.patch.object(batch.embedding_runtime, 'embed_texts',
                                                  side_effect=AssertionError('embedding called')))
        self.stack.enter_context(mock.patch.dict('os.environ', {'GLOSSARY_FILE': ''}))
        exported = work.export_work(output_dir=self.root / 'work')
        self.target = exported['manifest_path']
        self.package = json.loads(Path(exported['work_path']).read_text(encoding='utf-8'))

    def submission(self, indexes=None, sid='first', reviewed=False):
        rows = self.package['items']
        indexes = range(len(rows)) if indexes is None else indexes
        document = work.submission_template(self.package, submission_id=sid,
                                            producer={'type': 'agent', 'name': 'fixture test'})
        document['items'] = [
            {'occurrence_id': rows[index]['occurrence_id'], 'snapshot_digest': rows[index]['snapshot_digest'],
             'expected_candidate_digest': '', 'translation': TRANSLATIONS[index], 'reason': '原创夹具初译',
             'review': {'status': 'reviewed', 'reviewer': 'fixture reviewer'} if reviewed else {'status': 'unreviewed'}}
            for index in indexes
        ]
        return document

    def ready(self):
        work.submit_work(self.target, self.submission())
        checked = batch.check_results(self.target)
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
        return work.preview_work(self.target)

    def assert_refused(self, code, fn, *args, **kwargs):
        before = self.rpy.read_bytes()
        with self.assertRaises(MachineContractError) as caught:
            fn(*args, **kwargs)
        self.assertEqual(caught.exception.code_name, code)
        self.assertEqual(self.rpy.read_bytes(), before)

    def work_cli(self, *args):
        output, diagnostics = io.StringIO(), io.StringIO()
        with (mock.patch.object(batch.legacy, 'load_translator_settings'),
              mock.patch.object(batch, 'load_batch_settings'),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics)):
            code = batch.main([*map(str, args), '--output', 'json', '--strict-exit-codes'])
        self.assertNotIn('Traceback', diagnostics.getvalue())
        return code, json.loads(output.getvalue())

    def test_invalid_export_inputs_have_machine_diagnostics(self):
        invalid_utf8 = self.root / 'invalid.txt'
        invalid_utf8.write_bytes(b'\xff')
        for reference in (self.root / 'missing.txt', self.root, invalid_utf8):
            with self.subTest(reference=reference):
                code, result = self.work_cli('work-export', '--reference-file', reference,
                                             '--output-dir', self.root / 'invalid-export')
                self.assertEqual(code, 5)
                self.assertEqual(result['error']['code'], 'WORK_REFERENCE_INVALID')
                self.assertEqual(result['error']['suggested_action'], 'provide_readable_utf8_reference_file')
                self.assertFalse((self.root / 'invalid-export').exists())
        for glossary in (self.root, invalid_utf8):
            with self.subTest(glossary=glossary), mock.patch.object(batch.legacy, 'GLOSSARY_FILE', str(glossary)):
                code, result = self.work_cli('work-export', '--output-dir', self.root / 'invalid-glossary')
                self.assertEqual(code, 5)
                self.assertEqual(result['error']['code'], 'WORK_REFERENCE_INVALID')
                self.assertFalse((self.root / 'invalid-glossary').exists())
        before = Path(self.target).read_bytes()
        code, result = self.work_cli('work-export', '--output-dir', Path(self.target).parent)
        self.assertEqual(code, 5)
        self.assertEqual(result['error']['code'], 'WORK_OUTPUT_INVALID')
        self.assertEqual(result['error']['suggested_action'], 'choose_new_writable_output_directory_outside_tl')
        self.assertEqual(Path(self.target).read_bytes(), before)

    def test_unreadable_reference_and_unwritable_output_are_contract_errors(self):
        reference = self.root / 'private.txt'
        reference.write_text('style', encoding='utf-8')
        original_read = Path.read_bytes

        def deny_reference(path):
            if path.resolve() == reference.resolve():
                raise PermissionError('reference denied')
            return original_read(path)

        with mock.patch.object(Path, 'read_bytes', deny_reference):
            with self.assertRaises(MachineContractError) as caught:
                work.export_work(output_dir=self.root / 'unreadable', reference_files=[reference])
        self.assertEqual(caught.exception.code_name, 'WORK_REFERENCE_INVALID')
        with mock.patch.object(Path, 'mkdir', side_effect=PermissionError('output denied')):
            self.assert_refused('WORK_OUTPUT_INVALID', work.export_work, output_dir=self.root / 'unwritable')

    def test_invalid_candidate_version_does_not_persist_conflicts_or_revoke_check(self):
        for checked in (False, True):
            if checked:
                self.ready()
            manifest_before = Path(self.target).read_bytes()
            for fields in ({}, {'expected_candidate_digest': None},
                           {'expected_candidate_digest': 1}, {'expected_candidate_digest': []}):
                with self.subTest(checked=checked, fields=fields):
                    document = self.submission([0], sid='malformed')
                    document['items'][0].pop('expected_candidate_digest')
                    document['items'][0].update(fields)
                    self.assert_refused('WORK_SUBMISSION_INVALID', work.submit_work, self.target, document)
                    self.assertEqual(Path(self.target).read_bytes(), manifest_before)
        self.assertEqual(work.status_work(self.target)['conflicts'], [])

    def test_missing_preview_before_apply_can_be_rechecked_without_losing_candidates(self):
        preview = self.ready()
        Path(preview['preview_path']).unlink()
        status = work.status_work(self.target)
        self.assertEqual(status['writeback'], 'preview_unavailable')
        self.assertEqual(status['diagnostics'][0]['code'], 'WORK_PREVIEW_CHANGED')
        self.assert_refused('WORK_PREVIEW_CHANGED', work.apply_work, self.target)
        self.assertEqual(batch.check_results(self.target)['last_check_summary']['writeback_gate']['decision'], 'allow')
        self.assertEqual(work.status_work(self.target)['received_count'], 6)
        work.preview_work(self.target)
        self.assertEqual(work.apply_work(self.target)['status'], 'applied')

    def test_damaged_applied_preview_preserves_receipt_and_structured_status(self):
        preview = self.ready()
        receipt = work.apply_work(self.target)['receipt']
        path = Path(preview['preview_path'])
        before = path.read_bytes()
        manifest_before = Path(self.target).read_bytes()
        for contents in (None, b'{', b'{}', b'\xff'):
            with self.subTest(contents=contents):
                if contents is None:
                    path.unlink()
                else:
                    path.write_bytes(contents)
                code, result = self.work_cli('work-status', self.target)
                self.assertEqual(code, 5)
                self.assertTrue(result['ok'])
                self.assertEqual(result['result']['writeback'], 'applied')
                self.assertEqual(result['result']['diagnostics'][0]['code'], 'WORK_PREVIEW_CHANGED')
                code, result = self.work_cli('work-apply', self.target)
                self.assertEqual(code, 5)
                self.assertEqual(result['error']['code'], 'WORK_PREVIEW_CHANGED')
                self.assert_refused('WORK_ALREADY_APPLIED', batch.check_results, self.target)
                self.assertEqual(Path(self.target).read_bytes(), manifest_before)
        path.write_bytes(before)
        with mock.patch.object(sync_translation_preview, 'atomic_write_many_lines', side_effect=AssertionError('rewritten')):
            self.assertEqual(work.apply_work(self.target)['receipt'], receipt)

    def test_missing_preview_after_unrecorded_write_never_discards_recovery_binding(self):
        preview = self.ready()
        with mock.patch.object(work, '_save', side_effect=OSError('lost receipt')):
            with self.assertRaises(OSError):
                work.apply_work(self.target)
        path = Path(preview['preview_path'])
        contents = path.read_bytes()
        path.unlink()
        manifest_before = Path(self.target).read_bytes()
        self.assertEqual(work.status_work(self.target)['writeback'], 'preview_unavailable')
        self.assert_refused('WORK_PREVIEW_CHANGED', batch.check_results, self.target)
        self.assert_refused('WORK_PREVIEW_CHANGED', work.preview_work, self.target)
        self.assert_refused('WORK_PREVIEW_CHANGED', work.submit_work, self.target, self.submission([0], sid='later'))
        self.assertEqual(Path(self.target).read_bytes(), manifest_before)
        path.write_bytes(contents)
        with mock.patch.object(sync_translation_preview, 'atomic_write_many_lines', side_effect=AssertionError('rewritten')):
            self.assertEqual(work.apply_work(self.target)['status'], 'applied')

    def test_missing_preview_with_journal_never_rechecks_even_when_sources_unchanged(self):
        preview = self.ready()
        path = Path(preview['preview_path'])
        path.unlink()
        journal = path.parent / '.sync_writeback_transaction.json'
        journal.write_text('{}', encoding='utf-8')
        manifest_before = Path(self.target).read_bytes()
        self.assertEqual(work.status_work(self.target)['writeback'], 'recovery_required')
        for operation in (batch.check_results, work.preview_work):
            self.assert_refused('WORK_PREVIEW_CHANGED', operation, self.target)
        self.assert_refused('WORK_PREVIEW_CHANGED', work.submit_work, self.target, self.submission([0], sid='later'))
        self.assertEqual(Path(self.target).read_bytes(), manifest_before)
        self.assertEqual(journal.read_text(encoding='utf-8'), '{}')

    def test_after_apply_replays_receipt_but_new_submission_requires_revision_corpus(self):
        self.ready()
        original = work.submit_work(self.target, self.submission())
        work.apply_work(self.target)
        self.assertEqual(work.submit_work(self.target, self.submission()), original)
        self.assert_refused('WORK_ALREADY_APPLIED', work.submit_work, self.target, self.submission([0], sid='later'))

    def test_structure_refusal_uses_blocked_exit_code(self):
        document = self.submission([2])
        document['items'][0]['translation'] = '不要跟过去。'
        path = self.root / 'submission.json'
        path.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
        code, result = self.work_cli('work-submit', self.target, path)
        self.assertEqual(code, 4)
        self.assertEqual(result['error']['code'], 'WORK_STRUCTURE_BLOCKED')

    def test_partial_replay_revision_and_complete_shared_writeback(self):
        self.assertEqual(len(self.package['items']), 6)
        first, second = self.package['items'][:2]
        self.assertEqual(first['source'], second['source'])
        self.assertNotEqual(first['occurrence_id'], second['occurrence_id'])
        submission = self.submission([0])
        receipt = work.submit_work(self.target, submission)
        self.assertEqual(work.submit_work(self.target, copy.deepcopy(submission)), receipt)
        status = work.status_work(self.target)
        self.assertEqual(status['completeness'], 'partial')
        self.assertEqual(len(status['remaining_ids']), 5)
        checked = batch.check_results(self.target)
        self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'deny')
        with self.assertRaises(SystemExit):
            work.preview_work(self.target)
        revision = self.submission([0], sid='revise')
        revision['items'][0].update(expected_candidate_digest=receipt['candidate_digests'][first['occurrence_id']],
                                    translation='灯笼还带着余温。', reason='语感订正')
        work.submit_work(self.target, revision)
        work.submit_work(self.target, self.submission(range(1, 6), sid='rest'))
        self.assertEqual(batch.check_results(self.target)['last_check_summary']['writeback_gate']['decision'], 'allow')
        work.preview_work(self.target)
        result = batch.apply_results(self.target, force=True)
        self.assertEqual(result['status'], 'applied')
        self.assertIn('灯笼还带着余温。', self.rpy.read_text(encoding='utf-8'))
        self.assertEqual(work.apply_work(self.target)['receipt'], result['receipt'])
        manifest = batch.load_manifest(self.target)
        self.assertEqual(manifest['job_state'], 'LOCAL_CANDIDATES')
        self.assertEqual(manifest['external_work']['usage'], 'unknown')
        results = Path(batch.resolve_manifest_result_path(manifest)).read_text(encoding='utf-8')
        self.assertNotIn('"response":', results)
        self.assertFalse(Path(batch.LATEST_MANIFEST_FILE).exists())

    def test_competing_and_duplicate_submissions_are_rejected_atomically(self):
        first = self.submission([0])
        work.submit_work(self.target, first)
        modified = copy.deepcopy(first)
        modified['items'][0]['translation'] = '那盏灯还热着。'
        self.assert_refused('WORK_SUBMISSION_CONFLICT', work.submit_work, self.target, modified)
        modified['submission_id'] = 'competitor'
        modified['items'].append(self.submission([1])['items'][0])
        self.assert_refused('WORK_CANDIDATE_CONFLICT', work.submit_work, self.target, modified)
        self.assertEqual(work.status_work(self.target)['received_count'], 1)

    def test_wrong_identity_snapshot_and_broken_structure(self):
        for field, code, value in (
            ('occurrence_id', 'WORK_UNKNOWN_OCCURRENCE', 'unknown'),
            ('snapshot_digest', 'WORK_ITEM_STALE', 'wrong'),
            ('translation', 'WORK_STRUCTURE_BLOCKED', '不要跟过去。'),
        ):
            document = self.submission([2])
            document['items'][0][field] = value
            self.assert_refused(code, work.submit_work, self.target, document)
        for field in ('project_id', 'package_id', 'package_digest', 'reference_digest'):
            document = self.submission([0])
            document[field] = 'other'
            self.assert_refused('WORK_SUBMISSION_STALE', work.submit_work, self.target, document)

    def test_source_and_current_target_stale_even_with_force(self):
        self.ready()
        self.rpy.write_text(self.rpy.read_text(encoding='utf-8').replace('a ""', 'a "有人改过"', 1), encoding='utf-8')
        self.assertEqual(work.status_work(self.target)['status'], 'stale')
        self.assert_refused('WORK_SOURCE_STALE', batch.apply_results, self.target, force=True)

    def test_generation_target_is_bound_separately_from_catalog_language(self):
        self.ready()
        self.assertEqual(self.package['generation_target'], 'schinese')
        with mock.patch.object(batch.legacy, 'GENERATION_TARGET_LANGUAGE', 'japanese'):
            self.assert_refused('WORK_LANGUAGE_STALE', work.apply_work, self.target)
            with self.assertRaisesRegex(SystemExit, 'generation_target.unsupported'):
                work.export_work(output_dir=self.root / 'unsupported')
        self.assertFalse((self.root / 'unsupported').exists())

    def test_result_and_package_tampering_are_blocked(self):
        self.ready()
        manifest = batch.load_manifest(self.target)
        path = Path(batch.resolve_manifest_result_path(manifest))
        before = path.read_bytes()
        path.write_bytes(before + b'\n')
        self.assert_refused('WORK_RESULTS_CHANGED', batch.apply_results, self.target, force=True)
        path.write_bytes(before)
        package_path = Path(self.target).parent / 'work.json'
        package_path.write_text('{}', encoding='utf-8')
        self.assert_refused('WORK_PACKAGE_CHANGED', batch.apply_results, self.target, force=True)

    def test_result_generation_orphan_replays_without_partial_receipt(self):
        document = self.submission([0])
        with mock.patch.object(work, '_save', side_effect=OSError('interrupted before commit')):
            with self.assertRaises(OSError):
                work.submit_work(self.target, document)
        self.assertEqual(work.status_work(self.target)['received_count'], 0)
        receipt = work.submit_work(self.target, document)
        self.assertEqual(receipt['generation'], 1)
        self.assertEqual(work.submit_work(self.target, document), receipt)

    def test_committed_files_missing_state_recover_without_rewriting(self):
        self.ready()
        with mock.patch.object(work, '_save', side_effect=OSError('state not recorded')):
            with self.assertRaises(OSError):
                work.apply_work(self.target)
        self.assertEqual(work.status_work(self.target)['writeback'], 'recovery_required')
        committed = self.rpy.read_bytes()
        with mock.patch.object(sync_translation_preview, 'atomic_write_many_lines', side_effect=AssertionError('rewritten')):
            self.assertEqual(work.apply_work(self.target)['status'], 'applied')
        self.assertEqual(self.rpy.read_bytes(), committed)
        self.assertEqual(work.status_work(self.target)['writeback'], 'applied')

    def test_recovery_never_overwrites_subsequent_edits(self):
        self.ready()
        with mock.patch.object(work, '_save', side_effect=OSError('state not recorded')):
            with self.assertRaises(OSError):
                work.apply_work(self.target)
        self.rpy.write_bytes(self.rpy.read_bytes() + b'# subsequent edit\n')
        self.assert_refused('WORK_SOURCE_STALE', work.apply_work, self.target)

    def test_changed_reference_and_reviewed_candidate_reference(self):
        work.submit_work(self.target, self.submission([0], reviewed=True))
        exported = work.export_work(output_dir=self.root / 'later', reference_works=[self.target])
        later = exported['manifest_path']
        self.assertEqual(work.status_work(later)['status'], 'current')
        work.submit_work(self.target, self.submission([1], sid='unrelated'))
        self.assertEqual(work.status_work(later)['status'], 'current')
        candidate = work.status_work(self.target, include_items=True)['items'][0]['candidate']
        revision = self.submission([0], sid='new-review', reviewed=True)
        revision['items'][0].update(expected_candidate_digest=candidate['candidate_digest'], translation='这盏灯笼还有余温。')
        work.submit_work(self.target, revision)
        self.assertEqual(work.status_work(later)['status'], 'stale')

    def test_last_failed_check_revokes_old_preview(self):
        self.ready()
        with mock.patch.object(batch, 'check_translation_results', side_effect=OSError('interrupted check')):
            with self.assertRaises(OSError):
                batch.check_results(self.target)
        self.assert_refused('WORK_PREVIEW_REQUIRED', batch.apply_results, self.target, force=True)

    def test_direct_shared_preview_apply_requires_external_coordinator(self):
        preview = self.ready()
        with self.assertRaisesRegex(ValueError, 'work-apply'):
            sync_translation_preview.apply_sync_preview(preview['preview_path'], active_project_root=self.root,
                                                        active_tl_dir=self.tl)

    def test_scope_subset_rejects_other_known_occurrence(self):
        subset = work.export_work(output_dir=self.root / 'subset', occurrence_ids=[self.package['items'][0]['occurrence_id']])
        other = self.submission([1])
        other.update({key: value for key, value in json.loads(Path(subset['work_path']).read_text(encoding='utf-8')).items()
                      if key in ('package_id', 'package_digest', 'project_id', 'reference_digest')})
        self.assert_refused('WORK_UNKNOWN_OCCURRENCE', work.submit_work, subset['manifest_path'], other)

    def test_conflict_revokes_check_until_explicit_candidate_resolution(self):
        self.ready()
        conflicting = self.submission([0], sid='competitor')
        self.assert_refused('WORK_CANDIDATE_CONFLICT', work.submit_work, self.target, conflicting)
        self.assert_refused('WORK_UNRESOLVED_CONFLICT', batch.check_results, self.target)
        self.assert_refused('WORK_PREVIEW_REQUIRED', batch.apply_results, self.target, force=True)
        state = work.status_work(self.target, include_items=True)
        self.assertEqual(state['status'], 'conflict')
        self.assertEqual(strict_exit_code(success_envelope('work-status', status=state['status'], result=state)), 4)
        self.assertEqual(state['conflicts'][0]['status'], 'unresolved')
        resolved = self.submission([0], sid='resolve')
        resolved['items'][0]['expected_candidate_digest'] = state['items'][0]['candidate']['candidate_digest']
        work.submit_work(self.target, resolved)
        self.assertEqual(work.status_work(self.target)['conflicts'][0]['status'], 'resolved')
        self.assertEqual(batch.check_results(self.target)['last_check_summary']['writeback_gate']['decision'], 'allow')

    def test_damaged_external_marker_never_falls_back_to_legacy_check_or_apply(self):
        original = batch.load_manifest(self.target)
        for invalid in (None, [], 'missing'):
            with self.subTest(marker=invalid):
                manifest = copy.deepcopy(original)
                if invalid == 'missing':
                    manifest.pop('external_work')
                else:
                    manifest['external_work'] = invalid
                batch.save_manifest(manifest, update_latest=False)
                self.assert_refused('WORK_MANIFEST_INVALID', batch.check_results, self.target)
                self.assert_refused('WORK_MANIFEST_INVALID', batch.apply_results, self.target, force=True)
                self.assert_refused('WORK_PROVIDER_DISABLED', batch.validate_batch_translation_plan_before_dispatch,
                                    manifest, operation='submit')

    def test_damaged_state_fields_return_contract_errors_without_writes(self):
        original = batch.load_manifest(self.target)
        for field, value in (('preview', 'boom'), ('conflicts', ['oops']),
                             ('receipts', ['oops']), ('applied', 'yes')):
            with self.subTest(field=field):
                manifest = copy.deepcopy(original)
                manifest['external_work'][field] = value
                batch.save_manifest(manifest, update_latest=False)
                if field == 'preview':
                    status = work.status_work(self.target)
                    self.assertEqual(status['writeback'], 'preview_unavailable')
                    self.assertEqual(status['diagnostics'][0]['code'], 'WORK_PREVIEW_CHANGED')
                    self.assert_refused('WORK_PREVIEW_CHANGED', work.apply_work, self.target)
                    # Source unchanged and no writeback evidence: re-check may drop the damaged ref.
                    checked = batch.check_results(self.target)
                    self.assertNotIn('preview', checked['external_work'])
                else:
                    self.assert_refused('WORK_MANIFEST_INVALID', work.status_work, self.target)
                    self.assert_refused('WORK_MANIFEST_INVALID', batch.check_results, self.target)
        batch.save_manifest(original, update_latest=False)

    def test_provider_submit_on_work_manifest_is_refused_before_credentials(self):
        self.assert_refused('WORK_PROVIDER_DISABLED', batch.submit_manifest, self.target)
        code, result = self.work_cli('submit', self.target)
        self.assertEqual(code, 5)
        self.assertEqual(result['error']['code'], 'WORK_PROVIDER_DISABLED')
        self.assertEqual(result['error']['suggested_action'], 'use_work_submit')

    def test_malformed_submission_never_commits_receipt(self):
        for malformed in ([], True, float('nan')):
            document = self.submission([0])
            document['schema_version'] = malformed
            self.assert_refused('WORK_SUBMISSION_INVALID', work.submit_work, self.target, document)
        self.assertEqual(work.status_work(self.target)['generation'], 0)

    def test_work_changed_during_receive_does_not_commit_generation(self):
        path = Path(self.target).parent / 'work.json'
        before = path.read_bytes()
        original = work.atomic_write_jsonl

        def race(*args, **kwargs):
            original(*args, **kwargs)
            changed = copy.deepcopy(self.package)
            changed['items'][0]['source'] = 'changed during reception'
            path.write_text(json.dumps(changed), encoding='utf-8')

        with mock.patch.object(work, 'atomic_write_jsonl', side_effect=race):
            self.assert_refused('WORK_PACKAGE_CHANGED', work.submit_work, self.target, self.submission([0]))
        path.write_bytes(before)
        self.assertEqual(work.status_work(self.target)['generation'], 0)
        self.assertEqual(work.status_work(self.target)['received_count'], 0)

    def test_existing_language_allowance_does_not_bypass_structure(self):
        renamed = self.rpy.with_name('screens_patronlistitem.rpy')
        self.rpy.rename(renamed)
        self.rpy = renamed
        exported = work.export_work(output_dir=self.root / 'preserved-text-work')
        self.target = exported['manifest_path']
        self.package = json.loads(Path(exported['work_path']).read_text(encoding='utf-8'))
        broken = self.submission([2], sid='broken')
        broken['items'][0]['translation'] = "Don't follow the blue light."
        self.assert_refused('WORK_STRUCTURE_BLOCKED', work.submit_work, self.target, broken)
        document = self.submission()
        document['items'][0]['translation'] = self.package['items'][0]['source']
        work.submit_work(self.target, document)
        self.assertEqual(batch.check_results(self.target)['last_check_summary']['writeback_gate']['decision'], 'allow')
        work.preview_work(self.target)
        self.assertEqual(work.apply_work(self.target)['status'], 'applied')

    def test_revision_after_preview_cannot_use_old_preview(self):
        preview = self.ready()
        item = work.status_work(self.target, include_items=True)['items'][0]
        revision = self.submission([0], sid='revision')
        revision['items'][0]['expected_candidate_digest'] = item['candidate']['candidate_digest']
        work.submit_work(self.target, revision)
        self.assert_refused('WORK_PREVIEW_REQUIRED', batch.apply_results, self.target, force=True)
        old = sync_translation_preview.load_sync_preview(preview['preview_path'])
        self.assert_refused('WORK_PREVIEW_STALE', work.validate_preview_binding, old)

    def test_reference_file_changes_block_apply(self):
        reference = self.root / 'style.txt'
        reference.write_text('灯笼', encoding='utf-8')
        exported = work.export_work(output_dir=self.root / 'with-reference', reference_files=[reference])
        self.target = exported['manifest_path']
        self.package = json.loads(Path(exported['work_path']).read_text(encoding='utf-8'))
        self.ready()
        reference.write_text('提灯', encoding='utf-8')
        self.assert_refused('WORK_REFERENCE_STALE', batch.apply_results, self.target, force=True)

    def test_configured_glossary_is_shared_by_check_and_preview_and_versioned(self):
        glossary = self.root / 'glossary.json'
        glossary.write_text(json.dumps({'translations': {'lantern': '灯笼'}}), encoding='utf-8')
        with mock.patch.object(batch.legacy, 'GLOSSARY_FILE', str(glossary)):
            exported = work.export_work(output_dir=self.root / 'with-glossary')
            self.target = exported['manifest_path']
            self.package = json.loads(Path(exported['work_path']).read_text(encoding='utf-8'))
            preview = self.ready()
            checked = batch.load_manifest(self.target)['last_check_summary']
            self.assertEqual(checked['quality_glossary_entries'], 1)
            bound = sync_translation_preview.load_sync_preview(preview['preview_path'])
            self.assertEqual(Path(bound['quality_glossary_file']), glossary.resolve())
            glossary.write_text(json.dumps({'translations': {'lantern': '提灯'}}), encoding='utf-8')
            self.assert_refused('WORK_REFERENCE_STALE', batch.apply_results, self.target, force=True)

    def test_missing_glossary_creation_and_glossary_path_change_make_work_stale(self):
        glossary = self.root / 'later.json'
        with mock.patch.object(batch.legacy, 'GLOSSARY_FILE', str(glossary)):
            exported = work.export_work(output_dir=self.root / 'missing-glossary')
            self.assertEqual(work.status_work(exported['manifest_path'])['status'], 'current')
            glossary.write_text('{}', encoding='utf-8')
            self.assertEqual(work.status_work(exported['manifest_path'])['status'], 'stale')
        self.assertEqual(work.status_work(exported['manifest_path'])['status'], 'stale')

    def test_quality_warning_allows_shared_preview_and_apply(self):
        submission = self.submission()
        submission['items'][0]['translation'] = '灯笼里的iPhone还是温的。'
        work.submit_work(self.target, submission)
        summary = batch.check_results(self.target)['last_check_summary']
        self.assertGreater(summary['quality_gate']['warning_count'], 0)
        self.assertEqual(summary['writeback_gate']['decision'], 'allow')
        work.preview_work(self.target)
        self.assertEqual(work.apply_work(self.target)['status'], 'applied')

    def test_configured_quality_blocker_and_policy_change_cannot_force_apply(self):
        submission = self.submission()
        submission['items'][0]['translation'] = '灯笼里的iPhone还是温的。'
        work.submit_work(self.target, submission)
        batch.check_results(self.target)
        work.preview_work(self.target)
        with mock.patch.object(batch, 'BATCH_QUALITY_POLICY', translation_quality.normalize_policy(
            {'rules': {'cjk_latin_spacing': 'blocker'}}
        )):
            before = self.rpy.read_bytes()
            with self.assertRaises(SystemExit):
                batch.apply_results(self.target, force=True)
            summary = batch.check_results(self.target)['last_check_summary']
            self.assertGreater(summary['quality_gate']['blocker_count'], 0)
            self.assertEqual(summary['writeback_gate']['decision'], 'deny')
            with self.assertRaises(SystemExit):
                work.preview_work(self.target)
            self.assertEqual(before, self.rpy.read_bytes())

    def test_file_set_adapter_and_manifest_changes_fail_closed(self):
        self.ready()
        added = self.tl / 'new.rpy'
        added.write_text('# new file\n', encoding='utf-8')
        self.assert_refused('WORK_SOURCE_STALE', work.apply_work, self.target)
        added.unlink()
        with mock.patch.object(work.RenPyAdapter, 'adapter_version', 'future-version'):
            self.assert_refused('WORK_ADAPTER_STALE', work.apply_work, self.target)
        manifest = batch.load_manifest(self.target)
        manifest['chunks'][0]['items'][0]['source'] = 'edited source'
        batch.save_manifest(manifest, update_latest=False)
        self.assert_refused('WORK_PACKAGE_CHANGED', work.apply_work, self.target)

    def test_interruption_after_receipt_commit_replays_original_receipt(self):
        original = work._save
        document = self.submission([0])

        def interrupt(manifest):
            original(manifest)
            raise OSError('reply lost after commit')

        with mock.patch.object(work, '_save', side_effect=interrupt):
            with self.assertRaises(OSError):
                work.submit_work(self.target, document)
        persisted = work.status_work(self.target)['receipts'][0]
        self.assertEqual(work.submit_work(self.target, document), persisted)
        self.assertEqual(persisted['generation'], 1)

    def test_source_edit_between_validation_and_transaction_is_preserved(self):
        self.ready()
        original = sync_translation_preview.atomic_write_many_lines
        changed = self.rpy.read_bytes() + b'# concurrent edit\n'

        def race(*args, **kwargs):
            self.rpy.write_bytes(changed)
            return original(*args, **kwargs)

        with mock.patch.object(sync_translation_preview, 'atomic_write_many_lines', side_effect=race):
            with self.assertRaises(atomic_io.AtomicWritePreimageConflict):
                work.apply_work(self.target)
        self.assertEqual(self.rpy.read_bytes(), changed)

    def test_results_change_during_transaction_rolls_back_all_writes(self):
        self.ready()
        # Force an equivalent spelling, as Windows CI does with 8.3 temp paths.
        self.rpy = self.rpy.parent / '..' / self.rpy.parent.name / self.rpy.name
        before = self.rpy.read_bytes()
        manifest = batch.load_manifest(self.target)
        results = Path(batch.resolve_manifest_result_path(manifest))
        original = atomic_io.os.replace
        changed = False

        def race(source, target):
            nonlocal changed
            result = original(source, target)
            if Path(target).resolve() == self.rpy.resolve() and not changed:
                changed = True
                results.write_bytes(results.read_bytes() + b'\n')
            return result

        with mock.patch.object(atomic_io.os, 'replace', side_effect=race):
            with self.assertRaises(ValueError):
                work.apply_work(self.target)
        self.assertTrue(changed, 'The fault must occur after replacing the game file.')
        self.assertEqual(self.rpy.read_bytes(), before)

    def test_process_death_leaves_prepared_transaction_and_recovery_uses_it(self):
        self.ready()
        # Keep alias handling under regression coverage on every platform.
        self.rpy = self.rpy.parent / '..' / self.rpy.parent.name / self.rpy.name
        original = atomic_io.os.replace
        changed = False

        class SimulatedDeath(BaseException):
            pass

        def stop_after_replace(source, target):
            nonlocal changed
            result = original(source, target)
            if Path(target).resolve() == self.rpy.resolve() and not changed:
                changed = True
                raise SimulatedDeath()
            return result

        with mock.patch.object(atomic_io.os, 'replace', side_effect=stop_after_replace):
            with self.assertRaises(SimulatedDeath):
                work.apply_work(self.target)
        self.assertTrue(changed, 'The fault must occur after replacing the game file.')
        self.assertEqual(work.status_work(self.target)['writeback'], 'recovery_required')
        self.assertEqual(work.apply_work(self.target)['status'], 'applied')
        self.assertEqual(work.status_work(self.target)['writeback'], 'applied')

    def test_preview_state_record_failure_recovers_postimages(self):
        self.ready()
        original = sync_translation_preview.atomic_write_json

        def fail_state(path, payload, **kwargs):
            if payload.get('schema') == sync_translation_preview.SCHEMA and payload.get('state') == 'applied':
                raise OSError('lost preview state')
            return original(path, payload, **kwargs)

        with mock.patch.object(sync_translation_preview, 'atomic_write_json', side_effect=fail_state):
            with self.assertRaises(OSError):
                work.apply_work(self.target)
        self.assertEqual(work.status_work(self.target)['writeback'], 'recovery_required')
        with mock.patch.object(sync_translation_preview, 'atomic_write_many_lines', side_effect=AssertionError('rewrite')):
            self.assertEqual(work.apply_work(self.target)['status'], 'applied')

    def test_output_file_protection_covers_material_results_and_submission(self):
        args = batch.build_arg_parser().parse_args(['work-submit', self.target, str(self.root / 'submission.json')])
        for protected in (Path(self.target).parent / 'work.json', self.root / 'submission.json', self.rpy):
            self.assertIsNotNone(batch._find_output_file_path_conflict(args, str(protected)))

    def test_output_file_preflight_rejects_malformed_work_without_overwriting_output(self):
        original = json.loads(Path(self.target).read_text(encoding='utf-8'))
        output = self.root / 'status.json'
        output.write_bytes(b'keep existing output')
        for field, value in (('tl_dir', None), ('tl_dir', ''), ('tl_dir', []),
                             ('external_work', None), ('external_work', []),
                             ('package', None), ('file_digests', None), ('references', [None])):
            with self.subTest(field=field, value=value):
                manifest = copy.deepcopy(original)
                if field in ('package', 'file_digests', 'references'):
                    owner = manifest['external_work'] if field == 'package' else manifest['external_work']['package']
                else:
                    owner = manifest
                if field == 'tl_dir' and value is None:
                    owner.pop(field)
                else:
                    owner[field] = value
                Path(self.target).write_text(json.dumps(manifest), encoding='utf-8')
                code, result = self.work_cli('work-status', self.target, '--output-file', output)
                self.assertEqual(code, 5)
                self.assertEqual(result['error']['code'], 'WORK_MANIFEST_INVALID')
                self.assertFalse(result['error']['details']['workflow_started'])
                self.assertEqual(output.read_bytes(), b'keep existing output')

    def test_project_lock_path_errors_preserve_files_and_return_actionable_diagnostics(self):
        self.ready()
        before = self.rpy.read_bytes()
        manifest_before = Path(self.target).read_bytes()
        context = self.root / 'translation_context'
        context.write_bytes(b'existing project file')
        code, result = self.work_cli('work-apply', self.target)
        self.assertEqual(code, 5)
        self.assertEqual(result['error']['code'], 'WORK_LOCK_UNAVAILABLE')
        self.assertEqual(result['error']['suggested_action'], 'inspect_work_lock_path_permissions_and_owner')
        self.assertEqual(context.read_bytes(), b'existing project file')
        context.unlink()
        lock = context / '.external_work_apply.lock'
        lock.mkdir(parents=True)
        code, result = self.work_cli('work-apply', self.target)
        self.assertEqual(code, 5)
        self.assertEqual(result['error']['code'], 'WORK_LOCK_UNAVAILABLE')
        self.assertTrue(lock.is_dir())
        self.assertEqual(self.rpy.read_bytes(), before)
        self.assertEqual(Path(self.target).read_bytes(), manifest_before)

    def test_output_file_protection_covers_unselected_context_and_reference_artifacts(self):
        context = self.tl / 'context.rpy'
        context.write_text('# Read-only context file\n', encoding='utf-8')
        exported = work.export_work(output_dir=self.root / 'new-snapshot')
        self.target = exported['manifest_path']
        self.package = json.loads(Path(exported['work_path']).read_text(encoding='utf-8'))
        work.submit_work(self.target, self.submission([0], reviewed=True))
        downstream = work.export_work(output_dir=self.root / 'referencing', reference_works=[self.target])
        args = batch.build_arg_parser().parse_args(['work-status', downstream['manifest_path']])
        manifest = batch.load_manifest(self.target)
        for protected in (context, Path(self.target).parent / 'work.json',
                          Path(batch.resolve_manifest_result_path(manifest))):
            self.assertIsNotNone(batch._find_output_file_path_conflict(args, str(protected)))

    def test_post_apply_revision_uses_existing_proposal_contract_without_models(self):
        self.ready()
        work.apply_work(self.target)
        item = batch.collect_revision_file_jobs()[0]['items'][0]
        source_digest = revision_corpus.aggregate_digest(revision_corpus.collect_file_digests(dict(batch.collect_files_to_process())))
        row = {'schema_version': 1, 'occurrence_id': item['id'], 'identity_v2': item['id'],
               'file_rel_path': item['file_rel_path'], 'source': item['source'],
               'current_translation': item['current_translation'], 'proposed_translation': '灯笼还留着一点余温。',
               'reason': '写回后订正互操作验证', 'selected': True, 'disposition': 'accepted',
               'producer': {'type': 'agent', 'name': 'fixture reviewer'},
               'project_identity': {'tl_dir': str(self.tl)},
               'snapshot_digest': revision_corpus.item_snapshot_digest(item['source'], item['current_translation']),
               'corpus_snapshot_digest': source_digest}
        path = self.root / 'revision-proposal.jsonl'
        path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        imported = batch.import_revision_proposals(str(path))
        self.assertEqual(imported['status'], 'previewed')
        applied = batch.apply_revisions(imported['paths']['manifest'])
        self.assertEqual(applied['revision_apply_state'], 'applied')
        self.assertIn('灯笼还留着一点余温。', self.rpy.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
