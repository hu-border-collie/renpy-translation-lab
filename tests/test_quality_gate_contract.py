import json
import unittest
from unittest import mock

import cli_contract
import gemini_translate_batch as batch
import translation_quality


class WritebackGateContractTests(unittest.TestCase):
    def test_safety_safe_with_quality_warnings_still_allows_apply(self):
        safety = {
            'level': batch.CHECK_SAFETY_SAFE,
            'counts': {'safe': 10, 'warn': 0, 'block': 0},
            'reasons': {'warn': {}, 'block': {}},
        }
        quality_gate = {
            'decision': 'needs_review',
            'warning_count': 5,
            'blocker_count': 0,
        }

        gate = batch.summarize_writeback_gate(safety, quality_gate)

        self.assertTrue(gate['can_apply'])
        self.assertEqual(gate['decision'], 'allow')
        self.assertEqual(gate['blocker_count'], 0)

    def test_legacy_structural_warn_still_blocks_apply(self):
        safety = {
            'level': batch.CHECK_SAFETY_WARN,
            'counts': {'safe': 1, 'warn': 2, 'block': 0},
            'reasons': {'warn': {'source_mismatch': 2}, 'block': {}},
        }

        gate = batch.summarize_writeback_gate(safety, {'warning_count': 0, 'blocker_count': 0})

        self.assertFalse(gate['can_apply'])
        self.assertEqual(gate['decision'], 'deny')
        self.assertEqual(gate['structural_blocker_count'], 2)

    def test_quality_blocker_blocks_apply(self):
        safety = {
            'level': batch.CHECK_SAFETY_SAFE,
            'counts': {'safe': 10, 'warn': 0, 'block': 0},
            'reasons': {'warn': {}, 'block': {}},
        }

        gate = batch.summarize_writeback_gate(
            safety,
            {'warning_count': 0, 'blocker_count': 1},
        )

        self.assertFalse(gate['can_apply'])
        self.assertEqual(gate['decision'], 'deny')
        self.assertEqual(gate['quality_blocker_count'], 1)

    def test_attach_check_contract_persists_both_gates(self):
        manifest = {
            '_manifest_path': '/tmp/pkg/manifest.json',
            '_package_dir': '/tmp/pkg',
            'settings': {},
            'quality_acknowledged_finding_ids': [],
        }
        summary = {'reason_counts': {}, 'valid_items': 3, 'failure_items': 0}
        finding = {
            'finding_id': 'finding-1',
            'reason_code': translation_quality.REASON_CJK_LATIN_SPACING,
            'severity': 'medium',
            'disposition': 'warning',
            'item_id': 'item-1',
            'file': 'script.rpy',
            'line': 1,
            'source': 'Hello',
            'translation': '你好iPhone',
            'evidence': '{}',
            'suggestion': '',
            'rule_version': 1,
            'schema_version': 1,
        }

        with mock.patch.object(
            batch,
            'build_check_fingerprint',
            return_value={'fingerprint_sha256': 'fp'},
        ):
            summary = batch.attach_check_contract(manifest, summary, [finding])

        self.assertEqual(summary['safety_level'], 'safe')
        self.assertTrue(summary['can_apply'])
        self.assertTrue(summary['has_warnings'])
        self.assertEqual(summary['check_status'], 'ready_with_warnings')
        self.assertEqual(summary['writeback_gate']['decision'], 'allow')
        self.assertTrue(summary['writeback_gate']['can_apply'])
        self.assertEqual(summary['quality_gate']['warning_count'], 1)
        self.assertEqual(summary['check_contract_version'], batch.CHECK_CONTRACT_VERSION)

    def test_attach_check_contract_with_quality_blocker_marks_blocked(self):
        manifest = {
            '_manifest_path': '/tmp/pkg/manifest.json',
            '_package_dir': '/tmp/pkg',
            'settings': {},
            'quality_acknowledged_finding_ids': [],
        }
        summary = {'reason_counts': {}, 'valid_items': 3, 'failure_items': 0}
        finding = {
            'finding_id': 'finding-1',
            'reason_code': translation_quality.REASON_UNCLOSED_DELIMITERS,
            'severity': 'high',
            'disposition': 'blocker',
            'item_id': 'item-1',
            'file': 'script.rpy',
            'line': 1,
            'source': 'Hello',
            'translation': '你好 {w=0.5',
            'evidence': '{}',
            'suggestion': '',
            'rule_version': 1,
            'schema_version': 1,
        }

        with mock.patch.object(
            batch,
            'build_check_fingerprint',
            return_value={'fingerprint_sha256': 'fp'},
        ):
            summary = batch.attach_check_contract(manifest, summary, [finding])

        self.assertEqual(summary['check_status'], 'blocked')
        self.assertEqual(summary['writeback_gate']['decision'], 'deny')
        self.assertEqual(summary['safety_level'], 'block')
        self.assertEqual(summary['quality_gate']['blocker_count'], 1)


    def test_attach_check_contract_reuses_persisted_quality_gate_for_apply_recheck(self):
        manifest = {
            '_manifest_path': '/tmp/pkg/manifest.json',
            '_package_dir': '/tmp/pkg',
            'settings': {},
            'quality_acknowledged_finding_ids': [],
            'last_check_summary': {
                'quality_gate': {
                    'decision': 'needs_review',
                    'warning_count': 3,
                    'blocker_count': 1,
                    'acknowledged_count': 0,
                    'has_warnings': True,
                }
            },
        }
        summary = {'reason_counts': {}, 'valid_items': 3, 'failure_items': 0}

        with mock.patch.object(
            batch,
            'build_check_fingerprint',
            return_value={'fingerprint_sha256': 'fp'},
        ):
            summary = batch.attach_check_contract(manifest, summary)

        self.assertEqual(summary['quality_gate']['blocker_count'], 1)
        self.assertEqual(summary['writeback_gate']['decision'], 'deny')
        self.assertEqual(summary['check_status'], 'blocked')


class CheckFingerprintPolicyTests(unittest.TestCase):
    def test_quality_policy_change_changes_check_fingerprint(self):
        manifest = {
            '_manifest_path': '/tmp/pkg/manifest.json',
            '_package_dir': '/tmp/pkg',
            'settings': {},
            'files': {},
            'chunks': [],
            'base_dir': '/tmp',
            'tl_dir': '/tmp/tl',
        }
        old_policy = batch.BATCH_QUALITY_POLICY
        try:
            batch.BATCH_QUALITY_POLICY = translation_quality.normalize_policy(None)
            with mock.patch.object(batch, 'resolve_manifest_result_path', return_value='/tmp/pkg/results.jsonl'), \
                 mock.patch.object(batch, 'manifest_project_identity', return_value={'base_dir': '/tmp'}), \
                 mock.patch.object(batch, 'file_content_fingerprint', return_value={'sha256': 'r'}), \
                 mock.patch.object(
                    batch,
                    'stable_json_sha256',
                    side_effect=lambda payload: 'fp:' + json.dumps(
                        payload,
                        sort_keys=True,
                        default=str,
                    ),
                ):
                first = batch.build_check_fingerprint(manifest)
                batch.BATCH_QUALITY_POLICY = translation_quality.normalize_policy(
                    {'rules': {'cjk_latin_spacing': 'blocker'}}
                )
                second = batch.build_check_fingerprint(manifest)
            self.assertNotEqual(
                batch.check_fingerprint_id(first),
                batch.check_fingerprint_id(second),
            )
        finally:
            batch.BATCH_QUALITY_POLICY = old_policy


class RequireSafeCheckTests(unittest.TestCase):
    def _manifest(self, last_summary):
        return {
            '_manifest_path': '/tmp/pkg/manifest.json',
            '_package_dir': '/tmp/pkg',
            'settings': {},
            'last_check_summary': last_summary,
        }

    def test_allow_gate_with_quality_warnings_passes(self):
        manifest = self._manifest(
            {
                'check_contract_version': batch.CHECK_CONTRACT_VERSION,
                'check_fingerprint': {'fingerprint_sha256': 'current'},
                'safety_level': 'safe',
                'writeback_gate': {'decision': 'allow', 'can_apply': True},
                'quality_gate': {'decision': 'needs_review', 'warning_count': 7},
            }
        )
        with mock.patch.object(
            batch,
            'build_check_fingerprint',
            return_value={'fingerprint_sha256': 'current'},
        ):
            batch.require_safe_check_for_apply(manifest)

    def test_deny_gate_still_blocks_apply(self):
        manifest = self._manifest(
            {
                'check_contract_version': batch.CHECK_CONTRACT_VERSION,
                'check_fingerprint': {'fingerprint_sha256': 'current'},
                'safety_level': 'block',
                'writeback_gate': {'decision': 'deny', 'can_apply': False},
                'quality_gate': {'decision': 'pass', 'warning_count': 0},
            }
        )
        with (
            mock.patch.object(
                batch,
                'build_check_fingerprint',
                return_value={'fingerprint_sha256': 'current'},
            ),
            self.assertRaises(cli_contract.MachineContractError) as raised,
        ):
            batch.require_safe_check_for_apply(manifest)

        self.assertEqual(raised.exception.code_name, 'UNSAFE_CHECK_STATUS')

    def test_missing_gate_requires_recheck(self):
        manifest = self._manifest(
            {
                'check_contract_version': batch.CHECK_CONTRACT_VERSION,
                'check_fingerprint': {'fingerprint_sha256': 'current'},
                'safety_level': 'safe',
            }
        )
        with (
            mock.patch.object(
                batch,
                'build_check_fingerprint',
                return_value={'fingerprint_sha256': 'current'},
            ),
            self.assertRaises(cli_contract.MachineContractError) as raised,
        ):
            batch.require_safe_check_for_apply(manifest)

        self.assertEqual(raised.exception.code_name, 'APPLY_PREFLIGHT_FAILED')


class CheckResultsQualityIntegrationTests(unittest.TestCase):
    def test_check_results_persists_quality_gates_without_blocking_safety(self):
        import tempfile
        from pathlib import Path

        package_dir = Path(tempfile.mkdtemp())
        (package_dir / 'results.jsonl').write_text('{}\n', encoding='utf-8')
        manifest = {
            'mode': 'translation',
            'base_dir': str(package_dir),
            'tl_dir': str(package_dir),
            'target_language': 'schinese',
            'files': {},
            'settings': {},
            'execution': 'sync',
            'chunks': [
                {
                    'key': 'chunk-1',
                    'file_rel_path': 'script.rpy',
                    'items': [
                        {
                            'id': 'item-1',
                            'text': 'Hello Sir',
                            'line': 0,
                            'line_number': 1,
                            'start': 8,
                            'end': 17,
                            'speaker_name': 'Church Knight',
                        }
                    ],
                }
            ],
            '_manifest_path': str(package_dir / 'manifest.json'),
            '_package_dir': str(package_dir),
        }
        replacements = {
            'script.rpy': {
                0: [
                    (
                        8,
                        17,
                        '你好，Sir。',
                        '',
                        '"',
                        'Hello Sir',
                        'item-1',
                        'chunk-1',
                    )
                ]
            }
        }
        summary = {
            'expected_chunks': 1,
            'result_rows': 1,
            'processed_chunks': 1,
            'expected_items': 1,
            'valid_items': 1,
            'failure_items': 0,
            'chunk_row_errors': 0,
            'missing_response_chunks': 0,
            'partial_chunks': 0,
            'max_tokens_chunks': 0,
            'reason_counts': {},
        }

        with (
            mock.patch.object(batch, 'load_manifest', return_value=manifest),
            mock.patch.object(batch, 'require_manifest_mode'),
            mock.patch.object(batch, 'require_manifest_project_match'),
            mock.patch.object(
                batch,
                'collect_result_actions',
                return_value=(replacements, {}, [], summary),
            ),
            mock.patch.object(batch, 'save_manifest', side_effect=lambda m, **kwargs: None),
        ):
            checked = batch.check_results(manifest['_manifest_path'])

        last_summary = checked['last_check_summary']
        self.assertEqual(last_summary['safety_level'], 'safe')
        self.assertEqual(last_summary['check_status'], 'ready_with_warnings')
        self.assertEqual(last_summary['writeback_gate']['decision'], 'allow')
        self.assertGreater(last_summary['quality_gate']['warning_count'], 0)
        self.assertTrue(checked['last_quality_findings_path'].endswith('quality_findings.jsonl'))
        self.assertTrue(Path(checked['last_quality_findings_path']).exists())


class QualitySubjectCollectionTests(unittest.TestCase):
    def test_collect_quality_subjects_from_validated_replacements(self):
        manifest = {
            'chunks': [
                {
                    'key': 'chunk-1',
                    'file_rel_path': 'script.rpy',
                    'items': [
                        {
                            'id': 'item-1',
                            'text': 'Hello',
                            'line': 0,
                            'line_number': 1,
                            'start': 10,
                            'end': 17,
                            'speaker_id': 'ck',
                            'speaker_name': 'Church Knight',
                        }
                    ],
                }
            ]
        }
        replacements = {
            'script.rpy': {
                0: [
                    (
                        10,
                        17,
                        '你好',
                        '',
                        '"',
                        'Hello',
                        'item-1',
                        'chunk-1',
                    )
                ]
            }
        }

        subjects = batch.collect_quality_subjects(manifest, replacements)

        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0]['item_id'], 'item-1')
        self.assertEqual(subjects[0]['source'], 'Hello')
        self.assertEqual(subjects[0]['translation'], '你好')
        self.assertEqual(subjects[0]['speaker_name'], 'Church Knight')


if __name__ == '__main__':
    unittest.main()
