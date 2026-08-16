"""Acceptance tests for issue #363: one quality-finding schema across paths."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import final_review as fr
import gemini_translate_batch as batch
import sync_translation_preview as preview
import translation_quality as quality
import translator_runtime as runtime
from gui_qt.quality_findings_report import (
    filter_quality_items,
    normalize_quality_finding,
    quality_gate_from_manifest,
    reason_label,
    resolve_quality_findings_path,
)
from gui_qt.translation_workflow import extract_quality_gate


def mechanical_subject(**overrides):
    subject = {
        'item_id': 'item-1',
        'file_rel_path': 'script.rpy',
        'line': 0,
        'line_number': 1,
        'source': 'Hello',
        'translation': '你{w=0.5}好',
    }
    subject.update(overrides)
    return subject


class SharedSchemaContractTests(unittest.TestCase):
    def test_normalize_filling_and_coercion(self):
        normalized = quality.normalize_finding(
            {
                'reason_code': quality.REASON_CJK_LATIN_SPACING,
                'severity': 'high',
                'disposition': 'warning',
                'line': '7',
                'rule_version': '1',
            }
        )

        self.assertEqual(normalized['line'], 7)
        self.assertEqual(normalized['rule_version'], 1)
        self.assertEqual(normalized['rule_id'], 'cjk_latin_spacing')
        self.assertIn('finding_id', normalized)

    def test_validate_finding_reports_contract_errors(self):
        finding = quality.normalize_finding(
            quality.check_subject(mechanical_subject())[0]
        )
        self.assertEqual(quality.validate_finding(finding), [])

        invalid = dict(finding)
        invalid.pop('reason_code')
        invalid['severity'] = 'catastrophic'
        invalid['line'] = -1
        errors = quality.validate_finding(invalid)
        self.assertIn('missing required field: reason_code', errors)
        self.assertTrue(any('severity' in error for error in errors))
        self.assertTrue(any('line' in error for error in errors))

    def test_filter_findings_supports_shared_dimensions(self):
        findings = [
            {
                'finding_id': 'a',
                'schema_version': 1,
                'reason_code': quality.REASON_CJK_LATIN_SPACING,
                'rule_id': 'cjk_latin_spacing',
                'severity': 'medium',
                'disposition': 'warning',
                'item_id': 'i1',
                'file': 'a.rpy',
                'line': 1,
                'source': 'Hello',
                'translation': '你好iPhone',
                'evidence': '{}',
                'suggestion': '',
                'rule_version': 1,
            },
            {
                'finding_id': 'b',
                'schema_version': 1,
                'reason_code': quality.REASON_HALFWIDTH_PUNCTUATION,
                'rule_id': 'halfwidth_punctuation',
                'severity': 'high',
                'disposition': 'blocker',
                'item_id': 'i2',
                'file': 'b.rpy',
                'line': 2,
                'source': 'Hello',
                'translation': '你好,',
                'evidence': '{}',
                'suggestion': '',
                'rule_version': 1,
            },
        ]

        self.assertEqual(
            [row['finding_id'] for row in quality.filter_findings(
                findings,
                reason_codes=[quality.REASON_CJK_LATIN_SPACING],
            )],
            ['a'],
        )
        self.assertEqual(
            [row['finding_id'] for row in quality.filter_findings(
                findings,
                files=['a.rpy'],
            )],
            ['a'],
        )
        self.assertEqual(
            [row['finding_id'] for row in quality.filter_findings(
                findings,
                min_severity='high',
            )],
            ['b'],
        )
        self.assertEqual(
            [row['finding_id'] for row in quality.filter_findings(
                findings,
                dispositions=['blocker'],
            )],
            ['b'],
        )

    def test_file_filter_does_not_match_across_name_boundaries(self):
        findings = [
            {
                'finding_id': 'f1',
                'schema_version': 1,
                'reason_code': quality.REASON_CJK_LATIN_SPACING,
                'rule_id': 'cjk_latin_spacing',
                'severity': 'medium',
                'disposition': 'warning',
                'item_id': 'i1',
                'file': 'xscript.rpy',
                'line': 1,
                'source': '',
                'translation': '',
                'evidence': '',
                'suggestion': '',
                'rule_version': 1,
            },
        ]

        self.assertEqual(
            quality.filter_findings(findings, files=['script.rpy']),
            [],
        )
        self.assertEqual(
            len(quality.filter_findings(findings, files=['script'])),
            1,
        )
        self.assertEqual(
            len(quality.filter_findings(findings, files=['xscript'])),
            1,
        )

    def test_load_write_and_digest_round_trip(self):
        findings = quality.check_subject(mechanical_subject())
        with tempfile.TemporaryDirectory() as tmp:
            path = quality.write_findings(Path(tmp) / 'findings.jsonl', findings)
            loaded = quality.load_findings(path)

        self.assertEqual(len(loaded), len(findings))
        self.assertEqual(loaded[0]['finding_id'], findings[0]['finding_id'])
        self.assertEqual(
            quality.findings_digest(findings),
            quality.findings_digest(loaded),
        )
        changed = [dict(findings[0])]
        changed[0]['evidence'] = '{}'
        self.assertNotEqual(
            quality.findings_digest(findings),
            quality.findings_digest(changed),
        )

    def test_load_findings_is_lenient_by_default_and_strict_on_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'findings.jsonl'
            valid = quality.normalize_finding(
                quality.check_subject(mechanical_subject())[0]
            )
            path.write_text(
                json.dumps(valid, ensure_ascii=False) + '\n{bad\n',
                encoding='utf-8',
            )

            self.assertEqual(len(quality.load_findings(path)), 1)
            with self.assertRaisesRegex(ValueError, 'row 2'):
                quality.load_findings(path, strict=True)

    def test_strict_load_rejects_invalid_enum_values_before_normalizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'findings.jsonl'
            invalid = quality.normalize_finding(
                quality.check_subject(mechanical_subject())[0]
            )
            invalid['severity'] = 'catastrophic'
            invalid['disposition'] = 'sometimes'
            invalid['line'] = -1
            invalid['rule_version'] = 'not-an-int'
            path.write_text(
                json.dumps(invalid, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )

            with self.assertRaisesRegex(ValueError, 'severity'):
                quality.load_findings(path, strict=True)

    def test_final_review_adapter_preserves_semantic_fields(self):
        semantic = fr.normalize_finding(
            {
                'identity_v2': 'script.rpy:item-1',
                'file_rel_path': 'script.rpy',
                'source': 'Hello Sir',
                'current_translation': '你好Sir',
                'finding_type': 'mistranslation',
                'severity': 'high',
                'evidence': 'Sir kept in English',
                'reason': '称呼未本地化',
                'suggested_revision': '你好，先生',
            }
        )
        adapted = quality.adapt_final_review_finding(semantic)

        self.assertEqual(adapted['reason_code'], quality.FINAL_REVIEW_REASON_MISTRANSLATION)
        self.assertEqual(adapted['rule_id'], 'final_review')
        self.assertEqual(adapted['disposition'], quality.DISPOSITION_WARNING)
        self.assertEqual(adapted['severity'], 'high')
        self.assertEqual(adapted['item_id'], 'script.rpy:item-1')
        self.assertEqual(adapted['translation'], '你好Sir')
        self.assertEqual(adapted['finding_type'], 'mistranslation')
        self.assertEqual(adapted['reason'], '称呼未本地化')
        self.assertEqual(adapted['suggestion'], '你好，先生')
        self.assertEqual(quality.validate_finding(adapted), [])

    def test_final_review_finding_feeds_shared_gui_filters(self):
        semantic = fr.normalize_finding(
            {
                'identity_v2': 'script.rpy:item-1',
                'file_rel_path': 'script.rpy',
                'source': 'Hello',
                'current_translation': 'Hello',
                'finding_type': 'omission',
                'severity': 'high',
                'reason': '未翻译',
            }
        )
        adapted = quality.adapt_final_review_finding(semantic)
        item = normalize_quality_finding(adapted)

        self.assertEqual(item.reason_code, quality.FINAL_REVIEW_REASON_OMISSION)
        self.assertIn('最终审校', reason_label(item.reason_code))
        self.assertEqual(
            filter_quality_items(
                [item],
                min_severity='high',
            )[0].finding_id,
            adapted['finding_id'],
        )

    def test_gui_extracts_sync_quality_gate_text(self):
        gate = extract_quality_gate(
            "Sync quality gate: needs_review, warnings=3, blockers=1"
        )
        self.assertEqual(gate["decision"], "needs_review")
        self.assertEqual(gate["warning_count"], 3)
        self.assertEqual(gate["blocker_count"], 1)

    def test_gui_extracts_revision_quality_gate_with_parenthesized_counts(self):
        gate = extract_quality_gate(
            "Quality gate: needs_review (warnings=3, blockers=1)"
        )
        self.assertEqual(gate["decision"], "needs_review")
        self.assertEqual(gate["warning_count"], 3)
        self.assertEqual(gate["blocker_count"], 1)

    def test_gui_prefers_apply_time_revision_quality_gate(self):
        manifest = {
            'last_revision_preview': {
                'quality_gate': {'decision': 'pass', 'warning_count': 0},
            },
            'revision_apply_summary': {
                'quality_gate': {'decision': 'needs_review', 'warning_count': 1},
            },
        }
        self.assertEqual(
            quality_gate_from_manifest(manifest)['warning_count'],
            1,
        )

        blocked = {
            'last_revision_preview': {
                'quality_gate': {'decision': 'pass', 'warning_count': 0},
            },
            'last_revision_apply_summary': {
                'quality_gate': {'decision': 'needs_review', 'warning_count': 2},
            },
        }
        self.assertEqual(
            quality_gate_from_manifest(blocked)['warning_count'],
            2,
        )

    def test_gui_resolves_quality_paths_across_producers(self):
        sync_manifest = {
            'last_quality_findings_path': 'sync/quality_findings.jsonl',
            'last_revision_quality_findings_path': 'revision/quality_findings.jsonl',
        }
        self.assertEqual(
            resolve_quality_findings_path(sync_manifest),
            'sync/quality_findings.jsonl',
        )
        revision_manifest = {
            'last_revision_quality_findings_path': 'revision/quality_findings.jsonl',
        }
        self.assertEqual(
            resolve_quality_findings_path(revision_manifest),
            'revision/quality_findings.jsonl',
        )
        final_review_manifest = {
            'quality_findings_path': 'final/quality_findings.jsonl',
        }
        self.assertEqual(
            resolve_quality_findings_path(final_review_manifest),
            'final/quality_findings.jsonl',
        )

    def test_final_review_package_persists_shared_quality_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fr.write_campaign_package(
                tmp,
                manifest=fr.build_campaign_manifest(
                    package_dir=tmp,
                    display_name='quality-mapping-test',
                    base_dir='/tmp/project',
                    tl_dir='/tmp/project/game/tl/schinese',
                    snapshot={
                        'context_digest': 'c',
                        'snapshot_digest': 's',
                        'scope': {},
                    },
                    units=[],
                    readiness={'enabled': True, 'require_zero_pending': True},
                ),
                snapshot={'context_digest': 'c', 'snapshot_digest': 's', 'scope': {}},
                units=[],
                findings=[
                    {
                        'identity_v2': 'script.rpy:item-1',
                        'file_rel_path': 'script.rpy',
                        'source': 'Hello',
                        'current_translation': 'Hello',
                        'finding_type': 'omission',
                        'severity': 'medium',
                        'reason': '未翻译',
                    }
                ],
                write_report=False,
            )

            package = fr.load_campaign_package(paths['manifest'])
            self.assertTrue(Path(paths['quality_findings']).is_file())
            self.assertEqual(package['quality_findings'][0]['reason_code'],
                             quality.FINAL_REVIEW_REASON_OMISSION)
            self.assertIn('quality_gate', package['manifest']['summary'])

            stale_manifest = dict(package['manifest'])
            stale_manifest['summary'] = {
                **stale_manifest['summary'],
                'quality_mapping_version': 999,
            }
            Path(paths['manifest']).write_text(
                json.dumps(stale_manifest, ensure_ascii=False),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(fr.FinalReviewSchemaError, 'stale'):
                fr.load_campaign_package(paths['manifest'])


class SyncQualityFindingsTests(unittest.TestCase):
    def _make_preview(self, root: Path, subject=None, *, glossary_file=""):
        tl_dir = root / 'game' / 'tl' / 'schinese'
        tl_dir.mkdir(parents=True)
        target = tl_dir / 'a.rpy'
        source_text = '    "Hello"\n'
        preview_text = '    "你{w=0.5}好"\n'
        target.write_text(source_text, encoding='utf-8')
        rows = [{
            'relative_path': 'a.rpy',
            'source_text': source_text,
            'source_sha256': '',
            'preview_text': preview_text,
            'progress_entries': ['id:1'],
            'quality_subjects': [
                mechanical_subject(translation='你{w=0.5}好', item_id='id:1')
            ],
        }]
        from atomic_io import file_sha256
        rows[0]['source_sha256'] = file_sha256(target)
        return preview.create_sync_preview(
            log_dir=root / 'logs',
            project_root=root,
            tl_dir=tl_dir,
            files=rows,
            glossary_file=glossary_file,
        )

    def test_sync_preview_persists_findings_and_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, manifest = self._make_preview(root)

            report_path = Path(manifest_path).parent / 'quality_findings.jsonl'
            self.assertTrue(report_path.is_file())
            self.assertEqual(manifest['last_quality_findings_path'],
                             'quality_findings.jsonl')
            self.assertEqual(manifest['summary']['quality_gate']['decision'],
                             quality.GATE_NEEDS_REVIEW)
            self.assertGreater(
                manifest['summary']['quality_gate']['warning_count'], 0
            )
            loaded = quality.load_findings(str(report_path))
            self.assertTrue(any(
                row['reason_code'] == quality.REASON_WAIT_TAG_INSIDE_CJK
                for row in loaded
            ))
            self.assertIn('quality_policy_digest', manifest['summary'])
            self.assertIn('quality_rule_schema_version', manifest)

    def test_sync_apply_stales_when_quality_policy_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _manifest = self._make_preview(root)
            tl_dir = root / 'game' / 'tl' / 'schinese'

            with self.assertRaisesRegex(ValueError, 'Quality policy changed'):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                    active_quality_policy=quality.normalize_policy(
                        {'rules': {'renpy_wait_inside_cjk': 'off'}}
                    ),
                )

    def test_sync_apply_stales_when_glossary_is_cleared_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, _manifest = self._make_preview(
                root,
                glossary_file='glossary.json',
            )
            tl_dir = root / 'game' / 'tl' / 'schinese'

            with self.assertRaisesRegex(ValueError, 'Quality glossary changed'):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                    active_glossary_file='',
                )

    def test_sync_apply_stales_when_glossary_content_changes_on_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glossary = root / 'glossary.json'
            glossary.write_text(
                json.dumps({'normalize_map': {'Church Knight': '教会骑士'}}),
                encoding='utf-8',
            )
            manifest_path, _manifest = self._make_preview(
                root,
                glossary_file='glossary.json',
            )
            glossary.write_text(
                json.dumps({'normalize_map': {'Church Knight': '圣殿骑士'}}),
                encoding='utf-8',
            )
            tl_dir = root / 'game' / 'tl' / 'schinese'

            with self.assertRaisesRegex(ValueError, 'glossary content changed'):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                    active_glossary_file='glossary.json',
                )

    def test_sync_apply_stales_when_quality_rule_version_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, manifest = self._make_preview(root)
            manifest['quality_rule_schema_version'] += 1
            manifest['preview_fingerprint'] = preview._fingerprint(manifest)
            Path(manifest_path).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            tl_dir = root / 'game' / 'tl' / 'schinese'

            with self.assertRaisesRegex(ValueError, 'Quality rules changed'):
                preview.apply_sync_preview(
                    manifest_path,
                    active_project_root=root,
                    active_tl_dir=tl_dir,
                )


class SyncRuntimeQualityFindingsTests(unittest.TestCase):
    def test_runtime_sync_preview_runs_quality_rules_on_accepted_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl_dir = root / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            target = tl_dir / 'script.rpy'
            target.write_text('    "Hello"\n', encoding='utf-8')

            def translate_batch(batch, replacements, usage_run_id="", **_kwargs):
                task = batch[0]
                replacements.setdefault(task["line"], []).append(
                    (
                        task["start"],
                        task["end"],
                        '你好iPhone',
                        task.get("prefix") or "",
                        task["quote"],
                    )
                )
                return [task.get("progress_entry") or f"id:{task['line']}"]

            with (
                mock.patch.object(runtime, "BASE_DIR", str(root)),
                mock.patch.object(runtime, "TL_DIR", str(tl_dir)),
                mock.patch.object(runtime, "LOG_DIR", str(root / "logs")),
                mock.patch.object(runtime, "SYNC_BACKEND", "litellm"),
                mock.patch.object(runtime, "PREP_ENABLED", False),
                mock.patch.object(runtime, "INCLUDE_FILES", []),
                mock.patch.object(runtime, "INCLUDE_PREFIXES", []),
                mock.patch.object(runtime, "load_config"),
                mock.patch.object(runtime, "load_translator_settings"),
                mock.patch.object(runtime, "load_glossary"),
                mock.patch.object(runtime, "load_progress", return_value={}),
                mock.patch.object(
                    runtime,
                    "process_batch_with_retry",
                    side_effect=translate_batch,
                ),
            ):
                manifest_path = runtime.run_translation()

            manifest = preview.load_sync_preview(manifest_path)
            gate = manifest["summary"]["quality_gate"]
            self.assertEqual(gate["decision"], quality.GATE_NEEDS_REVIEW)
            self.assertGreater(gate["warning_count"], 0)
            loaded = quality.load_findings(
                str(Path(manifest_path).parent / "quality_findings.jsonl")
            )
            self.assertTrue(
                {finding["reason_code"] for finding in loaded}
                & {
                    quality.REASON_CJK_LATIN_SPACING,
                    quality.REASON_SUSPICIOUS_ENGLISH_RESIDUE,
                }
            )
            self.assertTrue(loaded)
            self.assertTrue(all(finding["line"] == 1 for finding in loaded))


class RevisionQualityStalenessTests(unittest.TestCase):
    def _preview(self, **overrides):
        preview = {
            'quality_rule_schema_version': quality.QUALITY_RULE_SCHEMA_VERSION,
            'quality_policy_runtime_digest': quality.policy_digest(
                batch.BATCH_QUALITY_POLICY
            ),
            'quality_policy_digest': quality.policy_digest(
                quality.normalize_policy(None)
            ),
            'quality_findings_path': '',
            'writeback_gate': {'decision': quality.GATE_ALLOW},
        }
        preview.update(overrides)
        return preview

    def test_fresh_revision_quality_preview_returns_none(self):
        manifest = {'quality_policy': quality.normalize_policy(None)}
        self.assertIsNone(
            batch._revision_quality_staleness(manifest, self._preview())
        )

    def test_rule_version_change_makes_revision_preview_stale(self):
        manifest = {'quality_policy': quality.normalize_policy(None)}
        reason, _message = batch._revision_quality_staleness(
            manifest,
            self._preview(
                quality_rule_schema_version=(
                    quality.QUALITY_RULE_SCHEMA_VERSION + 1
                )
            ),
        )
        self.assertEqual(reason, 'quality_rules_changed')

    def test_runtime_policy_change_makes_revision_preview_stale(self):
        manifest = {'quality_policy': quality.normalize_policy(None)}
        changed_policy = quality.normalize_policy(
            {'rules': {'renpy_wait_inside_cjk': 'off'}}
        )
        reason, _message = batch._revision_quality_staleness(
            manifest,
            self._preview(
                quality_policy_runtime_digest=quality.policy_digest(
                    changed_policy
                )
            ),
        )
        self.assertEqual(reason, 'quality_policy_changed')

    def test_manifest_policy_change_makes_revision_preview_stale(self):
        manifest = {'quality_policy': quality.normalize_policy(None)}
        changed_policy = quality.normalize_policy(
            {'rules': {'renpy_wait_inside_cjk': 'off'}}
        )
        reason, _message = batch._revision_quality_staleness(
            manifest,
            self._preview(
                quality_policy_digest=quality.policy_digest(changed_policy)
            ),
        )
        self.assertEqual(reason, 'quality_policy_changed')

    def test_denied_revision_writeback_gate_is_stale(self):
        manifest = {'quality_policy': quality.normalize_policy(None)}
        reason, _message = batch._revision_quality_staleness(
            manifest,
            self._preview(writeback_gate={'decision': quality.GATE_DENY}),
        )
        self.assertEqual(reason, 'revision_writeback_gate_denied')

    def test_revision_blocked_marker_raises_after_persisting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {
                '_manifest_path': str(Path(tmp) / 'manifest.json'),
                'execution': 'sync',
            }
            with mock.patch.object(batch, 'save_manifest') as save:
                with self.assertRaises(SystemExit):
                    batch._mark_revision_apply_blocked(
                        manifest,
                        'quality_blockers_present',
                        'configured quality blocker rules matched revision candidates.',
                    )

            save.assert_called_once()
            self.assertEqual(manifest['revision_apply_state'], 'blocked')
            self.assertEqual(
                manifest['revision_apply_blocked_reason'],
                'quality_blockers_present',
            )


class RevisionQualityFindingsTests(unittest.TestCase):
    def _manifest(self, package_dir: Path, policy=None):
        item = {
            'id': 'a.rpy:block:1:hash',
            'text': 'Hello',
            'source': 'Hello',
            'current_translation': '你好',
            'line': 0,
            'line_number': 1,
            'start': 4,
            'end': 9,
            'prefix': '',
            'quote': '"',
        }
        manifest = {
            '_manifest_path': str(package_dir / 'manifest.json'),
            '_package_dir': str(package_dir),
            'base_dir': str(package_dir),
            'chunks': [{
                'key': 'chunk-1',
                'file_rel_path': 'a.rpy',
                'items': [item],
            }],
            'quality_policy': quality.normalize_policy(policy),
            'quality_acknowledged_finding_ids': [],
        }
        return manifest

    def _replacements(self):
        action = (4, 9, '你{w=0.5}好', '', '"', '你好', 'a.rpy:block:1:hash', 'chunk-1')
        return {'a.rpy': {0: [action]}}

    def test_revision_quality_subjects_use_revision_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest = self._manifest(package_dir)
            subjects = batch.collect_revision_quality_subjects(
                manifest,
                self._replacements(),
            )

            self.assertEqual(len(subjects), 1)
            self.assertEqual(subjects[0]['item_id'], 'a.rpy:block:1:hash')
            self.assertEqual(subjects[0]['file_rel_path'], 'a.rpy')
            self.assertEqual(subjects[0]['line_number'], 1)
            self.assertEqual(subjects[0]['source'], 'Hello')
            self.assertEqual(subjects[0]['translation'], '你{w=0.5}好')

    def test_revision_quality_warning_does_not_block_but_blocker_does(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest = self._manifest(package_dir)
            summary = {'adapter_writeback_status': 'pass'}
            _findings, path = batch.run_revision_quality_check(
                manifest,
                summary,
                self._replacements(),
            )

            self.assertTrue(Path(path).is_file())
            self.assertEqual(summary['quality_gate']['decision'],
                             quality.GATE_NEEDS_REVIEW)
            self.assertTrue(summary['writeback_gate']['can_apply'])

            manifest = self._manifest(
                package_dir,
                policy={'rules': {'renpy_wait_inside_cjk': 'blocker'}},
            )
            summary = {'adapter_writeback_status': 'pass'}
            _findings, path = batch.run_revision_quality_check(
                manifest,
                summary,
                self._replacements(),
            )
            self.assertEqual(summary['quality_gate']['blocker_count'], 1)
            self.assertFalse(summary['writeback_gate']['can_apply'])
            self.assertEqual(summary['writeback_gate']['decision'],
                             quality.GATE_DENY)


if __name__ == '__main__':
    unittest.main()
