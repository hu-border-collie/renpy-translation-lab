import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import keyword_glossary_merge as merge_mod
import keyword_history


class KeywordGlossaryMergeTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _consistent_history(self, source: str, target: str) -> dict:
        return keyword_history.build_keyword_history_evidence(
            {'source': source, 'suggested_target': target},
            [
                {
                    'identity_v2': f'test/{source}:1',
                    'occurrence_id': f'test/{source}:1',
                    'file_rel_path': 'test/keywords.rpy',
                    'display_line': 1,
                    'locator': {'line_number': 1, 'start': 0},
                    'source': source,
                    'current_translation': target,
                }
            ],
        )

    def _sample_candidates(self) -> list[dict]:
        return [
            {
                'source': 'Void Gate',
                'suggested_target': '虚空门',
                'category': 'place',
                'confidence': 0.86,
                'evidence': 'Recurring gate name.',
                'history_evidence': self._consistent_history('Void Gate', '虚空门'),
            },
            {
                'source': 'Crystal Key',
                'suggested_target': '水晶钥匙',
                'category': 'item',
                'confidence': 0.55,
                'evidence': 'Quest key item.',
                'history_evidence': self._consistent_history('Crystal Key', '水晶钥匙'),
            },
            {
                'source': 'AR',
                'suggested_target': 'AR',
                'category': 'term',
                'confidence': 0.9,
                'evidence': 'Abbreviation kept unchanged.',
                'history_evidence': self._consistent_history('AR', 'AR'),
            },
        ]

    def test_merge_adds_new_normalize_map_and_preserve_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text(
                json.dumps({'preserve_terms': [], 'normalize_map': {}}, ensure_ascii=False),
                encoding='utf-8',
            )
            self._write_jsonl(candidates_path, self._sample_candidates())

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                min_confidence=0.8,
                accept_confidence=0.8,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 2)
            self.assertEqual(summary.skipped_low_confidence, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(data['normalize_map']['Void Gate'], '虚空门')
            self.assertIn('AR', data['preserve_terms'])

    def test_merge_skips_duplicate_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text(
                json.dumps(
                    {'preserve_terms': [], 'normalize_map': {'Void Gate': '虚空门'}},
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            self._write_jsonl(candidates_path, self._sample_candidates()[:1])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.skipped_duplicate, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(len(data['normalize_map']), 1)

    def test_merge_overwrite_updates_conflicting_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text(
                json.dumps(
                    {'preserve_terms': [], 'normalize_map': {'Void Gate': '旧译名'}},
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            self._write_jsonl(candidates_path, self._sample_candidates()[:1])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                overwrite=True,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 1)
            self.assertEqual(summary.overwritten, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(data['normalize_map']['Void Gate'], '虚空门')

    def test_dry_run_does_not_write_glossary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            original = {'preserve_terms': [], 'normalize_map': {}}
            glossary_path.write_text(json.dumps(original, ensure_ascii=False), encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates()[:1])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                dry_run=True,
                interactive=False,
                backup=False,
            )

            self.assertTrue(summary.dry_run)
            self.assertFalse(summary.wrote_glossary)
            self.assertEqual(json.loads(glossary_path.read_text(encoding='utf-8')), original)

    def test_min_confidence_filters_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates())

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                min_confidence=0.8,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 2)
            self.assertEqual(summary.skipped_low_confidence, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(list(data['normalize_map'].keys()), ['Void Gate'])
            self.assertIn('AR', data['preserve_terms'])

    def test_resolve_candidates_path_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            package = root / 'package'
            package.mkdir()
            jsonl_path = package / 'keyword_candidates.jsonl'
            self._write_jsonl(jsonl_path, self._sample_candidates()[:1])
            manifest_path = package / 'manifest.json'
            manifest_path.write_text(
                json.dumps(
                    {
                        'keyword_export': {
                            'jsonl_path': str(jsonl_path),
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )

            resolved = merge_mod.resolve_keyword_candidates_path(str(manifest_path))
            self.assertEqual(Path(resolved).resolve(), jsonl_path.resolve())

    def test_resolve_rejects_non_keyword_manifest_with_sibling_candidates(self):
        """A batch package must not be accepted merely because a stale JSONL exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir).resolve() / 'package'
            package.mkdir()
            jsonl_path = package / 'keyword_candidates.jsonl'
            self._write_jsonl(jsonl_path, self._sample_candidates()[:1])
            manifest_path = package / 'manifest.json'
            manifest_path.write_text(
                json.dumps({'mode': 'batch_translation'}, ensure_ascii=False),
                encoding='utf-8',
            )

            with self.assertRaises(SystemExit) as ctx:
                merge_mod.resolve_keyword_candidates_path(str(manifest_path))

        self.assertIn('not a keyword extraction package', str(ctx.exception))

    def test_resolve_keyword_lite_manifest_uses_sibling_candidates(self):
        """Lite keyword manifests may omit keyword_export but retain mode metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            package = Path(tmpdir).resolve() / 'package'
            package.mkdir()
            jsonl_path = package / 'keyword_candidates.jsonl'
            self._write_jsonl(jsonl_path, self._sample_candidates()[:1])
            manifest_path = package / 'manifest.json'
            manifest_path.write_text(
                json.dumps({'mode': 'keyword_extraction'}, ensure_ascii=False),
                encoding='utf-8',
            )

            resolved = merge_mod.resolve_keyword_candidates_path(str(package))
            self.assertEqual(Path(resolved).resolve(), jsonl_path.resolve())

    def test_resolve_missing_jsonl_reports_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir).resolve() / 'missing.jsonl'
            with self.assertRaises(SystemExit) as ctx:
                merge_mod.resolve_keyword_candidates_path(str(missing))
        self.assertIn('Keyword candidates JSONL not found', str(ctx.exception))

    def test_overwrite_moves_entry_across_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text(
                json.dumps(
                    {'preserve_terms': [], 'normalize_map': {'AR': '旧译名'}},
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            self._write_jsonl(
                candidates_path,
                [
                    {
                        'source': 'AR',
                        'suggested_target': 'AR',
                        'category': 'term',
                        'confidence': 0.95,
                        'evidence': 'Abbreviation kept unchanged.',
                        'history_evidence': self._consistent_history('AR', 'AR'),
                    }
                ],
            )

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                overwrite=True,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertIn('AR', data['preserve_terms'])
            self.assertNotIn('AR', data['normalize_map'])

    def test_quit_keeps_already_accepted_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates()[:2])

            responses = iter(['y', 'q'])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                input_func=lambda _prompt: next(responses),
                backup=False,
            )

            self.assertEqual(summary.accepted, 1)
            self.assertTrue(summary.wrote_glossary)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(data['normalize_map']['Void Gate'], '虚空门')
            self.assertNotIn('Crystal Key', data['normalize_map'])

    def test_dry_run_auto_accepts_without_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates()[:1])

            def fail_if_prompt(_prompt):
                raise AssertionError('dry-run should not prompt for input')

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                dry_run=True,
                input_func=fail_if_prompt,
                backup=False,
            )

            self.assertEqual(summary.accepted, 1)
            self.assertFalse(summary.wrote_glossary)

    def test_macro_preserve_line_does_not_trigger_translate_warning(self):
        candidate = {
            'source': 'AR',
            'suggested_target': 'AR',
            'category': 'item',
            'confidence': 0.9,
            'history_evidence': self._consistent_history('AR', 'AR'),
        }
        action = merge_mod.plan_merge_action(
            candidate,
            {'preserve_terms': [], 'normalize_map': {}},
        )
        self.assertIsNotNone(action)
        warnings = merge_mod.detect_candidate_warnings(
            candidate,
            action,
            macro_setting_text='AR 不翻译',
        )
        self.assertEqual(warnings, [])

    def test_is_likely_ui_noise_detects_launcher_labels(self):
        self.assertTrue(
            merge_mod.is_likely_ui_noise(
                {
                    'source': 'Start',
                    'suggested_target': '开始',
                    'evidence': 'common.rpy menu',
                }
            )
        )
        self.assertFalse(
            merge_mod.is_likely_ui_noise(
                {
                    'source': 'Void Gate',
                    'suggested_target': '虚空门',
                    'evidence': 'Recurring gate name.',
                }
            )
        )

    def test_merge_selected_candidates_honors_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates())

            candidates = merge_mod.load_keyword_candidates_jsonl(str(candidates_path))
            summary = merge_mod.merge_selected_candidates(
                candidates,
                {0},
                str(glossary_path),
                dry_run=True,
            )
            self.assertEqual(summary.accepted, 1)
            self.assertFalse(summary.wrote_glossary)

    def test_interactive_skip_counts_user_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            self._write_jsonl(candidates_path, self._sample_candidates()[:1])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                input_func=lambda _prompt: 'n',
                backup=False,
            )

            self.assertEqual(summary.skipped_user, 1)
            self.assertEqual(json.loads(glossary_path.read_text(encoding='utf-8')), {'normalize_map': {}})

    def test_history_conflict_is_not_auto_accepted_by_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            candidate = {
                'source': 'light',
                'suggested_target': '光',
                'category': 'term',
                'confidence': 0.99,
                'history_evidence': {
                    'status': 'ambiguous',
                    'first_occurrence': {
                        'file_rel_path': 'chapter01/a.rpy',
                        'line_number': 4,
                        'current_translation': '光',
                    },
                    'conflict_reasons': ['同一术语的历史 occurrence 存在多个不同现译'],
                },
            }
            self._write_jsonl(candidates_path, [candidate])

            rows = merge_mod.build_candidate_merge_rows(
                [candidate],
                {'normalize_map': {}},
            )
            self.assertFalse(rows[0].default_checked)
            self.assertTrue(rows[0].warnings)

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                accept_confidence=0.8,
                input_func=lambda _prompt: 'n',
                backup=False,
            )

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.skipped_user, 1)
            self.assertEqual(
                json.loads(glossary_path.read_text(encoding='utf-8')),
                {'normalize_map': {}},
            )

    def test_missing_history_is_not_auto_accepted_by_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            candidate = {
                'source': 'Legacy Term',
                'suggested_target': '旧术语',
                'category': 'term',
                'confidence': 0.99,
                'evidence': 'Legacy candidate without history export.',
            }
            self._write_jsonl(candidates_path, [candidate])

            rows = merge_mod.build_candidate_merge_rows(
                [candidate],
                {'normalize_map': {}},
            )
            self.assertFalse(rows[0].default_checked)
            self.assertTrue(any('历史证据缺失' in warning for warning in rows[0].warnings))

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                accept_confidence=0.8,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.skipped_user, 1)
            self.assertEqual(
                json.loads(glossary_path.read_text(encoding='utf-8')),
                {'normalize_map': {}},
            )

    def test_incomplete_consistent_history_is_not_auto_accepted_by_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            candidate = {
                'source': 'Legacy Term',
                'suggested_target': '旧术语',
                'category': 'term',
                'confidence': 0.99,
                'history_evidence': {'status': 'consistent'},
            }
            self._write_jsonl(candidates_path, [candidate])

            rows = merge_mod.build_candidate_merge_rows(
                [candidate],
                {'normalize_map': {}},
            )
            self.assertFalse(rows[0].default_checked)
            self.assertTrue(any('字段不完整' in warning for warning in rows[0].warnings))

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                accept_confidence=0.8,
                interactive=False,
                backup=False,
            )

            self.assertEqual(summary.accepted, 0)
            self.assertEqual(summary.skipped_user, 1)
            self.assertEqual(
                json.loads(glossary_path.read_text(encoding='utf-8')),
                {'normalize_map': {}},
            )

    def test_explicit_history_override_accepts_missing_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            candidates_path = root / 'keyword_candidates.jsonl'
            glossary_path = root / 'glossary.json'
            glossary_path.write_text('{"normalize_map": {}}', encoding='utf-8')
            candidate = {
                'source': 'Legacy Term',
                'suggested_target': '旧术语',
                'category': 'term',
                'confidence': 0.99,
            }
            self._write_jsonl(candidates_path, [candidate])

            summary = merge_mod.merge_keywords_to_glossary(
                str(candidates_path),
                str(glossary_path),
                accept_confidence=0.8,
                interactive=False,
                allow_history_review=True,
                backup=False,
            )

            self.assertEqual(summary.accepted, 1)
            data = json.loads(glossary_path.read_text(encoding='utf-8'))
            self.assertEqual(data['normalize_map']['Legacy Term'], '旧术语')


if __name__ == '__main__':
    unittest.main()
