import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import keyword_glossary_merge as merge_mod


class KeywordGlossaryMergeTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def _sample_candidates(self) -> list[dict]:
        return [
            {
                'source': 'Void Gate',
                'suggested_target': '虚空门',
                'category': 'place',
                'confidence': 0.86,
                'evidence': 'Recurring gate name.',
            },
            {
                'source': 'Crystal Key',
                'suggested_target': '水晶钥匙',
                'category': 'item',
                'confidence': 0.55,
                'evidence': 'Quest key item.',
            },
            {
                'source': 'AR',
                'suggested_target': 'AR',
                'category': 'term',
                'confidence': 0.9,
                'evidence': 'Abbreviation kept unchanged.',
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


if __name__ == '__main__':
    unittest.main()
