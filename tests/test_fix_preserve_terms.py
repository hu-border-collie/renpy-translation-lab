import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fix_preserve_terms as preserve_fix
import gemini_translate_batch as batch


class PreserveTermsPatchTests(unittest.TestCase):
    def test_results_patch_updates_authoritative_normalized_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            manifest_path = package_dir / 'manifest.json'
            result_path = package_dir / 'results.jsonl'
            failure_path = package_dir / 'failures.jsonl'
            item = {
                'id': 'a',
                'text': 'Hello Edgar',
                'line': 0,
                'start': 0,
                'end': 11,
                'prefix': '',
                'quote': '"',
            }
            raw_response = batch.response_payload_with_text(
                {},
                json.dumps([
                    {'id': 'a', 'translation': '你好'},
                ], ensure_ascii=False),
            )
            row = {
                'key': 'chunk-1',
                'response': raw_response,
                'normalized_response': {
                    'translations': [{'id': 'a', 'translation': '你好'}],
                },
                'contract_diagnostics': {
                    'complete': True,
                    'reason_counts': {},
                    'diagnostic_counts': {},
                    'issues': [],
                    'diagnostics': [],
                },
            }
            result_path.write_text(
                json.dumps(row, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            failure_path.write_text(
                json.dumps({
                    'key': 'chunk-1',
                    'id': 'a',
                    'text': 'Hello Edgar',
                }, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
            manifest_path.write_text(
                json.dumps({
                    'mode': batch.MANIFEST_MODE_TRANSLATION,
                    'result_jsonl_path': str(result_path),
                    'result_jsonl_sha256': 'stale',
                    'last_check_at': '2026-08-13T00:00:00',
                    'last_check_summary': {'safety_level': 'safe'},
                    'last_check_report_path': str(package_dir / 'check.jsonl'),
                    'chunks': [
                        {
                            'key': 'chunk-1',
                            'file_rel_path': 'script.rpy',
                            'items': [item],
                        }
                    ],
                }),
                encoding='utf-8',
            )

            with mock.patch.object(
                preserve_fix.legacy,
                'missing_preserved_terms',
                return_value=['Edgar'],
            ):
                changed = preserve_fix.patch_results_manifest(
                    str(manifest_path),
                    failure_path,
                )

            saved_row = json.loads(result_path.read_text(encoding='utf-8'))
            saved_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

        self.assertEqual(changed, 1)
        self.assertEqual(
            saved_row['normalized_response']['translations'][0]['translation'],
            'Edgar好',
        )
        self.assertEqual(saved_row['response'], raw_response)
        self.assertTrue(saved_row['contract_diagnostics']['complete'])
        self.assertNotIn('last_check_summary', saved_manifest)
        self.assertNotEqual(saved_manifest['result_jsonl_sha256'], 'stale')


if __name__ == '__main__':
    unittest.main()
