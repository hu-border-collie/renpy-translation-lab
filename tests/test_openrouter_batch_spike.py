"""Offline negative cases for #520; synthetic receipts are not remote evidence."""
from copy import deepcopy
import json
import unittest

from scripts import openrouter_batch_spike as spike


def synthetic_receipt(frozen):
    rows = []
    for m in frozen['mapping']:
        rows.append({
            'custom_id': m['custom_id'], 'error': None,
            'response': {'status_code': 200, 'body': {
                'id': 'synthetic-not-a-provider-receipt', 'model': spike.MODEL,
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                    'translations': [{'id': i, 'translation': '原创离线候选' + ''.join(
                        entry['marker'] for entry in m['request']['transport_metadata']
                        ['structure_protection']['items'][i]['entries'])}
                                     for i in m['request']['expected_ids']],
                }, ensure_ascii=False)}}],
            }},
        })
    return {'id': 'offline-only', 'endpoint': spike.ENDPOINT, 'model': spike.MODEL,
            'provider': 'OpenAI', 'status': 'completed', 'results': rows,
            'request_counts': {'total': 2, 'completed': 2, 'failed': 0}, 'usage': None}


class OpenRouterBatchSpikeTests(unittest.TestCase):
    def setUp(self):
        self.frozen = spike.freeze_fixture()
        self.receipt = synthetic_receipt(self.frozen)

    def inspect(self):
        return spike.map_receipt(self.frozen, self.receipt, batch_id='offline-only')

    def test_deterministic_freeze_and_ordered_wire_envelope(self):
        self.assertEqual(self.frozen, spike.freeze_fixture())
        body = json.loads(spike.serialize_envelope(self.frozen))
        self.assertEqual(list(body)[-1], 'requests')
        self.assertEqual(body['model'], 'openai/gpt-4.1-mini')
        self.assertEqual(body['provider'], {'only': ['openai'], 'allow_fallbacks': False})
        self.assertEqual(len(body['requests']), 2)
        for mapping, wire in zip(self.frozen['mapping'], body['requests']):
            req = mapping['request']
            self.assertEqual(wire['body']['messages'], [
                {'role': 'system', 'content': req['system_instruction']},
                {'role': 'user', 'content': req['user_prompt']},
            ])
            self.assertEqual(wire['body']['response_format']['json_schema']['schema'], req['response_schema'])
            self.assertEqual(wire['body']['max_tokens'], 1024)

    def test_source_prompt_schema_and_routing_tampering_fail_closed(self):
        for target, key in [('source', 'provenance'), ('envelope', 'model'), ('plan', 'plan_id')]:
            with self.subTest(target=target):
                frozen = deepcopy(self.frozen)
                frozen[target][key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'FROZEN_DIGEST_MISMATCH'):
                    spike.validate_frozen(frozen)
        frozen = deepcopy(self.frozen)
        frozen['mapping'][0]['request']['response_schema'] = {}
        payload = {k: v for k, v in frozen.items() if k != 'frozen_sha256'}
        frozen['frozen_sha256'] = spike.digest(payload)
        with self.assertRaisesRegex(ValueError, 'SOURCE_OR_REQUEST'):
            spike.validate_frozen(frozen)

    def test_unordered_results_map_by_id_and_never_authorize_writeback(self):
        self.receipt['results'].reverse()
        result = self.inspect()
        self.assertEqual(len(result['candidates']), 2)
        mapping = {m['custom_id']: m['request']['chunk_id'] for m in self.frozen['mapping']}
        for row in result['candidates']:
            self.assertEqual(row['chunk_id'], mapping[row['custom_id']])
        self.assertFalse(result['writeback_authorized'])
        self.assertIsNone(result['usage'])  # Unknown is not zero.

    def test_duplicate_unknown_and_wrong_batch_route_are_quarantined(self):
        for mode in ('duplicate', 'unknown', 'batch', 'provider', 'model', 'endpoint', 'state'):
            with self.subTest(mode=mode):
                original = deepcopy(self.receipt)
                if mode == 'duplicate':
                    self.receipt['results'].append(deepcopy(self.receipt['results'][0]))
                elif mode == 'unknown':
                    self.receipt['results'][0]['custom_id'] = 'other-job'
                elif mode == 'batch':
                    self.receipt['id'] = 'other-job'
                elif mode == 'state':
                    self.receipt['status'] = 'new-unrecognized-status'
                else:
                    self.receipt[mode] = 'other'
                with self.assertRaises(ValueError):
                    self.inspect()
                self.receipt = original

    def test_missing_result_preserves_unresolved_identity(self):
        missing = self.receipt['results'].pop()
        self.receipt['request_counts']['completed'] = 1
        result = self.inspect()
        self.assertEqual(result['failures'], [{'custom_id': missing['custom_id'], 'category': 'MISSING_CUSTOM_ID'}])
        self.assertEqual(result['local_state'], 'candidates_with_errors')

    def test_per_item_failure_retains_successful_sibling(self):
        row = self.receipt['results'][1]
        row.update(response=None, error={'code': 429, 'message': 'synthetic'})
        self.receipt['request_counts'].update(completed=1, failed=1)
        result = self.inspect()
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['failures'][0]['category'], 'REMOTE_ITEM_ERROR')

    def test_truncated_invalid_json_or_wrong_item_ids_never_become_candidates(self):
        for mode in ('length', 'refusal', 'json', 'unknown_id', 'duplicate_id', 'blank', 'missing_token'):
            with self.subTest(mode=mode):
                choice = self.receipt['results'][0]['response']['body']['choices'][0]
                saved = deepcopy(choice)
                if mode == 'length':
                    choice['finish_reason'] = 'length'
                elif mode == 'refusal':
                    choice['message']['refusal'] = 'synthetic'
                elif mode == 'json':
                    choice['message']['content'] = '{'
                else:
                    data = json.loads(choice['message']['content'])
                    if mode == 'unknown_id':
                        data['translations'][0]['id'] = 'other'
                    elif mode == 'duplicate_id':
                        data['translations'].append(data['translations'][0])
                    elif mode == 'blank':
                        data['translations'][0]['translation'] = ''
                    else:
                        data['translations'][1]['translation'] = '故意漏掉变量占位符'
                    choice['message']['content'] = json.dumps(data)
                self.assertEqual(len(self.inspect()['candidates']), 1)
                self.assertEqual(len(self.inspect()['failures']), 1)
                choice.clear()
                choice.update(saved)

    def test_noncompleted_partial_counts_do_not_invent_results(self):
        for state in sorted(spike.REMOTE_STATES - {'completed'}):
            with self.subTest(state=state):
                self.receipt.update(status=state, results=None)
                result = self.inspect()
                self.assertEqual(result['candidates'], [])
                self.assertEqual(result['local_state'], 'terminal_no_results' if state in spike.TERMINAL else 'pending')
        self.receipt['results'] = []
        with self.assertRaisesRegex(ValueError, 'NONCOMPLETED_RESULTS'):
            self.inspect()

    def test_count_and_response_error_conflicts_fail_closed(self):
        self.receipt['request_counts']['total'] = 3
        with self.assertRaisesRegex(ValueError, 'COUNTS'):
            self.inspect()
        self.receipt['request_counts']['total'] = 2
        self.receipt['results'][0]['error'] = {'code': 'synthetic'}
        with self.assertRaisesRegex(ValueError, 'EXCLUSIVITY'):
            self.inspect()

    def test_outcome_unknown_never_releases_budget_or_recommends_resubmit(self):
        for code in (None, 408, 429, 500, 502, 503):
            with self.subTest(code=code):
                self.assertEqual(spike.submission_recovery({
                    'state': 'dispatched_outcome_unknown', 'http_status': code,
                }), 'reconcile_list_or_support_keep_reservation')
        self.assertEqual(spike.submission_recovery({
            'state': 'dispatched_outcome_unknown', 'batch_id': 'known',
        }), 'poll_known_batch')
        self.assertEqual(spike.submission_recovery({
            'state': 'rejected', 'http_status': 403,
        }), 'stop_rejected_attempt')
        self.assertEqual(spike.submission_recovery({'state': 'unrecognized'}), 'attention_no_submit')

    def test_wrong_result_model_provider_and_malformed_rows_are_quarantined(self):
        body = self.receipt['results'][0]['response']['body']
        for key in ('model', 'provider'):
            with self.subTest(key=key):
                previous = body.get(key)
                body[key] = 'other'
                with self.assertRaises(ValueError):
                    self.inspect()
                body[key] = previous
        self.receipt['results'][0] = None
        with self.assertRaisesRegex(ValueError, 'INVALID_RESULT_ROW'):
            self.inspect()

    def test_modern_experiment_keeps_historical_digest_and_freezes_new_route(self):
        self.assertEqual(self.frozen['frozen_sha256'],
                         '0535796aa9890506f3ebfa8546f00017e513256e89f7f1a101da4275711d79e1')
        modern = spike.freeze_fixture(experiment='deepseek41-v2')
        spike.validate_frozen(modern)
        self.assertNotEqual(modern['frozen_sha256'], self.frozen['frozen_sha256'])
        self.assertEqual(modern['envelope']['model'], 'deepseek/deepseek-v4.1-flash')
        self.assertEqual(modern['envelope']['provider'], {'only': ['deepinfra/fp8'], 'allow_fallbacks': False})
        for m, r in zip(modern['mapping'], modern['envelope']['requests']):
            self.assertEqual(r['body']['temperature'], 0.2)
            self.assertEqual(r['body']['reasoning'], {'enabled': False})
            self.assertIn('Keep Mira and Oren in English.', m['request']['system_instruction'])
            self.assertNotIn('Mira = 米拉', m['request']['system_instruction'])
            self.assertEqual(r['body']['max_tokens'], 1024)

    def test_modern_mapping_preserves_actual_model_and_provider_boundaries(self):
        modern = spike.freeze_fixture(experiment='deepseek41-v2')
        receipt = synthetic_receipt(modern)
        receipt.update(model='deepseek/deepseek-v4.1-flash', provider='DeepInfra')
        for row in receipt['results']:
            row['response']['body']['model'] = 'deepseek/deepseek-v4.1-flash-20260910'
        mapped = spike.map_receipt(modern, receipt, batch_id='offline-only')
        self.assertEqual(len(mapped['candidates']), 2)
        receipt['model'] = 'deepseek/deepseek-v4.1-flash-20260910'
        mapped = spike.map_receipt(modern, receipt, batch_id='offline-only')
        self.assertEqual(len(mapped['candidates']), 2)
        receipt['provider'] = 'OpenAI'
        with self.assertRaisesRegex(ValueError, 'ROUTE_MISMATCH'):
            spike.map_receipt(modern, receipt, batch_id='offline-only')


if __name__ == '__main__':
    unittest.main()
