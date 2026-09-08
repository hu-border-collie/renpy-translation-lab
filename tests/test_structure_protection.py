"""Offline request/view, restore, lineage and writeback-boundary contracts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import structure_protection as protection
import translation_core as core
import translation_plan as plan
from engine_adapters import structure_rules


def build(strategy='sync', text='{b}Hello [name]{/b}', engine='renpy'):
    items = [{'id': 'one', 'text': text}, {'id': 'two', 'text': 'Goodbye [name]'}]
    result = plan.build_translation_plan(
        [{'file_rel_path': 'scene.rpy', 'tasks': items}],
        execution_strategy=strategy, source_identity={'engine': engine},
    )
    return result.requests[0], items


def response(request, items):
    maps = request.transport_metadata[protection.KEY]['items']
    return {'translations': [
        {'id': item['id'], 'translation': protection.protect(
            item['text'], engine=maps[item['id']]['engine'],
            request_id=request.request_id, item_id=item['id'], scope=maps[item['id']]['scope'],
            literal_brackets=maps[item['id']].get('literal_brackets', False),
        )[0].replace('Hello', '你好').replace('Goodbye', '再见')}
        for item in items
    ]}


def report(request, items, payload, canonical=False):
    result = core.validate_model_response(payload, expected_units=items)
    return protection.validate_report(result, request, items, canonical=canonical)


class ProtectionTests(unittest.TestCase):
    def test_retries_bind_current_context_and_reject_transplanted_maps(self):
        parent, items = build()
        left = plan.derive_translation_request(parent, items[:1], lineage_suffix='--retry',
                                              retrieval_blocks_text='context A')
        right = plan.derive_translation_request(parent, items[:1], lineage_suffix='--retry',
                                               retrieval_blocks_text='context B')
        self.assertEqual(left.request_id, right.request_id)
        self.assertNotEqual(left.transport_metadata[protection.KEY], right.transport_metadata[protection.KEY])
        raw = response(left, items[:1])
        self.assertFalse(report(right, items[:1], raw).complete)
        right.transport_metadata[protection.KEY] = left.transport_metadata[protection.KEY]
        self.assertIn('protection.mapping_mismatch', report(right, items[:1], raw).reason_counts())
        with self.assertRaisesRegex(ValueError, 'mapping_mismatch'):
            plan.derive_translation_request(right, items[:1], lineage_suffix='--again')

    def test_sensitive_looking_item_id_survives_child_redaction(self):
        items = [{'id': 'access_token', 'text': 'Hello [name]'}]
        parent = plan.build_translation_plan(
            [{'file_rel_path': 'x.rpy', 'tasks': items}], execution_strategy='sync',
            source_identity={'engine': 'renpy'},
        ).requests[0]
        parent.transport_metadata['api_key'] = 'synthetic-secret'
        child = plan.derive_translation_request(parent, items, lineage_suffix='--retry')
        self.assertEqual(child.transport_metadata['api_key'], '[redacted]')
        self.assertIsInstance(child.transport_metadata[protection.KEY]['items']['access_token'], dict)
        self.assertTrue(report(child, items, response(child, items)).complete)
        plan.derive_translation_request(child, items, lineage_suffix='--again')

    def test_missing_engine_and_v1_maps_fail_closed(self):
        for identity in ({}, {'engine': ''}, {'engine': 'unknown'}):
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, 'unsupported_engine'):
                plan.build_translation_plan([], execution_strategy='sync', source_identity=identity)
        request, items = build()
        raw = response(request, items)
        request.transport_metadata[protection.KEY]['version'] = 1
        self.assertIn('protection.stale_mapping', report(request, items, raw).reason_counts())

    def test_tyrano_literal_brackets_from_actual_catalog_extraction(self):
        from engine_adapters.tyrano import TyranoAdapter, build_translation_snapshot
        from engine_adapters.contracts import ProjectDiscoveryRequest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = root / 'data' / 'scenario'
            catalog = root / 'data' / 'others' / 'lang'
            scenario.mkdir(parents=True)
            catalog.mkdir(parents=True)
            (scenario / 'sample.ks').write_text('Hello \\[note\\]\nHello \\[\n', encoding='utf-8')
            (catalog / 'ch.json').write_text(json.dumps({
                'scenes': {'sample.ks': {'scenario': {'Hello [note]': '', 'Hello [': ''}, 'tag': {}}},
                'charas': {}, 'tags': {'glink': ['text'], 'ptext': ['text']},
            }), encoding='utf-8')
            adapter = TyranoAdapter()
            snapshot = build_translation_snapshot(adapter, ProjectDiscoveryRequest(
                project_root=str(root), localization_root=str(catalog), target_language='ch',
            ))
            occurrences = list(snapshot.occurrences)
            self.assertEqual(len(occurrences), 2)
            items = [core.unit_to_translation_item(item.unit) for item in occurrences]
            self.assertTrue(all(item['tyrano_literal_brackets'] for item in items))
            request = plan.build_translation_plan(
                [{'file_rel_path': 'sample.ks', 'tasks': items}], execution_strategy='sync',
                source_identity={'engine': 'tyrano'},
            ).requests[0]
            raw = {'translations': [{'id': item['id'], 'translation': item['text'].replace('Hello', '你好').replace('note', '说明')}
                                    for item in items]}
            self.assertTrue(report(request, items, raw).complete)
            for occurrence, translated in zip(occurrences, raw['translations']):
                self.assertEqual(adapter.validate_translation(occurrence, translated['translation']).status, 'pass')
        self.assertEqual(structure_rules.spans('Hello [', 'tyrano'), [])
        self.assertEqual(structure_rules.spans(r'Hello \[note\]', 'tyrano'), [])
        with self.assertRaisesRegex(ValueError, 'invalid_structure'):
            structure_rules.spans('Hello [', 'renpy')
        with self.assertRaisesRegex(ValueError, 'structure_order'):
            structure_rules.validate('Hello [r][p]', '你好 [p][r]', 'tyrano')

    def test_repair_refuses_foreign_engine(self):
        import gemini_translate_batch as batch
        with self.assertRaisesRegex(ValueError, 'unsupported_engine'):
            batch.build_repair_request({'engine': 'tyrano', 'items': [{'id': 'one', 'text': '[r][p]'}]})

    def test_round_trip_rules_and_literal_marker(self):
        for engine, source in [
            ('renpy', '{b}{i}Hi [a] [a] [data["x"]!r]{/i}{/b} %s %(name)s'),
            ('renpy', '[[literal {{literal %% \\n\n\r\n\t __RTL_example_0__'),
            ('tyrano', 'Hello [r][font color="red"]world[resetfont]'),
        ]:
            with self.subTest(engine=engine, source=source):
                args = dict(engine=engine, request_id='req', item_id='one')
                view, mapping = protection.protect(source, **args)
                self.assertEqual((view, mapping), protection.protect(source, **args))
                self.assertEqual(source, protection.restore(view, mapping, source=source, **args))

    def test_missing_duplicate_extra_modified_and_wrong_mapping(self):
        args = dict(engine='renpy', request_id='req', item_id='one')
        source = 'Hi [a]'
        view, mapping = protection.protect(source, **args)
        marker = mapping['entries'][0]['marker']
        for changed, code in [
            (view.replace(marker, ''), 'missing_token'),
            (view + marker, 'duplicate_token'),
            (view + '__RTL_foreign_0__', 'extra_token'),
            (view + '__RTL_broken', 'modified_token'),
            (view.replace(marker, marker.lower()), 'missing_token'),
        ]:
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, code):
                protection.restore(changed, mapping, source=source, **args)
        for key, value in [('request_id', 'other'), ('item_id', 'two'), ('engine', 'tyrano')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mapping_mismatch'):
                protection.restore(view, mapping, source=source, **{**args, key: value})
        broken = copy.deepcopy(mapping)
        broken['entries'][0]['value'] = '[injected]'
        with self.assertRaisesRegex(ValueError, 'mapping_mismatch'):
            protection.restore(view, broken, source=source, **args)

    def test_language_order_and_structural_order_are_distinct(self):
        structure_rules.validate('[a] versus [b]', '[b] 对比 [a]', 'renpy')
        for source, target in [
            ('{b}{i}Hi{/i}{/b}', '{b}{i}你好{/b}{/i}'),
            ('%s %d', '%d %s'),
            ('{b}Hi{/b}\n', '\n{b}你好{/b}'),
            ('[a]', '[a] [injected]'),
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                structure_rules.validate(source, target, 'renpy')
        with self.assertRaises(ValueError):
            structure_rules.validate('Hello [r][p]', '你好 [p][r]', 'tyrano')

    def test_sync_batch_model_views_match_and_canonical_source_is_unchanged(self):
        left, items = build('sync')
        right, _ = build('gemini_batch')
        self.assertEqual(left.user_prompt, right.user_prompt)
        self.assertEqual(left.system_instruction, right.system_instruction)
        self.assertEqual(items[0]['text'], '{b}Hello [name]{/b}')
        self.assertIn('__RTL_', left.user_prompt)
        self.assertNotEqual(left.request_id, right.request_id)
        self.assertEqual(plan.recompute_request_fingerprints(left),
                         (left.prompt_fingerprint, left.request_fingerprint))

    def test_partial_restore_never_accepts_damaged_item(self):
        request, items = build()
        payload = response(request, items)
        payload['translations'][0]['translation'] = '你好'
        result = report(request, items, payload)
        self.assertEqual(result.valid_ids, ['two'])
        self.assertEqual(result.retry_ids, ['one'])
        self.assertIn('protection.missing_token', result.reason_counts())

    def test_raw_remains_raw_and_canonical_is_not_restored_twice(self):
        request, items = build()
        raw = response(request, items)
        original = copy.deepcopy(raw)
        restored = report(request, items, raw)
        self.assertTrue(restored.complete)
        self.assertEqual(original, raw)
        self.assertEqual(restored.items[0]['translation'], '{b}你好 [name]{/b}')
        self.assertTrue(report(request, items, restored.to_envelope(), canonical=True).complete)
        self.assertFalse(report(request, items, restored.to_envelope()).complete)

    def test_stale_and_cross_request_maps_are_rejected(self):
        request, items = build()
        raw = response(request, items)
        for field, value in [('version', 999), ('request_id', 'foreign')]:
            changed = copy.deepcopy(request)
            changed.transport_metadata[protection.KEY][field] = value
            self.assertEqual(report(changed, items, raw).valid_ids, [])
        changed = copy.deepcopy(items)
        changed[0]['text'] += '!'
        self.assertEqual(report(request, changed, raw).valid_ids, ['two'])

    def test_derived_retry_binds_new_map_and_only_target_items(self):
        parent, items = build()
        child = plan.derive_translation_request(parent, items[:1], lineage_suffix='--retry')
        maps = child.transport_metadata[protection.KEY]
        self.assertEqual(set(maps['items']), {'one'})
        self.assertEqual(maps['request_id'], child.request_id)
        self.assertTrue(report(child, items[:1], response(child, items[:1])).complete)
        self.assertFalse(report(child, items[:1], response(parent, items[:1])).complete)
        child.transport_metadata[protection.KEY] = parent.transport_metadata[protection.KEY]
        self.assertFalse(report(child, items[:1], response(parent, items[:1])).complete)

    def test_serialized_restart_retains_binding(self):
        request, items = build()
        loaded = plan.TranslationRequest.from_dict(json.loads(json.dumps(request.to_dict())))
        self.assertTrue(report(loaded, items, response(request, items)).complete)
        self.assertEqual(plan.recompute_request_fingerprints(loaded),
                         (loaded.prompt_fingerprint, loaded.request_fingerprint))

    def test_legacy_never_fabricates_map(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'structure_protection' /
                              'legacy.json').read_text(encoding='utf-8'))
        request = plan.TranslationRequest.from_dict(fixture['request'])
        items, payload = fixture['items'], fixture['response']
        self.assertTrue(report(request, items, payload).complete)
        derived = plan.derive_translation_request(request, items[:1], lineage_suffix='--old')
        self.assertNotIn(protection.KEY, derived.transport_metadata)

    def test_batch_rows_restore_before_check_and_keep_raw(self):
        import gemini_translate_batch as batch
        request, items = build('gemini_batch')
        chunk = {**request.to_dict(), 'items': items, 'key': request.chunk_id}
        raw = response(request, items)
        row = {'key': chunk['key'], 'response': {'candidates': [{'content': {
            'parts': [{'text': json.dumps(raw)}]}}]}}
        canonical = batch.canonical_translation_result_row(row, chunk)
        self.assertEqual(canonical['response'], row['response'])
        self.assertEqual(canonical['normalized_response']['translations'][0]['translation'],
                         '{b}你好 [name]{/b}')
        restored = batch.result_items_from_row(row, 'test', items, chunk=chunk)
        self.assertEqual(restored, canonical['normalized_response']['translations'])

    def test_production_sync_accepts_only_restored_items(self):
        from sync_run_service import ProductionSyncBackendAdapter
        request, items = build()
        raw = response(request, items)
        adapter = ProductionSyncBackendAdapter(
            lambda *_args: {'parsed': raw, 'response_payload': {'raw': raw}},
            lambda *_args: items,
        )
        outcome = adapter.send(request.to_dict(), attempt={}, timeout_seconds=1)
        self.assertEqual(outcome.accepted_items['one']['translation'], '{b}你好 [name]{/b}')
        self.assertEqual(outcome.response_payload, {'raw': raw})

    def test_gui_has_stable_chinese_diagnostics(self):
        from gui_qt.check_failures_report import reason_code_label, classify_reason_category
        from gui_qt.user_copy import STRUCTURE_PROTECTION_COPY
        for reason in STRUCTURE_PROTECTION_COPY:
            self.assertNotEqual(reason_code_label(reason), reason)
            self.assertNotEqual(classify_reason_category(reason), 'unknown')

    def test_retry_rejects_stale_source_before_building_child(self):
        request, items = build()
        items[0]['text'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'mapping_mismatch'):
            plan.derive_translation_request(request, items[:1], lineage_suffix='--retry')

    def test_batch_item_retry_uses_own_map_and_merges_canonical_results(self):
        import gemini_translate_batch as batch
        request, items = build('gemini_batch')
        chunk = {**request.to_dict(), 'items': items, 'key': request.chunk_id,
                 'file_rel_path': 'scene.rpy', 'file_path': '', 'line_numbers': []}
        with mock.patch.object(batch, 'RAG_ENABLED', False), mock.patch.object(batch, 'SOURCE_INDEX_ENABLED', False):
            child = batch.build_retry_subchunk(chunk, 0, 1, 1)
        child_request = plan.TranslationRequest.from_dict(child)
        child_payload = response(child_request, items[:1])
        child_row = {'key': child['key'], 'response': {'text': json.dumps(child_payload)}}
        parent_payload = response(request, items)
        parent_payload['translations'][0]['translation'] = '失败'
        parent_row = {'key': chunk['key'], 'response': {'text': json.dumps(parent_payload)}}
        merged, _ = batch.merge_parent_row_with_retry_item_rows(
            parent_row, chunk, [child], {child['key']: child_row},
        )
        self.assertEqual(merged['normalized_response']['translations'][0]['translation'],
                         '{b}你好 [name]{/b}')
        self.assertEqual(merged['response'], parent_row['response'])

    def test_repair_request_and_result_share_mapping(self):
        import gemini_translate_batch as batch
        _, items = build()
        job = {'key': 'repair-one', 'items': items, 'context_past': [], 'context_future': []}
        row = batch.build_repair_request(job)
        self.assertEqual(row['transport_metadata'], job['transport_metadata'])
        request = plan.TranslationRequest.from_dict(job)
        restored = batch.validate_result_contract(response(request, items), core.MODE_TRANSLATION,
                                                 items, request=job)
        self.assertTrue(restored.complete)

    def test_failed_durable_response_is_persisted_without_winners_after_restart(self):
        from tests.test_sync_run_store import bootstrap_run
        from sync_run_contracts import ErrorCategory
        from sync_run_store import SyncRunStore
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = bootstrap_run(Path(tmp))
            store.acquire_lease(owner_token='owner')
            attempt_id = store.prepare_attempt(request_id='req-1', owner_token='owner')
            store.dispatch_attempt(attempt_id=attempt_id, owner_token='owner')
            raw = {'translations': [{'id': 'item-1', 'translation': '__RTL_broken'}]}
            store.record_failure(
                attempt_id=attempt_id, owner_token='owner',
                error_category=ErrorCategory.INVALID_STRUCTURED_RESPONSE,
                error_reason_code='protection.missing_token', terminal=True,
                response_payload=raw, normalized_payload={'translations': []},
                contract_diagnostics={'reason_counts': {'protection.missing_token': 1}},
            )
            reopened = SyncRunStore(Path(tmp), store.run_id)
            attempt = reopened.get_attempt(attempt_id)
            self.assertEqual(json.loads(attempt['response_payload_json']), raw)
            self.assertEqual(json.loads(attempt['normalized_payload_json']), {'translations': []})
            self.assertEqual(reopened.list_item_results(), [])

    def test_real_batch_check_blocks_missing_token_and_force_apply(self):
        import gemini_translate_batch as batch
        from tests.test_batch_golden_corpus import BatchGoldenCorpusTests
        fixture = BatchGoldenCorpusTests()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tl = fixture._copy_fixture_tl(root)
            path = tl / 'chapter01' / 'dialogue.rpy'
            path.write_text(path.read_text(encoding='utf-8').replace(
                'Welcome back, traveler.', '{b}Welcome back, [name].{/b}'), encoding='utf-8')
            original = path.read_bytes()
            previous = fixture._patch_batch_environment(root, tl)
            try:
                manifest_path = Path(batch.create_batch_package(skip_prepare=True))
                manifest = fixture._load_manifest(manifest_path)
                rows = []
                for chunk in manifest['chunks']:
                    mapping = chunk['transport_metadata'][protection.KEY]['items']
                    payload = {'translations': [
                        {'id': item['id'], 'translation': '译文' + ''.join(
                            entry['marker'] for entry in mapping[item['id']]['entries'])}
                        for item in chunk['items']
                    ]}
                    rows.append({'key': chunk['key'], 'response': {'text': json.dumps(payload)}})
                result_path = manifest_path.parent / 'results.jsonl'
                result_path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
                manifest['result_jsonl_path'] = 'results.jsonl'
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                checked = batch.check_results(str(manifest_path))
                self.assertEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
                for row in rows:
                    if '__RTL_' in row['response']['text']:
                        row['response']['text'] = protection.MARKER.sub('', row['response']['text'], count=1)
                        break
                result_path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
                checked = batch.check_results(str(manifest_path))
                self.assertNotEqual(checked['last_check_summary']['writeback_gate']['decision'], 'allow')
                self.assertIn('protection.missing_token', checked['last_check_summary']['reason_counts'])
                with self.assertRaises(SystemExit):
                    batch.apply_results(str(manifest_path), force=True)
                self.assertEqual(path.read_bytes(), original)
                self.assertNotEqual(fixture._load_manifest(manifest_path).get('apply_state'), 'applied')
            finally:
                fixture._restore_batch_environment(previous)


if __name__ == '__main__':
    unittest.main()
