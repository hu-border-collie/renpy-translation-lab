# -*- coding: utf-8 -*-
"""Tests for the execution-strategy-neutral translation plan (issue #346 P1).

The golden fixture under ``tests/fixtures/translation_plan_minimal`` freezes
the plan contract for both execution strategies. Regenerate the expected
snapshots after an intentional contract change with:

    RTP_FIXTURE_UPDATE=1 python -m unittest tests.test_translation_plan -q
"""

import hashlib
import json
import os
import unittest
from pathlib import Path

import translation_core
import translation_plan

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'translation_plan_minimal'
INPUTS_DIR = FIXTURE_DIR / 'inputs'
GAME_DIR = FIXTURE_DIR / 'game'
EXPECTED_DIR = FIXTURE_DIR / 'expected'


def _load_json(name):
    return json.loads((INPUTS_DIR / name).read_text(encoding='utf-8'))


def _load_text(name):
    # Normalize line endings: git checkout may convert fixture text files to
    # CRLF (core.autocrlf) and these bytes are hashed into prompt fingerprints.
    return (INPUTS_DIR / name).read_text(encoding='utf-8').replace('\r\n', '\n')


def _file_digest(rel_path):
    # Normalize line endings: git checkout may convert the fixture to CRLF
    # (core.autocrlf), and the golden identity must stay platform-stable.
    text = (GAME_DIR / rel_path).read_text(encoding='utf-8').replace('\r\n', '\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _fixture_source_identity():
    file_digests = {
        'chapter01/dialogue.rpy': _file_digest('chapter01/dialogue.rpy'),
        'chapter02/strings.rpy': _file_digest('chapter02/strings.rpy'),
    }
    combined = ''.join(file_digests[rel] for rel in sorted(file_digests))
    return {
        'engine': 'renpy',
        'adapter_version': '8.5.2',
        'project_identity_digest': hashlib.sha256(b'translation_plan_minimal').hexdigest(),
        'source_snapshot_fingerprint': hashlib.sha256(combined.encode('utf-8')).hexdigest(),
        'file_digests': file_digests,
    }


def build_fixture_plan(strategy, **overrides):
    glossary = _load_json('glossary.json')
    kwargs = dict(
        execution_strategy=strategy,
        source_identity=_fixture_source_identity(),
        config_snapshot=_load_json('config_snapshot.json'),
        model_profile_snapshot=_load_json('model_profile.json'),
        run_id='fixture-run',
        # Shrink the item budget so chapter01 splits inside and across its
        # two labels: the frozen golden plans then exercise both real block
        # boundaries and in-block CONTEXT BEFORE/AFTER rendered from task
        # dicts (the production D4 default 60/18000 is asserted in
        # ChunkingTests).
        chunk_policy=translation_plan.ChunkPolicy(max_items=3),
        preserve_terms=glossary['preserve_terms'],
        normalize_map=glossary['normalize_map'],
        non_translatable_exact=glossary['non_translatable_exact'],
        macro_setting=_load_text('macro_setting.txt').strip(),
        retrieval_blocks_provider=_load_text('retrieval_blocks.txt'),
        analysis_blocks_provider=_load_text('analysis_blocks.txt'),
    )
    kwargs.update(overrides)
    return translation_plan.build_translation_plan(_load_json('file_jobs.json'), **kwargs)


def _assert_golden(name, payload):
    path = EXPECTED_DIR / name
    if os.environ.get('RTP_FIXTURE_UPDATE') == '1':
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
            newline='\n',
        )
        return
    expected = json.loads(path.read_text(encoding='utf-8'))
    actual = translation_plan.canonical_json(payload)
    if translation_plan.canonical_json(expected) != actual:
        raise AssertionError(
            f'{name} drifted from the frozen golden plan; regenerate with '
            f'RTP_FIXTURE_UPDATE=1 if the change is intentional'
        )


class CanonicalJsonTests(unittest.TestCase):
    def test_sorts_keys_compactly_and_keeps_non_ascii(self):
        rendered = translation_plan.canonical_json({'b': 1, 'a': '中文'})
        self.assertEqual(rendered, '{"a":"中文","b":1}')

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            translation_plan.canonical_json({'value': float('nan')})

    def test_same_content_different_insertion_order_renders_identically(self):
        first = translation_plan.canonical_json({'x': {'b': 2, 'a': 1}, 'y': [3, 1]})
        second = translation_plan.canonical_json({'y': [3, 1], 'x': {'a': 1, 'b': 2}})
        self.assertEqual(first, second)

    def test_canonical_term_sequence_sorts_sets_and_keeps_list_order(self):
        self.assertEqual(
            translation_plan._canonical_term_sequence({'b', 'a'}),
            ['a', 'b'],
        )
        self.assertEqual(
            translation_plan._canonical_term_sequence(frozenset({'z', 'a'})),
            ['a', 'z'],
        )
        self.assertEqual(
            translation_plan._canonical_term_sequence(['x', 'x', 'y']),
            ['x', 'y'],
        )
        self.assertEqual(translation_plan._canonical_term_sequence(None), [])

    def test_bare_string_term_is_one_term_not_characters(self):
        self.assertEqual(
            translation_plan._canonical_term_sequence('Dawn Chorus'),
            ['Dawn Chorus'],
        )
        str_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            preserve_terms='Dawn Chorus',
            non_translatable_exact='B-side',
        )
        list_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            preserve_terms=['Dawn Chorus'],
            non_translatable_exact=['B-side'],
        )
        self.assertEqual(str_build.plan.plan_id, list_build.plan.plan_id)
        dawn_chorus_chunk = next(
            request for request in str_build.requests
            if '- Preserve: Dawn Chorus' in request.user_prompt
        )
        self.assertIn('- Preserve: Dawn Chorus', dawn_chorus_chunk.user_prompt)
        b_side_chunk = next(
            request for request in str_build.requests
            if '- Non-translatable: B-side' in request.user_prompt
        )
        self.assertIn('- Non-translatable: B-side', b_side_chunk.user_prompt)


class RedactionTests(unittest.TestCase):
    def test_replaces_credential_values_recursively(self):
        payload = {
            'api_key': 'sk-secret',
            'nested': {
                'extra_headers': {'Authorization': 'Bearer tok', 'X-Trace': 'keep'},
                'password': 'hunter2',
            },
            'model': 'fixture-model',
        }
        redacted = translation_plan.redact_sensitive(payload)
        self.assertEqual(redacted['api_key'], translation_plan.REDACTED_VALUE)
        self.assertEqual(
            redacted['nested']['extra_headers']['Authorization'],
            translation_plan.REDACTED_VALUE,
        )
        self.assertEqual(redacted['nested']['extra_headers']['X-Trace'], 'keep')
        self.assertEqual(redacted['nested']['password'], translation_plan.REDACTED_VALUE)
        self.assertEqual(redacted['model'], 'fixture-model')

    def test_credential_refs_are_references_and_survive_redaction(self):
        payload = {
            'credential_ref': {'kind': 'env', 'name': 'FIXTURE_MODEL_KEY', 'env_name': ''},
        }
        redacted = translation_plan.redact_sensitive(payload)
        self.assertEqual(redacted['credential_ref']['name'], 'FIXTURE_MODEL_KEY')

    def test_header_spellings_and_generic_token_redact(self):
        payload = {
            'extra_headers': {
                'X-Api-Key': 'secret-a',
                'x-api-key': 'secret-b',
                'token': 'secret-c',
                'AUTH_TOKEN': 'secret-d',
                'X-Token': 'secret-e',
                'x_token': 'secret-f',
                'session-token': 'secret-g',
                'X-Trace-Id': 'keep-me',
            },
            'credentials': {
                'access_key': 'secret-h',
                'private_key': 'secret-i',
                'client_key': 'secret-j',
                'aws_access_key_id': 'secret-k',
                'signing_key': 'secret-l',
            },
            'generation': {'max_output_tokens': 8192, 'temperature': 0.2},
            'usage': {'total_tokens': 12345, 'usages': [{'prompt_tokens': 1}]},
            'ui_hints': {'hotkey': 'F5', 'keyboard': 'qwerty'},
        }
        redacted = translation_plan.redact_sensitive(payload)
        headers = redacted['extra_headers']
        for key in ('X-Api-Key', 'x-api-key', 'token', 'AUTH_TOKEN', 'X-Token', 'x_token', 'session-token'):
            self.assertEqual(headers[key], translation_plan.REDACTED_VALUE, key)
        for key in ('access_key', 'private_key', 'client_key', 'aws_access_key_id', 'signing_key'):
            self.assertEqual(redacted['credentials'][key], translation_plan.REDACTED_VALUE, key)
        self.assertEqual(headers['X-Trace-Id'], 'keep-me')
        # Suffix matching must stay singular: plural token-count keys are
        # legitimate generation/usage data, not credentials.
        self.assertEqual(redacted['generation']['max_output_tokens'], 8192)
        self.assertEqual(redacted['usage']['total_tokens'], 12345)
        self.assertEqual(redacted['usage']['usages'][0]['prompt_tokens'], 1)
        # Non-credential key names that merely contain "key" stay intact.
        self.assertEqual(redacted['ui_hints']['hotkey'], 'F5')
        self.assertEqual(redacted['ui_hints']['keyboard'], 'qwerty')


class ChunkingTests(unittest.TestCase):
    @staticmethod
    def _reference_ranges(tasks, max_items, max_chars):
        # The pre-#346 batch algorithm, kept here as an oracle for semantics.
        total = len(tasks)
        start = 0
        while start < total:
            end = start
            current_chars = 0
            while end < total and (end - start) < max_items:
                item_chars = len(tasks[end].get('text', ''))
                if end > start and current_chars + item_chars > max_chars:
                    break
                current_chars += item_chars
                end += 1
            if end == start:
                end = start + 1
            yield start, end
            start = end

    def test_defaults_are_d4_values(self):
        policy = translation_plan.ChunkPolicy()
        self.assertEqual(policy.max_items, 60)
        self.assertEqual(policy.max_chars, 18000)

    def test_matches_reference_algorithm_for_varied_sizes(self):
        tasks = [{'text': 'x' * (7 * i + 3)} for i in range(40)]
        for max_items, max_chars in ((60, 18000), (5, 40), (3, 1000), (2, 10)):
            shared = list(translation_core.iter_translation_chunk_ranges(tasks, max_items, max_chars))
            reference = list(self._reference_ranges(tasks, max_items, max_chars))
            self.assertEqual(shared, reference, (max_items, max_chars))

    def test_single_oversized_item_forms_its_own_chunk(self):
        tasks = [{'text': 'a' * 50}, {'text': 'b' * 5}]
        self.assertEqual(
            list(translation_core.iter_translation_chunk_ranges(tasks, 60, 10)),
            [(0, 1), (1, 2)],
        )

    def test_chunk_id_keeps_legacy_batch_format(self):
        chunk_id = translation_plan.build_chunk_id('chapter01/dialogue.rpy', 1)
        prefix = hashlib.sha1('chapter01/dialogue.rpy'.encode('utf-8')).hexdigest()[:10]
        self.assertEqual(chunk_id, f'{prefix}-00001')


class StableIdTests(unittest.TestCase):
    def test_request_id_is_content_derived_and_stable(self):
        first = translation_plan.build_request_id('plan', 'chunk', ['a', 'b'])
        second = translation_plan.build_request_id('plan', 'chunk', ['a', 'b'])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertNotEqual(first, translation_plan.build_request_id('plan', 'chunk', ['a', 'c']))
        self.assertNotEqual(first, translation_plan.build_request_id('plan2', 'chunk', ['a', 'b']))


class LocalContextWindowTests(unittest.TestCase):
    @staticmethod
    def _task(block, text):
        return {'text': text, 'block_name': block}

    def test_stops_at_block_boundary(self):
        tasks = [self._task('block_a', 'a1'), self._task('block_b', 'b1'), self._task('block_b', 'b2')]
        window, diagnostics = translation_plan.build_local_context_window(tasks, 2, 3, 30, 10)
        self.assertEqual([item['text'] for item in window.before], ['b1'])
        self.assertTrue(diagnostics['block_bounded_before'])
        self.assertFalse(diagnostics['context_truncated'])

    def test_budget_truncation_is_reported(self):
        tasks = [self._task('block_b', f'b{i}') for i in range(5)]
        window, diagnostics = translation_plan.build_local_context_window(tasks, 3, 5, 2, 10)
        self.assertEqual([item['text'] for item in window.before], ['b1', 'b2'])
        self.assertTrue(diagnostics['context_truncated'])
        self.assertFalse(diagnostics['block_bounded_before'])
        self.assertEqual(diagnostics['context_before_items'], 2)

    def test_empty_block_name_never_bounds(self):
        tasks = [{'text': 's1', 'block_name': ''}, {'text': 's2', 'block_name': ''}]
        window, diagnostics = translation_plan.build_local_context_window(tasks, 1, 2, 30, 10)
        self.assertEqual([item['text'] for item in window.before], ['s1'])
        self.assertFalse(diagnostics['block_bounded_before'])
        self.assertFalse(diagnostics['block_bounded_after'])


class LexicalGlossaryTests(unittest.TestCase):
    def test_hit_order_is_normalize_preserve_non_translatable_with_dedup(self):
        hits = translation_plan.retrieve_lexical_glossary_hits(
            [{'text': 'The setlist and Mrs. Parker kept the B-side and the setlist.'}],
            normalize_map={'setlist': '曲目单'},
            preserve_terms=['Mrs. Parker', 'setlist'],
            non_translatable_exact=['B-side'],
        )
        self.assertEqual(
            hits,
            [
                {'source': 'setlist', 'target': '曲目单', 'kind': 'normalize'},
                {'source': 'Mrs. Parker', 'target': 'Mrs. Parker', 'kind': 'preserve'},
                {'source': 'B-side', 'target': '', 'kind': 'non_translatable'},
            ],
        )

    def test_empty_text_returns_no_hits(self):
        self.assertEqual(
            translation_plan.retrieve_lexical_glossary_hits([{'text': ''}], {'a': 'b'}),
            [],
        )

    def test_render_uses_issue338_wording(self):
        text = translation_plan.render_lexical_glossary_text([
            {'source': 'setlist', 'target': '曲目单', 'kind': 'normalize'},
            {'source': 'Dawn Chorus', 'target': 'Dawn Chorus', 'kind': 'preserve'},
            {'source': 'B-side', 'target': '', 'kind': 'non_translatable'},
        ])
        self.assertEqual(
            text.splitlines(),
            [
                '- Existing mapping: setlist -> 曲目单',
                '- Preserve: Dawn Chorus',
                '- Non-translatable: B-side',
            ],
        )


class PlanBuildTests(unittest.TestCase):
    def test_build_is_deterministic_across_invocations(self):
        first = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        second = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        self.assertEqual(
            translation_plan.canonical_json(first.plan.to_dict()),
            translation_plan.canonical_json(second.plan.to_dict()),
        )
        self.assertEqual(
            [request.to_dict() for request in first.requests],
            [request.to_dict() for request in second.requests],
        )

    def test_run_id_is_audit_only_and_excluded_from_fingerprints(self):
        first = build_fixture_plan(translation_plan.STRATEGY_SYNC, run_id='run-one')
        second = build_fixture_plan(translation_plan.STRATEGY_SYNC, run_id='run-two')
        self.assertEqual(first.plan.plan_id, second.plan.plan_id)
        self.assertEqual(first.plan.plan_fingerprint, second.plan.plan_fingerprint)
        self.assertEqual(first.plan.run_id, 'run-one')
        self.assertEqual(
            [request.prompt_fingerprint for request in first.requests],
            [request.prompt_fingerprint for request in second.requests],
        )

    def test_strategy_changes_identity_but_not_semantic_contract(self):
        sync = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        batch = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        self.assertNotEqual(sync.plan.plan_id, batch.plan.plan_id)
        self.assertEqual(len(sync.requests), len(batch.requests))
        for sync_request, batch_request in zip(sync.requests, batch.requests):
            self.assertEqual(sync_request.chunk_id, batch_request.chunk_id)
            self.assertEqual(
                sync_request.prompt_fingerprint,
                batch_request.prompt_fingerprint,
                'sync and batch must show the model the same semantic contract',
            )
            self.assertNotEqual(
                sync_request.request_fingerprint,
                batch_request.request_fingerprint,
                'transport metadata is expected to differ between strategies',
            )

    def test_transport_metadata_defaults_per_strategy(self):
        sync = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        batch = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        self.assertEqual(sync.requests[0].transport_metadata.get('sync_stage'), 'initial_translation')
        self.assertEqual(batch.requests[0].transport_metadata.get('batch_key'), batch.requests[0].chunk_id)

    def test_lexical_glossary_injected_without_retrieval(self):
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider='',
            analysis_blocks_provider='',
        )
        request = build.requests[0]
        self.assertIn('Existing glossary entries:', request.user_prompt)
        self.assertIn('setlist', request.user_prompt)
        project_layer = next(
            layer for layer in request.context_assembly['layers'] if layer['layer'] == 'project'
        )
        self.assertTrue(project_layer['diagnostics']['rag_independent'])
        self.assertGreater(project_layer['diagnostics']['lexical_glossary_hits'], 0)

    def test_generation_config_defaults_to_d6_baseline(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        self.assertEqual(build.requests[0].generation_config, {'temperature': 0.2})

    def test_invalid_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            build_fixture_plan('courier_pigeon')

    def test_expected_ids_and_schema_match_chunk_units(self):
        build = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        request = build.requests[0]
        self.assertTrue(request.expected_ids)
        translations = request.response_schema['properties']['translations']
        self.assertEqual(translations['minItems'], len(request.expected_ids))
        self.assertEqual(translations['maxItems'], len(request.expected_ids))
        self.assertEqual(request.plan_id, build.plan.plan_id)

    def test_plan_snapshot_redacts_credential_shaped_values(self):
        profile = _load_json('model_profile.json')
        profile['extra_headers'] = {'Authorization': 'Bearer super-secret'}
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            model_profile_snapshot=profile,
        )
        rendered = translation_plan.canonical_json(build.plan.to_dict())
        self.assertNotIn('super-secret', rendered)
        self.assertIn(translation_plan.REDACTED_VALUE, rendered)

    def test_request_redacts_credential_shaped_transport_metadata(self):
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            transport_metadata={
                'sync_stage': 'initial_translation',
                'Authorization': 'Bearer transport-secret',
                'x-api-key': 'transport-key',
            },
            generation_config={'temperature': 0.2, 'api_key': 'gen-secret'},
        )
        request = build.requests[0]
        self.assertEqual(
            request.transport_metadata['Authorization'],
            translation_plan.REDACTED_VALUE,
        )
        self.assertEqual(
            request.transport_metadata['x-api-key'],
            translation_plan.REDACTED_VALUE,
        )
        self.assertEqual(request.transport_metadata['sync_stage'], 'initial_translation')
        self.assertEqual(request.generation_config['api_key'], translation_plan.REDACTED_VALUE)
        self.assertEqual(request.generation_config['temperature'], 0.2)
        rendered = translation_plan.canonical_json(request.to_dict())
        self.assertNotIn('transport-secret', rendered)
        self.assertNotIn('gen-secret', rendered)

    def test_retrieval_budget_truncation_reaches_the_user_prompt(self):
        oversized = 'LOCKED TERMS:\n' + ('x' * 5000)
        baseline = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        oversized_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider=oversized,
        )
        request = oversized_build.requests[0]
        retrieval = next(
            layer['char_used']
            for layer in request.context_assembly['layers']
            if layer['layer'] == translation_plan.CONTEXT_LAYER_RETRIEVAL
        )
        truncated = next(
            layer['truncated']
            for layer in request.context_assembly['layers']
            if layer['layer'] == translation_plan.CONTEXT_LAYER_RETRIEVAL
        )
        self.assertTrue(truncated)
        self.assertEqual(retrieval, 220 + 1200)
        # The embedded prompt carries the budgeted text, not the provider's
        # full blob, and the truncation moves the fingerprint.
        self.assertIn('x' * 1400, request.user_prompt)
        self.assertNotIn('x' * 1500, request.user_prompt)
        self.assertNotEqual(request.prompt_fingerprint, baseline.requests[0].prompt_fingerprint)
        self.assertNotEqual(
            oversized_build.plan.plan_fingerprint,
            baseline.plan.plan_fingerprint,
        )

    def test_unordered_term_collections_yield_stable_identity(self):
        list_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            preserve_terms=['Dawn Chorus', 'Mrs. Parker'],
            non_translatable_exact=['B-side'],
        )
        set_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            preserve_terms={'Dawn Chorus', 'Mrs. Parker'},
            non_translatable_exact=frozenset({'B-side'}),
        )
        self.assertEqual(list_build.plan.plan_id, set_build.plan.plan_id)
        self.assertEqual(
            list_build.requests[0].user_prompt,
            set_build.requests[0].user_prompt,
        )

    def test_normalize_map_insertion_order_does_not_change_identity(self):
        tasks = [{'id': 'u1', 'text': 'The setlist and the encore were great.'}]
        first = translation_plan.retrieve_lexical_glossary_hits(
            tasks, normalize_map={'setlist': '曲目单', 'encore': '返场'},
        )
        second = translation_plan.retrieve_lexical_glossary_hits(
            tasks, normalize_map={'encore': '返场', 'setlist': '曲目单'},
        )
        self.assertEqual(first, second)
        self.assertEqual(
            [hit['source'] for hit in first],
            ['encore', 'setlist'],
        )
        first_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            normalize_map={'setlist': '曲目单', 'encore': '返场'},
        )
        second_build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            normalize_map={'encore': '返场', 'setlist': '曲目单'},
        )
        self.assertEqual(first_build.plan.plan_id, second_build.plan.plan_id)
        self.assertEqual(
            first_build.requests[0].user_prompt,
            second_build.requests[0].user_prompt,
        )

    def test_analysis_layer_budget_truncates_like_retrieval(self):
        oversized = 'PROJECT BRIEF:\n' + ('y' * 8000)
        baseline = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            analysis_blocks_provider=oversized,
        )
        request = build.requests[0]
        analysis = next(
            layer
            for layer in request.context_assembly['layers']
            if layer['layer'] == translation_plan.CONTEXT_LAYER_ANALYSIS
        )
        self.assertEqual(analysis['char_limit'], 4000)
        self.assertTrue(analysis['truncated'])
        self.assertEqual(analysis['char_used'], 4000)
        self.assertIn('y' * 3900, request.user_prompt)
        self.assertNotIn('y' * 4100, request.user_prompt)
        self.assertNotEqual(request.prompt_fingerprint, baseline.requests[0].prompt_fingerprint)

    def test_layer_sections_never_glue_after_truncation(self):
        # Retrieval truncated mid-line by the budget backstop must still end
        # on a clean section boundary before the analysis header.
        oversized_retrieval = 'LOCKED TERMS:\n' + ('z' * 5000)
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider=oversized_retrieval,
            analysis_blocks_provider='PROJECT BRIEF:\nclean brief.\n\n',
        )
        user_prompt = build.requests[0].user_prompt
        self.assertIn('z' * 1400 + '\n\nPROJECT BRIEF:', user_prompt)
        self.assertIn('\n\nPROJECT BRIEF:\nclean brief.', user_prompt)

    def test_block_layout_participates_in_plan_identity(self):
        jobs = _load_json('file_jobs.json')
        for task in jobs[0]['tasks']:
            if task['block_name'].endswith('chapter01_start'):
                task['block_name'] = task['block_name'].replace(
                    'chapter01_start', 'chapter01_renamed'
                )
        identity = _fixture_source_identity()
        glossary = _load_json('glossary.json')
        renamed_build = translation_plan.build_translation_plan(
            jobs,
            execution_strategy=translation_plan.STRATEGY_SYNC,
            source_identity=identity,
            config_snapshot=_load_json('config_snapshot.json'),
            model_profile_snapshot=_load_json('model_profile.json'),
            preserve_terms=glossary['preserve_terms'],
            normalize_map=glossary['normalize_map'],
            non_translatable_exact=glossary['non_translatable_exact'],
        )
        baseline = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        self.assertNotEqual(renamed_build.plan.plan_id, baseline.plan.plan_id)

    def test_retrieved_content_changes_prompts_but_not_plan_id(self):
        first = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider='LOCKED TERMS:\n- alpha\n\n',
            analysis_blocks_provider='PROJECT BRIEF:\nfirst brief.\n\n',
        )
        second = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider='LOCKED TERMS:\n- beta\n\n',
            analysis_blocks_provider='PROJECT BRIEF:\nsecond brief.\n\n',
        )
        self.assertEqual(first.plan.plan_id, second.plan.plan_id)
        self.assertNotEqual(first.plan.plan_fingerprint, second.plan.plan_fingerprint)
        for first_request, second_request in zip(first.requests, second.requests):
            self.assertNotEqual(
                first_request.prompt_fingerprint,
                second_request.prompt_fingerprint,
            )


class ContextAssemblyTests(unittest.TestCase):
    def test_layers_are_ordered_by_rank(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        assembly = build.requests[0].context_assembly
        ranks = [layer['rank'] for layer in assembly['layers']]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(
            [layer['layer'] for layer in assembly['layers']],
            [
                translation_plan.CONTEXT_LAYER_REQUIRED,
                translation_plan.CONTEXT_LAYER_LOCAL,
                translation_plan.CONTEXT_LAYER_PROJECT,
                translation_plan.CONTEXT_LAYER_RETRIEVAL,
                translation_plan.CONTEXT_LAYER_ANALYSIS,
            ],
        )
        self.assertEqual(
            assembly['total_char_used'],
            sum(layer['char_used'] for layer in assembly['layers']),
        )

    def test_retrieval_layer_budget_truncates_deterministically(self):
        long_text = 'LOCKED TERMS:\n' + ('x' * 5000)
        chunk_input = translation_plan.ChunkContextInput(
            target_units=[],
            retrieval_blocks_text=long_text,
        )
        assembly = translation_plan.assemble_context_layers(chunk_input, translation_plan.ContextPolicy())
        retrieval = next(
            layer for layer in assembly.layers if layer.layer == translation_plan.CONTEXT_LAYER_RETRIEVAL
        )
        self.assertEqual(retrieval.char_limit, 220 + 1200)
        self.assertTrue(retrieval.truncated)
        self.assertEqual(retrieval.char_used, retrieval.char_limit)

    def test_aggregate_budget_preserves_priority_and_explains_trimming(self):
        chunk_input = translation_plan.ChunkContextInput(
            target_units=[],
            macro_setting='Keep names stable.',
            retrieval_blocks_text='R' * 40,
            analysis_blocks_text='A' * 40,
        )
        baseline = translation_plan.assemble_context_layers(chunk_input)
        mandatory_used = sum(
            layer.char_used
            for layer in baseline.layers
            if layer.layer in {'required', 'local', 'project'}
        )
        assembly = translation_plan.assemble_context_layers(
            chunk_input,
            translation_plan.ContextPolicy(total_char_limit=mandatory_used + 20),
        )
        by_layer = {layer.layer: layer for layer in assembly.layers}
        self.assertEqual(by_layer['retrieval'].char_used, 20)
        self.assertEqual(by_layer['analysis'].char_used, 0)
        self.assertTrue(by_layer['retrieval'].truncated)
        self.assertTrue(by_layer['analysis'].truncated)
        self.assertEqual(
            [item['layer'] for item in assembly.dropped],
            ['retrieval', 'analysis'],
        )
        self.assertTrue(all(
            item['reason'] == 'aggregate_budget_exceeded'
            for item in assembly.dropped
        ))

    def test_duplicate_layer_texts_are_dropped_with_reason(self):
        chunk_input = translation_plan.ChunkContextInput(
            target_units=[],
            analysis_blocks_text='',
        )
        assembly = translation_plan.assemble_context_layers(chunk_input)
        # With no units and no providers, the project layer renders the first
        # empty text (kept) and retrieval/analysis duplicate it (dropped).
        # Removing the duplicate-drop logic must fail these assertions.
        self.assertEqual(len(assembly.layers), 3)
        self.assertTrue(assembly.dropped)
        self.assertEqual(
            {item['layer'] for item in assembly.dropped},
            {translation_plan.CONTEXT_LAYER_RETRIEVAL, translation_plan.CONTEXT_LAYER_ANALYSIS},
        )
        for dropped in assembly.dropped:
            self.assertEqual(dropped['reason'], 'duplicate_text')
            self.assertEqual(dropped['char_used'], 0)

    def test_local_layer_records_block_bounded_diagnostics(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        dialogue_chunks = [
            chunk for chunk in build.plan.chunks
            if chunk.file_rel_path == 'chapter01/dialogue.rpy'
        ]
        # max_items=3 splits chapter01 into [0:3] and [3:6] (chunking is
        # item/char based and does not see blocks): the first chunk's after
        # window stops at the hall block boundary but keeps the fourth
        # start-block task, and the second chunk keeps the full in-block
        # before window.
        self.assertEqual(len(dialogue_chunks), 2)
        by_spec = [chunk.context_window_spec for chunk in dialogue_chunks]
        self.assertEqual(by_spec[0]['context_before_items'], 0)
        self.assertEqual(by_spec[0]['context_after_items'], 1)
        self.assertEqual(by_spec[0]['block_bounded_after'], True)
        self.assertEqual(by_spec[1]['context_before_items'], 3)
        self.assertEqual(by_spec[1]['context_after_items'], 0)
        self.assertFalse(by_spec[1]['block_bounded_before'])

    def test_golden_prompt_embeds_in_block_context_from_task_dicts(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        # The first chapter01 chunk ([0:3]) carries the fourth start task as
        # CONTEXT AFTER and the second ([3:6]) carries the first three as
        # CONTEXT BEFORE — rendered from the task dicts with speaker labels
        # ("Name (id)"), so a regression in dict context rendering must move
        # the frozen prompt fingerprint.
        first = next(
            request for request in build.requests
            if request.chunk_id == translation_plan.build_chunk_id('chapter01/dialogue.rpy', 1)
        )
        self.assertIn(
            'CONTEXT AFTER:\n- Parker (p): Relax. I kept the B-side as the encore, like you asked.',
            first.user_prompt,
        )
        second = next(
            request for request in build.requests
            if request.chunk_id == translation_plan.build_chunk_id('chapter01/dialogue.rpy', 2)
        )
        self.assertIn(
            'CONTEXT BEFORE:\n- Gil (g): Hey [Gil_name!t], did you finish the Dawn Chorus setlist?',
            second.user_prompt,
        )
        self.assertIn('- Gil (g): Mrs. Parker will flip', second.user_prompt)
        self.assertIn('CONTEXT AFTER:\n(none)', second.user_prompt)

    def test_file_jobs_lines_index_source_lines_containing_task_text(self):
        jobs = _load_json('file_jobs.json')
        for job in jobs:
            source_lines = (GAME_DIR / job['file_rel_path']).read_text(
                encoding='utf-8'
            ).replace('\r\n', '\n').splitlines()
            for task in job['tasks']:
                self.assertLess(
                    task['line'], len(source_lines),
                    f"{task['id']} line {task['line']} out of range",
                )
                self.assertIn(
                    task['text'], source_lines[task['line']],
                    f"{task['id']} line {task['line']} does not contain its text",
                )

    def test_crlf_normalization_keeps_frozen_fingerprints_stable(self):
        # A CRLF checkout of the multi-line reference inputs must not move
        # the frozen fingerprints: _load_text normalizes line endings before
        # the text is hashed into the prompt.
        baseline = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        crlf_retrieval = _load_text('retrieval_blocks.txt').replace('\n', '\r\n')
        crlf_analysis = _load_text('analysis_blocks.txt').replace('\n', '\r\n')
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider=crlf_retrieval,
            analysis_blocks_provider=crlf_analysis,
        )
        self.assertEqual(
            [request.prompt_fingerprint for request in build.requests],
            [request.prompt_fingerprint for request in baseline.requests],
        )


class DerivedRequestTests(unittest.TestCase):
    def test_d7_child_is_deterministic_and_does_not_mutate_parent_plan(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        parent = build.requests[0]
        parent.capability_requirements.update({
            'context_budget_tokens': 1,
            'estimate_method': 'stale-parent-estimate',
            'response_format': 'named_translation_object',
            'provider_feature': {'mode': 'strict'},
        })
        parent_plan = translation_plan.canonical_json(build.plan.to_dict())
        parent_request = translation_plan.canonical_json(parent.to_dict())
        item = _load_json('file_jobs.json')[0]['tasks'][0]
        kwargs = dict(
            lineage_suffix='--L',
            file_rel_path='chapter01/dialogue.rpy',
            context_window=translation_core.ContextWindow(),
            preserve_terms=_load_json('glossary.json')['preserve_terms'],
            normalize_map=_load_json('glossary.json')['normalize_map'],
            non_translatable_exact=(
                _load_json('glossary.json')['non_translatable_exact']
            ),
            macro_setting=_load_text('macro_setting.txt').strip(),
            retrieval_blocks_text=_load_text('retrieval_blocks.txt'),
            analysis_blocks_text=_load_text('analysis_blocks.txt'),
            lineage_kind='split',
        )

        first = translation_plan.derive_translation_request(
            parent, [item], **kwargs
        )
        second = translation_plan.derive_translation_request(
            parent, [item], **kwargs
        )
        crlf_kwargs = dict(kwargs)
        crlf_kwargs['retrieval_blocks_text'] = kwargs[
            'retrieval_blocks_text'
        ].replace('\n', '\r\n')
        crlf_kwargs['analysis_blocks_text'] = kwargs[
            'analysis_blocks_text'
        ].replace('\n', '\r\n')
        crlf = translation_plan.derive_translation_request(
            parent, [item], **crlf_kwargs
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_dict(), crlf.to_dict())
        self.assertEqual(first.request_id, f'{parent.request_id}--L')
        self.assertEqual(first.plan_id, parent.plan_id)
        self.assertEqual(first.expected_ids, [str(item['id'])])
        self.assertEqual(
            first.transport_metadata['retry_parent_request_id'],
            parent.request_id,
        )
        self.assertEqual(
            first.capability_requirements['response_format'],
            'named_translation_object',
        )
        self.assertEqual(
            first.capability_requirements['provider_feature'],
            {'mode': 'strict'},
        )
        self.assertNotEqual(
            first.capability_requirements['context_budget_tokens'],
            1,
        )
        self.assertEqual(
            first.capability_requirements['estimate_method'],
            translation_plan.CONTEXT_TOKEN_ESTIMATE_METHOD,
        )
        self.assertEqual(
            translation_plan.TranslationRequest.from_dict(
                first.to_dict()
            ).capability_requirements,
            first.capability_requirements,
        )
        self.assertEqual(
            translation_plan.canonical_json(build.plan.to_dict()),
            parent_plan,
        )
        self.assertEqual(
            translation_plan.canonical_json(parent.to_dict()),
            parent_request,
        )


class GoldenPlanTests(unittest.TestCase):
    def test_normalized_sync_and_batch_requests_have_empty_readable_diff(self):
        sync = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        batch = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        report = translation_plan.plan_diff(sync.requests, batch.requests)
        self.assertTrue(report['equivalent'], translation_plan.format_plan_diff(report))
        self.assertEqual(
            translation_plan.format_plan_diff(report),
            'TranslationPlan semantic requests are equivalent (3 requests).',
        )
        for sync_request, batch_request in zip(sync.requests, batch.requests):
            self.assertEqual(
                translation_plan.canonical_semantic_request(sync_request),
                translation_plan.canonical_semantic_request(batch_request),
            )
            self.assertEqual(
                sync_request.prompt_fingerprint,
                batch_request.prompt_fingerprint,
            )

    def test_plan_diff_names_chunk_and_changed_semantic_field(self):
        sync = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        batch = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        batch.requests[0].user_prompt += '\nchanged'
        report = translation_plan.plan_diff(sync.requests, batch.requests)
        rendered = translation_plan.format_plan_diff(report)
        self.assertFalse(report['equivalent'])
        self.assertIn(sync.requests[0].chunk_id, rendered)
        self.assertIn('user_prompt', rendered)
        self.assertIn('+  "user_prompt"', rendered)

    def test_plan_diff_reports_missing_request_without_json_error(self):
        sync = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        report = translation_plan.plan_diff(sync.requests, sync.requests[:-1])
        rendered = translation_plan.format_plan_diff(report)
        self.assertFalse(report['equivalent'])
        self.assertIn('<missing>', rendered)
        self.assertIn('missing_request', rendered)

    def test_empty_optional_duplicate_is_not_a_material_drop(self):
        build = build_fixture_plan(
            translation_plan.STRATEGY_SYNC,
            retrieval_blocks_provider='',
            analysis_blocks_provider='',
        )
        diagnostics = translation_plan.summarize_request_diagnostics(
            build.plan.request_summaries
        )
        self.assertEqual(diagnostics['context_dropped_entries'], 0)

    def test_sync_plan_matches_frozen_golden(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        _assert_golden('plan.sync.json', build.plan.to_dict())

    def test_gemini_batch_plan_matches_frozen_golden(self):
        build = build_fixture_plan(translation_plan.STRATEGY_GEMINI_BATCH)
        _assert_golden('plan.gemini_batch.json', build.plan.to_dict())

    def test_golden_requests_round_trip_through_from_dict(self):
        build = build_fixture_plan(translation_plan.STRATEGY_SYNC)
        for request in build.requests:
            clone = translation_plan.TranslationRequest.from_dict(request.to_dict())
            self.assertEqual(clone.prompt_fingerprint, request.prompt_fingerprint)
            self.assertEqual(clone.request_fingerprint, request.request_fingerprint)
        plan_clone = translation_plan.TranslationPlan.from_dict(build.plan.to_dict())
        self.assertEqual(plan_clone.plan_fingerprint, build.plan.plan_fingerprint)


class PlanPurityTests(unittest.TestCase):
    def test_module_dependency_tree_stays_pure(self):
        # A fresh interpreter must not pull SDKs, network clients, or the
        # executor runtimes through translation_plan's import graph.
        import subprocess
        import sys

        repo_root = str(Path(__file__).resolve().parent.parent)
        code = (
            'import sys, json; import translation_plan; '
            "print(json.dumps(sorted(name for name in sys.modules if name in "
            "('google.genai', 'litellm', 'requests', 'urllib.request', "
            "'translator_runtime', 'gemini_translate_batch'))))"
        )
        result = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_fixture_files_contain_no_credential_values(self):
        patterns = ('sk-', 'Bearer ', 'API_KEY=', 'glsa_', 'AIza')
        for path in sorted(FIXTURE_DIR.rglob('*')):
            if not path.is_file():
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            for pattern in patterns:
                self.assertNotIn(pattern, text, f'{path} contains {pattern!r}')


if __name__ == '__main__':
    unittest.main()
