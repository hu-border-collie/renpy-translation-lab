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



class TranslationCoreRegressionTests(unittest.TestCase):
    def test_legacy_items_round_trip_through_translation_unit(self):
        sync_task = {
            'id': 'script.rpy:0:4',
            'text': 'Hello Alice',
            'line': 0,
            'start': 4,
            'end': 17,
            'prefix': '',
            'quote': '"',
            'speaker_id': 'e',
            'speaker_name': 'Eileen',
            'progress_entry': 'task:0:4',
        }
        sync_unit = translation_core.unit_from_sync_task(sync_task, file_rel_path='script.rpy')
        self.assertEqual(sync_unit.mode, translation_core.MODE_TRANSLATION)
        self.assertEqual(sync_unit.source_text, 'Hello Alice')
        self.assertEqual(sync_unit.line_number, 0)
        self.assertEqual(sync_unit.display_line_number, 1)
        self.assertEqual(sync_unit.progress_entry, 'task:0:4')
        self.assertEqual(
            translation_core.unit_to_translation_item(sync_unit),
            {
                'id': 'script.rpy:0:4',
                'text': 'Hello Alice',
                'line': 0,
                'start': 4,
                'end': 17,
                'prefix': '',
                'quote': '"',
                'speaker_id': 'e',
                'speaker': 'e',
                'speaker_name': 'Eileen',
            },
        )

        revision_item = {
            'id': 'script.rpy:1:4:revision:0',
            'source': 'Open the Void Gate',
            'current_translation': '\u6253\u5f00\u95e8',
            'file_rel_path': 'script.rpy',
            'line': 1,
            'line_number': 2,
            'start': 8,
            'end': 14,
            'prefix': '',
            'quote': '"',
        }
        revision_unit = translation_core.unit_from_revision_item(revision_item)
        self.assertEqual(revision_unit.mode, translation_core.MODE_REVISION)
        self.assertEqual(revision_unit.source_text, 'Open the Void Gate')
        self.assertEqual(
            translation_core.unit_to_revision_item(revision_unit)['current_translation'],
            '\u6253\u5f00\u95e8',
        )

        keyword_item = {
            'id': 'script.rpy:2:keyword:0',
            'text': 'Void Gate',
            'file_rel_path': 'script.rpy',
            'line_number': 2,
            'translation_line_number': 3,
        }
        keyword_unit = translation_core.unit_from_keyword_item(keyword_item)
        keyword_legacy = translation_core.unit_to_keyword_item(keyword_unit)
        self.assertEqual(keyword_unit.mode, translation_core.MODE_KEYWORD_EXTRACTION)
        self.assertEqual(keyword_legacy['translation_line_number'], 3)

    def test_prompt_wrappers_use_core_schema_for_all_modes(self):
        translation_prompt = batch_mod.build_user_prompt(
            [{'id': 'script.rpy:0:0', 'text': 'Before line', 'speaker_id': 'n', 'speaker_name': 'Noah'}],
            [{'id': 'script.rpy:0:4', 'text': 'Hello Alice', 'speaker_id': 'e', 'speaker_name': 'Eileen'}],
            ['After line'],
            glossary_hits=[{'source': 'Alice', 'target': 'Alice'}],
        )
        translation_schema = batch_mod.build_response_json_schema(
            [{'id': 'script.rpy:0:4', 'text': 'Hello Alice'}]
        )
        self.assertIn('LOCKED TERMS', translation_prompt)
        self.assertIn('TARGET', translation_prompt)
        self.assertIn('script.rpy:0:4', translation_prompt)
        self.assertIn('Noah (n): Before line', translation_prompt)
        self.assertIn('"speaker_id":"e"', translation_prompt)
        self.assertIn('"speaker_name":"Eileen"', translation_prompt)
        translation_items = translation_schema['properties']['translations']['items']
        self.assertEqual(translation_schema['required'], ['translations'])
        self.assertEqual(translation_items['required'], ['id', 'translation'])
        self.assertNotIn('enum', translation_items['properties']['id'])

        revision_chunk = {
            'file_rel_path': 'script.rpy',
            'items': [
                {
                    'id': 'script.rpy:1:4:revision:0',
                    'source': 'Open the Void Gate',
                    'current_translation': '\u6253\u5f00\u95e8',
                    'file_rel_path': 'script.rpy',
                    'line': 1,
                    'line_number': 2,
                    'start': 8,
                    'end': 14,
                    'prefix': '',
                    'quote': '"',
                }
            ],
        }
        revision_prompt = batch_mod.build_revision_user_prompt(revision_chunk)
        revision_schema = batch_mod.build_revision_response_json_schema(revision_chunk['items'])
        self.assertIn('current_translation', revision_prompt)
        self.assertIn('should_update', revision_prompt)
        self.assertEqual(
            revision_schema['properties']['revisions']['items']['required'],
            ['id', 'should_update', 'revised_translation', 'reason'],
        )
        self.assertNotIn(
            'enum',
            revision_schema['properties']['revisions']['items']['properties']['id'],
        )

        keyword_prompt = batch_mod.build_keyword_user_prompt(
            [{'id': 'script.rpy:2:keyword:0', 'text': 'Void Gate', 'line_number': 2}]
        )
        keyword_schema = batch_mod.build_keyword_response_json_schema(5)
        self.assertIn('source_item_ids', keyword_prompt)
        self.assertIn('chunk_summary', keyword_prompt)
        self.assertEqual(keyword_schema['properties']['candidates']['maxItems'], 5)
        candidate_schema = keyword_schema['properties']['candidates']['items']
        self.assertNotIn('enum', candidate_schema['properties']['category'])
        source_ids_schema = candidate_schema['properties']['source_item_ids']
        self.assertEqual(source_ids_schema['minItems'], 1)
        self.assertIn('chunk_summary', keyword_schema['required'])

    def test_translation_contract_reports_stable_id_and_field_reasons(self):
        expected = [
            {'id': 'a', 'text': 'Hello'},
            {'id': 'b', 'text': 'World'},
            {'id': 'c', 'text': 'Again'},
            {'id': 'd', 'text': 'Empty'},
        ]
        report = translation_core.validate_model_response(
            {
                'translations': [
                    {'id': 'a', 'translation': '你好'},
                    {'id': 'a', 'translation': '重复'},
                    {'id': 'unknown', 'translation': '未知'},
                    {'id': 'b'},
                    {'id': 'd', 'translation': ''},
                ]
            },
            expected_units=expected,
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, ['a', 'b', 'c', 'd'])
        reasons = report.reason_counts()
        self.assertEqual(reasons['result_duplicate_id'], 1)
        self.assertEqual(reasons['result_unknown_id'], 1)
        self.assertEqual(reasons['result_missing_field'], 1)
        self.assertEqual(reasons['result_empty_translation'], 1)
        self.assertEqual(reasons['response_missing_expected_id'], 4)

    def test_translation_contract_reports_envelope_level_reasons(self):
        expected = [{'id': 'a', 'text': 'Hello'}]
        missing = translation_core.validate_model_response(
            {},
            expected_units=expected,
        )
        wrong_type = translation_core.validate_model_response(
            {'translations': {}},
            expected_units=expected,
        )

        self.assertEqual(
            missing.reason_counts(),
            {'response_envelope_missing': 1},
        )
        self.assertEqual(
            wrong_type.reason_counts(),
            {'response_items_not_array': 1},
        )

    def test_translation_contract_rejects_results_when_no_ids_were_requested(self):
        report = translation_core.validate_model_response(
            {
                'translations': [
                    {'id': 'outside', 'translation': '未知'},
                ]
            },
            expected_units=[],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.reason_counts(), {'result_unknown_id': 1})

    def test_keyword_contract_rejects_provenance_when_no_ids_were_requested(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'term',
                        'confidence': 0.9,
                        'evidence': 'line',
                        'source_item_ids': ['outside'],
                    }
                ],
                'chunk_summary': '',
                'summary_evidence_item_ids': [],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.reason_counts(), {'result_unknown_source_id': 1})

    def test_translation_contract_ignores_unexpected_fields_with_diagnostic(self):
        report = translation_core.validate_model_response(
            {
                'translations': [
                    {'id': 'a', 'translation': '你好', 'notes': 'extra'},
                ]
            },
            expected_units=[{'id': 'a', 'text': 'Hello'}],
        )

        self.assertTrue(report.complete)
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.items, [{'id': 'a', 'translation': '你好'}])
        self.assertEqual(
            report.diagnostic_counts(),
            {'result_unexpected_field': 1},
        )

    def test_translation_contract_keeps_legacy_array_read_compatibility(self):
        report = translation_core.validate_model_response(
            [
                {'id': 'b', 'translation': '世界'},
                {'id': 'a', 'translation': '你好'},
            ],
            expected_units=[
                {'id': 'a', 'text': 'Hello'},
                {'id': 'b', 'text': 'World'},
            ],
        )

        self.assertTrue(report.complete)
        self.assertTrue(report.legacy_shape)
        self.assertEqual([item['id'] for item in report.items], ['a', 'b'])
        self.assertEqual(
            report.to_envelope(),
            {
                'translations': [
                    {'id': 'a', 'translation': '你好'},
                    {'id': 'b', 'translation': '世界'},
                ]
            },
        )

    def test_keyword_contract_rejects_unknown_provenance_ids(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'term',
                        'confidence': 0.9,
                        'evidence': 'line',
                        'source_item_ids': ['outside'],
                    }
                ],
                'chunk_summary': '',
                'summary_evidence_item_ids': [],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[{'id': 'a', 'text': 'Void Gate'}],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, ['a'])
        self.assertEqual(report.reason_counts()['result_unknown_source_id'], 1)

    def test_keyword_contract_rejects_empty_candidate_provenance(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'term',
                        'confidence': 0.9,
                        'evidence': 'line a',
                        'source_item_ids': [],
                    }
                ],
                'chunk_summary': '',
                'summary_evidence_item_ids': [],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[{'id': 'a', 'text': 'Void Gate'}],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, ['a'])
        self.assertEqual(report.reason_counts(), {'result_missing_field': 1})

    def test_keyword_contract_keeps_legacy_empty_provenance_read_compatibility(self):
        report = translation_core.validate_model_response(
            [
                {
                    'source': 'Void Gate',
                    'suggested_target': '虚空门',
                    'category': 'term',
                    'confidence': 0.9,
                    'evidence': 'legacy artifact',
                    'source_item_ids': [],
                }
            ],
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[{'id': 'a', 'text': 'Void Gate'}],
        )

        self.assertTrue(report.complete)
        self.assertTrue(report.legacy_shape)
        self.assertEqual(len(report.items), 1)
        self.assertEqual(report.items[0]['source_item_ids'], [])

    def test_keyword_contract_rejects_summary_without_provenance(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [],
                'chunk_summary': 'The party opens the Void Gate.',
                'summary_evidence_item_ids': [],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[{'id': 'a', 'text': 'Open the Void Gate'}],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.items, [])
        self.assertEqual(report.retry_ids, ['a'])
        self.assertEqual(report.reason_counts(), {'result_missing_field': 1})

    def test_keyword_contract_rejects_summary_with_only_unknown_provenance(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [],
                'chunk_summary': 'The party opens the Void Gate.',
                'summary_evidence_item_ids': ['outside'],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[{'id': 'a', 'text': 'Open the Void Gate'}],
        )

        self.assertFalse(report.complete)
        self.assertEqual(report.metadata['summary_evidence_item_ids'], [])
        self.assertEqual(report.retry_ids, ['a'])
        self.assertEqual(
            report.reason_counts(),
            {
                'result_unknown_source_id': 1,
                'result_missing_field': 1,
            },
        )

    def test_keyword_contract_allows_valid_candidates_to_cover_only_some_lines(self):
        report = translation_core.validate_model_response(
            {
                'candidates': [
                    {
                        'source': 'Void Gate',
                        'suggested_target': '虚空门',
                        'category': 'term',
                        'confidence': 0.9,
                        'evidence': 'line a',
                        'source_item_ids': ['a'],
                    }
                ],
                'chunk_summary': 'Only one line contains a glossary candidate.',
                'summary_evidence_item_ids': ['a'],
            },
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            expected_units=[
                {'id': 'a', 'text': 'Void Gate'},
                {'id': 'b', 'text': 'She walks away.'},
            ],
        )

        self.assertTrue(report.complete)
        self.assertEqual(report.valid_ids, ['a'])
        self.assertEqual(report.retry_ids, [])
        self.assertEqual(report.completeness, 1.0)

    def test_project_analysis_optional_inputs_collect_all_available_layers(self):
        fake_store = mock.Mock()
        fake_store.load_summaries.return_value = [
            {
                "summary": "Alice enters the library.",
                "source_files": ["script.rpy"],
            }
        ]
        with (
            mock.patch(
                "project_analysis.resolve_project_analysis_store",
                return_value=fake_store,
            ),
            mock.patch.object(
                batch_mod,
                "_load_final_review_glossary_terms",
                return_value=[{"source": "Alice", "target": "爱丽丝"}],
            ),
            mock.patch.object(batch_mod, "BATCH_MACRO_SETTING", "Tone: restrained"),
            mock.patch.object(batch_mod, "SOURCE_INDEX_ENABLED", True),
            mock.patch.object(batch_mod, "SOURCE_INDEX_STORE_DIR", "context/source_index"),
            mock.patch.object(batch_mod, "STORY_MEMORY_ENABLED", True),
            mock.patch.object(batch_mod, "STORY_MEMORY_GRAPH_FILE", "context/story_graph.json"),
            mock.patch.object(
                batch_mod,
                "retrieve_source_hits",
                return_value=(
                    [
                        {
                            "source_id": "source-1",
                            "file_rel_path": "script.rpy",
                            "line_start": 1,
                            "line_end": 3,
                            "source_text": "Alice enters.",
                            "score": 0.9,
                        }
                    ],
                    {"store_dir": "context/source_index"},
                ),
            ),
            mock.patch.object(batch_mod, "retrieve_batch_story_hits", return_value={"hit": True}),
            mock.patch.object(batch_mod.story_memory, "has_story_hits", return_value=True),
            mock.patch.object(
                batch_mod.story_memory,
                "format_story_hits_block",
                return_value="Alice knows Bob.",
            ),
        ):
            inputs = batch_mod.collect_project_analysis_optional_inputs(
                store_dir="context/project_analysis",
                base_dir="C:/Game",
            )

        self.assertEqual(
            set(inputs),
            {"glossary", "macro_setting", "source_index", "story_memory"},
        )
        self.assertEqual(inputs["source_index"]["provenance"]["source_ids"], ["source-1"])
        self.assertEqual(inputs["story_memory"]["provenance"]["query_file"], "script.rpy")

    def test_project_context_artifacts_are_cached_across_targets(self):
        fake_store = mock.Mock()
        fake_store.load_summaries.return_value = [
            {
                "id": "label:a",
                "label_id": "a",
                "summary": "Scene A",
                "source_files": ["script.rpy"],
                "line_span": [1, 10],
            },
            {
                "id": "label:b",
                "label_id": "b",
                "summary": "Scene B",
                "source_files": ["script.rpy"],
                "line_span": [20, 30],
            },
        ]
        fake_store.load_routes.return_value = [
            {
                "id": "route:a",
                "route_id": "route:a",
                "summary": "Route A",
                "metadata": {"label_ids": ["a"]},
            },
            {
                "id": "route:b",
                "route_id": "route:b",
                "summary": "Route B",
                "metadata": {"label_ids": ["b"]},
            },
        ]
        global_payload = {
            "injectable": True,
            "text": "Global brief",
            "diagnostics": "status=published",
            "labels": [],
            "routes": [],
            "local_diagnostics": "",
        }
        with (
            mock.patch.object(batch_mod, "PROJECT_ANALYSIS_ENABLED", True),
            mock.patch.object(batch_mod, "PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF", True),
            mock.patch.object(batch_mod, "_PROJECT_BRIEF_CACHE", None),
            mock.patch.object(batch_mod, "_PROJECT_BRIEF_CACHE_KEY", None),
            mock.patch.object(
                batch_mod,
                "compute_current_project_analysis_fingerprint",
                return_value="fp-1",
            ),
            mock.patch(
                "project_analysis.load_injectable_project_context",
                return_value=global_payload,
            ) as load_global,
            mock.patch(
                "project_analysis.resolve_project_analysis_store",
                return_value=fake_store,
            ) as resolve_store,
        ):
            first = batch_mod.load_injectable_project_context_for_prompts(
                "script.rpy", [5]
            )
            second = batch_mod.load_injectable_project_context_for_prompts(
                "script.rpy", [25]
            )

        self.assertEqual(first["labels"][0]["label_id"], "a")
        self.assertEqual(second["labels"][0]["label_id"], "b")
        load_global.assert_called_once()
        resolve_store.assert_called_once()
        fake_store.load_summaries.assert_called_once()
        fake_store.load_routes.assert_called_once()

    def test_project_context_store_failure_is_non_blocking_and_not_cached(self):
        with (
            mock.patch.object(batch_mod, "PROJECT_ANALYSIS_ENABLED", True),
            mock.patch.object(batch_mod, "PROJECT_ANALYSIS_INJECT_PUBLISHED_BRIEF", True),
            mock.patch.object(batch_mod, "_PROJECT_BRIEF_CACHE", None),
            mock.patch.object(batch_mod, "_PROJECT_BRIEF_CACHE_KEY", None),
            mock.patch.object(
                batch_mod,
                "compute_current_project_analysis_fingerprint",
                return_value="fp-1",
            ),
            mock.patch(
                "project_analysis.load_injectable_project_context",
                side_effect=ValueError("corrupt analysis store"),
            ),
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            result = batch_mod.load_injectable_project_context_for_prompts(
                "script.rpy", [5]
            )
            cached = batch_mod._PROJECT_BRIEF_CACHE

        self.assertEqual(
            result,
            {"text": "", "diagnostics": "", "labels": [], "routes": [], "local_diagnostics": ""},
        )
        self.assertIsNone(cached)
        self.assertIn("project analysis local context unavailable", stderr.getvalue())

    def test_translation_and_revision_prompts_select_target_local_context(self):
        payload = {
            "text": "Global brief",
            "diagnostics": "status=published",
            "labels": [{"label_id": "scene_a", "summary": "Local label"}],
            "routes": [{"route_id": "route:start", "summary": "Local route"}],
            "local_diagnostics": "target=script.rpy labels=scene_a",
        }
        with mock.patch(
            "gemini_translate_batch.load_injectable_project_context_for_prompts",
            return_value=payload,
        ) as load_context:
            translation_prompt = batch_mod.build_user_prompt(
                [],
                [{"id": "line-1", "text": "Hello", "line": 9}],
                [],
                file_rel_path="script.rpy",
            )
            revision_prompt = batch_mod.build_revision_user_prompt(
                {
                    "file_rel_path": "script.rpy",
                    "line_numbers": [20],
                    "items": [
                        {
                            "id": "line-2",
                            "source": "Hello",
                            "current_translation": "你好",
                            "line_number": 20,
                        }
                    ],
                }
            )

        self.assertIn("PROJECT LOCAL CONTEXT", translation_prompt)
        self.assertIn("Local label", translation_prompt)
        self.assertIn("Local route", revision_prompt)
        self.assertEqual(load_context.call_args_list[0].args, ("script.rpy", [10]))
        self.assertEqual(load_context.call_args_list[1].args, ("script.rpy", [20]))

    def test_translation_prompts_emphasize_renpy_interpolation_preservation(self):
        system_text = translation_core.build_translation_system_instruction(['[Gil_name!t]'])
        self.assertIn('[Gil_name!t]', system_text)
        self.assertIn('never replace', system_text)
        self.assertIn('never omit an item', system_text)

        sync_text = translation_core.build_sync_translation_prompt(
            [{'id': 'line-1', 'text': 'Coach [Gil_name!t] is here.'}],
            ['[Gil_name!t]'],
        )
        self.assertIn('[Gil_name!t]', sync_text)
        self.assertIn('never turn them into literal names', sync_text)

    def test_core_result_parsers_and_writeback_actions_are_mode_aware(self):
        translation_results = translation_core.normalize_model_results(
            {'translations': [{'id': 'a', 'translation': '\u4f60\u597d'}]},
            mode=translation_core.MODE_TRANSLATION,
        )
        revision_results = translation_core.normalize_model_results(
            {'revisions': [{'id': 'b', 'should_update': 'yes', 'translation': '\u65b0\u8bd1'}]},
            mode=translation_core.MODE_REVISION,
        )
        keyword_results = translation_core.normalize_model_results(
            {'keywords': [{'source': 'Void Gate', 'category': 'unknown', 'confidence': 2}]},
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
        )

        self.assertEqual(translation_results, [{'id': 'a', 'translation': '\u4f60\u597d'}])
        self.assertTrue(revision_results[0]['should_update'])
        self.assertEqual(revision_results[0]['revised_translation'], '\u65b0\u8bd1')
        self.assertEqual(keyword_results[0]['category'], 'other')
        self.assertEqual(keyword_results[0]['confidence'], 1.0)

        unit = translation_core.unit_from_translation_item(
            {
                'id': 'script.rpy:0:4',
                'text': 'Hello',
                'line': 0,
                'start': 4,
                'end': 11,
                'quote': '"',
            },
            file_rel_path='script.rpy',
        )
        action = translation_core.translation_writeback_action(unit, translation_results[0], chunk_key='chunk-1')
        self.assertEqual(
            translation_core.writeback_tuple(action),
            (4, 11, '\u4f60\u597d', '', '"', 'Hello', 'script.rpy:0:4', 'chunk-1'),
        )
        keyword_unit = translation_core.unit_from_keyword_item(
            {'id': 'kw-1', 'text': 'Void Gate'},
            file_rel_path='script.rpy',
        )
        self.assertIsNone(
            translation_core.build_writeback_action(
                keyword_unit,
                keyword_results[0],
                mode=translation_core.MODE_KEYWORD_EXTRACTION,
            )
        )

    def test_manifest_item_dispatches_by_mode_with_chunk_defaults(self):
        revision_unit = translation_core.unit_from_manifest_item(
            {
                'id': 'script.rpy:2:4',
                'text': '',
                'source': 'Hello Alice',
                'current_translation': None,
                'line_number': 3,
            },
            mode=translation_core.MODE_REVISION,
            chunk={'file_rel_path': 'script.rpy', 'file_path': '/tmp/script.rpy'},
        )
        self.assertEqual(revision_unit.mode, translation_core.MODE_REVISION)
        self.assertEqual(revision_unit.file_rel_path, 'script.rpy')
        self.assertEqual(revision_unit.text, 'Hello Alice')
        self.assertEqual(revision_unit.current_translation, '')
        self.assertEqual(revision_unit.line, 2)

        keyword_unit = translation_core.unit_from_manifest_item(
            {'id': 'kw-1', 'text': 'Void Gate', 'translation_line_number': 8},
            mode=translation_core.MODE_KEYWORD_EXTRACTION,
            chunk={'file_rel_path': 'script.rpy'},
        )
        self.assertEqual(keyword_unit.mode, translation_core.MODE_KEYWORD_EXTRACTION)
        self.assertEqual(keyword_unit.file_rel_path, 'script.rpy')
        self.assertEqual(keyword_unit.metadata['translation_line_number'], 8)

    def test_revision_context_block_accepts_units_and_dicts(self):
        block = translation_core.format_revision_context_block(
            [
                translation_core.TranslationUnit(
                    id='unit-1',
                    source='Hello Alice',
                    current_translation='\u4f60\u597d Alice',
                ),
                {'source': 'Goodbye', 'current_translation': '\u518d\u89c1'},
            ]
        )

        self.assertIn('- Hello Alice => \u4f60\u597d Alice', block)
        self.assertIn('- Goodbye => \u518d\u89c1', block)



if __name__ == '__main__':
    unittest.main()
