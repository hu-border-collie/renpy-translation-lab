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
        self.assertEqual(translation_schema['items']['required'], ['id', 'translation'])

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
            revision_schema['items']['required'],
            ['id', 'should_update', 'revised_translation', 'reason'],
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
        self.assertIn('chunk_summary', keyword_schema['required'])

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
