import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from relation_analyzer import cli
from relation_analyzer import semantic
from relation_analyzer.parsing import extract_character_definitions, extract_units_from_rpy, infer_characters_from_units, load_text_units, parse_dialogue_line, resolve_speaker_name
from relation_analyzer.plotting import project_vectors_2d
from relation_analyzer.relations import compute_relation_data, write_relation_csv
from relation_analyzer.story_seed import build_story_graph_seed, write_story_graph_seed


class RelationAnalyzerTests(unittest.TestCase):
    def test_parse_dialogue_line_extracts_speaker_and_text(self):
        parsed = parse_dialogue_line('spencer_no_side "Hello there."')
        self.assertEqual(parsed['speaker'], 'spencer_no_side')
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_parse_dialogue_line_accepts_attribute_qualified_say_statement(self):
        parsed = parse_dialogue_line('e happy "Hello there."')
        self.assertEqual(parsed['speaker'], 'e')
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_parse_dialogue_line_rejects_assignment_like_statement(self):
        self.assertIsNone(parse_dialogue_line('title = "Hello there."'))

    def test_parse_dialogue_line_allows_new_as_regular_speaker_name(self):
        parsed = parse_dialogue_line('new "Hello there."')
        self.assertEqual(parsed['speaker'], 'new')
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_parse_dialogue_line_rejects_bare_string_python_literal(self):
        self.assertIsNone(parse_dialogue_line('"dismiss": ["mouseup_1"],'))

    def test_parse_dialogue_line_accepts_bare_string_with_narrator_clause(self):
        parsed = parse_dialogue_line('"Hello there." nointeract')
        self.assertIsNone(parsed['speaker'])
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_parse_dialogue_line_rejects_python_keyword_speaker(self):
        self.assertIsNone(parse_dialogue_line('assert "Hello there."'))
        self.assertIsNone(parse_dialogue_line('yield "Hello there."'))

    def test_parse_dialogue_line_rejects_style_property_line(self):
        self.assertIsNone(parse_dialogue_line('font "DejaVuSans.ttf"'))

    def test_parse_dialogue_line_rejects_screen_property_line(self):
        self.assertIsNone(parse_dialogue_line('text "Overlay"'))
        self.assertIsNone(parse_dialogue_line('id "window"'))

    def test_parse_dialogue_line_rejects_assignment_like_text_commands(self):
        self.assertIsNone(parse_dialogue_line('narrator = Character("Narrator")'))
        self.assertIsNone(parse_dialogue_line('extend = "Hello there."'))

    def test_resolve_speaker_name_uses_generic_suffix_heuristic(self):
        self.assertEqual(resolve_speaker_name('spencer_no_side'), 'Spencer')
        self.assertEqual(resolve_speaker_name('mr_smith'), 'Mr Smith')

    def test_extract_character_definitions_reads_renpy_character_names(self):
        definitions = extract_character_definitions([
            'define e = Character("Eileen")\n',
            'define n = Character(_("Noah"), color="#fff")\n',
            'define c = Character(name="Cora", image="cora")\n',
            'define a = Character("艾")\n',
            'define img = Character(None, image="not_a_name")\n',
            'default title = "Not a character"\n',
        ])

        self.assertEqual(definitions, {'e': 'Eileen', 'n': 'Noah', 'c': 'Cora', 'a': '艾'})

    def test_resolve_speaker_name_prefers_explicit_mapping_over_character_define(self):
        with patch.dict('relation_analyzer.parsing.SPEAKER_TO_CHARACTER', {'e': '艾琳'}):
            self.assertEqual(resolve_speaker_name('e', {'e': 'Eileen'}), '艾琳')

    def test_infer_characters_from_units_prefers_most_frequent_speakers(self):
        units = [
            {'speaker_name': 'Spencer'},
            {'speaker_name': 'Ian'},
            {'speaker_name': 'Spencer'},
            {'speaker_name': None},
            {'speaker_name': 'Andrew'},
            {'speaker_name': 'Andrew'},
        ]
        self.assertEqual(infer_characters_from_units(units, 2), ['Andrew', 'Spencer'])

    def test_compute_relation_data_builds_nonzero_pair_scores(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'spencer_no_side', 'speaker_name': 'Spencer', 'text': 'Ian, are you there?', 'context': 'Spencer: Ian, are you there?'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'ian', 'speaker_name': 'Ian', 'text': 'Yeah, I am here.', 'context': 'Ian: Yeah, I am here.'},
            {'source': 'scene_1.rpy', 'line_no': 3, 'speaker': 'spencer_no_side', 'speaker_name': 'Spencer', 'text': 'Andrew should join us later.', 'context': 'Spencer: Andrew should join us later.'},
            {'source': 'scene_1.rpy', 'line_no': 4, 'speaker': 'andrew', 'speaker_name': 'Andrew', 'text': 'Sorry, I am late.', 'context': 'Andrew: Sorry, I am late.'},
        ]
        data = compute_relation_data(units, ['Spencer', 'Ian', 'Andrew'], segment_size=4)
        pair_rows = {frozenset((row['left'], row['right'])): row for row in data['pair_rows']}
        self.assertGreater(pair_rows[frozenset(('Ian', 'Spencer'))]['score'], 0.0)
        self.assertGreater(pair_rows[frozenset(('Andrew', 'Spencer'))]['mention'], 0.0)
        self.assertEqual(data['characters'], ['Spencer', 'Ian', 'Andrew'])

    def test_compute_relation_data_deduplicates_character_list(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'ian', 'speaker_name': 'Ian', 'text': 'Hello Spencer.', 'context': 'Ian: Hello Spencer.'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'spencer_no_side', 'speaker_name': 'Spencer', 'text': 'Hi Ian.', 'context': 'Spencer: Hi Ian.'},
        ]
        data = compute_relation_data(units, ['Ian', 'Spencer', 'Ian'], segment_size=2)

        self.assertEqual(data['characters'], ['Ian', 'Spencer'])
        self.assertEqual(len(data['pair_rows']), 1)
        self.assertEqual(data['pair_rows'][0]['left'], 'Ian')
        self.assertEqual(data['pair_rows'][0]['right'], 'Spencer')

    def test_build_story_graph_seed_exports_reviewable_candidates(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'eileen_side', 'speaker_name': 'Eileen', 'text': 'Noah, open the gate.', 'context': 'Eileen: Noah, open the gate.'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'noah', 'speaker_name': 'Noah', 'text': 'I am opening it.', 'context': 'Noah: I am opening it.'},
            {'source': 'scene_2.rpy', 'line_no': 1, 'speaker': 'eileen_happy', 'speaker_name': 'Eileen', 'text': 'Good work, Noah.', 'context': 'Eileen: Good work, Noah.'},
        ]
        relation_data = compute_relation_data(units, ['Eileen', 'Noah'], segment_size=2)

        seed = build_story_graph_seed(units, ['Eileen', 'Noah'], relation_data)

        self.assertEqual(seed['schema_version'], 1)
        self.assertEqual(seed['characters']['eileen']['name'], 'Eileen')
        self.assertEqual(seed['characters']['eileen']['speaker_ids'], ['eileen_side', 'eileen_happy'])
        self.assertEqual(seed['characters']['eileen']['seed_stats']['speaker_count'], 2)
        self.assertEqual(seed['characters']['noah']['speaker_ids'], ['noah'])
        self.assertEqual(len(seed['relations']), 1)
        relation = seed['relations'][0]
        self.assertEqual(relation['left'], 'eileen')
        self.assertEqual(relation['right'], 'noah')
        self.assertEqual(relation['type'], 'candidate')
        self.assertIn('人工确认', relation['note'])
        self.assertGreater(relation['seed_stats']['dialogue_raw'], 0.0)
        self.assertIn('scene_1.rpy', relation['seed_stats']['source_files'])
        self.assertTrue(relation['seed_stats']['needs_human_review'])

    def test_build_story_graph_seed_uses_character_aliases_for_speaker_ids(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'e', 'speaker_name': 'Eileen', 'text': 'Noah, open the gate.', 'context': 'Eileen: Noah, open the gate.'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'noah', 'speaker_name': 'Noah', 'text': 'I am opening it.', 'context': 'Noah: I am opening it.'},
        ]
        with patch.dict('relation_analyzer.parsing.CHARACTER_ALIASES', {'艾琳': ['Eileen']}):
            relation_data = compute_relation_data(units, ['艾琳', 'Noah'], segment_size=2)
            seed = build_story_graph_seed(units, ['艾琳', 'Noah'], relation_data)

        self.assertEqual(seed['characters']['艾琳']['speaker_ids'], ['e'])

    def test_build_story_graph_seed_uses_relative_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scene_path = root / 'scene_1.rpy'
            units = [
                {'source': str(scene_path), 'line_no': 1, 'speaker': 'e', 'speaker_name': 'E', 'text': 'Hello Noah.', 'context': 'E: Hello Noah.'},
                {'source': str(scene_path), 'line_no': 2, 'speaker': 'noah', 'speaker_name': 'Noah', 'text': 'Hello.', 'context': 'Noah: Hello.'},
            ]
            relation_data = compute_relation_data(units, ['E', 'Noah'], segment_size=2)

            seed = build_story_graph_seed(units, ['E', 'Noah'], relation_data, source_root=root)

        self.assertEqual(seed['characters']['e']['seed_stats']['source_files'], ['scene_1.rpy'])
        self.assertEqual(seed['relations'][0]['seed_stats']['source_files'], ['scene_1.rpy'])

    def test_build_story_graph_seed_filters_zero_evidence_pairs(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'a', 'speaker_name': 'A', 'text': 'Solo.', 'context': 'A: Solo.'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'c', 'speaker_name': 'C', 'text': 'Solo.', 'context': 'C: Solo.'},
            {'source': 'scene_1.rpy', 'line_no': 3, 'speaker': 'b', 'speaker_name': 'B', 'text': 'Solo.', 'context': 'B: Solo.'},
        ]
        relation_data = compute_relation_data(units, ['A', 'B', 'C'], segment_size=1)

        seed = build_story_graph_seed(units, ['A', 'B', 'C'], relation_data)
        relation_pairs = {(item['left'], item['right']) for item in seed['relations']}

        self.assertNotIn(('a', 'b'), relation_pairs)
        self.assertIn(('a', 'c'), relation_pairs)
        self.assertIn(('b', 'c'), relation_pairs)

    def test_write_story_graph_seed_creates_parent_directory(self):
        units = [
            {'source': 'scene_1.rpy', 'line_no': 1, 'speaker': 'e', 'speaker_name': 'E', 'text': 'Hello Noah.', 'context': 'E: Hello Noah.'},
            {'source': 'scene_1.rpy', 'line_no': 2, 'speaker': 'noah', 'speaker_name': 'Noah', 'text': 'Hello.', 'context': 'Noah: Hello.'},
        ]
        relation_data = compute_relation_data(units, ['E', 'Noah'], segment_size=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'logs' / 'story_memory' / 'story_graph.seed.json'
            write_story_graph_seed(output_path, units, ['E', 'Noah'], relation_data)
            loaded = json.loads(output_path.read_text(encoding='utf-8'))

        self.assertEqual(loaded['characters']['e']['speaker_ids'], ['e'])
        self.assertEqual(loaded['relations'][0]['type'], 'candidate')

    def test_load_text_units_reads_utf8_sig_for_rpy_and_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rpy_path = root / 'scene.rpy'
            txt_path = root / 'notes.txt'
            rpy_path.write_text(
                'translate schinese start:\n    e happy "Hello there."\n',
                encoding='utf-8-sig',
            )
            txt_path.write_text(
                'First paragraph line.\n\nSecond paragraph line.\n',
                encoding='utf-8-sig',
            )

            rpy_units = load_text_units(rpy_path, context_window=0)
            txt_units = load_text_units(txt_path, context_window=0)

            self.assertEqual(len(rpy_units), 1)
            self.assertEqual(rpy_units[0]['speaker'], 'e')
            self.assertEqual(rpy_units[0]['text'], 'Hello there.')
            self.assertEqual(len(txt_units), 2)
            self.assertEqual(txt_units[0]['text'], 'First paragraph line.')

    def test_load_text_units_preserves_previous_speaker_for_extend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rpy_path = root / 'scene.rpy'
            rpy_path.write_text(
                'translate schinese start:\n'
                '    e happy "Hello there."\n'
                '    extend "And another line."\n',
                encoding='utf-8-sig',
            )

            rpy_units = load_text_units(rpy_path, context_window=0)

            self.assertEqual(len(rpy_units), 2)
            self.assertEqual(rpy_units[0]['speaker'], 'e')
            self.assertEqual(rpy_units[1]['speaker'], 'e')
            self.assertEqual(rpy_units[1]['speaker_name'], 'E')
            self.assertEqual(rpy_units[1]['text'], 'And another line.')

    def test_load_text_units_uses_character_defines_across_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'characters.rpy').write_text(
                'define e = Character("Eileen")\n',
                encoding='utf-8-sig',
            )
            (root / 'scene.rpy').write_text(
                'label start:\n'
                '    e happy "Hello there."\n',
                encoding='utf-8-sig',
            )

            rpy_units = load_text_units(root, context_window=0)

            self.assertEqual(len(rpy_units), 1)
            self.assertEqual(rpy_units[0]['speaker'], 'e')
            self.assertEqual(rpy_units[0]['speaker_name'], 'Eileen')

    def test_load_text_units_reuses_cached_rpy_lines_for_character_defines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            characters_path = root / 'characters.rpy'
            scene_path = root / 'scene.rpy'
            characters_path.write_text(
                'define e = Character("Eileen")\n',
                encoding='utf-8-sig',
            )
            scene_path.write_text(
                'label start:\n'
                '    e happy "Hello there."\n',
                encoding='utf-8-sig',
            )
            original_read_text = Path.read_text
            read_counts = {}

            def counting_read_text(path, *args, **kwargs):
                if path.suffix.lower() == '.rpy':
                    read_counts[path] = read_counts.get(path, 0) + 1
                return original_read_text(path, *args, **kwargs)

            with patch.object(Path, 'read_text', counting_read_text):
                rpy_units = load_text_units(root, context_window=0)

            self.assertEqual(len(rpy_units), 1)
            self.assertEqual(read_counts[characters_path], 1)
            self.assertEqual(read_counts[scene_path], 1)

    def test_load_text_units_resets_speaker_context_after_narrator_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rpy_path = root / 'scene.rpy'
            rpy_path.write_text(
                'translate schinese start:\n'
                '    e happy "Hello there."\n'
                '    "A narrator line."\n'
                '    extend "Still narrator."\n',
                encoding='utf-8-sig',
            )

            rpy_units = load_text_units(rpy_path, context_window=0)

            self.assertEqual(len(rpy_units), 3)
            self.assertEqual(rpy_units[0]['speaker'], 'e')
            self.assertIsNone(rpy_units[1]['speaker'])
            self.assertIsNone(rpy_units[2]['speaker'])
            self.assertEqual(rpy_units[2]['text'], 'Still narrator.')

    def test_extract_units_from_rpy_skips_strings_only_translation_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rpy_path = root / 'common.rpy'
            rpy_path.write_text(
                'translate schinese strings:\n'
                '    old "Default"\n'
                '    new "Default"\n',
                encoding='utf-8-sig',
            )

            units = extract_units_from_rpy(rpy_path)

            self.assertEqual(units, [])

    def test_extract_units_from_rpy_preserves_raw_dialogue_around_strings_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rpy_path = root / 'mixed.rpy'
            rpy_path.write_text(
                'label start:\n'
                '    e "Before."\n'
                '\n'
                'translate schinese strings:\n'
                '    old "Default"\n'
                '    new "Default"\n'
                '\n'
                'label later:\n'
                '    "After." nointeract\n',
                encoding='utf-8-sig',
            )

            units = extract_units_from_rpy(rpy_path)

            self.assertEqual(len(units), 2)
            self.assertEqual(units[0]['speaker'], 'e')
            self.assertEqual(units[0]['text'], 'Before.')
            self.assertIsNone(units[1]['speaker'])
            self.assertEqual(units[1]['text'], 'After.')

    def test_project_vectors_2d_handles_one_dimensional_embeddings(self):
        class FakePCA:
            def __init__(self, n_components):
                self.n_components = n_components

            def fit_transform(self, matrix):
                rows = []
                for row in matrix:
                    rows.append([float(row[0])][:self.n_components])
                return __import__('numpy').array(rows, dtype=float)

        vectors_2d = project_vectors_2d([[1.0], [2.0], [3.0]], FakePCA)

        self.assertEqual(vectors_2d.shape, (3, 2))
        self.assertTrue((vectors_2d[:, 1] == 0).all())

    def test_cli_relation_mode_exits_nonzero_when_active_characters_drop_below_two(self):
        args = SimpleNamespace(
            input='input.rpy',
            output='output.png',
            cache_dir='cache',
            characters='Solo,Missing',
            auto_characters=0,
            portraits='off',
            mode='relation',
            batch_size=1,
            context_window=0,
            model='gemini-embedding-001',
            output_dimensionality=768,
            max_texts_per_character=0,
            relation_window_size=12,
            csv_output=None,
            story_seed_output=None,
        )
        with patch.object(cli, 'parse_args', return_value=args), \
             patch.object(cli, 'resolve_path', side_effect=lambda value: Path(value)), \
             patch.object(cli, 'load_text_units', return_value=[{'text': 'x'}]), \
             patch.object(cli, 'compute_relation_data', return_value={'characters': ['Solo']}):
            with self.assertRaises(SystemExit) as exc:
                cli.main()

        self.assertEqual(str(exc.exception), '❌ 提取到的有效角色少于 2 个，无法计算关系。')

    def test_cli_relation_mode_writes_story_seed_with_input_root(self):
        args = SimpleNamespace(
            input='input_dir',
            output='output.png',
            cache_dir='cache',
            characters='E,Noah',
            auto_characters=0,
            portraits='off',
            mode='relation',
            batch_size=1,
            context_window=0,
            model='gemini-embedding-001',
            output_dimensionality=768,
            max_texts_per_character=0,
            relation_window_size=12,
            csv_output=None,
            story_seed_output='seed.json',
        )
        units = [{'source': 'scene.rpy', 'line_no': 1, 'speaker': 'e', 'speaker_name': 'E', 'text': 'Hello.', 'context': 'E: Hello.'}]
        relation_data = {'characters': ['E', 'Noah'], 'pair_rows': [], 'segment_size': 12, 'pair_source_files': {}}

        with patch.object(cli, 'parse_args', return_value=args), \
             patch.object(cli, 'resolve_path', side_effect=lambda value: Path(value)), \
             patch.object(cli, 'load_text_units', return_value=units), \
             patch.object(cli, 'compute_relation_data', return_value=relation_data), \
             patch.object(cli, 'write_story_graph_seed') as write_seed, \
             patch.object(cli, 'analyze_and_plot_relation'):
            cli.main()

        write_seed.assert_called_once_with(
            Path('seed.json'),
            units,
            ['E', 'Noah'],
            relation_data,
            source_root=Path('input_dir'),
        )

    def test_cli_semantic_mode_exits_nonzero_when_active_characters_drop_below_two(self):
        args = SimpleNamespace(
            input='input.rpy',
            output='output.png',
            cache_dir='cache',
            characters='Solo,Missing',
            auto_characters=0,
            portraits='off',
            mode='semantic',
            batch_size=1,
            context_window=0,
            model='gemini-embedding-001',
            output_dimensionality=1,
            max_texts_per_character=0,
            relation_window_size=12,
            csv_output=None,
            story_seed_output=None,
        )
        with patch.object(cli, 'parse_args', return_value=args), \
             patch.object(cli, 'resolve_path', side_effect=lambda value: Path(value)), \
             patch.object(cli, 'load_text_units', return_value=[{'text': 'x'}]), \
             patch.object(cli, 'collect_character_texts', return_value={'Solo': ['x'], 'Missing': []}), \
             patch.object(cli, 'extract_character_vectors', return_value={'Solo': [1.0]}):
            with self.assertRaises(SystemExit) as exc:
                cli.main()

        self.assertEqual(str(exc.exception), '❌ 提取到的有效角色少于 2 个，无法计算关系。')

    def test_write_relation_csv_uses_proper_csv_escaping(self):
        relation_data = {
            'pair_rows': [
                {
                    'left': 'A,One',
                    'right': 'B"Two',
                    'score': 0.5,
                    'co_scene': 0.4,
                    'dialogue': 0.3,
                    'mention': 0.2,
                    'co_scene_raw': 2.0,
                    'dialogue_raw': 1.0,
                    'mention_raw': 0.5,
                    'dominant_component': '提及,对话',
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'relations.csv'
            write_relation_csv(csv_path, relation_data)
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]['left'], 'A,One')
        self.assertEqual(rows[0]['right'], 'B"Two')
        self.assertEqual(rows[0]['dominant_component'], '提及,对话')

    def test_embed_texts_rotates_past_retry_count_until_valid_key(self):
        class FakeModels:
            def __init__(self, behavior):
                self.behavior = behavior

            def embed_content(self, **kwargs):
                return self.behavior()

        class FakeClient:
            def __init__(self, behavior):
                self.models = FakeModels(behavior)

        def auth_error():
            raise RuntimeError('API key expired. Please renew the API key.')

        def success():
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 2.0])]
            )

        behaviors = [auth_error, auth_error, auth_error, success]
        clients = [FakeClient(behavior) for behavior in behaviors]
        client_iter = iter(clients)
        state = {'index': 0}

        def fake_rotate():
            if state['index'] >= len(clients) - 1:
                return False
            state['index'] += 1
            return True

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(semantic, 'API_RETRIES', 3), \
             patch.object(semantic, 'load_numpy', return_value=__import__('numpy')), \
             patch.object(semantic, 'load_embedding_libs', return_value=(None, SimpleNamespace(EmbedContentConfig=lambda **kwargs: kwargs))), \
             patch.object(semantic, 'get_client', side_effect=lambda: next(client_iter)), \
             patch.object(semantic, 'rotate_api_key', side_effect=fake_rotate), \
             patch.object(semantic, 'get_api_key_source', side_effect=lambda: f'key{state["index"] + 1}'), \
             patch.object(semantic, 'load_cached_embedding', return_value=None), \
             patch.object(semantic, 'save_cached_embedding', return_value=None):
            embeddings = semantic.embed_texts(
                ['hello'],
                batch_size=1,
                model_name='gemini-embedding-001',
                output_dimensionality=2,
                cache_dir=Path(tmpdir),
            )

        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(embeddings.tolist(), [[1.0, 2.0]])

    def test_embed_texts_uses_cache_without_loading_gemini_client(self):
        cached_vector = __import__('numpy').array([3.0, 4.0], dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch.object(semantic, 'load_numpy', return_value=__import__('numpy')), \
             patch.object(semantic, 'load_cached_embedding', return_value=cached_vector), \
             patch.object(semantic, 'load_embedding_libs', side_effect=AssertionError('should not load embedding libs')), \
             patch.object(semantic, 'get_client', side_effect=AssertionError('should not create client')):
            embeddings = semantic.embed_texts(
                ['hello'],
                batch_size=1,
                model_name='gemini-embedding-001',
                output_dimensionality=2,
                cache_dir=Path(tmpdir),
            )

        self.assertEqual(embeddings.shape, (1, 2))
        self.assertEqual(embeddings.tolist(), [[3.0, 4.0]])


if __name__ == '__main__':
    unittest.main()
