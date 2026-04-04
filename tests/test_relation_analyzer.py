import csv
import tempfile
import unittest
from pathlib import Path

from relation_analyzer.parsing import infer_characters_from_units, load_text_units, parse_dialogue_line, resolve_speaker_name
from relation_analyzer.relations import compute_relation_data, write_relation_csv


class RelationAnalyzerTests(unittest.TestCase):
    def test_parse_dialogue_line_extracts_speaker_and_text(self):
        parsed = parse_dialogue_line('spencer_no_side "Hello there."')
        self.assertEqual(parsed['speaker'], 'spencer_no_side')
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_parse_dialogue_line_accepts_attribute_qualified_say_statement(self):
        parsed = parse_dialogue_line('e happy "Hello there."')
        self.assertEqual(parsed['speaker'], 'e')
        self.assertEqual(parsed['text'], 'Hello there.')

    def test_resolve_speaker_name_uses_generic_suffix_heuristic(self):
        self.assertEqual(resolve_speaker_name('spencer_no_side'), 'Spencer')
        self.assertEqual(resolve_speaker_name('mr_smith'), 'Mr Smith')

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


if __name__ == '__main__':
    unittest.main()
