import unittest

from relation_analyzer.parsing import infer_characters_from_units, parse_dialogue_line, resolve_speaker_name
from relation_analyzer.relations import compute_relation_data


class RelationAnalyzerTests(unittest.TestCase):
    def test_parse_dialogue_line_extracts_speaker_and_text(self):
        parsed = parse_dialogue_line('spencer_no_side "Hello there."')
        self.assertEqual(parsed['speaker'], 'spencer_no_side')
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


if __name__ == '__main__':
    unittest.main()
