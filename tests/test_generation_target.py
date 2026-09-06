import unittest

import generation_target


class GenerationTargetContractTests(unittest.TestCase):
    def test_omitted_value_is_schinese(self):
        resolution = generation_target.resolve_generation_target(None)
        self.assertEqual(resolution.canonical, 'schinese')
        self.assertTrue(resolution.supported)
        self.assertFalse(resolution.explicit)
        self.assertEqual(generation_target.require_supported(resolution), 'schinese')

    def test_aliases_map_to_schinese(self):
        for raw in ('schinese', 'zh-CN', 'zh_hans', 'Simplified Chinese'):
            resolution = generation_target.resolve_generation_target(raw)
            self.assertEqual(resolution.canonical, 'schinese', raw)
            self.assertTrue(resolution.supported, raw)

    def test_japanese_is_unsupported(self):
        resolution = generation_target.resolve_generation_target('japanese')
        self.assertFalse(resolution.supported)
        with self.assertRaises(generation_target.GenerationTargetError) as raised:
            generation_target.require_supported(resolution)
        self.assertEqual(raised.exception.code, 'generation_target.unsupported')
        self.assertIn('generation.target_language', str(raised.exception))

    def test_config_reader_uses_generation_target_language(self):
        resolution = generation_target.resolve_generation_target_from_config(
            {'generation': {'target_language': 'korean'}}
        )
        self.assertEqual(resolution.canonical, 'korean')
        self.assertFalse(resolution.supported)

    def test_catalog_hint_ignores_custom_slugs(self):
        self.assertEqual(
            generation_target.catalog_language_hint('my_pack', 'game/tl/my_pack'),
            '',
        )
        self.assertEqual(
            generation_target.catalog_language_hint('japanese', 'game/tl/japanese'),
            'japanese',
        )
        self.assertEqual(
            generation_target.catalog_language_hint('schinese', 'game/tl/schinese'),
            '',
        )
