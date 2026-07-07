import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import batch_non_chinese_rules
import gemini_translate_batch as batch_mod


class BatchNonChineseFileCacheTests(unittest.TestCase):
    def _write_old_new_fixture(self, tl_dir):
        tl_path = tl_dir / 'misc_labels.rpy'
        tl_path.write_text(
            '# game/misc_labels.rpy:10\nold "Avi"\nnew "Avi"\n',
            encoding='utf-8',
        )
        manifest = {
            'tl_dir': str(tl_dir),
            'non_chinese_rules': copy.deepcopy(batch_non_chinese_rules.DEFAULT_NON_CHINESE_RULES),
        }
        chunk = {'file_rel_path': 'misc_labels.rpy'}
        item = {'line_number': 2}
        return manifest, chunk, item

    def test_old_new_static_item_reuses_tl_line_read_within_call(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest, chunk, item = self._write_old_new_fixture(Path(tmpdir).resolve())
            cache = batch_mod.NonChineseFileReadCache()
            open_paths = []
            real_open = open

            def counting_open(path, *args, **kwargs):
                open_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with mock.patch('builtins.open', side_effect=counting_open):
                allowed = batch_mod.is_manifest_static_non_chinese_item(
                    manifest,
                    chunk,
                    'Avi',
                    'Avi',
                    item=item,
                    file_read_cache=cache,
                )

            self.assertTrue(allowed)
            self.assertEqual(len(open_paths), 1)

    def test_allow_non_chinese_reuses_single_tl_open_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest, chunk, item = self._write_old_new_fixture(Path(tmpdir).resolve())
            open_paths = []
            real_open = open

            def counting_open(path, *args, **kwargs):
                open_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with mock.patch('builtins.open', side_effect=counting_open):
                allowed = batch_mod.allow_non_chinese_batch_translation(
                    manifest,
                    chunk,
                    'Avi',
                    'Avi',
                    item=item,
                )

            self.assertTrue(allowed)
            self.assertEqual(len(open_paths), 1)

    def test_allow_non_chinese_matches_results_with_and_without_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest, chunk, item = self._write_old_new_fixture(Path(tmpdir).resolve())
            kwargs = {
                'manifest': manifest,
                'chunk': chunk,
                'original': 'Avi',
                'translated': 'Avi',
                'item': item,
            }
            with_cache = batch_mod.is_manifest_static_non_chinese_item(
                **kwargs,
                file_read_cache=batch_mod.NonChineseFileReadCache(),
            )
            without_cache = batch_mod.is_manifest_static_non_chinese_item(**kwargs)
            via_allow = batch_mod.allow_non_chinese_batch_translation(
                manifest,
                chunk,
                'Avi',
                'Avi',
                item=item,
            )

            self.assertTrue(with_cache)
            self.assertEqual(with_cache, without_cache)
            self.assertEqual(via_allow, without_cache)

    def test_player_name_comparison_uses_cached_reads_without_extra_tl_opens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir).resolve()
            source_path = base_dir / 'game' / 'script.rpy'
            source_path.parent.mkdir(parents=True)
            source_path.write_text('if Main == _("Herbert"):\n    pass\n', encoding='utf-8')
            tl_dir = base_dir / 'game' / 'tl' / 'schinese'
            tl_dir.mkdir(parents=True)
            tl_path = tl_dir / 'script.rpy'
            tl_path.write_text('# game/script.rpy:1\n    "Herbert"\n', encoding='utf-8')
            manifest = {
                'base_dir': str(base_dir),
                'tl_dir': str(tl_dir),
                'non_chinese_rules': copy.deepcopy(batch_non_chinese_rules.DEFAULT_NON_CHINESE_RULES),
            }
            chunk = {'file_rel_path': 'script.rpy'}
            item = {'line_number': 2}
            cache = batch_mod.NonChineseFileReadCache()
            open_paths = []
            real_open = open

            def counting_open(path, *args, **kwargs):
                open_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with mock.patch('builtins.open', side_effect=counting_open):
                allowed = batch_mod.is_manifest_player_name_comparison_item(
                    manifest,
                    chunk,
                    'Herbert',
                    item,
                    file_read_cache=cache,
                )
                cached_tl_reads = cache.read_lines(str(tl_path))
                cached_source_reads = cache.read_line(str(source_path), 1)

            self.assertTrue(allowed)
            self.assertEqual(len(open_paths), 2)
            self.assertEqual(len(cached_tl_reads), 2)
            self.assertIn('Main', cached_source_reads)

    def test_read_line_and_read_lines_share_same_path_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tl_path = Path(tmpdir).resolve() / 'shared.rpy'
            tl_path.write_text('line one\nline two\n', encoding='utf-8')
            cache = batch_mod.NonChineseFileReadCache()
            open_paths = []
            real_open = open

            def counting_open(path, *args, **kwargs):
                open_paths.append(str(path))
                return real_open(path, *args, **kwargs)

            with mock.patch('builtins.open', side_effect=counting_open):
                self.assertEqual(cache.read_line(str(tl_path), 1), 'line one\n')
                self.assertEqual(cache.read_lines(str(tl_path))[1], 'line two\n')

            self.assertEqual(len(open_paths), 1)


if __name__ == '__main__':
    unittest.main()
