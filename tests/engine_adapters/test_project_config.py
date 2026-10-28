"""Explicit engine configuration, audited windows and GUI invalidation evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from engine_adapters.project_config import (project_engine, read_mv_project_config,
                                            read_project_configuration, require_engine_operation, mv_material_revision,
                                            detect_project_engine_mismatch, ProjectEngineMismatchError)
from tests.support.mv_project import make_mv_project


class ProjectConfigTests(unittest.TestCase):
    def test_mv_asset_roots_admit_only_ancestors_resolving_to_the_same_content(self):
        from engine_adapters.project_config import mv_project_asset_roots

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            project = root / 'Game'
            content = project / 'work' / 'www'
            direct = root / 'direct-content'
            for path in (content, direct):
                (path / 'data').mkdir(parents=True)
                (path / 'js').mkdir()
                (path / 'js/rpg_core.js').write_text('// Original inert MV marker.\n')
            (project / 'original').mkdir()
            self.assertEqual(mv_project_asset_roots(content),
                             (content, project / 'work', project))
            self.assertEqual(mv_project_asset_roots(direct), (direct,))
            # A different content root in the package takes precedence over www.
            (project / 'work' / 'data').mkdir()
            (project / 'work' / 'js').mkdir()
            (project / 'work' / 'js/rpg_core.js').write_text('// Other inert MV marker.\n')
            self.assertEqual(mv_project_asset_roots(content), (content,))

    def test_auto_detection_keeps_full_game_scope_assets_and_unapproved_source_distinct(self):
        from engine_adapters.project_setup import mv_project_setup_status
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ('work', 'original'):
                content = root / 'Game' / name / 'www'
                (content / 'data').mkdir(parents=True)
                (content / 'js').mkdir()
                (content / 'js/rpg_core.js').write_text('// Original inert MV marker.\n')
            file = root / 'translator_config.json'
            config = {'game_root': 'Game/work', 'glossary_file': 'glossary.json',
                      'sync': {'macro_setting_file': 'macro_setting.md'},
                      'include_files': [], 'include_prefixes': [], 'unknown': {'keep': True}}
            file.write_text(json.dumps(config), encoding='utf-8')
            before = file.read_bytes()
            resolved = read_project_configuration(file)
            self.assertEqual(resolved['engine'], 'rpgmaker_mv')
            self.assertEqual(Path(resolved['game_root']), root / 'Game/work/www')
            self.assertEqual(Path(resolved['glossary_file']), root / 'Game/work/glossary.json')
            self.assertEqual(Path(resolved['sync']['macro_setting_file']), root / 'Game/work/macro_setting.md')
            self.assertNotIn('rpgmaker_mv', resolved)
            self.assertEqual(resolved['include_files'], [])
            self.assertEqual(resolved['unknown'], config['unknown'])
            setup = mv_project_setup_status(resolved, file)
            self.assertEqual(setup['scope'], 'full_project')
            self.assertEqual(Path(setup['source_suggestion']), root / 'Game/original/www')
            self.assertEqual(setup['missing'], ['source_root', 'support_window_file'])
            self.assertEqual(file.read_bytes(), before)
            self.assertEqual(config['sync']['macro_setting_file'], 'macro_setting.md')

    def test_renpy_selection_reports_mv_content_root_without_changing_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file = root / 'translator_config.json'
            for packaged in (False, True):
                target = root / ('packaged' if packaged else 'direct')
                content = target / 'www' if packaged else target
                (content / 'data').mkdir(parents=True)
                (content / 'js').mkdir()
                (content / 'js/rpg_core.js').write_text('// Original inert MV marker.\n')
                for explicit in (False, True):
                    config = {'game_root': str(target.relative_to(root))}
                    if explicit:
                        config['engine'] = 'renpy'
                    file.write_text(json.dumps(config), encoding='utf-8')
                    before = file.read_bytes()
                    with self.subTest(packaged=packaged, explicit=explicit):
                        issue = detect_project_engine_mismatch(config, file)
                        if not explicit:
                            self.assertIsNone(issue)
                            resolved = read_project_configuration(file)
                            self.assertEqual(resolved['engine'], 'rpgmaker_mv')
                            self.assertEqual(resolved['game_root'], str(content))
                            self.assertEqual(file.read_bytes(), before)
                            continue
                        self.assertEqual(issue['details']['detected_engine'], 'rpgmaker_mv')
                        self.assertEqual(issue['details']['content_root'], str(content))
                        self.assertIn('support window', issue['message'])
                        self.assertEqual(file.read_bytes(), before)
                        for operation in ('doctor', 'sync-status', 'sync-cancel', 'usage-report',
                                          'review-index-build', 'review-index-status', 'quality-report'):
                            require_engine_operation(config, operation, config_file=file)
                        config['engine'] = 'rpgmaker_mv'
                        self.assertIsNone(detect_project_engine_mismatch(config, file))

    def test_detection_requires_data_and_core_and_never_treats_mz_as_mv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            config = {'game_root': str(root)}
            file = root / 'translator_config.json'
            self.assertIsNone(detect_project_engine_mismatch(config, file))
            (root / 'js').mkdir()
            (root / 'js/rpg_core.js').write_text('// Original inert marker.\n')
            self.assertIsNone(detect_project_engine_mismatch(config, file))
            (root / 'js/rpg_core.js').unlink()
            (root / 'data').mkdir()
            self.assertIsNone(detect_project_engine_mismatch(config, file))
            (root / 'js/rmmz_core.js').write_text('// Original inert MZ marker.\n')
            issue = detect_project_engine_mismatch(config, file)
            self.assertEqual(issue['details']['detected_engine'], 'rpgmaker_mz')
            self.assertIn('MZ is unsupported', issue['message'])
            with self.assertRaises(ProjectEngineMismatchError):
                require_engine_operation(config, 'prepare', config_file=file)

    def test_default_and_invalid_engine_are_distinct(self):
        self.assertEqual(project_engine({}), 'renpy')
        for value in ('', 'rpgmaker_mz', None, False, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project_engine({'engine': value})

    def test_routes_refuse_unsupported_mv_operations(self):
        for operation in ('build', 'work-export', 'generate-template', 'prepare', 'legacy-sync',
                          'build-keywords', 'preview-revisions', 'project-analysis-build-structure'):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'durable Sync'):
                require_engine_operation({'engine': 'rpgmaker_mv'}, operation)
        for operation in ('sync-start', 'sync-status', 'sync-resume', 'sync-derive',
                          'sync-cancel', 'check', 'apply', 'doctor', 'coverage-status',
                          'sync-keywords', 'export-keywords', 'merge-keywords-to-glossary',
                          'bootstrap-rag', 'bootstrap-source-index'):
            require_engine_operation({'engine': 'rpgmaker_mv'}, operation)

    def test_malformed_configuration_is_not_interpreted_as_default_renpy(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'translator_config.json'
            for content in ('[]', '{"engine":', '{"engine":"renpy","engine":"rpgmaker_mv"}',
                            '{"engine":"rpgmaker_mv","metadata":NaN}'):
                with self.subTest(content=content), self.assertRaises(ValueError):
                    file.write_text(content, encoding='utf-8')
                    read_project_configuration(file)

    def test_relative_roots_window_schema_and_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            file, config, _positions = make_mv_project(root)
            config['game_root'] = 'target'
            config['rpgmaker_mv'] = {'source_root': 'source', 'support_window_file': 'approved-window.json'}
            settings = read_mv_project_config(config, file)
            self.assertEqual(settings.target_root, root / 'target')
            self.assertEqual(settings.source_root, root / 'source')
            config['rpgmaker_mv']['reviewed_history_file'] = 'reviewed.jsonl'
            self.assertEqual(read_mv_project_config(config, file).reviewed_history_file, root / 'reviewed.jsonl')
            config['rpgmaker_mv']['reviewed_history_file'] = False
            with self.assertRaisesRegex(ValueError, 'reviewed_history_file'):
                read_mv_project_config(config, file)
            config['rpgmaker_mv']['reviewed_history_file'] = ''
            revision = mv_material_revision(config, file)
            path = root / 'source/data/Map001.json'
            path.write_bytes(path.read_bytes() + b' ')
            self.assertNotEqual(revision, mv_material_revision(config, file))
            original = json.loads((root / 'approved-window.json').read_text())
            for key, value in (('schema_version', True), ('message_styles', 'yes'), ('material_sha256', [])):
                with self.subTest(key=key):
                    changed = copy.deepcopy(original)
                    changed[key] = value
                    (root / 'approved-window.json').write_text(json.dumps(changed), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'schema'):
                        read_mv_project_config(config, file)


if __name__ == '__main__':
    unittest.main()
